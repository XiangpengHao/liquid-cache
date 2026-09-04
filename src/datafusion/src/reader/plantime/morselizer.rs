use std::{collections::VecDeque, fmt, future::Future, sync::Arc};

use arrow_schema::SchemaRef;
use datafusion::{
    common::{exec_err, internal_err},
    datasource::{
        listing::{FileRange, PartitionedFile},
        physical_plan::{
            ParquetFileMetrics, ParquetFileReaderFactory,
            parquet::{
                BloomFilterStatistics, PagePruningAccessPlanFilter, ParquetAccessPlan,
                RowGroupAccessPlanFilter,
            },
        },
        table_schema::TableSchema,
    },
    error::Result,
    execution::object_store::ObjectStoreUrl,
    physical_expr::{
        DynamicFilterTracking, PhysicalExpr, PhysicalExprSimplifier, projection::ProjectionExprs,
        utils::reassign_expr_columns,
    },
    physical_expr_adapter::{PhysicalExprAdapterFactory, replace_columns_with_literals},
    physical_optimizer::pruning::{FilePruner, PruningPredicate, build_pruning_predicate},
    physical_plan::metrics::{Count, ExecutionPlanMetricsSet, MetricBuilder},
};
#[cfg(test)]
use datafusion_datasource::morsel::Morsel;
use datafusion_datasource::morsel::{MorselPlan, MorselPlanner, Morselizer};
use futures::{FutureExt, future::BoxFuture};
use log::debug;
use parquet::{
    arrow::{
        ParquetRecordBatchStreamBuilder, ProjectionMask,
        arrow_reader::{ArrowReaderMetadata, ArrowReaderOptions, RowSelection},
        async_reader::AsyncFileReader,
        parquet_column,
    },
    errors::ParquetError,
    file::metadata::PageIndexPolicy,
};

use crate::{
    cache::{
        BatchID, ColumnLineages, InsertArrowArrayError, LiquidCacheParquetRef, ParquetFileIdentity,
        PrefetchOutcome, RowGroupSnapshots,
    },
    reader::{
        plantime::row_filter::build_row_filter,
        runtime::{
            LiquidRowGroupPlanner, apply_predicates, build_projection_schema, get_root_column_ids,
            take_next_batch,
        },
    },
    utils::row_selector_to_boolean_buffer,
};
#[cfg(test)]
use liquid_cache::cache::{CachedBatchType, EntryID, LiquidCache};

pub(crate) struct LiquidMorselizer {
    pub(crate) partition_index: usize,
    pub(crate) projection: ProjectionExprs,
    pub(crate) batch_size: usize,
    pub(crate) predicate: Option<Arc<dyn PhysicalExpr>>,
    pub(crate) table_schema: TableSchema,
    pub(crate) metrics: ExecutionPlanMetricsSet,
    pub(crate) parquet_file_reader_factory: Arc<dyn ParquetFileReaderFactory>,
    pub(crate) object_store_url: ObjectStoreUrl,
    pub(crate) reorder_filters: bool,
    pub(crate) liquid_cache: LiquidCacheParquetRef,
    pub(crate) expr_adapter_factory: Arc<dyn PhysicalExprAdapterFactory>,
    pub(crate) span: Option<Arc<fastrace::Span>>,
    pub(crate) lineages: Arc<ColumnLineages>,
    pub(crate) prefetch: bool,
}

type ParquetInput = Box<dyn AsyncFileReader>;

#[derive(Clone)]
pub(crate) struct LiquidFileReaderFactory {
    pub(crate) factory: Arc<dyn ParquetFileReaderFactory>,
    pub(crate) partition_index: usize,
    pub(crate) partitioned_file: PartitionedFile,
    pub(crate) metadata_size_hint: Option<usize>,
    pub(crate) metrics: ExecutionPlanMetricsSet,
}

impl LiquidFileReaderFactory {
    pub(crate) fn create(&self) -> parquet::errors::Result<ParquetInput> {
        let reader = self
            .factory
            .create_reader(
                self.partition_index,
                self.partitioned_file.clone(),
                self.metadata_size_hint,
                &self.metrics,
            )
            .map_err(|error| ParquetError::External(Box::new(error)))?;
        Ok(reader)
    }
}

impl fmt::Debug for LiquidMorselizer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("LiquidMorselizer")
            .field("partition_index", &self.partition_index)
            .field("batch_size", &self.batch_size)
            .finish_non_exhaustive()
    }
}

impl Morselizer for LiquidMorselizer {
    fn plan_file(&self, partitioned_file: PartitionedFile) -> Result<Box<dyn MorselPlanner>> {
        let file_range = partitioned_file.range.clone();
        let access_plan = partitioned_file.extensions.get_arc::<ParquetAccessPlan>();
        let file_name = partitioned_file.object_meta.location.to_string();
        let metrics = LiquidFileMetrics::new(self.partition_index, &file_name, &self.metrics);
        let metadata_size_hint = partitioned_file.metadata_size_hint;
        let file_identity = ParquetFileIdentity::new(
            self.object_store_url.clone(),
            partitioned_file.object_meta.location.to_string(),
        );
        let reader_factory = Arc::new(LiquidFileReaderFactory {
            factory: Arc::clone(&self.parquet_file_reader_factory),
            partition_index: self.partition_index,
            partitioned_file: partitioned_file.clone(),
            metadata_size_hint,
            metrics: self.metrics.clone(),
        });

        let logical_file_schema = Arc::clone(self.table_schema.file_schema());
        let output_schema = Arc::new(
            self.projection
                .project_schema(self.table_schema.table_schema())?,
        );
        let mut projection = self.projection.clone();
        let mut predicate = self.predicate.clone();
        let mut literal_columns = std::collections::HashMap::new();
        for (field, value) in self
            .table_schema
            .table_partition_cols()
            .iter()
            .zip(&partitioned_file.partition_values)
        {
            literal_columns.insert(field.name().clone(), value.clone());
        }
        if !literal_columns.is_empty() {
            projection = projection.try_map_exprs(|expr| {
                replace_columns_with_literals(Arc::clone(&expr), &literal_columns)
            })?;
            predicate = predicate
                .map(|predicate| replace_columns_with_literals(predicate, &literal_columns))
                .transpose()?;
        }

        let file_pruner = predicate
            .as_ref()
            .filter(|predicate| {
                DynamicFilterTracking::classify(predicate).contains_dynamic_filter()
                    || partitioned_file.has_statistics()
            })
            .and_then(|predicate| {
                FilePruner::try_new(
                    Arc::clone(predicate),
                    &logical_file_schema,
                    &partitioned_file,
                    metrics.predicate_creation_errors.clone(),
                )
            });
        let span = self.span.as_ref().map(|span| {
            Arc::new(fastrace::Span::enter_with_parent(
                format!("file_{file_name}"),
                span,
            ))
        });

        Ok(Box::new(LiquidFilePlanner {
            state: LiquidOpenState::PruneFile(Box::new(PreparedLiquidOpen {
                file_range,
                access_plan,
                file_name,
                metrics,
                file_pruner,
                reader_factory,
                batch_size: self.batch_size,
                logical_file_schema,
                output_schema,
                projection,
                predicate,
                reorder_filters: self.reorder_filters,
                liquid_cache: self.liquid_cache.clone(),
                expr_adapter_factory: Arc::clone(&self.expr_adapter_factory),
                file_identity,
                span,
                lineages: Arc::clone(&self.lineages),
                prefetch: self.prefetch,
            })),
        }))
    }
}

#[derive(Clone)]
pub(crate) struct LiquidFileMetrics {
    pub(crate) file_metrics: ParquetFileMetrics,
    pub(crate) predicate_creation_errors: Count,
    pub(crate) batches_prefetched: Count,
    pub(crate) prefetch_skipped: Count,
}

impl LiquidFileMetrics {
    pub(crate) fn new(
        partition_index: usize,
        file_name: &str,
        metrics: &ExecutionPlanMetricsSet,
    ) -> Self {
        Self {
            file_metrics: ParquetFileMetrics::new(partition_index, file_name, metrics),
            predicate_creation_errors: MetricBuilder::new(metrics)
                .global_counter("num_predicate_creation_errors"),
            batches_prefetched: MetricBuilder::new(metrics)
                .counter("batches_prefetched", partition_index),
            prefetch_skipped: MetricBuilder::new(metrics)
                .counter("prefetch_skipped", partition_index),
        }
    }
}

struct PreparedLiquidOpen {
    file_range: Option<FileRange>,
    access_plan: Option<Arc<ParquetAccessPlan>>,
    file_name: String,
    metrics: LiquidFileMetrics,
    file_pruner: Option<FilePruner>,
    reader_factory: Arc<LiquidFileReaderFactory>,
    batch_size: usize,
    logical_file_schema: SchemaRef,
    output_schema: SchemaRef,
    projection: ProjectionExprs,
    predicate: Option<Arc<dyn PhysicalExpr>>,
    reorder_filters: bool,
    liquid_cache: LiquidCacheParquetRef,
    expr_adapter_factory: Arc<dyn PhysicalExprAdapterFactory>,
    file_identity: ParquetFileIdentity,
    span: Option<Arc<fastrace::Span>>,
    lineages: Arc<ColumnLineages>,
    prefetch: bool,
}

struct MetadataLoadedLiquidOpen {
    prepared: Box<PreparedLiquidOpen>,
    reader_metadata: ArrowReaderMetadata,
    options: ArrowReaderOptions,
}

struct PreparedRowGroups {
    context: RowGroupPlanningContext,
    row_groups: RowGroupAccessPlanFilter,
}

struct RowGroupPlanningContext {
    prepared: Box<PreparedLiquidOpen>,
    reader_metadata: ArrowReaderMetadata,
    physical_file_schema: SchemaRef,
    cache_full_schema: SchemaRef,
    builder: ParquetRecordBatchStreamBuilder<ParquetInput>,
    projection_mask: ProjectionMask,
    row_filter: Option<super::LiquidRowFilter>,
    pruning_predicate: Option<Arc<PruningPredicate>>,
    page_pruning_predicate: Option<Arc<PagePruningAccessPlanFilter>>,
}

struct BloomFiltersLoadedLiquidOpen {
    prepared: PreparedRowGroups,
    bloom_filters: Vec<BloomFilterStatistics>,
}

struct PlannedRowGroups {
    context: RowGroupPlanningContext,
    access_plan: ParquetAccessPlan,
}

enum LiquidOpenState {
    PruneFile(Box<PreparedLiquidOpen>),
    LoadMetadata(BoxFuture<'static, Result<MetadataLoadedLiquidOpen>>),
    PrepareAndPruneByStats(Box<MetadataLoadedLiquidOpen>),
    LoadBloomFilters(BoxFuture<'static, Result<BloomFiltersLoadedLiquidOpen>>),
    PruneBloomAndPages(Box<BloomFiltersLoadedLiquidOpen>),
    PlanRowGroups(Box<PlannedRowGroups>),
    Done,
}

impl fmt::Debug for LiquidOpenState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::PruneFile(_) => "PruneFile",
            Self::LoadMetadata(_) => "LoadMetadata",
            Self::PrepareAndPruneByStats(_) => "PrepareAndPruneByStats",
            Self::LoadBloomFilters(_) => "LoadBloomFilters",
            Self::PruneBloomAndPages(_) => "PruneBloomAndPages",
            Self::PlanRowGroups(_) => "PlanRowGroups",
            Self::Done => "Done",
        })
    }
}

impl LiquidOpenState {
    fn transition(self) -> Result<Self> {
        match self {
            Self::PruneFile(mut prepared) => {
                if let Some(file_pruner) = &mut prepared.file_pruner
                    && file_pruner.should_prune()?
                {
                    prepared
                        .metrics
                        .file_metrics
                        .files_ranges_pruned_statistics
                        .add_pruned(1);
                    return Ok(Self::Done);
                }

                prepared
                    .metrics
                    .file_metrics
                    .files_ranges_pruned_statistics
                    .add_matched(1);
                Ok(Self::LoadMetadata(
                    async move {
                        let options = ArrowReaderOptions::new()
                            .with_page_index_policy(PageIndexPolicy::Required);
                        let metadata_load_time =
                            prepared.metrics.file_metrics.metadata_load_time.clone();
                        let mut timer = metadata_load_time.timer();
                        let mut reader = prepared.reader_factory.create()?;
                        let reader_metadata =
                            ArrowReaderMetadata::load_async(&mut reader, options.clone()).await?;
                        timer.stop();
                        Ok(MetadataLoadedLiquidOpen {
                            prepared,
                            reader_metadata,
                            options,
                        })
                    }
                    .boxed(),
                ))
            }
            Self::LoadMetadata(future) => Ok(Self::LoadMetadata(future)),
            Self::PrepareAndPruneByStats(loaded) => prepare_and_prune_by_stats(*loaded),
            Self::LoadBloomFilters(future) => Ok(Self::LoadBloomFilters(future)),
            Self::PruneBloomAndPages(loaded) => {
                let mut prepared = loaded.prepared;
                let predicate = prepared
                    .context
                    .pruning_predicate
                    .as_deref()
                    .expect("bloom filters are loaded only with a pruning predicate");
                prepared.row_groups.prune_by_bloom_filters(
                    predicate,
                    &prepared.context.prepared.metrics.file_metrics,
                    &loaded.bloom_filters,
                );
                Ok(Self::PlanRowGroups(Box::new(prune_pages(prepared))))
            }
            Self::PlanRowGroups(planned) => Ok(Self::PlanRowGroups(planned)),
            Self::Done => Ok(Self::Done),
        }
    }
}

fn prepare_and_prune_by_stats(mut loaded: MetadataLoadedLiquidOpen) -> Result<LiquidOpenState> {
    let metadata_load_time = loaded
        .prepared
        .metrics
        .file_metrics
        .metadata_load_time
        .clone();
    let mut metadata_timer = metadata_load_time.timer();
    let physical_file_schema = Arc::clone(loaded.reader_metadata.schema());
    let cache_full_schema = Arc::clone(&physical_file_schema);
    loaded.options = loaded
        .options
        .with_schema(Arc::clone(&physical_file_schema));
    loaded.reader_metadata = ArrowReaderMetadata::try_new(
        Arc::clone(loaded.reader_metadata.metadata()),
        loaded.options,
    )?;
    let rewriter = loaded.prepared.expr_adapter_factory.create(
        Arc::clone(&loaded.prepared.logical_file_schema),
        Arc::clone(&physical_file_schema),
    )?;
    let simplifier = PhysicalExprSimplifier::new(&physical_file_schema);
    loaded.prepared.predicate = loaded
        .prepared
        .predicate
        .take()
        .map(|predicate| simplifier.simplify(rewriter.rewrite(predicate)?))
        .transpose()?;
    loaded.prepared.projection = loaded
        .prepared
        .projection
        .try_map_exprs(|expr| simplifier.simplify(rewriter.rewrite(expr)?))?;

    let (pruning_predicate, page_pruning_predicate) = build_pruning_predicates(
        loaded.prepared.predicate.as_ref(),
        &physical_file_schema,
        &loaded.prepared.metrics.predicate_creation_errors,
    );
    metadata_timer.stop();
    let builder = ParquetRecordBatchStreamBuilder::new_with_metadata(
        loaded.prepared.reader_factory.create()?,
        loaded.reader_metadata.clone(),
    );
    let projection_mask = ProjectionMask::roots(
        builder.parquet_schema(),
        loaded.prepared.projection.column_indices(),
    );
    let row_filter =
        loaded.prepared.predicate.as_ref().and_then(|predicate| {
            match build_row_filter(
                predicate,
                &physical_file_schema,
                loaded.reader_metadata.metadata(),
                loaded.prepared.reorder_filters,
                &loaded.prepared.metrics.file_metrics,
            ) {
                Ok(filter) => filter,
                Err(error) => {
                    debug!(
                        "Ignoring error building row filter for '{:?}': {error:?}",
                        loaded.prepared.predicate
                    );
                    None
                }
            }
        });

    let metadata = builder.metadata();
    let row_group_metadata = metadata.row_groups();
    let access_plan = create_initial_plan(
        &loaded.prepared.file_name,
        loaded.prepared.access_plan.take(),
        row_group_metadata.len(),
    )?;
    let mut row_groups = RowGroupAccessPlanFilter::new(access_plan);
    if let Some(range) = &loaded.prepared.file_range {
        row_groups.prune_by_range(row_group_metadata, range);
    }
    if let Some(predicate) = pruning_predicate.as_deref() {
        row_groups.prune_by_statistics(
            &physical_file_schema,
            builder.parquet_schema(),
            row_group_metadata,
            predicate,
            &loaded.prepared.metrics.file_metrics,
        );
    }

    let prepared = PreparedRowGroups {
        context: RowGroupPlanningContext {
            prepared: loaded.prepared,
            reader_metadata: loaded.reader_metadata,
            physical_file_schema,
            cache_full_schema,
            builder,
            projection_mask,
            row_filter,
            pruning_predicate,
            page_pruning_predicate,
        },
        row_groups,
    };
    if prepared.context.pruning_predicate.is_some() && !prepared.row_groups.is_empty() {
        Ok(LiquidOpenState::LoadBloomFilters(
            async move {
                let mut prepared = prepared;
                let predicate = Arc::clone(
                    prepared
                        .context
                        .pruning_predicate
                        .as_ref()
                        .expect("pruning predicate was checked before scheduling bloom I/O"),
                );
                let bloom_filters = load_bloom_filters(
                    &mut prepared.context.builder,
                    predicate.as_ref(),
                    &prepared.context.prepared.metrics.file_metrics,
                    &prepared.row_groups,
                )
                .await;
                Ok(BloomFiltersLoadedLiquidOpen {
                    prepared,
                    bloom_filters,
                })
            }
            .boxed(),
        ))
    } else {
        Ok(LiquidOpenState::PlanRowGroups(Box::new(prune_pages(
            prepared,
        ))))
    }
}

fn prune_pages(prepared: PreparedRowGroups) -> PlannedRowGroups {
    let PreparedRowGroups {
        context,
        row_groups,
    } = prepared;
    let mut access_plan = row_groups.build();
    if !access_plan.is_empty()
        && let Some(predicate) = &context.page_pruning_predicate
    {
        access_plan = predicate.prune_plan_with_page_index(
            access_plan,
            &context.physical_file_schema,
            context.builder.parquet_schema(),
            context.builder.metadata().as_ref(),
            &context.prepared.metrics.file_metrics,
        );
    }
    PlannedRowGroups {
        context,
        access_plan,
    }
}

struct LiquidFilePlanner {
    state: LiquidOpenState,
}

impl fmt::Debug for LiquidFilePlanner {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("LiquidFilePlanner")
            .field(&self.state)
            .finish()
    }
}

impl LiquidFilePlanner {
    fn schedule_io<F>(future: F) -> MorselPlan
    where
        F: Future<Output = Result<LiquidOpenState>> + Send + 'static,
    {
        let future = async move {
            let state = future.await?;
            Ok(Box::new(Self { state }) as Box<dyn MorselPlanner>)
        };
        MorselPlan::new().with_pending_planner(future)
    }
}

impl MorselPlanner for LiquidFilePlanner {
    fn plan(self: Box<Self>) -> Result<Option<MorselPlan>> {
        let state = self.state.transition()?;
        match state {
            LiquidOpenState::LoadMetadata(future) => Ok(Some(Self::schedule_io(async move {
                Ok(LiquidOpenState::PrepareAndPruneByStats(Box::new(
                    future.await?,
                )))
            }))),
            LiquidOpenState::LoadBloomFilters(future) => Ok(Some(Self::schedule_io(async move {
                Ok(LiquidOpenState::PruneBloomAndPages(Box::new(future.await?)))
            }))),
            LiquidOpenState::PlanRowGroups(planned) => plan_row_group_morsels(*planned),
            LiquidOpenState::Done => Ok(None),
            cpu_state => Ok(Some(
                MorselPlan::new().with_planners(vec![Box::new(Self { state: cpu_state })]),
            )),
        }
    }
}

fn plan_row_group_morsels(planned: PlannedRowGroups) -> Result<Option<MorselPlan>> {
    let PlannedRowGroups {
        context,
        access_plan,
    } = planned;
    let prefetch = context.prepared.prefetch;
    let cached_file = context
        .prepared
        .liquid_cache
        .register_or_get_file_with_hints(
            context.prepared.file_identity.clone(),
            Arc::clone(&context.cache_full_schema),
            Arc::clone(&context.prepared.lineages),
        );
    let metadata = Arc::clone(context.reader_metadata.metadata());
    let schema_descriptor = metadata.file_metadata().schema_descr();
    let projection_column_ids = get_root_column_ids(schema_descriptor, &context.projection_mask);
    let stream_schema = build_projection_schema(&cached_file.schema(), &projection_column_ids);
    let replace_schema = !stream_schema.eq(&context.prepared.output_schema);
    let projection = context
        .prepared
        .projection
        .try_map_exprs(|expr| reassign_expr_columns(expr, &stream_schema))?;
    let projector = Arc::new(projection.make_projector(&stream_schema)?);
    let row_group_planner = Arc::new(LiquidRowGroupPlanner {
        metadata: Arc::clone(&metadata),
        reader_factory: context.prepared.reader_factory,
        row_filter: context.row_filter,
        cached_file,
        projection: context.projection_mask,
        batch_size: context.prepared.batch_size,
        stream_schema,
        output_schema: Arc::clone(&context.prepared.output_schema),
        projector,
        replace_schema,
        span: context.prepared.span,
        liquid_cache: context.prepared.liquid_cache,
        metrics: context.prepared.metrics.clone(),
    });

    let row_group_indexes = access_plan.row_group_indexes();
    let row_group_metadata = metadata.row_groups();
    let mut selection = access_plan.into_overall_row_selection(row_group_metadata)?;
    let mut queue = VecDeque::with_capacity(row_group_indexes.len());
    for row_group_idx in row_group_indexes {
        let row_count = row_group_metadata[row_group_idx].num_rows() as usize;
        let row_group_selection = selection
            .as_mut()
            .map(|selection| selection.split_off(row_count));
        let row_group_selection = row_group_selection.unwrap_or_else(|| {
            vec![parquet::arrow::arrow_reader::RowSelector::select(row_count)].into()
        });
        if row_group_selection.row_count() > 0 {
            queue.push_back((row_group_idx, row_group_selection));
        }
    }

    if queue.is_empty() {
        return Ok(None);
    }

    let chain = LiquidRowGroupChain {
        planner: row_group_planner,
        queue,
        snapshots: Arc::default(),
        prefetch,
    };
    let plan = if prefetch {
        MorselPlan::new().with_pending_planner(prefetch_future(chain))
    } else {
        MorselPlan::new().with_planners(vec![Box::new(chain)])
    };
    Ok(Some(plan))
}

struct LiquidRowGroupChain {
    planner: Arc<LiquidRowGroupPlanner>,
    queue: VecDeque<(usize, RowSelection)>,
    snapshots: Arc<RowGroupSnapshots>,
    prefetch: bool,
}

impl fmt::Debug for LiquidRowGroupChain {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("LiquidRowGroupChain")
            .field("remaining_row_groups", &self.queue.len())
            .finish_non_exhaustive()
    }
}

impl MorselPlanner for LiquidRowGroupChain {
    fn plan(mut self: Box<Self>) -> Result<Option<MorselPlan>> {
        let (row_group_idx, selection) = self
            .queue
            .pop_front()
            .expect("a row group chain is never empty");
        let snapshots = std::mem::take(&mut self.snapshots);
        let Some(morsel) = self.planner.plan(row_group_idx, Some(selection), snapshots) else {
            return internal_err!("selected row group {row_group_idx} produced no morsel");
        };
        let mut plan = MorselPlan::new().with_morsels(vec![Box::new(morsel)]);
        if self.queue.is_empty() {
            return Ok(Some(plan));
        }

        if !self.prefetch {
            return Ok(Some(plan.with_planners(vec![self])));
        }

        let next_row_group = self.queue.front().unwrap().0;
        let estimate = self.planner.estimated_bytes(next_row_group);
        let headroom = self
            .planner
            .liquid_cache
            .max_memory_bytes()
            .saturating_sub(self.planner.liquid_cache.memory_usage_bytes());
        if headroom >= estimate {
            plan = plan.with_pending_planner(prefetch_future(*self));
        } else {
            self.planner.metrics.prefetch_skipped.add(1);
            plan = plan.with_planners(vec![self]);
        }
        Ok(Some(plan))
    }
}

async fn prefetch_future(chain: LiquidRowGroupChain) -> Result<Box<dyn MorselPlanner>> {
    Ok(Box::new(prefetch_front(chain).await) as Box<dyn MorselPlanner>)
}

async fn prefetch_front(chain: LiquidRowGroupChain) -> LiquidRowGroupChain {
    let (row_group_idx, selection) = chain.queue.front().expect("prefetch chain has work");
    let mut selectors: VecDeque<_> = selection.clone().into();
    let batch_size = chain.planner.cached_file.batch_size();
    let selected_batches = std::iter::from_fn(|| take_next_batch(&mut selectors, batch_size))
        .enumerate()
        .filter_map(|(idx, selection)| {
            let selection = row_selector_to_boolean_buffer(&selection);
            (selection.count_set_bits() > 0).then_some((BatchID::from_raw(idx as u16), selection))
        })
        .collect::<Vec<_>>();
    let estimate = chain.planner.estimated_bytes(*row_group_idx);
    let per_batch_estimate = estimate / selected_batches.len().max(1);
    let snapshots = Arc::clone(&chain.snapshots);
    let mut context = chain
        .planner
        .prefetch_context(*row_group_idx, Arc::clone(&snapshots));

    for (batch_id, input_selection) in selected_batches {
        let mut produced_snapshots = false;
        let predicate_summary = prefetch_columns(
            &context.cached_row_group,
            batch_id,
            &context.predicate_column_ids,
        )
        .await;
        produced_snapshots |= predicate_summary.any_snapshotted;

        if predicate_summary.any_missing {
            match materialize_prefetch_batch(&mut context, batch_id).await {
                Ok(()) => produced_snapshots = true,
                Err(error) => {
                    debug!("Stopping row group {row_group_idx} prefetch: {error}");
                    break;
                }
            }
        }

        let filtered_selection = if let Some(filter) = context.row_filter.as_mut() {
            match apply_predicates(&context.cached_row_group, batch_id, input_selection, filter)
                .await
            {
                Ok(selection) => selection,
                Err(error) => {
                    debug!("Stopping row group {row_group_idx} prefetch: {error}");
                    break;
                }
            }
        } else {
            Some(input_selection)
        };

        if let Some(filtered_selection) = filtered_selection {
            snapshots.insert_selection(batch_id, filtered_selection.clone());

            if filtered_selection.count_set_bits() > 0 {
                let projection_summary = prefetch_columns(
                    &context.cached_row_group,
                    batch_id,
                    &context.projection_column_ids,
                )
                .await;
                produced_snapshots |= projection_summary.any_snapshotted;
                if projection_summary.any_missing {
                    match materialize_prefetch_batch(&mut context, batch_id).await {
                        Ok(()) => produced_snapshots = true,
                        Err(error) => {
                            debug!("Stopping row group {row_group_idx} prefetch: {error}");
                            break;
                        }
                    }
                }
            }
        }

        if produced_snapshots {
            chain.planner.metrics.batches_prefetched.add(1);
        }
        let headroom = chain
            .planner
            .liquid_cache
            .max_memory_bytes()
            .saturating_sub(chain.planner.liquid_cache.memory_usage_bytes());
        if headroom < per_batch_estimate {
            break;
        }
    }

    chain
}

struct PrefetchColumnsSummary {
    any_missing: bool,
    any_snapshotted: bool,
}

async fn prefetch_columns(
    row_group: &crate::cache::CachedRowGroupRef,
    batch_id: BatchID,
    column_ids: &[usize],
) -> PrefetchColumnsSummary {
    let mut summary = PrefetchColumnsSummary {
        any_missing: false,
        any_snapshotted: false,
    };
    for column_id in column_ids {
        let column = row_group.get_column(*column_id as u64).unwrap();
        match column.prefetch_snapshot(batch_id).await {
            PrefetchOutcome::Snapshotted => summary.any_snapshotted = true,
            PrefetchOutcome::Missing => summary.any_missing = true,
            PrefetchOutcome::AlreadySnapshotted => {}
        }
    }
    summary
}

async fn materialize_prefetch_batch(
    context: &mut crate::reader::runtime::LiquidRowGroupPrefetchContext,
    batch_id: BatchID,
) -> std::result::Result<(), parquet::errors::ParquetError> {
    let record_batch = context.fallback.fetch_batch(batch_id).await?;
    for (position, column_id) in context.cache_column_ids.iter().enumerate() {
        let column = context
            .cached_row_group
            .get_column(*column_id as u64)
            .unwrap();
        let array = Arc::clone(record_batch.column(position));
        match column.insert(batch_id, Arc::clone(&array)).await {
            Ok(()) | Err(InsertArrowArrayError::AlreadyCached) => {}
            Err(InsertArrowArrayError::CacheFull) => {}
        }
        column.insert_snapshot(batch_id, array);
    }
    Ok(())
}

async fn load_bloom_filters(
    builder: &mut ParquetRecordBatchStreamBuilder<ParquetInput>,
    predicate: &PruningPredicate,
    file_metrics: &ParquetFileMetrics,
    row_groups: &RowGroupAccessPlanFilter,
) -> Vec<BloomFilterStatistics> {
    let mut row_group_bloom_filters =
        vec![BloomFilterStatistics::new(); builder.metadata().num_row_groups()];
    let parquet_columns = predicate
        .literal_columns()
        .into_iter()
        .filter_map(|column_name| {
            let parquet_schema = builder.parquet_schema();
            let (column_idx, _) = parquet_column(parquet_schema, predicate.schema(), &column_name)?;
            let column = parquet_schema.column(column_idx);
            Some((
                column_name,
                column_idx,
                column.physical_type(),
                column.type_length(),
            ))
        })
        .collect::<Vec<_>>();

    for row_group_idx in row_groups.row_group_indexes() {
        let mut bloom_filters = BloomFilterStatistics::with_capacity(parquet_columns.len());
        for (column_name, column_idx, physical_type, type_length) in &parquet_columns {
            let bloom_filter = match builder
                .get_row_group_column_bloom_filter(row_group_idx, *column_idx)
                .await
            {
                Ok(Some(bloom_filter)) => bloom_filter,
                Ok(None) => continue,
                Err(error) => {
                    debug!("Ignoring error reading bloom filter: {error}");
                    file_metrics.predicate_evaluation_errors.add(1);
                    continue;
                }
            };
            bloom_filters.insert(column_name, bloom_filter, *physical_type, *type_length);
        }
        row_group_bloom_filters[row_group_idx] = bloom_filters;
    }

    row_group_bloom_filters
}

fn create_initial_plan(
    file_name: &str,
    access_plan: Option<Arc<ParquetAccessPlan>>,
    row_group_count: usize,
) -> Result<ParquetAccessPlan> {
    if let Some(access_plan) = access_plan {
        let plan_len = access_plan.len();
        if plan_len != row_group_count {
            return exec_err!(
                "Invalid ParquetAccessPlan for {file_name}. Specified {plan_len} row groups, but file has {row_group_count}"
            );
        }
        return Ok(access_plan.as_ref().clone());
    }

    Ok(ParquetAccessPlan::new_all(row_group_count))
}

pub(crate) fn build_pruning_predicates(
    predicate: Option<&Arc<dyn PhysicalExpr>>,
    file_schema: &SchemaRef,
    predicate_creation_errors: &Count,
) -> (
    Option<Arc<PruningPredicate>>,
    Option<Arc<PagePruningAccessPlanFilter>>,
) {
    let Some(predicate) = predicate else {
        return (None, None);
    };
    let pruning_predicate = build_pruning_predicate(
        Arc::clone(predicate),
        file_schema,
        predicate_creation_errors,
    );
    let page_pruning_predicate = build_page_pruning_predicate(predicate, file_schema);
    (pruning_predicate, Some(page_pruning_predicate))
}

pub(crate) fn build_page_pruning_predicate(
    predicate: &Arc<dyn PhysicalExpr>,
    file_schema: &SchemaRef,
) -> Arc<PagePruningAccessPlanFilter> {
    Arc::new(PagePruningAccessPlanFilter::new(
        predicate,
        Arc::clone(file_schema),
    ))
}

#[cfg(test)]
mod tests {
    use std::{
        collections::VecDeque,
        fs::File,
        sync::atomic::{AtomicUsize, Ordering},
    };

    use arrow::{
        array::{Array, ArrayRef, Int32Array, RecordBatch},
        datatypes::{DataType, Field, Schema},
    };
    use datafusion::{
        common::ScalarValue,
        datasource::{
            listing::PartitionedFile,
            physical_plan::{
                FileScanConfigBuilder, FileSource, ParquetSource,
                parquet::{CachedParquetFileReaderFactory, DefaultParquetFileReaderFactory},
            },
        },
        execution::{object_store::ObjectStoreUrl, runtime_env::RuntimeEnv},
        logical_expr::Operator,
        physical_expr::{
            PhysicalExpr,
            expressions::{BinaryExpr, Column, Literal},
            projection::ProjectionExprs,
        },
        physical_expr_adapter::DefaultPhysicalExprAdapterFactory,
        physical_plan::metrics::ExecutionPlanMetricsSet,
    };
    use futures::StreamExt;
    use liquid_cache::{
        cache::{AlwaysHydrate, Evict},
        cache_policies::LiquidPolicy,
    };
    use object_store::{ObjectStore, local::LocalFileSystem, path::Path};
    use parquet::arrow::ArrowWriter;

    use crate::{
        cache::{BatchID, CachedFileRef, CachedRowGroupRef, LiquidCacheParquet},
        reader::{LiquidParquetSource, extract_multi_column_or},
    };

    use super::*;

    static NEXT_FILE_ID: AtomicUsize = AtomicUsize::new(0);

    struct PlannedTestFile {
        morsels: Vec<Box<dyn Morsel>>,
        _cache: Arc<LiquidCacheParquet>,
        cached_file: CachedFileRef,
        _tmp_dir: tempfile::TempDir,
    }

    struct TestFilePlanner {
        planner: Box<dyn MorselPlanner>,
        cache: Arc<LiquidCacheParquet>,
        cached_file: CachedFileRef,
        tmp_dir: tempfile::TempDir,
    }

    fn schema() -> SchemaRef {
        Arc::new(Schema::new(vec![
            Field::new("a", DataType::Int32, false),
            Field::new("b", DataType::Int32, false),
        ]))
    }

    fn write_two_row_group_file(path: &std::path::Path, schema: SchemaRef) {
        let file = File::create(path).unwrap();
        let mut writer = ArrowWriter::try_new(file, Arc::clone(&schema), None).unwrap();
        writer
            .write(
                &RecordBatch::try_new(
                    Arc::clone(&schema),
                    vec![
                        Arc::new(Int32Array::from(vec![0, 1, 2, 3])),
                        Arc::new(Int32Array::from(vec![10, 11, 12, 13])),
                    ],
                )
                .unwrap(),
            )
            .unwrap();
        writer.flush().unwrap();
        writer
            .write(
                &RecordBatch::try_new(
                    schema,
                    vec![
                        Arc::new(Int32Array::from(vec![4, 5, 6, 7])),
                        Arc::new(Int32Array::from(vec![14, 15, 16, 17])),
                    ],
                )
                .unwrap(),
            )
            .unwrap();
        writer.close().unwrap();
    }

    fn write_single_row_group_file(path: &std::path::Path, schema: SchemaRef, a: Vec<i32>) {
        let file = File::create(path).unwrap();
        let b = a.iter().map(|value| value + 1000).collect::<Vec<_>>();
        let batch = RecordBatch::try_new(
            Arc::clone(&schema),
            vec![Arc::new(Int32Array::from(a)), Arc::new(Int32Array::from(b))],
        )
        .unwrap();
        let mut writer = ArrowWriter::try_new(file, schema, None).unwrap();
        writer.write(&batch).unwrap();
        writer.close().unwrap();
    }

    async fn drive_planner(planner: Box<dyn MorselPlanner>) -> Vec<Box<dyn Morsel>> {
        let mut planners = VecDeque::from([planner]);
        let mut morsels = Vec::new();
        while let Some(planner) = planners.pop_front() {
            let Some(mut plan) = planner.plan().unwrap() else {
                continue;
            };
            morsels.extend(plan.take_morsels());
            planners.extend(plan.take_ready_planners());
            if let Some(pending) = plan.take_pending_planner() {
                planners.push_back(pending.await.unwrap());
            }
        }
        morsels
    }

    struct PlanOptions {
        max_memory_bytes: usize,
        max_disk_bytes: usize,
        predicate: Option<Arc<dyn PhysicalExpr>>,
        projection_columns: Vec<usize>,
        single_row_group_values: Option<Vec<i32>>,
    }

    impl Default for PlanOptions {
        fn default() -> Self {
            Self {
                max_memory_bytes: usize::MAX,
                max_disk_bytes: usize::MAX,
                predicate: None,
                projection_columns: vec![0, 1],
                single_row_group_values: None,
            }
        }
    }

    async fn create_test_cache(
        path: &std::path::Path,
        max_memory_bytes: usize,
        max_disk_bytes: usize,
    ) -> Arc<LiquidCacheParquet> {
        let store = t4::mount(path.join("liquid_cache.t4")).await.unwrap();
        Arc::new(
            LiquidCacheParquet::new(
                4,
                max_memory_bytes,
                max_disk_bytes,
                store,
                Box::new(LiquidPolicy::new()),
                Box::new(Evict),
                Box::new(AlwaysHydrate::new()),
            )
            .await,
        )
    }

    async fn prepare_test_file(options: PlanOptions) -> TestFilePlanner {
        let schema = schema();
        let tmp_dir = tempfile::tempdir().unwrap();
        let file_id = NEXT_FILE_ID.fetch_add(1, Ordering::Relaxed);
        let file_name = "data.parquet".to_string();
        let parquet_path = tmp_dir.path().join(&file_name);
        if let Some(values) = options.single_row_group_values {
            write_single_row_group_file(&parquet_path, Arc::clone(&schema), values);
        } else {
            write_two_row_group_file(&parquet_path, Arc::clone(&schema));
        }
        let partitioned_file = PartitionedFile::new(
            file_name.clone(),
            std::fs::metadata(&parquet_path).unwrap().len(),
        );
        let object_store = Arc::new(LocalFileSystem::new_with_prefix(tmp_dir.path()).unwrap());
        let cache = create_test_cache(
            tmp_dir.path(),
            options.max_memory_bytes,
            options.max_disk_bytes,
        )
        .await;
        let metrics = ExecutionPlanMetricsSet::new();
        let object_store_url = ObjectStoreUrl::parse(format!("test-{file_id}:///")).unwrap();
        let morselizer = LiquidMorselizer {
            partition_index: 0,
            projection: ProjectionExprs::from_indices(&options.projection_columns, schema.as_ref()),
            batch_size: 4,
            predicate: options.predicate,
            table_schema: TableSchema::from(Arc::clone(&schema)),
            metrics: metrics.clone(),
            parquet_file_reader_factory: Arc::new(DefaultParquetFileReaderFactory::new(
                object_store,
            )),
            object_store_url: object_store_url.clone(),
            reorder_filters: false,
            liquid_cache: cache.clone(),
            expr_adapter_factory: Arc::new(DefaultPhysicalExprAdapterFactory),
            span: None,
            lineages: Arc::default(),
            prefetch: true,
        };
        let cached_file = cache.register_or_get_file(
            ParquetFileIdentity::new(object_store_url, file_name),
            schema,
        );
        TestFilePlanner {
            planner: morselizer.plan_file(partitioned_file).unwrap(),
            cache,
            cached_file,
            tmp_dir,
        }
    }

    async fn plan_test_file(options: PlanOptions) -> PlannedTestFile {
        let prepared = prepare_test_file(options).await;
        let morsels = drive_planner(prepared.planner).await;
        PlannedTestFile {
            morsels,
            _cache: prepared.cache,
            cached_file: prepared.cached_file,
            _tmp_dir: prepared.tmp_dir,
        }
    }

    async fn advance_to_row_group_chain(
        mut planner: Box<dyn MorselPlanner>,
    ) -> Box<dyn MorselPlanner> {
        loop {
            let mut plan = planner.plan().unwrap().expect("file has row groups");
            assert!(plan.take_morsels().is_empty());
            if let Some(ready) = plan.take_ready_planners().pop() {
                planner = ready;
                continue;
            }
            let pending = plan.take_pending_planner().expect("planner has more work");
            planner = pending.await.unwrap();
            if format!("{planner:?}").contains("LiquidRowGroupChain") {
                return planner;
            }
        }
    }

    async fn metadata_cache_hits(cache_limit: usize) -> (Option<usize>, Option<usize>) {
        let schema = schema();
        let tmp_dir = tempfile::tempdir().unwrap();
        let parquet_path = tmp_dir.path().join("data.parquet");
        write_two_row_group_file(&parquet_path, Arc::clone(&schema));
        let file = PartitionedFile::new(
            "data.parquet",
            std::fs::metadata(&parquet_path).unwrap().len(),
        );
        let object_store: Arc<dyn ObjectStore> =
            Arc::new(LocalFileSystem::new_with_prefix(tmp_dir.path()).unwrap());
        let metadata_cache = RuntimeEnv::default()
            .cache_manager
            .get_file_metadata_cache();
        metadata_cache.update_cache_limit(cache_limit);
        let parquet_file_reader_factory = Arc::new(CachedParquetFileReaderFactory::new(
            Arc::clone(&object_store),
            Arc::clone(&metadata_cache),
        ));
        let parquet_source = ParquetSource::new(Arc::clone(&schema))
            .with_parquet_file_reader_factory(parquet_file_reader_factory);
        let liquid_cache = create_test_cache(tmp_dir.path(), usize::MAX, usize::MAX).await;
        let source = LiquidParquetSource::from_parquet_source(parquet_source, liquid_cache);
        let base_config = FileScanConfigBuilder::new(
            ObjectStoreUrl::local_filesystem(),
            Arc::new(source.clone()),
        )
        .with_file(file.clone())
        .build();
        let morselizer = source
            .create_morselizer(object_store, &base_config, 0)
            .unwrap();
        let path = Path::from("data.parquet");

        advance_to_row_group_chain(morselizer.plan_file(file.clone()).unwrap()).await;
        let first_hits = metadata_cache
            .list_entries()
            .get(&path)
            .map(|entry| entry.hits);
        advance_to_row_group_chain(morselizer.plan_file(file).unwrap()).await;
        let second_hits = metadata_cache
            .list_entries()
            .get(&path)
            .map(|entry| entry.hits);
        (first_hits, second_hits)
    }

    #[tokio::test]
    async fn datafusion_metadata_cache_ablation() {
        assert_eq!(metadata_cache_hits(usize::MAX).await, (Some(0), Some(1)));
        assert_eq!(metadata_cache_hits(0).await, (None, None));
    }

    fn gt_expr(column_name: &str, column_index: usize, literal: i32) -> Arc<dyn PhysicalExpr> {
        Arc::new(BinaryExpr::new(
            Arc::new(Column::new(column_name, column_index)),
            Operator::Gt,
            Arc::new(Literal::new(ScalarValue::Int32(Some(literal)))),
        ))
    }

    fn eq_expr(column_name: &str, column_index: usize, literal: i32) -> Arc<dyn PhysicalExpr> {
        Arc::new(BinaryExpr::new(
            Arc::new(Column::new(column_name, column_index)),
            Operator::Eq,
            Arc::new(Literal::new(ScalarValue::Int32(Some(literal)))),
        ))
    }

    #[tokio::test]
    async fn data_cache_is_scoped_to_object_store() {
        let schema = schema();
        let dir_a = tempfile::tempdir().unwrap();
        let dir_b = tempfile::tempdir().unwrap();
        let cache_dir = tempfile::tempdir().unwrap();
        let path_a = dir_a.path().join("data.parquet");
        let path_b = dir_b.path().join("data.parquet");
        write_single_row_group_file(&path_a, Arc::clone(&schema), vec![1, 2]);
        write_single_row_group_file(&path_b, Arc::clone(&schema), vec![10, 20]);

        let cache = create_test_cache(cache_dir.path(), usize::MAX, usize::MAX).await;
        let metrics = ExecutionPlanMetricsSet::new();
        let projection = ProjectionExprs::from_indices(&[0, 1], schema.as_ref());
        let morselizer_a = LiquidMorselizer {
            partition_index: 0,
            projection: projection.clone(),
            batch_size: 4,
            predicate: None,
            table_schema: TableSchema::from(Arc::clone(&schema)),
            metrics: metrics.clone(),
            parquet_file_reader_factory: Arc::new(DefaultParquetFileReaderFactory::new(Arc::new(
                LocalFileSystem::new_with_prefix(dir_a.path()).unwrap(),
            ))),
            object_store_url: ObjectStoreUrl::parse("data-cache-a:///").unwrap(),
            reorder_filters: false,
            liquid_cache: Arc::clone(&cache),
            expr_adapter_factory: Arc::new(DefaultPhysicalExprAdapterFactory),
            span: None,
            lineages: Arc::default(),
            prefetch: true,
        };
        let morselizer_b = LiquidMorselizer {
            partition_index: 0,
            projection,
            batch_size: 4,
            predicate: None,
            table_schema: TableSchema::from(Arc::clone(&schema)),
            metrics,
            parquet_file_reader_factory: Arc::new(DefaultParquetFileReaderFactory::new(Arc::new(
                LocalFileSystem::new_with_prefix(dir_b.path()).unwrap(),
            ))),
            object_store_url: ObjectStoreUrl::parse("data-cache-b:///").unwrap(),
            reorder_filters: false,
            liquid_cache: cache,
            expr_adapter_factory: Arc::new(DefaultPhysicalExprAdapterFactory),
            span: None,
            lineages: Arc::default(),
            prefetch: true,
        };

        let file_a = PartitionedFile::new("data.parquet", std::fs::metadata(path_a).unwrap().len());
        let file_b = PartitionedFile::new("data.parquet", std::fs::metadata(path_b).unwrap().len());
        let rows_a =
            collect_columns(drive_planner(morselizer_a.plan_file(file_a).unwrap()).await).await;
        let rows_b =
            collect_columns(drive_planner(morselizer_b.plan_file(file_b).unwrap()).await).await;

        assert_eq!(rows_a, (vec![1, 2], vec![1001, 1002]));
        assert_eq!(rows_b, (vec![10, 20], vec![1010, 1020]));
    }

    async fn collect_columns(morsels: Vec<Box<dyn Morsel>>) -> (Vec<i32>, Vec<i32>) {
        let mut a = Vec::new();
        let mut b = Vec::new();
        for morsel in morsels {
            let batches = morsel.into_stream().collect::<Vec<_>>().await;
            for batch in batches {
                let batch = batch.unwrap();
                a.extend(
                    batch
                        .column(0)
                        .as_any()
                        .downcast_ref::<Int32Array>()
                        .unwrap()
                        .values(),
                );
                if batch.num_columns() > 1 {
                    b.extend(
                        batch
                            .column(1)
                            .as_any()
                            .downcast_ref::<Int32Array>()
                            .unwrap()
                            .values(),
                    );
                }
            }
        }
        (a, b)
    }

    async fn insert_batches(
        row_group: &CachedRowGroupRef,
        column_id: usize,
        batches: &[(u16, &[i32])],
    ) {
        let column = row_group.get_column(column_id as u64).unwrap();
        for (batch_idx, values) in batches {
            let array: ArrayRef = Arc::new(Int32Array::from(values.to_vec()));
            column
                .insert(BatchID::from_raw(*batch_idx), array)
                .await
                .unwrap();
        }
    }

    async fn contains(row_group: &CachedRowGroupRef, column_id: usize, batch_idx: u16) -> bool {
        row_group
            .get_column(column_id as u64)
            .unwrap()
            .get_arrow_array_test_only(BatchID::from_raw(batch_idx))
            .await
            .is_some()
    }

    fn kind_of(cache: &LiquidCache, id: &EntryID) -> Option<CachedBatchType> {
        let mut kind = None;
        cache.for_each_entry(|entry_id, entry| {
            if entry_id == id {
                kind = Some(CachedBatchType::from(entry));
            }
        });
        kind
    }

    #[tokio::test]
    async fn plans_one_morsel_per_selected_row_group() {
        let all = plan_test_file(PlanOptions {
            ..Default::default()
        })
        .await;
        assert_eq!(all.morsels.len(), 2);
        assert_eq!(
            collect_columns(all.morsels).await.0,
            vec![0, 1, 2, 3, 4, 5, 6, 7]
        );

        let pruned = plan_test_file(PlanOptions {
            predicate: Some(gt_expr("a", 0, 3)),
            ..Default::default()
        })
        .await;
        assert_eq!(pruned.morsels.len(), 1);
        assert_eq!(collect_columns(pruned.morsels).await.0, vec![4, 5, 6, 7]);
    }

    #[tokio::test]
    async fn prefetch_hands_snapshots_to_next_morsel() {
        let file = prepare_test_file(PlanOptions::default()).await;
        let chain = advance_to_row_group_chain(file.planner).await;
        let mut first_plan = chain.plan().unwrap().unwrap();
        let mut morsels = first_plan.take_morsels();
        assert_eq!(morsels.len(), 1);
        let next = first_plan.take_pending_planner().unwrap().await.unwrap();

        let row_group = file.cached_file.create_row_group(1, vec![]);
        for column_id in 0..2 {
            let id = row_group
                .get_column(column_id)
                .unwrap()
                .entry_id(BatchID::from_raw(0))
                .into();
            assert_eq!(
                kind_of(file.cache.storage(), &id),
                Some(CachedBatchType::MemoryArrow)
            );
        }

        morsels.extend(next.plan().unwrap().unwrap().take_morsels());
        assert_eq!(
            collect_columns(morsels).await.0,
            vec![0, 1, 2, 3, 4, 5, 6, 7]
        );
    }

    #[tokio::test]
    async fn prefetched_multi_column_or_uses_snapshots() {
        let predicate: Arc<dyn PhysicalExpr> = Arc::new(BinaryExpr::new(
            eq_expr("a", 0, 3),
            Operator::Or,
            eq_expr("b", 1, 20),
        ));
        assert!(extract_multi_column_or(&predicate).is_some());
        let file = prepare_test_file(PlanOptions {
            predicate: Some(predicate),
            ..Default::default()
        })
        .await;
        file.cache.storage().stats();

        let chain = advance_to_row_group_chain(file.planner).await;
        let morsels = chain.plan().unwrap().unwrap().take_morsels();
        assert_eq!(collect_columns(morsels).await, (vec![3], vec![13]));
        assert_eq!(
            file.cache.storage().stats().runtime.try_read_liquid_calls,
            0
        );
    }

    #[tokio::test]
    async fn prefetch_fetches_absent_predicate_columns() {
        let file = prepare_test_file(PlanOptions {
            predicate: Some(gt_expr("a", 0, 7)),
            projection_columns: vec![1],
            single_row_group_values: Some((0..12).collect()),
            ..Default::default()
        })
        .await;

        let chain = advance_to_row_group_chain(file.planner).await;
        let predicate = file
            .cached_file
            .create_row_group(0, vec![0])
            .get_column(0)
            .unwrap();
        for batch_idx in 0..3 {
            let entry_id = predicate.entry_id(BatchID::from_raw(batch_idx)).into();
            assert_eq!(
                kind_of(file.cache.storage(), &entry_id),
                Some(CachedBatchType::MemoryArrow)
            );
        }

        let morsels = chain.plan().unwrap().unwrap().take_morsels();
        assert_eq!(
            collect_columns(morsels).await.0,
            vec![1008, 1009, 1010, 1011]
        );
    }

    #[tokio::test]
    async fn prefetch_skips_projection_for_filtered_batches() {
        let file = prepare_test_file(PlanOptions {
            predicate: Some(gt_expr("a", 0, 7)),
            projection_columns: vec![1],
            single_row_group_values: Some((0..12).collect()),
            ..Default::default()
        })
        .await;
        let row_group = file.cached_file.create_row_group(0, vec![0]);
        insert_batches(
            &row_group,
            0,
            &[(0, &[0, 1, 2, 3]), (1, &[4, 5, 6, 7]), (2, &[8, 9, 10, 11])],
        )
        .await;
        insert_batches(
            &row_group,
            1,
            &[
                (0, &[1000, 1001, 1002, 1003]),
                (1, &[1004, 1005, 1006, 1007]),
                (2, &[1008, 1009, 1010, 1011]),
            ],
        )
        .await;
        file.cache.flush_data().await.unwrap();

        let chain = advance_to_row_group_chain(file.planner).await;
        let projection = row_group.get_column(1).unwrap();
        for batch_idx in 0..2 {
            let entry_id = projection.entry_id(BatchID::from_raw(batch_idx)).into();
            assert_eq!(
                kind_of(file.cache.storage(), &entry_id),
                Some(CachedBatchType::DiskArrow)
            );
        }
        let surviving_entry = projection.entry_id(BatchID::from_raw(2)).into();
        assert_eq!(
            kind_of(file.cache.storage(), &surviving_entry),
            Some(CachedBatchType::MemoryArrow)
        );

        let morsels = chain.plan().unwrap().unwrap().take_morsels();
        assert_eq!(
            collect_columns(morsels).await.0,
            vec![1008, 1009, 1010, 1011]
        );
    }

    #[tokio::test]
    async fn snapshots_survive_eviction() {
        let file = prepare_test_file(PlanOptions::default()).await;
        let chain = advance_to_row_group_chain(file.planner).await;
        let mut first = chain.plan().unwrap().unwrap();
        let next = first.take_pending_planner().unwrap().await.unwrap();
        let second = next.plan().unwrap().unwrap().take_morsels();
        file.cache.flush_data().await.unwrap();

        let row_group = file.cached_file.create_row_group(1, vec![]);
        assert_eq!(collect_columns(second).await.0, vec![4, 5, 6, 7]);
        for column_id in 0..2 {
            let id = row_group
                .get_column(column_id)
                .unwrap()
                .entry_id(BatchID::from_raw(0))
                .into();
            assert_eq!(
                kind_of(file.cache.storage(), &id),
                Some(CachedBatchType::DiskArrow)
            );
        }
    }

    #[tokio::test]
    async fn headroom_gate_skips_prefetch() {
        let file = prepare_test_file(PlanOptions {
            max_memory_bytes: 1,
            max_disk_bytes: 0,
            ..Default::default()
        })
        .await;
        let chain = advance_to_row_group_chain(file.planner).await;
        let mut first = chain.plan().unwrap().unwrap();
        assert!(first.take_pending_planner().is_none());
        let next = first.take_ready_planners().pop().unwrap();

        let mut morsels = first.take_morsels();
        morsels.extend(next.plan().unwrap().unwrap().take_morsels());
        assert_eq!(
            collect_columns(morsels).await.0,
            vec![0, 1, 2, 3, 4, 5, 6, 7]
        );
    }

    #[tokio::test]
    async fn cache_full_keeps_inserted_batches_and_skips_failed_inserts() {
        let one_array_memory = Arc::new(Int32Array::from(vec![0, 1, 2, 3])).get_array_memory_size();
        let planned = plan_test_file(PlanOptions {
            max_memory_bytes: one_array_memory * 3,
            max_disk_bytes: 0,
            ..Default::default()
        })
        .await;
        let row_group0 = planned.cached_file.create_row_group(0, vec![]);
        let row_group1 = planned.cached_file.create_row_group(1, vec![]);

        let (a, b) = collect_columns(planned.morsels).await;
        assert_eq!(a, vec![0, 1, 2, 3, 4, 5, 6, 7]);
        assert_eq!(b, vec![10, 11, 12, 13, 14, 15, 16, 17]);
        assert!(contains(&row_group0, 0, 0).await);
        assert!(contains(&row_group0, 1, 0).await);
        assert!(contains(&row_group1, 0, 0).await);
        assert!(!contains(&row_group1, 1, 0).await);
    }

    #[tokio::test]
    async fn cache_full_with_filter_keeps_results_correct() {
        let one_array_memory = Arc::new(Int32Array::from(vec![0, 1, 2, 3])).get_array_memory_size();
        let planned = plan_test_file(PlanOptions {
            max_memory_bytes: one_array_memory * 3,
            max_disk_bytes: 0,
            predicate: Some(gt_expr("a", 0, 2)),
            ..Default::default()
        })
        .await;
        let row_group0 = planned.cached_file.create_row_group(0, vec![]);
        let row_group1 = planned.cached_file.create_row_group(1, vec![]);
        let (a, b) = collect_columns(planned.morsels).await;
        assert_eq!(a, vec![3, 4, 5, 6, 7]);
        assert_eq!(b, vec![13, 14, 15, 16, 17]);
        assert!(contains(&row_group0, 0, 0).await);
        assert!(contains(&row_group0, 1, 0).await);
        assert!(contains(&row_group1, 0, 0).await);
        assert!(!contains(&row_group1, 1, 0).await);
    }

    #[tokio::test]
    async fn mid_scan_eviction_recovers() {
        let planned = plan_test_file(PlanOptions {
            max_memory_bytes: 0,
            max_disk_bytes: 0,
            ..Default::default()
        })
        .await;
        let row_group0 = planned.cached_file.create_row_group(0, vec![]);
        let row_group1 = planned.cached_file.create_row_group(1, vec![]);
        let (a, b) = collect_columns(planned.morsels).await;
        assert_eq!(a, vec![0, 1, 2, 3, 4, 5, 6, 7]);
        assert_eq!(b, vec![10, 11, 12, 13, 14, 15, 16, 17]);
        for row_group in [&row_group0, &row_group1] {
            assert!(!contains(row_group, 0, 0).await);
            assert!(!contains(row_group, 1, 0).await);
        }
    }

    #[tokio::test]
    async fn predicate_fallback_uses_predicate_projection() {
        let one_array_memory = Arc::new(Int32Array::from(vec![0, 1, 2, 3])).get_array_memory_size();
        let planned = plan_test_file(PlanOptions {
            max_memory_bytes: one_array_memory * 3,
            max_disk_bytes: 0,
            predicate: Some(gt_expr("b", 1, 12)),
            projection_columns: vec![0],
            ..Default::default()
        })
        .await;
        let row_group0 = planned.cached_file.create_row_group(0, vec![]);
        let row_group1 = planned.cached_file.create_row_group(1, vec![]);
        assert_eq!(
            collect_columns(planned.morsels).await.0,
            vec![3, 4, 5, 6, 7]
        );
        assert!(contains(&row_group0, 0, 0).await);
        assert!(contains(&row_group0, 1, 0).await);
        assert!(contains(&row_group1, 0, 0).await);
        assert!(!contains(&row_group1, 1, 0).await);
    }

    #[tokio::test]
    async fn missing_column_falls_back_to_parquet() {
        let file = prepare_test_file(PlanOptions::default()).await;
        let row_group0 = file.cached_file.create_row_group(0, vec![]);
        let row_group1 = file.cached_file.create_row_group(1, vec![]);
        insert_batches(&row_group0, 0, &[(0, &[0, 1, 2, 3])]).await;
        insert_batches(&row_group1, 0, &[(0, &[4, 5, 6, 7])]).await;

        let (a, b) = collect_columns(drive_planner(file.planner).await).await;
        assert_eq!(a, vec![0, 1, 2, 3, 4, 5, 6, 7]);
        assert_eq!(b, vec![10, 11, 12, 13, 14, 15, 16, 17]);
        assert!(contains(&row_group0, 1, 0).await);
        assert!(contains(&row_group1, 1, 0).await);
    }

    #[tokio::test]
    async fn fallback_stream_advances_across_misses() {
        let parquet_a = vec![
            100, 101, 102, 103, 4, 5, 6, 7, 200, 201, 202, 203, 12, 13, 14, 15,
        ];
        let file = prepare_test_file(PlanOptions {
            projection_columns: vec![0],
            single_row_group_values: Some(parquet_a),
            ..Default::default()
        })
        .await;
        let row_group = file.cached_file.create_row_group(0, vec![]);
        insert_batches(&row_group, 0, &[(0, &[0, 1, 2, 3]), (2, &[8, 9, 10, 11])]).await;

        assert_eq!(
            collect_columns(drive_planner(file.planner).await).await.0,
            (0..16).collect::<Vec<_>>()
        );
        for batch_idx in 0..4 {
            assert!(contains(&row_group, 0, batch_idx).await);
        }
    }

    #[tokio::test]
    async fn source_uses_native_morsel_api() {
        let schema = schema();
        let tmp_dir = tempfile::tempdir().unwrap();
        let parquet_path = tmp_dir.path().join("data.parquet");
        write_two_row_group_file(&parquet_path, Arc::clone(&schema));
        let file = PartitionedFile::new(
            "data.parquet",
            std::fs::metadata(&parquet_path).unwrap().len(),
        );
        let cache = create_test_cache(tmp_dir.path(), usize::MAX, usize::MAX).await;
        let source = LiquidParquetSource::from_parquet_source(
            ParquetSource::new(Arc::clone(&schema)),
            cache,
        );
        let base_config = FileScanConfigBuilder::new(
            ObjectStoreUrl::local_filesystem(),
            Arc::new(source.clone()),
        )
        .with_file(file.clone())
        .build();
        let object_store = Arc::new(LocalFileSystem::new_with_prefix(tmp_dir.path()).unwrap());

        assert!(
            source
                .create_file_opener(object_store.clone(), &base_config, 0)
                .is_err()
        );
        let morselizer = source
            .create_morselizer(object_store, &base_config, 0)
            .unwrap();
        assert!(morselizer.plan_file(file).is_ok());
    }
}
