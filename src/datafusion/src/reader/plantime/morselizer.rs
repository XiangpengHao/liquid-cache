use std::{fmt, future::Future, sync::Arc};

use arrow_schema::SchemaRef;
use datafusion::{
    common::exec_err,
    datasource::{
        listing::{FileRange, PartitionedFile},
        physical_plan::{
            ParquetFileMetrics,
            parquet::{
                BloomFilterStatistics, PagePruningAccessPlanFilter, ParquetAccessPlan,
                RowGroupAccessPlanFilter,
            },
        },
        table_schema::TableSchema,
    },
    error::Result,
    physical_expr::{
        DynamicFilterTracking, PhysicalExpr, PhysicalExprSimplifier, projection::ProjectionExprs,
        utils::reassign_expr_columns,
    },
    physical_expr_adapter::{PhysicalExprAdapterFactory, replace_columns_with_literals},
    physical_optimizer::pruning::{FilePruner, PruningPredicate, build_pruning_predicate},
    physical_plan::metrics::{Count, ExecutionPlanMetricsSet, MetricBuilder},
};
use datafusion_datasource::morsel::{Morsel, MorselPlan, MorselPlanner, Morselizer};
use futures::{FutureExt, future::BoxFuture};
use log::debug;
use parquet::{
    arrow::{
        ParquetRecordBatchStreamBuilder, ProjectionMask,
        arrow_reader::{ArrowReaderMetadata, ArrowReaderOptions},
        parquet_column,
    },
    file::metadata::PageIndexPolicy,
};

use crate::{
    cache::{ColumnSqueezeHints, LiquidCacheParquetRef},
    reader::{
        plantime::row_filter::build_row_filter,
        runtime::{LiquidRowGroupPlanner, build_projection_schema, get_root_column_ids},
    },
};

use super::source::{CachedMetaReaderFactory, ParquetMetadataCacheReader};

pub(crate) struct LiquidMorselizer {
    pub(crate) partition_index: usize,
    pub(crate) projection: ProjectionExprs,
    pub(crate) batch_size: usize,
    pub(crate) predicate: Option<Arc<dyn PhysicalExpr>>,
    pub(crate) table_schema: TableSchema,
    pub(crate) metrics: ExecutionPlanMetricsSet,
    pub(crate) parquet_file_reader_factory: Arc<CachedMetaReaderFactory>,
    pub(crate) reorder_filters: bool,
    pub(crate) liquid_cache: LiquidCacheParquetRef,
    pub(crate) expr_adapter_factory: Arc<dyn PhysicalExprAdapterFactory>,
    pub(crate) span: Option<Arc<fastrace::Span>>,
    pub(crate) squeeze_hints: Arc<ColumnSqueezeHints>,
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
        let file_metrics = ParquetFileMetrics::new(self.partition_index, &file_name, &self.metrics);
        let metadata_size_hint = partitioned_file.metadata_size_hint;
        let file_location = partitioned_file.object_meta.location.to_string();
        let reader = self.parquet_file_reader_factory.create_liquid_reader(
            self.partition_index,
            partitioned_file.clone(),
            metadata_size_hint,
            &self.metrics,
        );

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

        let predicate_creation_errors =
            MetricBuilder::new(&self.metrics).global_counter("num_predicate_creation_errors");
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
                    predicate_creation_errors.clone(),
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
                file_metrics,
                file_pruner,
                reader,
                batch_size: self.batch_size,
                logical_file_schema,
                output_schema,
                projection,
                predicate,
                predicate_creation_errors,
                reorder_filters: self.reorder_filters,
                liquid_cache: self.liquid_cache.clone(),
                expr_adapter_factory: Arc::clone(&self.expr_adapter_factory),
                file_location,
                span,
                squeeze_hints: Arc::clone(&self.squeeze_hints),
            })),
        }))
    }
}

struct PreparedLiquidOpen {
    file_range: Option<FileRange>,
    access_plan: Option<Arc<ParquetAccessPlan>>,
    file_name: String,
    file_metrics: ParquetFileMetrics,
    file_pruner: Option<FilePruner>,
    reader: ParquetMetadataCacheReader,
    batch_size: usize,
    logical_file_schema: SchemaRef,
    output_schema: SchemaRef,
    projection: ProjectionExprs,
    predicate: Option<Arc<dyn PhysicalExpr>>,
    predicate_creation_errors: Count,
    reorder_filters: bool,
    liquid_cache: LiquidCacheParquetRef,
    expr_adapter_factory: Arc<dyn PhysicalExprAdapterFactory>,
    file_location: String,
    span: Option<Arc<fastrace::Span>>,
    squeeze_hints: Arc<ColumnSqueezeHints>,
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
    builder: ParquetRecordBatchStreamBuilder<ParquetMetadataCacheReader>,
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
                        .file_metrics
                        .files_ranges_pruned_statistics
                        .add_pruned(1);
                    return Ok(Self::Done);
                }

                prepared
                    .file_metrics
                    .files_ranges_pruned_statistics
                    .add_matched(1);
                Ok(Self::LoadMetadata(
                    async move {
                        let options = ArrowReaderOptions::new()
                            .with_page_index_policy(PageIndexPolicy::Required);
                        let metadata_load_time = prepared.file_metrics.metadata_load_time.clone();
                        let mut timer = metadata_load_time.timer();
                        let reader_metadata =
                            ArrowReaderMetadata::load_async(&mut prepared.reader, options.clone())
                                .await?;
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
                    &prepared.context.prepared.file_metrics,
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
    let metadata_load_time = loaded.prepared.file_metrics.metadata_load_time.clone();
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
    debug_assert!(
        Arc::strong_count(loaded.reader_metadata.metadata()) > 1,
        "meta data must be cached already"
    );

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
        &loaded.prepared.predicate_creation_errors,
    );
    metadata_timer.stop();
    let builder = ParquetRecordBatchStreamBuilder::new_with_metadata(
        loaded.prepared.reader.clone(),
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
                &loaded.prepared.file_metrics,
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
            &loaded.prepared.file_metrics,
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
                    &prepared.context.prepared.file_metrics,
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
            &context.prepared.file_metrics,
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
    let cached_file = context
        .prepared
        .liquid_cache
        .register_or_get_file_with_hints(
            context.prepared.file_location.clone(),
            Arc::clone(&context.cache_full_schema),
            Arc::clone(&context.prepared.squeeze_hints),
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
    let row_group_planner = LiquidRowGroupPlanner {
        metadata: Arc::clone(&metadata),
        input: context.prepared.reader.clone(),
        row_filter: context.row_filter,
        cached_file,
        projection: context.projection_mask,
        batch_size: context.prepared.batch_size,
        stream_schema,
        output_schema: Arc::clone(&context.prepared.output_schema),
        projector,
        replace_schema,
        span: context.prepared.span,
    };

    let row_group_indexes = access_plan.row_group_indexes();
    let row_group_metadata = metadata.row_groups();
    let mut selection = access_plan.into_overall_row_selection(row_group_metadata)?;
    let mut morsels: Vec<Box<dyn Morsel>> = Vec::with_capacity(row_group_indexes.len());
    for row_group_idx in row_group_indexes {
        let row_count = row_group_metadata[row_group_idx].num_rows() as usize;
        let row_group_selection = selection
            .as_mut()
            .map(|selection| selection.split_off(row_count));
        if let Some(morsel) = row_group_planner.plan(row_group_idx, row_group_selection) {
            morsels.push(Box::new(morsel));
        }
    }

    Ok((!morsels.is_empty()).then(|| MorselPlan::new().with_morsels(morsels)))
}

async fn load_bloom_filters(
    builder: &mut ParquetRecordBatchStreamBuilder<ParquetMetadataCacheReader>,
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
            physical_plan::{FileScanConfigBuilder, FileSource, ParquetSource},
        },
        execution::object_store::ObjectStoreUrl,
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
        cache::{AlwaysHydrate, squeeze_policies::Evict},
        cache_policies::LiquidPolicy,
    };
    use object_store::local::LocalFileSystem;
    use parquet::arrow::{ArrowWriter, async_reader::AsyncFileReader};

    use crate::{
        cache::{BatchID, CachedFileRef, CachedRowGroupRef, LiquidCacheParquet},
        reader::LiquidParquetSource,
    };

    use super::*;

    static NEXT_FILE_ID: AtomicUsize = AtomicUsize::new(0);

    struct PlannedTestFile {
        morsels: Vec<Box<dyn Morsel>>,
        _cache: Arc<LiquidCacheParquet>,
        cached_file: CachedFileRef,
        _tmp_dir: tempfile::TempDir,
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

    async fn plan_test_file(options: PlanOptions) -> PlannedTestFile {
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
        let morselizer = LiquidMorselizer {
            partition_index: 0,
            projection: ProjectionExprs::from_indices(&options.projection_columns, schema.as_ref()),
            batch_size: 4,
            predicate: options.predicate,
            table_schema: TableSchema::from(Arc::clone(&schema)),
            metrics: metrics.clone(),
            parquet_file_reader_factory: Arc::new(CachedMetaReaderFactory::new(
                object_store,
                ObjectStoreUrl::parse(format!("test-{file_id}:///")).unwrap(),
            )),
            reorder_filters: false,
            liquid_cache: cache.clone(),
            expr_adapter_factory: Arc::new(DefaultPhysicalExprAdapterFactory),
            span: None,
            squeeze_hints: Arc::default(),
        };
        let morsels = drive_planner(morselizer.plan_file(partitioned_file).unwrap()).await;
        let cached_file = cache.register_or_get_file(file_name, schema);
        PlannedTestFile {
            morsels,
            _cache: cache,
            cached_file,
            _tmp_dir: tmp_dir,
        }
    }

    fn gt_expr(column_name: &str, column_index: usize, literal: i32) -> Arc<dyn PhysicalExpr> {
        Arc::new(BinaryExpr::new(
            Arc::new(Column::new(column_name, column_index)),
            Operator::Gt,
            Arc::new(Literal::new(ScalarValue::Int32(Some(literal)))),
        ))
    }

    #[tokio::test]
    async fn metadata_cache_is_scoped_to_object_store() {
        let schema = schema();
        let dir_a = tempfile::tempdir().unwrap();
        let dir_b = tempfile::tempdir().unwrap();
        let path_a = dir_a.path().join("data.parquet");
        let path_b = dir_b.path().join("data.parquet");
        write_single_row_group_file(&path_a, schema.clone(), vec![1]);
        write_single_row_group_file(&path_b, schema, vec![1, 2]);
        let metrics = ExecutionPlanMetricsSet::new();
        let mut reader_a = CachedMetaReaderFactory::new(
            Arc::new(LocalFileSystem::new_with_prefix(dir_a.path()).unwrap()),
            ObjectStoreUrl::parse("store-a:///").unwrap(),
        )
        .create_liquid_reader(
            0,
            PartitionedFile::new("data.parquet", std::fs::metadata(path_a).unwrap().len()),
            None,
            &metrics,
        );
        let mut reader_b = CachedMetaReaderFactory::new(
            Arc::new(LocalFileSystem::new_with_prefix(dir_b.path()).unwrap()),
            ObjectStoreUrl::parse("store-b:///").unwrap(),
        )
        .create_liquid_reader(
            0,
            PartitionedFile::new("data.parquet", std::fs::metadata(path_b).unwrap().len()),
            None,
            &metrics,
        );

        let metadata_a = reader_a.get_metadata(None).await.unwrap();
        let metadata_b = reader_b.get_metadata(None).await.unwrap();

        assert_eq!(metadata_a.file_metadata().num_rows(), 1);
        assert_eq!(metadata_b.file_metadata().num_rows(), 2);
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

    async fn is_cached(row_group: &CachedRowGroupRef, column_id: usize, batch_idx: u16) -> bool {
        row_group
            .get_column(column_id as u64)
            .unwrap()
            .get_arrow_array_test_only(BatchID::from_raw(batch_idx))
            .await
            .is_some()
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
        assert!(is_cached(&row_group0, 0, 0).await);
        assert!(is_cached(&row_group0, 1, 0).await);
        assert!(is_cached(&row_group1, 0, 0).await);
        assert!(!is_cached(&row_group1, 1, 0).await);
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
        assert!(is_cached(&row_group0, 0, 0).await);
        assert!(is_cached(&row_group0, 1, 0).await);
        assert!(is_cached(&row_group1, 0, 0).await);
        assert!(!is_cached(&row_group1, 1, 0).await);
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
            assert!(!is_cached(row_group, 0, 0).await);
            assert!(!is_cached(row_group, 1, 0).await);
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
        assert!(is_cached(&row_group0, 0, 0).await);
        assert!(is_cached(&row_group0, 1, 0).await);
        assert!(is_cached(&row_group1, 0, 0).await);
        assert!(!is_cached(&row_group1, 1, 0).await);
    }

    #[tokio::test]
    async fn missing_column_falls_back_to_parquet() {
        let planned = plan_test_file(PlanOptions {
            ..Default::default()
        })
        .await;
        let row_group0 = planned.cached_file.create_row_group(0, vec![]);
        let row_group1 = planned.cached_file.create_row_group(1, vec![]);
        insert_batches(&row_group0, 0, &[(0, &[0, 1, 2, 3])]).await;
        insert_batches(&row_group1, 0, &[(0, &[4, 5, 6, 7])]).await;

        let (a, b) = collect_columns(planned.morsels).await;
        assert_eq!(a, vec![0, 1, 2, 3, 4, 5, 6, 7]);
        assert_eq!(b, vec![10, 11, 12, 13, 14, 15, 16, 17]);
        assert!(is_cached(&row_group0, 1, 0).await);
        assert!(is_cached(&row_group1, 1, 0).await);
    }

    #[tokio::test]
    async fn fallback_stream_advances_across_misses() {
        let parquet_a = vec![
            100, 101, 102, 103, 4, 5, 6, 7, 200, 201, 202, 203, 12, 13, 14, 15,
        ];
        let planned = plan_test_file(PlanOptions {
            projection_columns: vec![0],
            single_row_group_values: Some(parquet_a),
            ..Default::default()
        })
        .await;
        let row_group = planned.cached_file.create_row_group(0, vec![]);
        insert_batches(&row_group, 0, &[(0, &[0, 1, 2, 3]), (2, &[8, 9, 10, 11])]).await;

        assert_eq!(
            collect_columns(planned.morsels).await.0,
            (0..16).collect::<Vec<_>>()
        );
        for batch_idx in 0..4 {
            assert!(is_cached(&row_group, 0, batch_idx).await);
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
