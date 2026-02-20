use std::sync::Arc;

use crate::{
    cache::LiquidCacheParquetRef,
    optimizers::enrich_schema_for_cache,
    reader::{
        plantime::{row_filter::build_row_filter, row_group_filter::RowGroupAccessPlanFilter},
        runtime::LiquidStreamBuilder,
    },
};
use ahash::AHashMap;
use arrow::array::{RecordBatch, RecordBatchOptions};
use arrow_schema::{Field, Schema, SchemaRef};
use datafusion::{
    common::exec_err,
    datasource::{
        listing::PartitionedFile,
        physical_plan::{
            FileOpenFuture, FileOpener, ParquetFileMetrics,
            parquet::{PagePruningAccessPlanFilter, ParquetAccessPlan},
        },
        table_schema::TableSchema,
    },
    error::DataFusionError,
    physical_expr::PhysicalExprSimplifier,
    physical_expr::projection::ProjectionExprs,
    physical_expr::utils::reassign_expr_columns,
    physical_expr_adapter::{PhysicalExprAdapterFactory, replace_columns_with_literals},
    physical_expr_common::physical_expr::is_dynamic_physical_expr,
    physical_optimizer::pruning::{FilePruner, PruningPredicate, build_pruning_predicate},
    physical_plan::{
        PhysicalExpr,
        metrics::{Count, ExecutionPlanMetricsSet, MetricBuilder},
    },
};
use futures::StreamExt;
use futures::TryStreamExt;
use log::debug;
use parquet::arrow::{
    ParquetRecordBatchStreamBuilder, ProjectionMask,
    arrow_reader::{ArrowReaderMetadata, ArrowReaderOptions},
};
use parquet::file::metadata::ParquetMetaData;

use super::source::CachedMetaReaderFactory;

pub struct LiquidParquetOpener {
    partition_index: usize,
    projection: ProjectionExprs,
    batch_size: usize,
    limit: Option<usize>,
    predicate: Option<Arc<dyn PhysicalExpr>>,
    table_schema: TableSchema,
    metrics: ExecutionPlanMetricsSet,
    parquet_file_reader_factory: Arc<CachedMetaReaderFactory>,
    reorder_filters: bool,
    liquid_cache: LiquidCacheParquetRef,
    expr_adapter_factory: Arc<dyn PhysicalExprAdapterFactory>,
    span: Option<Arc<fastrace::Span>>,
}

impl LiquidParquetOpener {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        partition_index: usize,
        projection: ProjectionExprs,
        batch_size: usize,
        limit: Option<usize>,
        predicate: Option<Arc<dyn PhysicalExpr>>,
        table_schema: TableSchema,
        metrics: ExecutionPlanMetricsSet,
        liquid_cache: LiquidCacheParquetRef,
        parquet_file_reader_factory: Arc<CachedMetaReaderFactory>,
        reorder_filters: bool,
        expr_adapter_factory: Arc<dyn PhysicalExprAdapterFactory>,
        span: Option<Arc<fastrace::Span>>,
    ) -> Self {
        Self {
            partition_index,
            projection,
            batch_size,
            limit,
            predicate,
            table_schema,
            metrics,
            liquid_cache,
            parquet_file_reader_factory,
            reorder_filters,
            expr_adapter_factory,
            span,
        }
    }
}

// transfer lineage metadata from tagged schema to dst schema
// The two schema must from the same file.
fn transfer_lineage_metadata_to_file_schema(
    tagged_schema: SchemaRef,
    dst_schema: SchemaRef,
) -> Schema {
    let mut new_fields = vec![];

    let mut tagged_fields = AHashMap::new();
    for field in tagged_schema.fields().iter() {
        tagged_fields.insert(field.name().to_string(), field.clone());
    }
    for field in dst_schema.fields().iter() {
        let tagged_field = match tagged_fields.get(field.name()) {
            Some(tagged_field) => {
                let new_field = Field::clone(field).with_metadata(tagged_field.metadata().clone());
                Arc::new(new_field)
            }
            None => field.clone(),
        };
        new_fields.push(tagged_field);
    }
    let dst_metadata = dst_schema.metadata().clone();
    Schema::new(new_fields).with_metadata(dst_metadata)
}

impl FileOpener for LiquidParquetOpener {
    fn open(&self, partitioned_file: PartitionedFile) -> Result<FileOpenFuture, DataFusionError> {
        let file_range = partitioned_file.range.clone();
        let extensions = partitioned_file.extensions.clone();
        let file_name = partitioned_file.object_meta.location.to_string();
        let file_metrics = ParquetFileMetrics::new(self.partition_index, &file_name, &self.metrics);

        let metadata_size_hint = partitioned_file.metadata_size_hint;

        let lc = self.liquid_cache.clone();
        let file_loc = partitioned_file.object_meta.location.to_string();

        let mut async_file_reader = self.parquet_file_reader_factory.create_liquid_reader(
            self.partition_index,
            partitioned_file.clone(),
            metadata_size_hint,
            &self.metrics,
        );

        let batch_size = self.batch_size;
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
            .zip(partitioned_file.partition_values.iter())
        {
            literal_columns.insert(field.name().clone(), value.clone());
        }
        if !literal_columns.is_empty() {
            projection = projection.try_map_exprs(|expr| {
                replace_columns_with_literals(Arc::clone(&expr), &literal_columns)
            })?;
            predicate = predicate
                .map(|p| replace_columns_with_literals(p, &literal_columns))
                .transpose()?;
        }
        let reorder_predicates = self.reorder_filters;
        let limit = self.limit;

        let predicate_creation_errors =
            MetricBuilder::new(&self.metrics).global_counter("num_predicate_creation_errors");

        let expr_adapter_factory = Arc::clone(&self.expr_adapter_factory);
        let span = self.span.clone();
        Ok(Box::pin(async move {
            // Prune this file using the file level statistics and partition values.
            // Since dynamic filters may have been updated since planning it is possible that we are able
            // to prune files now that we couldn't prune at planning time.
            // It is assumed that there is no point in doing pruning here if the predicate is not dynamic,
            // as it would have been done at planning time.
            // We'll also check this after every record batch we read,
            // and if at some point we are able to prove we can prune the file using just the file level statistics
            // we can end the stream early.
            let mut file_pruner = predicate
                .as_ref()
                .filter(|p| is_dynamic_physical_expr(p) || partitioned_file.has_statistics())
                .and_then(|p| {
                    FilePruner::try_new(
                        Arc::clone(p),
                        &logical_file_schema,
                        &partitioned_file,
                        predicate_creation_errors.clone(),
                    )
                });

            if let Some(file_pruner) = &mut file_pruner
                && file_pruner.should_prune()?
            {
                file_metrics.files_ranges_pruned_statistics.add_pruned(1);
                return Ok(futures::stream::empty().boxed());
            }

            file_metrics.files_ranges_pruned_statistics.add_matched(1);

            let mut options = ArrowReaderOptions::new().with_page_index(true);
            let mut metadata_timer = file_metrics.metadata_load_time.timer();

            // Begin by loading the metadata from the underlying reader (note
            // the returned metadata may actually include page indexes as some
            // readers may return page indexes even when not requested -- for
            // example when they are cached)
            let mut reader_metadata =
                ArrowReaderMetadata::load_async(&mut async_file_reader, options.clone()).await?;

            // Note about schemas: we are actually dealing with **3 different schemas** here:
            // - The table schema as defined by the TableProvider.
            //   This is what the user sees, what they get when they `SELECT * FROM table`, etc.
            // - The logical file schema: this is the table schema minus any hive partition columns and projections.
            //   This is what the physical file schema is coerced to.
            // - The physical file schema: this is the schema as defined by the parquet file. This is what the parquet file actually contains.
            let physical_file_schema = Arc::clone(reader_metadata.schema());
            let physical_file_schema = Arc::new(transfer_lineage_metadata_to_file_schema(
                Arc::clone(&logical_file_schema),
                Arc::clone(&physical_file_schema),
            ));
            let cache_full_schema = enrich_schema_for_cache(&physical_file_schema);
            options = options.with_schema(Arc::clone(&physical_file_schema));
            reader_metadata =
                ArrowReaderMetadata::try_new(Arc::clone(reader_metadata.metadata()), options)?;
            debug_assert!(
                Arc::strong_count(reader_metadata.metadata()) > 1,
                "meta data must be cached already"
            );

            let rewriter = expr_adapter_factory.create(
                Arc::clone(&logical_file_schema),
                Arc::clone(&physical_file_schema),
            );
            let simplifier = PhysicalExprSimplifier::new(&physical_file_schema);
            predicate = predicate
                .map(|p| simplifier.simplify(rewriter.rewrite(p)?))
                .transpose()?;
            projection = projection.try_map_exprs(|p| simplifier.simplify(rewriter.rewrite(p)?))?;

            let (pruning_predicate, page_pruning_predicate) = build_pruning_predicates(
                predicate.as_ref(),
                &physical_file_schema,
                &predicate_creation_errors,
            );

            metadata_timer.stop();

            let mut builder = ParquetRecordBatchStreamBuilder::new_with_metadata(
                async_file_reader.clone(),
                reader_metadata.clone(),
            );
            let indices = projection.column_indices();
            let mask = ProjectionMask::roots(builder.parquet_schema(), indices);

            // Filter pushdown: evaluate predicates during scan
            let row_filter = predicate.as_ref().and_then(|p| {
                let row_filter = build_row_filter(
                    p,
                    &physical_file_schema,
                    reader_metadata.metadata(),
                    reorder_predicates,
                    &file_metrics,
                );

                match row_filter {
                    Ok(Some(filter)) => Some(filter),
                    Ok(None) => None,
                    Err(e) => {
                        debug!("Ignoring error building row filter for '{predicate:?}': {e:?}");
                        None
                    }
                }
            });

            // Determine which row groups to actually read. The idea is to skip
            // as many row groups as possible based on the metadata and query
            let file_metadata: Arc<ParquetMetaData> = Arc::clone(builder.metadata());
            let predicate = pruning_predicate.as_ref().map(|p| p.as_ref());
            let rg_metadata = file_metadata.row_groups();
            // track which row groups to actually read
            let access_plan = create_initial_plan(&file_name, extensions, rg_metadata.len())?;
            let mut row_groups = RowGroupAccessPlanFilter::new(access_plan);
            // if there is a range restricting what parts of the file to read
            if let Some(range) = file_range.as_ref() {
                row_groups.prune_by_range(rg_metadata, range);
            }
            // If there is a predicate that can be evaluated against the metadata
            if let Some(predicate) = predicate.as_ref() {
                row_groups.prune_by_statistics(
                    &physical_file_schema,
                    builder.parquet_schema(),
                    rg_metadata,
                    predicate,
                    &file_metrics,
                );

                if !row_groups.is_empty() {
                    row_groups
                        .prune_by_bloom_filters(
                            &physical_file_schema,
                            &mut builder,
                            predicate,
                            &file_metrics,
                        )
                        .await;
                }
            }

            let mut access_plan = row_groups.build();

            // page index pruning: if all data on individual pages can
            // be ruled using page metadata, rows from other columns
            // with that range can be skipped as well
            if !access_plan.is_empty()
                && let Some(p) = page_pruning_predicate
            {
                access_plan = p.prune_plan_with_page_index(
                    access_plan,
                    &physical_file_schema,
                    builder.parquet_schema(),
                    file_metadata.as_ref(),
                    &file_metrics,
                );
            }

            let row_group_indexes = access_plan.row_group_indexes();
            let row_selection = access_plan.into_overall_row_selection(rg_metadata)?;

            let mut liquid_builder =
                LiquidStreamBuilder::new(async_file_reader, Arc::clone(reader_metadata.metadata()))
                    .with_batch_size(batch_size)
                    .with_row_groups(row_group_indexes)
                    .with_projection(mask)
                    .with_selection(row_selection)
                    .with_limit(limit);

            if let Some(row_filter) = row_filter {
                liquid_builder = liquid_builder.with_row_filter(row_filter);
            }

            if let Some(s) = &span {
                let span = fastrace::Span::enter_with_parent("liquid_stream", s);
                liquid_builder = liquid_builder.with_span(span);
            }

            let liquid_cache = lc.register_or_get_file(file_loc, Arc::clone(&cache_full_schema));

            let stream = liquid_builder.build(liquid_cache)?;

            let stream_schema = Arc::clone(stream.schema());
            let replace_schema = !stream_schema.eq(&output_schema);
            let projection =
                projection.try_map_exprs(|expr| reassign_expr_columns(expr, &stream_schema))?;
            let projector = projection.make_projector(&stream_schema)?;

            let adapted = stream
                .map_err(|e| DataFusionError::External(Box::new(e)))
                .map(move |batch| {
                    batch.and_then(|batch| {
                        let batch = projector.project_batch(&batch)?;
                        if replace_schema {
                            let (_schema, arrays, num_rows) = batch.into_parts();
                            let options = RecordBatchOptions::new().with_row_count(Some(num_rows));
                            RecordBatch::try_new_with_options(
                                Arc::clone(&output_schema),
                                arrays,
                                &options,
                            )
                            .map_err(Into::into)
                        } else {
                            Ok(batch)
                        }
                    })
                });

            Ok(adapted.boxed())
        }))
    }
}

fn create_initial_plan(
    file_name: &str,
    extensions: Option<Arc<dyn std::any::Any + Send + Sync>>,
    row_group_count: usize,
) -> Result<ParquetAccessPlan, DataFusionError> {
    if let Some(extensions) = extensions {
        if let Some(access_plan) = extensions.downcast_ref::<ParquetAccessPlan>() {
            let plan_len = access_plan.len();
            if plan_len != row_group_count {
                return exec_err!(
                    "Invalid ParquetAccessPlan for {file_name}. Specified {plan_len} row groups, but file has {row_group_count}"
                );
            }

            // check row group count matches the plan
            return Ok(access_plan.clone());
        } else {
            debug!("ParquetExec Ignoring unknown extension specified for {file_name}");
        }
    }

    // default to scanning all row groups
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
    let Some(predicate) = predicate.as_ref() else {
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

/// Build a page pruning predicate from an optional predicate expression.
/// If the predicate is None or the predicate cannot be converted to a page pruning
/// predicate, return None.
pub(crate) fn build_page_pruning_predicate(
    predicate: &Arc<dyn PhysicalExpr>,
    file_schema: &SchemaRef,
) -> Arc<PagePruningAccessPlanFilter> {
    Arc::new(PagePruningAccessPlanFilter::new(
        predicate,
        Arc::clone(file_schema),
    ))
}
