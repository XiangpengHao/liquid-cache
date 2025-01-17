use std::sync::Arc;

use arrow_schema::{ArrowError, SchemaRef};
use datafusion::{
    common::exec_err,
    datasource::{
        physical_plan::{
            parquet::ParquetAccessPlan, FileMeta, FileOpenFuture, FileOpener, ParquetFileMetrics,
            ParquetFileReaderFactory,
        },
        schema_adapter::SchemaAdapterFactory,
    },
    error::DataFusionError,
    physical_optimizer::pruning::PruningPredicate,
    physical_plan::{metrics::ExecutionPlanMetricsSet, PhysicalExpr},
};
use futures::StreamExt;
use futures::TryStreamExt;
use log::debug;
use parquet::arrow::{
    arrow_reader::{ArrowReaderMetadata, ArrowReaderOptions},
    async_reader::AsyncFileReader,
    ParquetRecordBatchStreamBuilder, ProjectionMask,
};

use crate::liquid_parquet::reader::plantime::{
    coerce_file_schema_to_string_type, coerce_file_schema_to_view_type,
    page_filter::PagePruningAccessPlanFilter, row_filter,
    row_group_filter::RowGroupAccessPlanFilter,
};

pub struct LiquidParquetOpener {
    pub partition_index: usize,
    pub projection: Arc<[usize]>,
    pub batch_size: usize,
    pub limit: Option<usize>,
    pub predicate: Option<Arc<dyn PhysicalExpr>>,
    pub pruning_predicate: Option<Arc<PruningPredicate>>,
    pub page_pruning_predicate: Option<Arc<PagePruningAccessPlanFilter>>,
    pub table_schema: SchemaRef,
    pub metrics: ExecutionPlanMetricsSet,
    pub parquet_file_reader_factory: Arc<dyn ParquetFileReaderFactory>,
    pub reorder_filters: bool,
    pub schema_adapter_factory: Arc<dyn SchemaAdapterFactory>,
}

impl FileOpener for LiquidParquetOpener {
    fn open(&self, file_meta: FileMeta) -> Result<FileOpenFuture, DataFusionError> {
        let file_range = file_meta.range.clone();
        let extensions = file_meta.extensions.clone();
        let file_name = file_meta.location().to_string();
        let file_metrics = ParquetFileMetrics::new(self.partition_index, &file_name, &self.metrics);

        let metadata_size_hint = file_meta.metadata_size_hint;

        let mut reader: Box<dyn AsyncFileReader> = self.parquet_file_reader_factory.create_reader(
            self.partition_index,
            file_meta,
            metadata_size_hint,
            &self.metrics,
        )?;

        let batch_size = self.batch_size;

        let projected_schema = SchemaRef::from(self.table_schema.project(&self.projection)?);
        let schema_adapter = self
            .schema_adapter_factory
            .create(projected_schema, Arc::clone(&self.table_schema));
        let predicate = self.predicate.clone();
        let pruning_predicate = self.pruning_predicate.clone();
        let page_pruning_predicate = self.page_pruning_predicate.clone();
        let table_schema = Arc::clone(&self.table_schema);
        let reorder_predicates = self.reorder_filters;
        let enable_page_index = should_enable_page_index(&self.page_pruning_predicate);
        let limit = self.limit;

        Ok(Box::pin(async move {
            let options = ArrowReaderOptions::new().with_page_index(enable_page_index);

            let mut metadata_timer = file_metrics.metadata_load_time.timer();
            let metadata = ArrowReaderMetadata::load_async(&mut reader, options.clone()).await?;
            let mut schema = Arc::clone(metadata.schema());

            if let Some(merged) = coerce_file_schema_to_string_type(&table_schema, &schema) {
                schema = Arc::new(merged);
            }

            // read with view types
            if let Some(merged) = coerce_file_schema_to_view_type(&table_schema, &schema) {
                schema = Arc::new(merged);
            }

            let options = ArrowReaderOptions::new()
                .with_page_index(enable_page_index)
                .with_schema(Arc::clone(&schema));
            let metadata = ArrowReaderMetadata::try_new(Arc::clone(metadata.metadata()), options)?;

            metadata_timer.stop();

            let mut builder = ParquetRecordBatchStreamBuilder::new_with_metadata(reader, metadata);

            let file_schema = Arc::clone(builder.schema());

            let (schema_mapping, adapted_projections) = schema_adapter.map_schema(&file_schema)?;

            let mask = ProjectionMask::roots(
                builder.parquet_schema(),
                adapted_projections.iter().cloned(),
            );

            // Filter pushdown: evaluate predicates during scan
            if let Some(predicate) = predicate {
                let row_filter = row_filter::build_row_filter(
                    &predicate,
                    &file_schema,
                    &table_schema,
                    builder.metadata(),
                    reorder_predicates,
                    &file_metrics,
                    Arc::clone(&schema_mapping),
                );

                match row_filter {
                    Ok(Some(filter)) => {
                        builder = builder.with_row_filter(filter);
                    }
                    Ok(None) => {}
                    Err(e) => {
                        debug!(
                            "Ignoring error building row filter for '{:?}': {}",
                            predicate, e
                        );
                    }
                };
            };

            // Determine which row groups to actually read. The idea is to skip
            // as many row groups as possible based on the metadata and query
            let file_metadata = Arc::clone(builder.metadata());
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
                    &file_schema,
                    builder.parquet_schema(),
                    rg_metadata,
                    predicate,
                    &file_metrics,
                );

                if !row_groups.is_empty() {
                    row_groups
                        .prune_by_bloom_filters(
                            &file_schema,
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
            if enable_page_index && !access_plan.is_empty() {
                if let Some(p) = page_pruning_predicate {
                    access_plan = p.prune_plan_with_page_index(
                        access_plan,
                        &file_schema,
                        builder.parquet_schema(),
                        file_metadata.as_ref(),
                        &file_metrics,
                    );
                }
            }

            let row_group_indexes = access_plan.row_group_indexes();
            if let Some(row_selection) = access_plan.into_overall_row_selection(rg_metadata)? {
                builder = builder.with_row_selection(row_selection);
            }

            if let Some(limit) = limit {
                builder = builder.with_limit(limit)
            }

            let stream = builder
                .with_projection(mask)
                .with_batch_size(batch_size)
                .with_row_groups(row_group_indexes)
                .build()?;

            let adapted = stream
                .map_err(|e| ArrowError::ExternalError(Box::new(e)))
                .map(move |maybe_batch| {
                    maybe_batch.and_then(|b| schema_mapping.map_batch(b).map_err(Into::into))
                });

            Ok(adapted.boxed())
        }))
    }
}

fn should_enable_page_index(
    page_pruning_predicate: &Option<Arc<PagePruningAccessPlanFilter>>,
) -> bool {
    page_pruning_predicate.is_some()
        && page_pruning_predicate
            .as_ref()
            .map(|p| p.filter_number() > 0)
            .unwrap_or(false)
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
