use std::sync::Arc;

use arrow_schema::{ArrowError, SchemaRef};
use datafusion::{
    common::exec_err,
    datasource::{
        physical_plan::{
            FileMeta, FileOpenFuture, FileOpener, ParquetFileMetrics, ParquetFileReaderFactory,
            parquet::ParquetAccessPlan,
        },
        schema_adapter::SchemaAdapterFactory,
    },
    error::DataFusionError,
    physical_optimizer::pruning::PruningPredicate,
    physical_plan::{PhysicalExpr, metrics::ExecutionPlanMetricsSet},
};
use futures::StreamExt;
use futures::TryStreamExt;
use log::debug;
use parquet::arrow::{
    ParquetRecordBatchStreamBuilder, ProjectionMask,
    arrow_reader::{ArrowReaderMetadata, ArrowReaderOptions},
    async_reader::AsyncFileReader,
};

use crate::{
    cache::LiquidCacheRef,
    reader::{
        plantime::{
            page_filter::PagePruningAccessPlanFilter, row_filter,
            row_group_filter::RowGroupAccessPlanFilter,
        },
        runtime::ArrowReaderBuilderBridge,
    },
};

use super::{coerce_to_parquet_reader_types, transform_to_liquid_cache_types};

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
    pub liquid_cache: LiquidCacheRef,
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
        let liquid_cache = self.liquid_cache.clone();

        Ok(Box::pin(async move {
            let options = ArrowReaderOptions::new().with_page_index(enable_page_index);

            let mut metadata_timer = file_metrics.metadata_load_time.timer();

            let parquet_metadata = reader.get_metadata().await?;
            let metadata = ArrowReaderMetadata::try_new(parquet_metadata, options)?;
            debug_assert!(
                Arc::strong_count(metadata.metadata()) > 1,
                "meta data must be cached already"
            );
            let schema = Arc::clone(metadata.schema());

            let reader_schema = Arc::new(coerce_to_parquet_reader_types(&schema));
            let output_schema = Arc::new(transform_to_liquid_cache_types(&schema));

            let options = ArrowReaderOptions::new()
                .with_page_index(enable_page_index)
                .with_schema(Arc::clone(&reader_schema));
            let metadata = ArrowReaderMetadata::try_new(Arc::clone(metadata.metadata()), options)?;

            metadata_timer.stop();

            let mut builder = ParquetRecordBatchStreamBuilder::new_with_metadata(reader, metadata);

            let file_schema = Arc::clone(&output_schema);

            let (_schema_mapping, adapted_projections) = schema_adapter.map_schema(&file_schema)?;

            let mask = ProjectionMask::roots(
                builder.parquet_schema(),
                adapted_projections.iter().cloned(),
            );

            // Filter pushdown: evaluate predicates during scan
            let row_filter = predicate.as_ref().and_then(|p| {
                let row_filter = row_filter::build_row_filter(
                    p,
                    &file_schema,
                    &table_schema,
                    builder.metadata(),
                    reorder_predicates,
                    &file_metrics,
                );

                match row_filter {
                    Ok(Some(filter)) => Some(filter),
                    Ok(None) => None,
                    Err(e) => {
                        debug!(
                            "Ignoring error building row filter for '{:?}': {}",
                            predicate, e
                        );
                        None
                    }
                }
            });

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

            let builder = builder
                .with_projection(mask)
                .with_batch_size(batch_size)
                .with_row_groups(row_group_indexes);

            let mut liquid_builder =
                unsafe { ArrowReaderBuilderBridge::from_parquet(builder).into_liquid_builder() };

            if let Some(row_filter) = row_filter {
                liquid_builder = liquid_builder.with_row_filter(row_filter);
            }

            let cached_file = liquid_cache.file(file_name);
            let stream = liquid_builder.build(cached_file)?;

            let adapted = stream.map_err(|e| ArrowError::ExternalError(Box::new(e)));

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
