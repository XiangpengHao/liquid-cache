use std::sync::Arc;

use crate::{
    cache::LiquidCacheRef,
    reader::{
        plantime::{row_filter, row_group_filter::RowGroupAccessPlanFilter},
        runtime::ArrowReaderBuilderBridge,
    },
};
use arrow_schema::{ArrowError, SchemaRef};
use datafusion::{
    common::exec_err,
    datasource::{
        physical_plan::{
            FileMeta, FileOpenFuture, FileOpener, ParquetFileMetrics,
            parquet::{PagePruningAccessPlanFilter, ParquetAccessPlan},
        },
        schema_adapter::SchemaAdapterFactory,
    },
    error::DataFusionError,
    physical_optimizer::pruning::PruningPredicate,
    physical_plan::{PhysicalExpr, metrics::ExecutionPlanMetricsSet},
};
use futures::StreamExt;
use futures::TryStreamExt;
use liquid_cache_common::{CacheSchema, ClientSchema, LiquidCacheMode, ParquetSchema};
use log::debug;
use parquet::arrow::{
    ParquetRecordBatchStreamBuilder, ProjectionMask,
    arrow_reader::{ArrowReaderMetadata, ArrowReaderOptions},
    async_reader::AsyncFileReader,
};

use super::source::CachedMetaReaderFactory;

pub struct LiquidParquetOpener {
    partition_index: usize,
    projection: Arc<[usize]>,
    batch_size: usize,
    limit: Option<usize>,
    predicate: Option<Arc<dyn PhysicalExpr>>,
    pruning_predicate: Option<Arc<PruningPredicate>>,
    page_pruning_predicate: Option<Arc<PagePruningAccessPlanFilter>>,
    // Schema mess:
    // 1. The client pass in a schema with UTF8 type (disabled string view), as client schema
    // 2. Our reader will read as string view, so we need to coerce it to string view, as file schema
    // 3. LiquidCache stores Dict<UInt16, UTF8> as string, so we have a liquid schema.
    cache_schema: CacheSchema,
    parquet_schema: ParquetSchema,
    client_schema: ClientSchema,
    metrics: ExecutionPlanMetricsSet,
    parquet_file_reader_factory: Arc<CachedMetaReaderFactory>,
    reorder_filters: bool,
    liquid_cache: LiquidCacheRef,
    schema_adapter_factory: Arc<dyn SchemaAdapterFactory>,
}

impl LiquidParquetOpener {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        partition_index: usize,
        projection: Arc<[usize]>,
        batch_size: usize,
        limit: Option<usize>,
        predicate: Option<Arc<dyn PhysicalExpr>>,
        pruning_predicate: Option<Arc<PruningPredicate>>,
        page_pruning_predicate: Option<Arc<PagePruningAccessPlanFilter>>,
        cache_schema: CacheSchema,
        metrics: ExecutionPlanMetricsSet,
        liquid_cache: LiquidCacheRef,
        liquid_cache_mode: LiquidCacheMode,
        parquet_file_reader_factory: Arc<CachedMetaReaderFactory>,
        reorder_filters: bool,
        schema_adapter_factory: Arc<dyn SchemaAdapterFactory>,
    ) -> Self {
        let parquet_schema = ParquetSchema::from(&cache_schema, &liquid_cache_mode);
        let client_schema = ClientSchema::from(&cache_schema, &liquid_cache_mode);

        Self {
            partition_index,
            projection,
            batch_size,
            limit,
            predicate,
            pruning_predicate,
            page_pruning_predicate,
            cache_schema,
            parquet_schema,
            client_schema,
            metrics,
            liquid_cache,
            parquet_file_reader_factory,
            reorder_filters,
            schema_adapter_factory,
        }
    }
}

impl FileOpener for LiquidParquetOpener {
    fn open(&self, file_meta: FileMeta) -> Result<FileOpenFuture, DataFusionError> {
        let file_range = file_meta.range.clone();
        let extensions = file_meta.extensions.clone();
        let file_name = file_meta.location().to_string();
        let file_metrics = ParquetFileMetrics::new(self.partition_index, &file_name, &self.metrics);

        let metadata_size_hint = file_meta.metadata_size_hint;

        let liquid_cache = self
            .liquid_cache
            .register_or_get_file(file_meta.location().to_string());

        let mut reader = self.parquet_file_reader_factory.create_liquid_reader(
            self.partition_index,
            file_meta,
            metadata_size_hint,
            &self.metrics,
        );

        let batch_size = self.batch_size;

        let projected_schema = SchemaRef::from(self.cache_schema.project(&self.projection)?);
        let schema_adapter = self
            .schema_adapter_factory
            .create(projected_schema, Arc::clone(&self.cache_schema));
        let predicate = self.predicate.clone();
        let pruning_predicate = self.pruning_predicate.clone();
        let page_pruning_predicate = self.page_pruning_predicate.clone();
        let cache_schema = Arc::clone(&self.cache_schema);
        let client_schema = Arc::clone(&self.client_schema);
        let parquet_schema = Arc::clone(&self.parquet_schema);
        let reorder_predicates = self.reorder_filters;
        let enable_page_index = should_enable_page_index(&self.page_pruning_predicate);
        let limit = self.limit;

        Ok(Box::pin(async move {
            let options = ArrowReaderOptions::new().with_page_index(enable_page_index);

            let mut metadata_timer = file_metrics.metadata_load_time.timer();

            let parquet_metadata = reader.get_metadata(Some(&options)).await?;
            let metadata = ArrowReaderMetadata::try_new(parquet_metadata, options)?;
            debug_assert!(
                Arc::strong_count(metadata.metadata()) > 1,
                "meta data must be cached already"
            );

            let options = ArrowReaderOptions::new()
                .with_page_index(enable_page_index)
                .with_schema(Arc::clone(&parquet_schema));
            let metadata = ArrowReaderMetadata::try_new(Arc::clone(metadata.metadata()), options)?;

            metadata_timer.stop();

            let mut builder = ParquetRecordBatchStreamBuilder::new_with_metadata(reader, metadata);

            let (_schema_mapping, adapted_projections) =
                schema_adapter.map_schema(&parquet_schema)?;

            let mask = ProjectionMask::roots(
                builder.parquet_schema(),
                adapted_projections.iter().cloned(),
            );

            // Filter pushdown: evaluate predicates during scan
            let row_filter = predicate.as_ref().and_then(|p| {
                let row_filter = row_filter::build_row_filter(
                    p,
                    &cache_schema,
                    &client_schema,
                    builder.metadata(),
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
                    &parquet_schema,
                    builder.parquet_schema(),
                    rg_metadata,
                    predicate,
                    &file_metrics,
                );

                if !row_groups.is_empty() {
                    row_groups
                        .prune_by_bloom_filters(
                            &parquet_schema,
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
            if enable_page_index
                && !access_plan.is_empty()
                && let Some(p) = page_pruning_predicate
            {
                access_plan = p.prune_plan_with_page_index(
                    access_plan,
                    &parquet_schema,
                    builder.parquet_schema(),
                    file_metadata.as_ref(),
                    &file_metrics,
                );
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

            let stream = liquid_builder.build(liquid_cache)?;

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
