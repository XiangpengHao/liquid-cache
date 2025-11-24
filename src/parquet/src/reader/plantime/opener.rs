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
use arrow_schema::{Field, Schema, SchemaRef};
use datafusion::{
    common::exec_err,
    datasource::{
        listing::PartitionedFile,
        physical_plan::{
            FileOpenFuture, FileOpener, ParquetFileMetrics,
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
use log::debug;
use parquet::arrow::{
    ParquetRecordBatchStreamBuilder, ProjectionMask,
    arrow_reader::{ArrowReaderMetadata, ArrowReaderOptions},
};
use parquet::file::metadata::ParquetMetaData;

use super::source::CachedMetaReaderFactory;

pub struct LiquidParquetOpener {
    partition_index: usize,
    projection: Arc<[usize]>,
    batch_size: usize,
    limit: Option<usize>,
    predicate: Option<Arc<dyn PhysicalExpr>>,
    pruning_predicate: Option<Arc<PruningPredicate>>,
    page_pruning_predicate: Option<Arc<PagePruningAccessPlanFilter>>,
    downstream_full_schema: SchemaRef,
    metrics: ExecutionPlanMetricsSet,
    parquet_file_reader_factory: Arc<CachedMetaReaderFactory>,
    reorder_filters: bool,
    liquid_cache: LiquidCacheParquetRef,
    schema_adapter_factory: Arc<dyn SchemaAdapterFactory>,
    span: Option<Arc<fastrace::Span>>,
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
        downstream_full_schema: SchemaRef,
        metrics: ExecutionPlanMetricsSet,
        liquid_cache: LiquidCacheParquetRef,
        parquet_file_reader_factory: Arc<CachedMetaReaderFactory>,
        reorder_filters: bool,
        schema_adapter_factory: Arc<dyn SchemaAdapterFactory>,
        span: Option<Arc<fastrace::Span>>,
    ) -> Self {
        Self {
            partition_index,
            projection,
            batch_size,
            limit,
            predicate,
            pruning_predicate,
            page_pruning_predicate,
            downstream_full_schema,
            metrics,
            liquid_cache,
            parquet_file_reader_factory,
            reorder_filters,
            schema_adapter_factory,
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
    fn open(&self, file: PartitionedFile) -> Result<FileOpenFuture, DataFusionError> {
        let file_range = file.range.clone();
        let extensions = file.extensions.clone();
        let file_name = file.object_meta.location.to_string();
        let file_metrics = ParquetFileMetrics::new(self.partition_index, &file_name, &self.metrics);

        let metadata_size_hint = file.metadata_size_hint;

        let lc = self.liquid_cache.clone();
        let file_loc = file.object_meta.location.to_string();

        let mut async_file_reader = self.parquet_file_reader_factory.create_liquid_reader(
            self.partition_index,
            file,
            metadata_size_hint,
            &self.metrics,
        );

        let batch_size = self.batch_size;

        let projected_schema =
            SchemaRef::from(self.downstream_full_schema.project(&self.projection)?);
        let schema_adapter = self.schema_adapter_factory.create(
            Arc::clone(&projected_schema),
            Arc::clone(&self.downstream_full_schema),
        );
        let predicate = self.predicate.clone();
        let pruning_predicate = self.pruning_predicate.clone();
        let page_pruning_predicate = self.page_pruning_predicate.clone();
        let downstream_full_schema = Arc::clone(&self.downstream_full_schema);
        let reorder_predicates = self.reorder_filters;
        let enable_page_index = should_enable_page_index(&self.page_pruning_predicate);
        let limit = self.limit;
        let schema_adapter_factory = Arc::clone(&self.schema_adapter_factory);
        let span = self.span.clone();
        Ok(Box::pin(async move {
            let mut options = ArrowReaderOptions::new().with_page_index(enable_page_index);

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
                Arc::clone(&downstream_full_schema),
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

            metadata_timer.stop();

            let mut builder = ParquetRecordBatchStreamBuilder::new_with_metadata(
                async_file_reader.clone(),
                reader_metadata.clone(),
            );

            let (schema_mapping, adapted_projections) =
                schema_adapter.map_schema(&physical_file_schema)?;

            let mask = ProjectionMask::roots(
                builder.parquet_schema(),
                adapted_projections.iter().cloned(),
            );

            // Filter pushdown: evaluate predicates during scan
            let row_filter = predicate.as_ref().and_then(|p| {
                let row_filter = build_row_filter(
                    p,
                    &physical_file_schema,
                    &downstream_full_schema,
                    reader_metadata.metadata(),
                    reorder_predicates,
                    &file_metrics,
                    &schema_adapter_factory,
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
            if enable_page_index
                && !access_plan.is_empty()
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

            let adapted = stream
                .map_err(|e| DataFusionError::External(Box::new(e)))
                .map(move |batch| batch.and_then(|batch| schema_mapping.map_batch(batch)));

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
