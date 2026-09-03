use super::LiquidMorselizer;
use crate::cache::{ColumnSqueezeHints, LiquidCacheParquetRef};
use ahash::{HashMap, HashMapExt};
use bytes::Bytes;
use datafusion::{
    common::{internal_err, tree_node::TreeNodeRecursion},
    config::{ConfigOptions, TableParquetOptions},
    datasource::{
        listing::PartitionedFile,
        physical_plan::{
            FileScanConfig, FileSource, ParquetFileMetrics, ParquetFileReaderFactory,
            ParquetSource, parquet::can_expr_be_pushed_down_with_schemas,
        },
        table_schema::TableSchema,
    },
    error::Result,
    execution::object_store::ObjectStoreUrl,
    physical_expr::projection::ProjectionExprs,
    physical_expr::utils::conjunction,
    physical_expr_adapter::DefaultPhysicalExprAdapterFactory,
    physical_plan::{
        DisplayFormatType, PhysicalExpr, apply_expression_roots,
        filter_pushdown::{FilterPushdownPropagation, PushedDown, PushedDownPredicate},
        metrics::ExecutionPlanMetricsSet,
    },
};
use datafusion_datasource::morsel::Morselizer;
use futures::{FutureExt, future::BoxFuture};
use object_store::{ObjectStore, ObjectStoreExt, path::Path};
use parquet::{
    arrow::{arrow_reader::ArrowReaderOptions, async_reader::AsyncFileReader},
    errors::ParquetError,
    file::metadata::{PageIndexPolicy, ParquetMetaData, ParquetMetaDataReader},
};
use std::{
    fmt::{self, Formatter},
    ops::Range,
    sync::{Arc, LazyLock},
};
use tokio::sync::RwLock;

static META_CACHE: LazyLock<MetadataCache> = LazyLock::new(MetadataCache::new);

#[derive(Debug)]
pub(crate) struct CachedMetaReaderFactory {
    store: Arc<dyn ObjectStore>,
    store_url: ObjectStoreUrl,
}

impl CachedMetaReaderFactory {
    pub(crate) fn new(store: Arc<dyn ObjectStore>, store_url: ObjectStoreUrl) -> Self {
        Self { store, store_url }
    }

    pub(crate) fn object_store_url(&self) -> &ObjectStoreUrl {
        &self.store_url
    }

    pub(crate) fn create_liquid_reader(
        &self,
        partition_index: usize,
        partitioned_file: PartitionedFile,
        metadata_size_hint: Option<usize>,
        metrics: &ExecutionPlanMetricsSet,
    ) -> ParquetMetadataCacheReader {
        let path = partitioned_file.object_meta.location.clone();

        ParquetMetadataCacheReader {
            file_metrics: ParquetFileMetrics::new(partition_index, path.as_ref(), metrics),
            store: Arc::clone(&self.store),
            store_url: self.store_url.clone(),
            file_size: partitioned_file.object_meta.size,
            metadata_size_hint,
            path,
        }
    }
}

impl ParquetFileReaderFactory for CachedMetaReaderFactory {
    fn create_reader(
        &self,
        partition_index: usize,
        partitioned_file: PartitionedFile,
        metadata_size_hint: Option<usize>,
        metrics: &ExecutionPlanMetricsSet,
    ) -> Result<Box<dyn AsyncFileReader + Send>> {
        let reader = self.create_liquid_reader(
            partition_index,
            partitioned_file,
            metadata_size_hint,
            metrics,
        );
        Ok(Box::new(reader))
    }
}

struct MetadataCache {
    val: RwLock<HashMap<(ObjectStoreUrl, Path), Arc<ParquetMetaData>>>,
}

impl MetadataCache {
    fn new() -> Self {
        Self {
            val: RwLock::new(HashMap::new()),
        }
    }
}

#[derive(Clone)]
pub struct ParquetMetadataCacheReader {
    file_metrics: ParquetFileMetrics,
    store: Arc<dyn ObjectStore>,
    store_url: ObjectStoreUrl,
    file_size: u64,
    metadata_size_hint: Option<usize>,
    path: Path,
}

fn to_parquet_err(error: object_store::Error) -> ParquetError {
    ParquetError::External(Box::new(error))
}

impl AsyncFileReader for ParquetMetadataCacheReader {
    fn get_byte_ranges(
        &mut self,
        ranges: Vec<Range<u64>>,
    ) -> BoxFuture<'_, parquet::errors::Result<Vec<Bytes>>> {
        let total: u64 = ranges.iter().map(|r| r.end - r.start).sum();
        self.file_metrics.bytes_scanned.add(total as usize);
        async move {
            self.store
                .get_ranges(&self.path, &ranges)
                .await
                .map_err(to_parquet_err)
        }
        .boxed()
    }

    fn get_bytes(&mut self, range: Range<u64>) -> BoxFuture<'_, parquet::errors::Result<Bytes>> {
        self.file_metrics
            .bytes_scanned
            .add((range.end - range.start) as usize);
        async move {
            self.store
                .get_range(&self.path, range)
                .await
                .map_err(to_parquet_err)
        }
        .boxed()
    }

    fn get_metadata(
        &mut self,
        options: Option<&ArrowReaderOptions>,
    ) -> BoxFuture<'_, parquet::errors::Result<Arc<ParquetMetaData>>> {
        let cache_key = (self.store_url.clone(), self.path.clone());
        let options = options.cloned();
        async move {
            // First check with read lock
            {
                let cache = META_CACHE.val.read().await;
                if let Some(meta) = cache.get(&cache_key) {
                    return Ok(meta.clone());
                }
            }

            // Upgrade to write lock and double-check
            let mut cache = META_CACHE.val.write().await;
            match cache.entry(cache_key) {
                std::collections::hash_map::Entry::Occupied(entry) => Ok(entry.get().clone()),
                std::collections::hash_map::Entry::Vacant(entry) => {
                    let file_size = self.file_size;
                    let meta = ParquetMetaDataReader::new()
                        .with_arrow_reader_options(options.as_ref())
                        .with_prefetch_hint(self.metadata_size_hint)
                        .load_and_finish(&mut *self, file_size)
                        .await?;
                    let mut reader = ParquetMetaDataReader::new_with_metadata(meta.clone())
                        .with_page_index_policy(PageIndexPolicy::Optional);
                    reader.load_page_index(&mut *self).await?;
                    let meta = Arc::new(reader.finish()?);
                    entry.insert(meta.clone());
                    Ok(meta)
                }
            }
        }
        .boxed()
    }
}

/// The data source for LiquidCache
#[derive(Clone)]
pub struct LiquidParquetSource {
    metrics: ExecutionPlanMetricsSet,
    predicate: Option<Arc<dyn PhysicalExpr>>,
    table_parquet_options: TableParquetOptions,
    liquid_cache: LiquidCacheParquetRef,
    batch_size: Option<usize>,
    projection: ProjectionExprs,
    table_schema: TableSchema,
    span: Option<Arc<fastrace::Span>>,
    squeeze_hints: Arc<ColumnSqueezeHints>,
    prefetch: bool,
}

impl LiquidParquetSource {
    fn reorder_filters(&self) -> bool {
        self.table_parquet_options.global.reorder_filters
    }

    /// Set the span for the LiquidParquetSource
    pub fn with_span(&self, span: fastrace::Span) -> Self {
        Self {
            span: Some(Arc::new(span)),
            ..self.clone()
        }
    }

    /// Set the table schema for the LiquidParquetSource
    pub fn with_table_schema(&self, table_schema: TableSchema) -> Self {
        Self {
            table_schema,
            ..self.clone()
        }
    }

    /// Attach typed squeeze hints (keyed by file-schema column name) derived
    /// from the query plan. These flow to the cache when the file is opened.
    pub fn with_squeeze_hints(&self, squeeze_hints: Arc<ColumnSqueezeHints>) -> Self {
        Self {
            squeeze_hints,
            ..self.clone()
        }
    }

    /// Enable or disable row-group prefetching.
    pub fn with_prefetch(mut self, prefetch: bool) -> Self {
        self.prefetch = prefetch;
        self
    }

    /// The typed squeeze hints currently attached to this source.
    pub fn squeeze_hints(&self) -> &Arc<ColumnSqueezeHints> {
        &self.squeeze_hints
    }

    /// Set predicate information.
    pub fn with_predicate(mut self, predicate: Arc<dyn PhysicalExpr>) -> Self {
        self.predicate = Some(predicate);
        self
    }

    /// Create a new LiquidParquetSource from a ParquetSource
    pub fn from_parquet_source(source: ParquetSource, liquid_cache: LiquidCacheParquetRef) -> Self {
        let predicate = source.filter();

        let table_schema = source.table_schema().clone();
        let projection = source.projection().cloned().unwrap_or_else(|| {
            let table_schema = table_schema.table_schema();
            ProjectionExprs::from_indices(
                &(0..table_schema.fields().len()).collect::<Vec<_>>(),
                table_schema,
            )
        });
        let mut v = Self {
            table_schema,
            table_parquet_options: source.table_parquet_options().clone(),
            batch_size: Some(liquid_cache.batch_size()),
            liquid_cache,
            projection,
            metrics: source.metrics().clone(),
            predicate: None,
            span: None,
            squeeze_hints: Arc::default(),
            prefetch: true,
        };

        if let Some(predicate) = predicate {
            v = v.with_predicate(predicate);
        }

        v
    }

    /// Get the predicate for the LiquidParquetSource
    pub fn predicate(&self) -> Option<Arc<dyn PhysicalExpr>> {
        self.predicate.clone()
    }
}

impl FileSource for LiquidParquetSource {
    fn create_file_opener(
        &self,
        _object_store: Arc<dyn ObjectStore>,
        _base_config: &FileScanConfig,
        _partition: usize,
    ) -> Result<Arc<dyn datafusion::datasource::physical_plan::FileOpener>> {
        internal_err!(
            "LiquidParquetSource::create_file_opener called but it supports the Morsel API, please use that instead"
        )
    }

    fn create_morselizer(
        &self,
        object_store: Arc<dyn ObjectStore>,
        base_config: &FileScanConfig,
        partition: usize,
    ) -> Result<Box<dyn Morselizer>> {
        let expr_adapter_factory = base_config
            .expr_adapter_factory
            .clone()
            .unwrap_or_else(|| Arc::new(DefaultPhysicalExprAdapterFactory) as _);

        let reader_factory = Arc::new(CachedMetaReaderFactory::new(
            object_store,
            base_config.object_store_url.clone(),
        ));

        let execution_span = self
            .span
            .clone()
            .map(|span| fastrace::Span::enter_with_parent(format!("opener_{partition}"), &span));
        Ok(Box::new(LiquidMorselizer {
            partition_index: partition,
            projection: self.projection.clone(),
            batch_size: self
                .batch_size
                .expect("Batch size must be set before creating LiquidMorselizer"),
            predicate: self.predicate.clone(),
            table_schema: self.table_schema.clone(),
            metrics: self.metrics.clone(),
            liquid_cache: self.liquid_cache.clone(),
            parquet_file_reader_factory: reader_factory,
            reorder_filters: self.reorder_filters(),
            expr_adapter_factory,
            span: execution_span.map(Arc::new),
            squeeze_hints: Arc::clone(&self.squeeze_hints),
            prefetch: self.prefetch,
        }))
    }

    fn with_batch_size(&self, batch_size: usize) -> Arc<dyn FileSource> {
        let mut conf = self.clone();
        conf.batch_size = Some(batch_size);
        Arc::new(conf)
    }

    fn filter(&self) -> Option<Arc<dyn PhysicalExpr>> {
        self.predicate.clone()
    }

    fn table_schema(&self) -> &TableSchema {
        &self.table_schema
    }

    fn try_pushdown_projection(
        &self,
        projection: &ProjectionExprs,
    ) -> Result<Option<Arc<dyn FileSource>>> {
        let mut source = self.clone();
        source.projection = self.projection.try_merge(projection)?;
        Ok(Some(Arc::new(source)))
    }

    fn projection(&self) -> Option<&ProjectionExprs> {
        Some(&self.projection)
    }

    fn metrics(&self) -> &ExecutionPlanMetricsSet {
        &self.metrics
    }

    fn file_type(&self) -> &str {
        "liquid_parquet"
    }

    fn fmt_extra(&self, t: DisplayFormatType, f: &mut Formatter) -> fmt::Result {
        match t {
            DisplayFormatType::Default | DisplayFormatType::Verbose => {
                if let Some(predicate) = self.filter() {
                    write!(f, ", predicate={predicate}")?;
                }
                Ok(())
            }
            DisplayFormatType::TreeRender => Ok(()),
        }
    }

    fn try_pushdown_filters(
        &self,
        filters: Vec<Arc<dyn PhysicalExpr>>,
        _config: &ConfigOptions,
    ) -> Result<FilterPushdownPropagation<Arc<dyn FileSource>>> {
        let filters: Vec<_> = filters
            .into_iter()
            .map(|filter| {
                if can_expr_be_pushed_down_with_schemas(&filter, self.table_schema.file_schema()) {
                    PushedDownPredicate::supported(filter)
                } else {
                    PushedDownPredicate::unsupported(filter)
                }
            })
            .collect();

        if filters
            .iter()
            .all(|filter| matches!(filter.discriminant, PushedDown::No))
        {
            return Ok(FilterPushdownPropagation::with_parent_pushdown_result(
                vec![PushedDown::No; filters.len()],
            ));
        }

        let supported = filters
            .iter()
            .filter_map(|filter| match filter.discriminant {
                PushedDown::Yes => Some(Arc::clone(&filter.predicate)),
                PushedDown::No => None,
            });
        let predicate = conjunction(self.predicate.iter().cloned().chain(supported));
        let source = Arc::new(self.clone().with_predicate(predicate));

        Ok(FilterPushdownPropagation::with_parent_pushdown_result(
            filters.iter().map(|filter| filter.discriminant).collect(),
        )
        .with_updated_node(source))
    }

    fn apply_expressions(
        &self,
        f: &mut dyn FnMut(&Arc<dyn PhysicalExpr>) -> Result<TreeNodeRecursion>,
    ) -> Result<TreeNodeRecursion> {
        apply_expression_roots(
            self.predicate
                .iter()
                .chain(self.projection.iter().map(|projection| &projection.expr)),
            f,
        )
    }
}
