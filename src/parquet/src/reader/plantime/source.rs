use super::opener::LiquidParquetOpener;
use crate::cache::LiquidCacheParquetRef;
use ahash::{HashMap, HashMapExt};
use bytes::Bytes;
use datafusion::{
    config::TableParquetOptions,
    datasource::{
        listing::PartitionedFile,
        physical_plan::{
            FileScanConfig, FileSource, ParquetFileMetrics, ParquetFileReaderFactory, ParquetSource,
        },
        table_schema::TableSchema,
    },
    error::Result,
    physical_expr::{conjunction, projection::ProjectionExprs},
    physical_expr_adapter::DefaultPhysicalExprAdapterFactory,
    physical_plan::{
        PhysicalExpr,
        filter_pushdown::{FilterPushdownPropagation, PushedDown, PushedDownPredicate},
        metrics::ExecutionPlanMetricsSet,
    },
};
use futures::{FutureExt, future::BoxFuture};
use object_store::{ObjectStore, path::Path};
use parquet::{
    arrow::{
        arrow_reader::ArrowReaderOptions,
        async_reader::{AsyncFileReader, ParquetObjectReader},
    },
    file::metadata::{PageIndexPolicy, ParquetMetaData, ParquetMetaDataReader},
};
use std::{
    any::Any,
    ops::Range,
    sync::{Arc, LazyLock},
};
use tokio::sync::RwLock;

static META_CACHE: LazyLock<MetadataCache> = LazyLock::new(MetadataCache::new);

#[derive(Debug)]
pub(crate) struct CachedMetaReaderFactory {
    store: Arc<dyn ObjectStore>,
}

impl CachedMetaReaderFactory {
    pub(crate) fn new(store: Arc<dyn ObjectStore>) -> Self {
        Self { store }
    }

    pub(crate) fn create_liquid_reader(
        &self,
        partition_index: usize,
        partitioned_file: PartitionedFile,
        metadata_size_hint: Option<usize>,
        metrics: &ExecutionPlanMetricsSet,
    ) -> ParquetMetadataCacheReader {
        let path = partitioned_file.object_meta.location.clone();
        let store = Arc::clone(&self.store);
        let mut inner = ParquetObjectReader::new(store, path.clone());

        if let Some(hint) = metadata_size_hint {
            inner = inner.with_footer_size_hint(hint);
        }

        ParquetMetadataCacheReader {
            file_metrics: ParquetFileMetrics::new(partition_index, path.as_ref(), metrics),
            inner,
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
    val: RwLock<HashMap<Path, Arc<ParquetMetaData>>>,
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
    inner: ParquetObjectReader,
    path: Path,
}

impl AsyncFileReader for ParquetMetadataCacheReader {
    fn get_byte_ranges(
        &mut self,
        ranges: Vec<Range<u64>>,
    ) -> BoxFuture<'_, parquet::errors::Result<Vec<Bytes>>> {
        let total: u64 = ranges.iter().map(|r| r.end - r.start).sum();
        self.file_metrics.bytes_scanned.add(total as usize);
        self.inner.get_byte_ranges(ranges)
    }

    fn get_bytes(&mut self, range: Range<u64>) -> BoxFuture<'_, parquet::errors::Result<Bytes>> {
        self.file_metrics
            .bytes_scanned
            .add((range.end - range.start) as usize);
        self.inner.get_bytes(range)
    }

    fn get_metadata(
        &mut self,
        options: Option<&ArrowReaderOptions>,
    ) -> BoxFuture<'_, parquet::errors::Result<Arc<ParquetMetaData>>> {
        let path = self.path.clone();
        let options = options.cloned();
        async move {
            // First check with read lock
            {
                let cache = META_CACHE.val.read().await;
                if let Some(meta) = cache.get(&path) {
                    return Ok(meta.clone());
                }
            }

            // Upgrade to write lock and double-check
            let mut cache = META_CACHE.val.write().await;
            match cache.entry(path.clone()) {
                std::collections::hash_map::Entry::Occupied(entry) => Ok(entry.get().clone()),
                std::collections::hash_map::Entry::Vacant(entry) => {
                    let meta = self.inner.get_metadata(options.as_ref()).await?;
                    let meta = Arc::try_unwrap(meta).unwrap_or_else(|e| e.as_ref().clone());
                    let mut reader = ParquetMetaDataReader::new_with_metadata(meta.clone())
                        .with_page_index_policy(PageIndexPolicy::Optional);
                    reader.load_page_index(&mut self.inner).await?;
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
    table_schema: TableSchema,
    projection: ProjectionExprs,
    span: Option<Arc<fastrace::Span>>,
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

    /// Set predicate information, also sets pruning_predicate and page_pruning_predicate attributes
    pub fn with_predicate(&self, predicate: Arc<dyn PhysicalExpr>) -> Self {
        let mut conf = self.clone();
        conf.predicate = Some(Arc::clone(&predicate));
        conf
    }

    /// Create a new LiquidParquetSource from a ParquetSource
    pub fn from_parquet_source(source: ParquetSource, liquid_cache: LiquidCacheParquetRef) -> Self {
        let predicate = source.filter();

        let table_schema = source.table_schema().clone();
        let projection = source.projection().cloned().unwrap_or_else(|| {
            let full_schema = table_schema.table_schema();
            let indices: Vec<usize> = (0..full_schema.fields().len()).collect();
            ProjectionExprs::from_indices(&indices, full_schema.as_ref())
        });
        let mut v = Self {
            table_schema,
            table_parquet_options: source.table_parquet_options().clone(),
            batch_size: Some(liquid_cache.batch_size()),
            liquid_cache,
            metrics: source.metrics().clone(),
            predicate: None,
            projection,
            span: None,
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
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn create_file_opener(
        &self,
        object_store: Arc<dyn ObjectStore>,
        base_config: &FileScanConfig,
        partition: usize,
    ) -> Result<Arc<dyn datafusion::datasource::physical_plan::FileOpener>> {
        let expr_adapter_factory = base_config
            .expr_adapter_factory
            .clone()
            .unwrap_or_else(|| Arc::new(DefaultPhysicalExprAdapterFactory) as _);

        let reader_factory = Arc::new(CachedMetaReaderFactory::new(object_store));

        let execution_span = self
            .span
            .clone()
            .map(|span| fastrace::Span::enter_with_parent(format!("opener_{partition}"), &span));
        let opener = LiquidParquetOpener::new(
            partition,
            self.projection.clone(),
            self.batch_size
                .expect("Batch size must be set before creating LiquidParquetOpener"),
            base_config.limit,
            self.predicate.clone(),
            self.table_schema.clone(),
            self.metrics.clone(),
            self.liquid_cache.clone(),
            reader_factory,
            self.pushdown_filters(),
            self.reorder_filters(),
            expr_adapter_factory,
            execution_span.map(Arc::new),
        );

        Ok(Arc::new(opener))
    }

    fn with_batch_size(&self, batch_size: usize) -> Arc<dyn FileSource> {
        let mut conf = self.clone();
        conf.batch_size = Some(batch_size);
        Arc::new(conf)
    }

    fn table_schema(&self) -> &TableSchema {
        &self.table_schema
    }

    fn filter(&self) -> Option<Arc<dyn PhysicalExpr>> {
        self.predicate.clone()
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

    fn try_pushdown_filters(
        &self,
        filters: Vec<Arc<dyn PhysicalExpr>>,
        config: &datafusion::config::ConfigOptions,
    ) -> Result<FilterPushdownPropagation<Arc<dyn FileSource>>> {
        use super::row_filter::can_expr_be_pushed_down_with_schemas;

        let table_schema = self.table_schema.table_schema();
        let config_pushdown_enabled = config.execution.parquet.pushdown_filters;
        let table_pushdown_enabled = self.pushdown_filters();
        let pushdown_filters = table_pushdown_enabled || config_pushdown_enabled;

        let mut source = self.clone();
        let filters: Vec<PushedDownPredicate> = filters
            .into_iter()
            .map(|filter| {
                if can_expr_be_pushed_down_with_schemas(&filter, table_schema) {
                    PushedDownPredicate::supported(filter)
                } else {
                    PushedDownPredicate::unsupported(filter)
                }
            })
            .collect();

        if filters
            .iter()
            .all(|f| matches!(f.discriminant, PushedDown::No))
        {
            return Ok(FilterPushdownPropagation::with_parent_pushdown_result(
                vec![PushedDown::No; filters.len()],
            ));
        }

        let allowed_filters = filters
            .iter()
            .filter_map(|f| match f.discriminant {
                PushedDown::Yes => Some(Arc::clone(&f.predicate)),
                PushedDown::No => None,
            })
            .collect::<Vec<_>>();

        let predicate = match source.predicate {
            Some(predicate) => conjunction(std::iter::once(predicate).chain(allowed_filters)),
            None => conjunction(allowed_filters),
        };
        source.predicate = Some(predicate);
        source = source.with_pushdown_filters(pushdown_filters);
        let source = Arc::new(source);

        if !pushdown_filters {
            return Ok(FilterPushdownPropagation::with_parent_pushdown_result(vec![
                PushedDown::No;
                filters.len()
            ])
            .with_updated_node(source));
        }

        Ok(FilterPushdownPropagation::with_parent_pushdown_result(
            filters.iter().map(|f| f.discriminant).collect(),
        )
        .with_updated_node(source))
    }
}

impl LiquidParquetSource {
    /// Configure whether filter pushdown should be enabled for this source.
    pub fn with_pushdown_filters(mut self, pushdown_filters: bool) -> Self {
        self.table_parquet_options.global.pushdown_filters = pushdown_filters;
        self
    }

    fn pushdown_filters(&self) -> bool {
        self.table_parquet_options.global.pushdown_filters
    }
}
