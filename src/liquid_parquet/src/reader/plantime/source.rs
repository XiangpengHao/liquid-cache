use super::opener::LiquidParquetOpener;
use crate::{LiquidCacheMode, LiquidCacheRef};
use ahash::{HashMap, HashMapExt};
use arrow_schema::{Schema, SchemaRef};
use bytes::Bytes;
use datafusion::{
    common::Statistics,
    config::TableParquetOptions,
    datasource::{
        physical_plan::{
            FileMeta, FileScanConfig, FileSource, ParquetFileMetrics, ParquetFileReaderFactory,
            ParquetSource, parquet::PagePruningAccessPlanFilter,
        },
        schema_adapter::DefaultSchemaAdapterFactory,
    },
    error::Result,
    physical_optimizer::pruning::PruningPredicate,
    physical_plan::{
        PhysicalExpr,
        metrics::{ExecutionPlanMetricsSet, MetricBuilder},
    },
};
use futures::{FutureExt, future::BoxFuture};
use object_store::{ObjectStore, path::Path};
use parquet::{
    arrow::async_reader::{AsyncFileReader, ParquetObjectReader},
    file::metadata::{ParquetMetaData, ParquetMetaDataReader},
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
        file_meta: FileMeta,
        metadata_size_hint: Option<usize>,
        metrics: &ExecutionPlanMetricsSet,
    ) -> ParquetMetadataCacheReader {
        let path = file_meta.location().clone();
        let store = Arc::clone(&self.store);
        let mut inner = ParquetObjectReader::new(store, file_meta.object_meta);

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
        file_meta: FileMeta,
        metadata_size_hint: Option<usize>,
        metrics: &ExecutionPlanMetricsSet,
    ) -> Result<Box<dyn AsyncFileReader + Send>> {
        let reader =
            self.create_liquid_reader(partition_index, file_meta, metadata_size_hint, metrics);
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

pub struct ParquetMetadataCacheReader {
    file_metrics: ParquetFileMetrics,
    inner: ParquetObjectReader,
    path: Path,
}

impl AsyncFileReader for ParquetMetadataCacheReader {
    fn get_byte_ranges(
        &mut self,
        ranges: Vec<Range<usize>>,
    ) -> BoxFuture<'_, parquet::errors::Result<Vec<Bytes>>> {
        let total = ranges.iter().map(|r| r.end - r.start).sum();
        self.file_metrics.bytes_scanned.add(total);
        self.inner.get_byte_ranges(ranges)
    }

    fn get_bytes(&mut self, range: Range<usize>) -> BoxFuture<'_, parquet::errors::Result<Bytes>> {
        self.file_metrics.bytes_scanned.add(range.end - range.start);
        self.inner.get_bytes(range)
    }

    fn get_metadata(&mut self) -> BoxFuture<'_, parquet::errors::Result<Arc<ParquetMetaData>>> {
        let path = self.path.clone();
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
                    let meta = self.inner.get_metadata().await?;
                    let meta = Arc::try_unwrap(meta).unwrap_or_else(|e| e.as_ref().clone());
                    let mut reader = ParquetMetaDataReader::new_with_metadata(meta.clone())
                        .with_page_indexes(true);
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
#[derive(Debug, Clone)]
pub struct LiquidParquetSource {
    metrics: ExecutionPlanMetricsSet,
    predicate: Option<Arc<dyn PhysicalExpr>>,
    pruning_predicate: Option<Arc<PruningPredicate>>,
    page_pruning_predicate: Option<Arc<PagePruningAccessPlanFilter>>,
    table_parquet_options: TableParquetOptions,
    liquid_cache: LiquidCacheRef,
    liquid_cache_mode: LiquidCacheMode,
    batch_size: Option<usize>,
    metadata_size_hint: Option<usize>,
    projected_statistics: Option<Statistics>,
}

impl LiquidParquetSource {
    pub(crate) fn new(
        table_parquet_options: TableParquetOptions,
        liquid_cache: LiquidCacheRef,
        liquid_cache_mode: LiquidCacheMode,
    ) -> Self {
        Self {
            table_parquet_options,
            liquid_cache,
            liquid_cache_mode,
            metrics: ExecutionPlanMetricsSet::default(),
            predicate: None,
            pruning_predicate: None,
            page_pruning_predicate: None,
            batch_size: None,
            projected_statistics: None,
            metadata_size_hint: None,
        }
    }

    fn reorder_filters(&self) -> bool {
        self.table_parquet_options.global.reorder_filters
    }

    pub(crate) fn with_metadata_size_hint(mut self, metadata_size_hint: usize) -> Self {
        self.metadata_size_hint = Some(metadata_size_hint);
        self
    }

    fn with_metrics(mut self, metrics: ExecutionPlanMetricsSet) -> Self {
        self.metrics = metrics;
        self
    }

    /// Set predicate information, also sets pruning_predicate and page_pruning_predicate attributes
    pub fn with_predicate(
        &self,
        file_schema: Arc<Schema>,
        predicate: Arc<dyn PhysicalExpr>,
    ) -> Self {
        let mut conf = self.clone();

        let metrics = ExecutionPlanMetricsSet::new();
        let predicate_creation_errors =
            MetricBuilder::new(&metrics).global_counter("num_predicate_creation_errors");

        conf = conf.with_metrics(metrics);
        conf.predicate = Some(Arc::clone(&predicate));

        match PruningPredicate::try_new(Arc::clone(&predicate), Arc::clone(&file_schema)) {
            Ok(pruning_predicate) => {
                if !pruning_predicate.always_true() {
                    conf.pruning_predicate = Some(Arc::new(pruning_predicate));
                }
            }
            Err(e) => {
                log::debug!("Could not create pruning predicate for: {e}");
                predicate_creation_errors.add(1);
            }
        };

        let page_pruning_predicate = Arc::new(PagePruningAccessPlanFilter::new(
            &predicate,
            Arc::clone(&file_schema),
        ));
        conf.page_pruning_predicate = Some(page_pruning_predicate);

        conf
    }

    /// Create a new LiquidParquetSource from a ParquetSource
    pub fn from_parquet_source(
        source: ParquetSource,
        liquid_cache: LiquidCacheRef,
        liquid_cache_mode: LiquidCacheMode,
    ) -> Self {
        Self {
            table_parquet_options: source.table_parquet_options().clone(),
            batch_size: Some(liquid_cache.batch_size()),
            liquid_cache,
            liquid_cache_mode,
            metrics: source.metrics().clone(),
            predicate: source.predicate().cloned(),
            pruning_predicate: source.pruning_predicate().cloned(),
            page_pruning_predicate: source.page_pruning_predicate().cloned(),
            metadata_size_hint: source.table_parquet_options().global.metadata_size_hint,
            projected_statistics: Some(source.statistics().unwrap()),
        }
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
    ) -> Arc<dyn datafusion::datasource::physical_plan::FileOpener> {
        let projection: Vec<usize> = base_config
            .projection
            .as_ref()
            .map(|p| {
                p.iter()
                    .filter(|col_idx| **col_idx < base_config.file_schema.fields().len())
                    .copied()
                    .collect()
            })
            .unwrap_or_else(|| (0..base_config.file_schema.fields().len()).collect());

        let reader_factory = Arc::new(CachedMetaReaderFactory::new(object_store));
        let schema_adapter = Arc::new(DefaultSchemaAdapterFactory);

        let opener = LiquidParquetOpener {
            partition_index: partition,
            projection: Arc::from(projection),
            batch_size: self
                .batch_size
                .expect("Batch size must be set before creating LiquidParquetOpener"),
            limit: base_config.limit,
            predicate: self.predicate.clone(),
            pruning_predicate: self.pruning_predicate.clone(),
            page_pruning_predicate: self.page_pruning_predicate.clone(),
            table_schema: Arc::clone(&base_config.file_schema),
            metrics: self.metrics.clone(),
            parquet_file_reader_factory: reader_factory,
            reorder_filters: self.reorder_filters(),
            schema_adapter_factory: schema_adapter,
            liquid_cache: self.liquid_cache.clone(),
            liquid_cache_mode: self.liquid_cache_mode,
        };

        Arc::new(opener)
    }

    fn with_batch_size(&self, batch_size: usize) -> Arc<dyn FileSource> {
        let mut conf = self.clone();
        conf.batch_size = Some(batch_size);
        Arc::new(conf)
    }

    fn with_schema(&self, _schema: SchemaRef) -> Arc<dyn FileSource> {
        Arc::new(Self { ..self.clone() })
    }

    fn with_projection(&self, _config: &FileScanConfig) -> Arc<dyn FileSource> {
        Arc::new(Self { ..self.clone() })
    }

    fn with_statistics(&self, statistics: Statistics) -> Arc<dyn FileSource> {
        Arc::new(Self {
            projected_statistics: Some(statistics),
            ..self.clone()
        })
    }

    fn metrics(&self) -> &ExecutionPlanMetricsSet {
        &self.metrics
    }

    fn statistics(&self) -> Result<Statistics> {
        let statistics = &self.projected_statistics;
        let statistics = statistics
            .clone()
            .expect("projected_statistics must be set");
        // When filters are pushed down, we have no way of knowing the exact statistics.
        // Note that pruning predicate is also a kind of filter pushdown.
        // (bloom filters use `pruning_predicate` too)
        if self.pruning_predicate.is_some()
            || self.page_pruning_predicate.is_some()
            || (self.predicate.is_some() && self.reorder_filters())
        {
            Ok(statistics.to_inexact())
        } else {
            Ok(statistics)
        }
    }

    fn file_type(&self) -> &str {
        "liquid_parquet"
    }
}
