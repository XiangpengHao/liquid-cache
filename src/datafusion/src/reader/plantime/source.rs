use super::LiquidMorselizer;
use crate::cache::{ColumnLineages, LiquidCacheParquetRef};
use datafusion::{
    common::{internal_err, tree_node::TreeNodeRecursion},
    config::{ConfigOptions, TableParquetOptions},
    datasource::{
        physical_plan::{
            FileScanConfig, FileSource, ParquetFileReaderFactory, ParquetSource,
            parquet::{DefaultParquetFileReaderFactory, can_expr_be_pushed_down_with_schemas},
        },
        table_schema::TableSchema,
    },
    error::Result,
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
use object_store::ObjectStore;
use std::{
    fmt::{self, Formatter},
    sync::Arc,
};

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
    parquet_file_reader_factory: Option<Arc<dyn ParquetFileReaderFactory>>,
    span: Option<Arc<fastrace::Span>>,
    lineages: Arc<ColumnLineages>,
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

    /// Attach typed lineage expressions (keyed by file-schema column name) derived
    /// from the query plan. These flow to the cache when the file is opened.
    pub fn with_lineages(&self, lineages: Arc<ColumnLineages>) -> Self {
        Self {
            lineages,
            ..self.clone()
        }
    }

    /// Enable or disable row-group prefetching.
    pub fn with_prefetch(mut self, prefetch: bool) -> Self {
        self.prefetch = prefetch;
        self
    }

    /// The typed lineage expressions currently attached to this source.
    pub fn lineages(&self) -> &Arc<ColumnLineages> {
        &self.lineages
    }

    /// Set predicate information.
    pub fn with_predicate(mut self, predicate: Arc<dyn PhysicalExpr>) -> Self {
        self.predicate = Some(predicate);
        self
    }

    /// Create a new LiquidParquetSource from a ParquetSource
    pub fn from_parquet_source(source: ParquetSource, liquid_cache: LiquidCacheParquetRef) -> Self {
        let predicate = source.filter();
        let parquet_file_reader_factory = source.parquet_file_reader_factory().cloned();

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
            parquet_file_reader_factory,
            predicate: None,
            span: None,
            lineages: Arc::default(),
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

        let parquet_file_reader_factory = self
            .parquet_file_reader_factory
            .clone()
            .unwrap_or_else(|| Arc::new(DefaultParquetFileReaderFactory::new(object_store)));

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
            parquet_file_reader_factory,
            object_store_url: base_config.object_store_url.clone(),
            reorder_filters: self.reorder_filters(),
            expr_adapter_factory,
            span: execution_span.map(Arc::new),
            lineages: Arc::clone(&self.lineages),
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
