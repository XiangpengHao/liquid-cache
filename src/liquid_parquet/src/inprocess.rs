use std::any::Any;
use std::fmt::Formatter;
use std::path::PathBuf;
use std::sync::Arc;
use std::task::{Context, Poll};

use arrow::array::RecordBatch;
use arrow_schema::SchemaRef;
use datafusion::common::Statistics;
use datafusion::common::tree_node::{Transformed, TreeNode, TreeNodeRecursion};
use datafusion::config::ConfigOptions;
use datafusion::datasource::physical_plan::{FileScanConfig, ParquetSource};
use datafusion::datasource::schema_adapter::{DefaultSchemaAdapterFactory, SchemaMapper};
use datafusion::datasource::source::{DataSource, DataSourceExec};
use datafusion::error::Result;
use datafusion::execution::{RecordBatchStream, SendableRecordBatchStream, TaskContext};
use datafusion::physical_optimizer::PhysicalOptimizerRule;
use datafusion::physical_plan::{
    DisplayAs, DisplayFormatType, Distribution, ExecutionPlan, PlanProperties,
    execution_plan::CardinalityEffect,
    metrics::{ExecutionPlanMetricsSet, MetricsSet},
    projection::ProjectionExec,
};
use datafusion::prelude::{SessionConfig, SessionContext};
use futures::{Stream, StreamExt};
use liquid_cache_common::{CacheEvictionStrategy, LiquidCacheMode, coerce_to_liquid_cache_types};

use crate::cache::policies::CachePolicy;
use crate::{LiquidCache, LiquidCacheRef, LiquidParquetSource};

/// Builder for in-process liquid cache session context
///
/// This allows you to use liquid cache within the same process,
/// instead of using the client-server architecture as in the default mode.
///
/// # Example
/// ```rust
/// use liquid_cache_parquet::{
///     common::{CacheEvictionStrategy, LiquidCacheMode},
///     LiquidCacheInProcessBuilder,
/// };
/// use datafusion::prelude::{SessionConfig, SessionContext};
/// use tempfile::TempDir;
///
/// #[tokio::main]
/// async fn main() -> Result<(), Box<dyn std::error::Error>> {
///     let temp_dir = TempDir::new().unwrap();
///
///     let (ctx, _) = LiquidCacheInProcessBuilder::new()
///         .with_max_cache_bytes(1024 * 1024 * 1024) // 1GB
///         .with_cache_dir(temp_dir.path().to_path_buf())
///         .with_cache_mode(LiquidCacheMode::Liquid { transcode_in_background: false })
///         .with_cache_strategy(CacheEvictionStrategy::Discard)
///         .build(SessionConfig::new())?;
///
///     // Register the test parquet file
///     ctx.register_parquet("hits", "../../examples/nano_hits.parquet", Default::default())
///         .await?;
///
///     ctx.sql("SELECT COUNT(*) FROM hits").await?.show().await?;
///     Ok(())
/// }
/// ```
#[derive(Debug, Clone)]
pub struct LiquidCacheInProcessBuilder {
    /// Size of batches for caching
    batch_size: usize,
    /// Maximum cache size in bytes
    max_cache_bytes: usize,
    /// Directory for disk cache
    cache_dir: PathBuf,
    /// Cache mode (InMemoryArrow or InMemoryLiquid)
    cache_mode: LiquidCacheMode,
    /// Cache eviction strategy
    cache_strategy: CacheEvictionStrategy,
}

impl Default for LiquidCacheInProcessBuilder {
    fn default() -> Self {
        Self {
            batch_size: 8192 * 2,
            max_cache_bytes: 1024 * 1024 * 1024, // 1GB
            cache_dir: std::env::temp_dir().join("liquid_cache"),
            cache_mode: LiquidCacheMode::default(),
            cache_strategy: CacheEvictionStrategy::Discard,
        }
    }
}

impl LiquidCacheInProcessBuilder {
    /// Create a new builder with defaults
    pub fn new() -> Self {
        Self::default()
    }

    /// Set batch size
    pub fn with_batch_size(mut self, batch_size: usize) -> Self {
        self.batch_size = batch_size;
        self
    }

    /// Set maximum cache size in bytes
    pub fn with_max_cache_bytes(mut self, max_cache_bytes: usize) -> Self {
        self.max_cache_bytes = max_cache_bytes;
        self
    }

    /// Set cache directory
    pub fn with_cache_dir(mut self, cache_dir: PathBuf) -> Self {
        self.cache_dir = cache_dir;
        self
    }

    /// Set cache mode
    pub fn with_cache_mode(mut self, cache_mode: LiquidCacheMode) -> Self {
        self.cache_mode = cache_mode;
        self
    }

    /// Set cache strategy
    pub fn with_cache_strategy(mut self, cache_strategy: CacheEvictionStrategy) -> Self {
        self.cache_strategy = cache_strategy;
        self
    }

    /// Build a SessionContext with liquid cache configured
    /// Returns the SessionContext and the liquid cache reference
    pub fn build(self, mut config: SessionConfig) -> Result<(SessionContext, LiquidCacheRef)> {
        config.options_mut().execution.parquet.pushdown_filters = true;
        config
            .options_mut()
            .execution
            .parquet
            .schema_force_view_types = false;
        config.options_mut().execution.parquet.binary_as_string = true;
        config.options_mut().execution.batch_size = self.batch_size;

        // Create the cache
        let policy: Box<dyn CachePolicy> = match self.cache_strategy {
            CacheEvictionStrategy::Discard => Box::new(crate::policies::DiscardPolicy),
            CacheEvictionStrategy::Lru => Box::new(crate::policies::LruPolicy::new()),
            CacheEvictionStrategy::Filo => Box::new(crate::policies::FiloPolicy::new()),
            CacheEvictionStrategy::ToDisk => Box::new(crate::policies::ToDiskPolicy::new()),
        };
        let cache = LiquidCache::new(
            self.batch_size,
            self.max_cache_bytes,
            self.cache_dir,
            self.cache_mode,
            policy,
        );
        let cache_ref = Arc::new(cache);

        // Create the optimizer
        let optimizer = InProcessOptimizer::with_cache(cache_ref.clone());

        // Build the session state with the optimizer
        let state = datafusion::execution::SessionStateBuilder::new()
            .with_config(config)
            .with_default_features()
            .with_physical_optimizer_rule(Arc::new(optimizer))
            .build();

        Ok((SessionContext::new_with_state(state), cache_ref))
    }
}

/// Execution plan that wraps a DataSourceExec with liquid cache and handles schema adaptation
#[derive(Debug)]
struct InProcessLiquidCacheExec {
    /// The wrapped execution plan (DataSourceExec with LiquidParquetSource)
    wrapped_plan: Arc<dyn ExecutionPlan>,
    /// The original schema that parent nodes expect
    original_schema: SchemaRef,
    /// Metrics for this execution plan
    metrics: ExecutionPlanMetricsSet,
    /// Plan properties with the original schema
    plan_properties: PlanProperties,
}

impl InProcessLiquidCacheExec {
    /// Create a new InProcessLiquidCacheExec
    fn new(wrapped_plan: Arc<dyn ExecutionPlan>, original_schema: SchemaRef) -> Self {
        use datafusion::physical_expr::EquivalenceProperties;

        // Create equivalence properties with the original schema
        let eq_properties = EquivalenceProperties::new(original_schema.clone());

        // Get properties from wrapped plan but use our schema
        let wrapped_props = wrapped_plan.properties();
        let plan_properties = PlanProperties::new(
            eq_properties,
            wrapped_props.output_partitioning().clone(),
            wrapped_props.emission_type,
            wrapped_props.boundedness,
        );

        Self {
            wrapped_plan,
            original_schema,
            metrics: ExecutionPlanMetricsSet::new(),
            plan_properties,
        }
    }
}

impl DisplayAs for InProcessLiquidCacheExec {
    fn fmt_as(&self, t: DisplayFormatType, f: &mut Formatter<'_>) -> std::fmt::Result {
        match t {
            DisplayFormatType::Default | DisplayFormatType::Verbose => {
                write!(f, "InProcessLiquidCacheExec")
            }
            DisplayFormatType::TreeRender => {
                write!(f, "InProcessLiquidCache")
            }
        }
    }
}

impl ExecutionPlan for InProcessLiquidCacheExec {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn name(&self) -> &str {
        "InProcessLiquidCacheExec"
    }

    fn properties(&self) -> &datafusion::physical_plan::PlanProperties {
        &self.plan_properties
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        vec![&self.wrapped_plan]
    }

    fn with_new_children(
        self: Arc<Self>,
        children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        Ok(Arc::new(Self::new(
            children.into_iter().next().unwrap(),
            self.original_schema.clone(),
        )))
    }

    fn execute(
        &self,
        partition: usize,
        context: Arc<TaskContext>,
    ) -> Result<SendableRecordBatchStream> {
        let wrapped_stream = self.wrapped_plan.execute(partition, context)?;
        let original_schema = self.original_schema.clone();

        Ok(Box::pin(SchemaAdaptingStream::new(
            wrapped_stream,
            original_schema,
        )))
    }

    fn statistics(&self) -> Result<Statistics> {
        self.wrapped_plan.statistics()
    }

    fn required_input_distribution(&self) -> Vec<Distribution> {
        self.wrapped_plan.required_input_distribution()
    }

    fn benefits_from_input_partitioning(&self) -> Vec<bool> {
        self.wrapped_plan.benefits_from_input_partitioning()
    }

    fn supports_limit_pushdown(&self) -> bool {
        self.wrapped_plan.supports_limit_pushdown()
    }

    fn with_fetch(&self, limit: Option<usize>) -> Option<Arc<dyn ExecutionPlan>> {
        self.wrapped_plan.with_fetch(limit).map(|new_wrapped| {
            Arc::new(Self::new(new_wrapped, self.original_schema.clone())) as Arc<dyn ExecutionPlan>
        })
    }

    fn fetch(&self) -> Option<usize> {
        self.wrapped_plan.fetch()
    }

    fn cardinality_effect(&self) -> CardinalityEffect {
        self.wrapped_plan.cardinality_effect()
    }

    fn try_swapping_with_projection(
        &self,
        projection: &ProjectionExec,
    ) -> Result<Option<Arc<dyn ExecutionPlan>>> {
        self.wrapped_plan.try_swapping_with_projection(projection)
    }

    fn metrics(&self) -> Option<MetricsSet> {
        Some(self.metrics.clone_inner())
    }
}

/// Stream that adapts schema from liquid cache types back to original types
struct SchemaAdaptingStream {
    wrapped_stream: SendableRecordBatchStream,
    original_schema: SchemaRef,
    schema_mapper: Option<Arc<dyn SchemaMapper>>,
}

impl SchemaAdaptingStream {
    fn new(wrapped_stream: SendableRecordBatchStream, original_schema: SchemaRef) -> Self {
        Self {
            wrapped_stream,
            original_schema,
            schema_mapper: None,
        }
    }
}

impl Stream for SchemaAdaptingStream {
    type Item = Result<RecordBatch>;

    fn poll_next(
        mut self: std::pin::Pin<&mut Self>,
        cx: &mut Context<'_>,
    ) -> Poll<Option<Self::Item>> {
        match self.wrapped_stream.poll_next_unpin(cx) {
            Poll::Ready(Some(Ok(batch))) => {
                let adapted_batch = if let Some(schema_mapper) = &self.schema_mapper {
                    schema_mapper.map_batch(batch).unwrap()
                } else {
                    // Create schema mapper on first batch
                    let (schema_mapper, _) =
                        DefaultSchemaAdapterFactory::from_schema(self.original_schema.clone())
                            .map_schema(&batch.schema())
                            .unwrap();
                    let adapted_batch = schema_mapper.map_batch(batch).unwrap();
                    self.schema_mapper = Some(schema_mapper);
                    adapted_batch
                };
                Poll::Ready(Some(Ok(adapted_batch)))
            }
            Poll::Ready(Some(Err(e))) => Poll::Ready(Some(Err(e))),
            Poll::Ready(None) => Poll::Ready(None),
            Poll::Pending => Poll::Pending,
        }
    }
}

impl RecordBatchStream for SchemaAdaptingStream {
    fn schema(&self) -> SchemaRef {
        self.original_schema.clone()
    }
}

/// Physical optimizer rule for in-process liquid cache
///
/// This optimizer rewrites DataSourceExec nodes that read Parquet files
/// to use LiquidParquetSource instead of the default ParquetSource
#[derive(Debug)]
struct InProcessOptimizer {
    cache: LiquidCacheRef,
}

impl InProcessOptimizer {
    /// Create an optimizer with an existing cache instance
    fn with_cache(cache: LiquidCacheRef) -> Self {
        Self { cache }
    }

    /// Rewrite a data source plan to use liquid cache
    fn rewrite_data_source_plan(&self, plan: Arc<dyn ExecutionPlan>) -> Arc<dyn ExecutionPlan> {
        let cache_mode = self.cache.cache_mode();

        let rewritten = plan
            .transform_up(|node| {
                let any_plan = node.as_any();
                if let Some(plan) = any_plan.downcast_ref::<DataSourceExec>() {
                    let data_source = plan.data_source();
                    let any_source = data_source.as_any();
                    if let Some(file_scan_config) = any_source.downcast_ref::<FileScanConfig>() {
                        let file_source = file_scan_config.file_source();
                        let any_file_source = file_source.as_any();

                        // Check if this is a ParquetSource (same logic as server code)
                        if let Some(parquet_source) =
                            any_file_source.downcast_ref::<ParquetSource>()
                        {
                            // Save the original schema before coercion
                            let original_schema = plan.schema();

                            // Create a new LiquidParquetSource from the existing ParquetSource
                            let liquid_source = LiquidParquetSource::from_parquet_source(
                                parquet_source.clone(),
                                file_scan_config.file_schema.clone(),
                                self.cache.clone(),
                                *cache_mode,
                            );

                            let mut new_config = file_scan_config.clone();
                            new_config.file_source = Arc::new(liquid_source);

                            // Coerce schema types for liquid cache compatibility
                            let coerced_schema = coerce_to_liquid_cache_types(
                                new_config.file_schema.as_ref(),
                                cache_mode,
                            );

                            new_config.file_schema = Arc::new(coerced_schema);

                            let new_data_source: Arc<dyn DataSource> = Arc::new(new_config);
                            let wrapped_plan = Arc::new(DataSourceExec::new(new_data_source));

                            // Wrap with InProcessLiquidCacheExec to handle schema adaptation
                            let adapter_plan = Arc::new(InProcessLiquidCacheExec::new(
                                wrapped_plan,
                                original_schema,
                            ));

                            return Ok(Transformed::new(
                                adapter_plan,
                                true,
                                TreeNodeRecursion::Continue,
                            ));
                        }
                    }
                }
                Ok(Transformed::no(node))
            })
            .unwrap();

        rewritten.data
    }
}

impl PhysicalOptimizerRule for InProcessOptimizer {
    fn optimize(
        &self,
        plan: Arc<dyn ExecutionPlan>,
        _config: &ConfigOptions,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        Ok(self.rewrite_data_source_plan(plan))
    }

    fn name(&self) -> &str {
        "InProcessLiquidCacheOptimizer"
    }

    fn schema_check(&self) -> bool {
        true
    }
}
