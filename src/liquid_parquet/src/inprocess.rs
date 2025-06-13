use std::path::PathBuf;
use std::sync::Arc;

use datafusion::config::ConfigOptions;
use datafusion::error::Result;
use datafusion::physical_optimizer::PhysicalOptimizerRule;
use datafusion::physical_plan::ExecutionPlan;
use datafusion::prelude::{SessionConfig, SessionContext};
use liquid_cache_common::{CacheEvictionStrategy, LiquidCacheMode};

use crate::cache::policies::CachePolicy;
use crate::utils::rewrite_data_source_plan;
use crate::{LiquidCache, LiquidCacheRef};

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

        let optimizer = InProcessOptimizer::with_cache(cache_ref.clone());

        let state = datafusion::execution::SessionStateBuilder::new()
            .with_config(config)
            .with_default_features()
            .with_physical_optimizer_rule(Arc::new(optimizer))
            .build();

        Ok((SessionContext::new_with_state(state), cache_ref))
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
}

impl PhysicalOptimizerRule for InProcessOptimizer {
    fn optimize(
        &self,
        plan: Arc<dyn ExecutionPlan>,
        _config: &ConfigOptions,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        Ok(rewrite_data_source_plan(plan, &self.cache))
    }

    fn name(&self) -> &str {
        "InProcessLiquidCacheOptimizer"
    }

    fn schema_check(&self) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use datafusion::datasource::{
        file_format::parquet::ParquetFormat,
        listing::{ListingOptions, ListingTableUrl},
    };

    use super::*;

    #[tokio::test(flavor = "multi_thread")]
    async fn register_with_listing_table() -> Result<()> {
        let file_format = ParquetFormat::default().with_enable_pruning(true);
        let listing_options =
            ListingOptions::new(Arc::new(file_format)).with_file_extension(".parquet");
        let (ctx, _) = LiquidCacheInProcessBuilder::new().build(SessionConfig::new())?;
        let table_path = ListingTableUrl::parse("../../examples/nano_hits.parquet")?;
        ctx.register_listing_table("hits", &table_path, listing_options.clone(), None, None)
            .await?;

        ctx.sql("SELECT * FROM hits where \"URL\" like '%google%'")
            .await?
            .show()
            .await?;
        Ok(())
    }
}
