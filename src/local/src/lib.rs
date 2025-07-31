#[cfg(test)]
mod tests;

use std::path::PathBuf;
use std::sync::Arc;

use datafusion::config::ConfigOptions;
use datafusion::error::Result;
use datafusion::physical_optimizer::PhysicalOptimizerRule;
use datafusion::physical_plan::ExecutionPlan;
use datafusion::prelude::{SessionConfig, SessionContext};
use liquid_cache_common::LiquidCacheMode;
use liquid_cache_parquet::{LiquidCache, LiquidCacheRef, rewrite_data_source_plan};
use liquid_cache_store::store::CachePolicy;
use liquid_cache_store::store::policies::FiloPolicy;

pub use liquid_cache_common as common;
pub use liquid_cache_store as store;

/// Builder for in-process liquid cache session context
///
/// This allows you to use liquid cache within the same process,
/// instead of using the client-server architecture as in the default mode.
///
/// # Example
/// ```rust
/// use liquid_cache_local::{
///     common::{LiquidCacheMode},
///     LiquidCacheInProcessBuilder,
/// };
/// use datafusion::prelude::{SessionConfig, SessionContext};
/// use tempfile::TempDir;
///
/// #[tokio::main]
/// async fn main() -> Result<(), Box<dyn std::error::Error>> {
///     use liquid_cache_local::store::policies::FiloPolicy;
/// let temp_dir = TempDir::new().unwrap();
///
///     let (ctx, _) = LiquidCacheInProcessBuilder::new()
///         .with_max_cache_bytes(1024 * 1024 * 1024) // 1GB
///         .with_cache_dir(temp_dir.path().to_path_buf())
///         .with_cache_mode(LiquidCacheMode::Liquid { transcode_in_background: false })
///         .with_cache_strategy(Box::new(FiloPolicy::new()))
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
#[derive(Debug)]
pub struct LiquidCacheInProcessBuilder {
    /// Size of batches for caching
    batch_size: usize,
    /// Maximum cache size in bytes
    max_cache_bytes: usize,
    /// Directory for disk cache
    cache_dir: PathBuf,
    /// Cache mode (`LiquidCacheMode::Arrow` or `LiquidCacheMode::Liquid`)
    cache_mode: LiquidCacheMode,
    /// Cache eviction strategy
    cache_strategy: Box<dyn CachePolicy>,
}

impl Default for LiquidCacheInProcessBuilder {
    fn default() -> Self {
        Self {
            batch_size: 8192 * 2,
            max_cache_bytes: 1024 * 1024 * 1024, // 1GB
            cache_dir: std::env::temp_dir().join("liquid_cache"),
            cache_mode: LiquidCacheMode::default(),
            cache_strategy: Box::new(FiloPolicy::new()),
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
    pub fn with_cache_strategy(mut self, cache_strategy: Box<dyn CachePolicy>) -> Self {
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
        config.options_mut().execution.batch_size = self.batch_size;

        let cache = LiquidCache::new(
            self.batch_size,
            self.max_cache_bytes,
            self.cache_dir,
            self.cache_mode,
            self.cache_strategy,
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
mod local_tests {
    use arrow_schema::{DataType, Field, Schema};
    use datafusion::datasource::{
        file_format::parquet::ParquetFormat,
        listing::{ListingOptions, ListingTableUrl},
    };

    use super::*;

    #[tokio::test]
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

    #[tokio::test]
    async fn test_provide_schema() -> Result<()> {
        let (ctx, _) = LiquidCacheInProcessBuilder::new().build(SessionConfig::new())?;

        let file_format = ParquetFormat::default().with_enable_pruning(true);
        let listing_options =
            ListingOptions::new(Arc::new(file_format)).with_file_extension(".parquet");

        let table_path = ListingTableUrl::parse("../../examples/nano_hits.parquet")?;
        let schema = Schema::new(vec![Field::new("WatchID", DataType::Int64, false)]);

        ctx.register_listing_table(
            "hits",
            &table_path,
            listing_options.clone(),
            Some(Arc::new(schema)),
            None,
        )
        .await?;

        ctx.sql("SELECT \"WatchID\" FROM hits limit 1")
            .await?
            .show()
            .await?;
        Ok(())
    }

    #[tokio::test]
    async fn test_provide_schema2() -> Result<()> {
        let df_ctx = SessionContext::new();
        let liquid_ctx = {
            let (ctx, _) = LiquidCacheInProcessBuilder::new()
                .with_cache_mode(LiquidCacheMode::Liquid {
                    transcode_in_background: false,
                })
                .build(SessionConfig::new())?;
            ctx
        };

        let file_format = ParquetFormat::default().with_enable_pruning(true);
        let listing_options =
            ListingOptions::new(Arc::new(file_format)).with_file_extension(".parquet");

        let table_path = ListingTableUrl::parse("../../dev/test_parquet/openobserve.parquet")?;
        let schema = Arc::new(Schema::new(vec![
            Field::new("_timestamp", DataType::Int64, false),
            Field::new("log", DataType::Utf8, true),
            Field::new("message", DataType::Utf8, true),
            Field::new("kubernetes_namespace_name", DataType::Utf8, false),
        ]));

        df_ctx
            .register_listing_table(
                "default",
                &table_path,
                listing_options.clone(),
                Some(schema.clone()),
                None,
            )
            .await?;
        liquid_ctx
            .register_listing_table(
                "default",
                &table_path,
                listing_options.clone(),
                Some(schema.clone()),
                None,
            )
            .await?;

        let sql_to_tests = [
            "SELECT * from default where log like '%hhj%' order by _timestamp",
            "SELECT date_bin(interval '10 second', to_timestamp_micros(_timestamp), to_timestamp('2001-01-01T00:00:00')) AS zo_sql_key, count(*) AS zo_sql_num from default WHERE log like '%hhj%' or message like '%hhj%' GROUP BY zo_sql_key ORDER BY zo_sql_key",
            "SELECT _timestamp, kubernetes_namespace_name from default order by _timestamp desc limit 100",
        ];
        for sql in sql_to_tests {
            for _i in 0..3 {
                let df_results = df_ctx.sql(sql).await?.collect().await?;
                let liquid_results = liquid_ctx.sql(sql).await?.collect().await?;
                assert_eq!(df_results, liquid_results);
            }
        }
        Ok(())
    }
}
