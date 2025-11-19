#![warn(missing_docs)]
#![doc = include_str!("../README.md")]

#[cfg(test)]
mod tests;

use std::path::PathBuf;
use std::sync::Arc;

use datafusion::error::Result;
use datafusion::logical_expr::ScalarUDF;
use datafusion::prelude::{SessionConfig, SessionContext};
use liquid_cache_common::IoMode;
use liquid_cache_parquet::optimizers::{LineageOptimizer, LocalModeOptimizer};
use liquid_cache_parquet::{
    LiquidCache, LiquidCacheRef, VariantGetUdf, VariantPretty, VariantToJsonUdf,
};
use liquid_cache_storage::cache::squeeze_policies::{SqueezePolicy, TranscodeSqueezeEvict};
use liquid_cache_storage::cache_policies::CachePolicy;
use liquid_cache_storage::cache_policies::LiquidPolicy;

pub use liquid_cache_common as common;
pub use liquid_cache_storage as storage;

/// Builder for in-process liquid cache session context
///
/// This allows you to use liquid cache within the same process,
/// instead of using the client-server architecture as in the default mode.
///
/// # Example
/// ```rust
/// use liquid_cache_local::{
///     storage::cache_policies::LiquidPolicy,
///     LiquidCacheLocalBuilder,
/// };
/// use datafusion::prelude::{SessionConfig, SessionContext};
/// use tempfile::TempDir;
///
/// #[tokio::main]
/// async fn main() -> Result<(), Box<dyn std::error::Error>> {
///     let temp_dir = TempDir::new().unwrap();
///
///     let (ctx, _) = LiquidCacheLocalBuilder::new()
///         .with_max_cache_bytes(1024 * 1024 * 1024) // 1GB
///         .with_cache_dir(temp_dir.path().to_path_buf())
///         .with_cache_policy(Box::new(LiquidPolicy::new()))
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
pub struct LiquidCacheLocalBuilder {
    /// Size of batches for caching
    batch_size: usize,
    /// Maximum cache size in bytes
    max_cache_bytes: usize,
    /// Directory for disk cache
    cache_dir: PathBuf,
    /// Cache policy
    cache_policy: Box<dyn CachePolicy>,
    /// Squeeze policy
    squeeze_policy: Box<dyn SqueezePolicy>,

    span: fastrace::Span,

    io_mode: IoMode,

    eager_shredding: bool,
}

impl Default for LiquidCacheLocalBuilder {
    fn default() -> Self {
        Self {
            batch_size: 8192,
            max_cache_bytes: 1024 * 1024 * 1024, // 1GB
            cache_dir: std::env::temp_dir().join("liquid_cache"),
            cache_policy: Box::new(LiquidPolicy::new()),
            squeeze_policy: Box::new(TranscodeSqueezeEvict),
            span: fastrace::Span::enter_with_local_parent("liquid_cache_local_builder"),
            io_mode: IoMode::StdBlocking,
            eager_shredding: false,
        }
    }
}

impl LiquidCacheLocalBuilder {
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

    /// Set squeeze policy
    pub fn with_squeeze_policy(mut self, squeeze_policy: Box<dyn SqueezePolicy>) -> Self {
        self.squeeze_policy = squeeze_policy;
        self
    }

    /// Set cache strategy
    pub fn with_cache_policy(mut self, cache_policy: Box<dyn CachePolicy>) -> Self {
        self.cache_policy = cache_policy;
        self
    }

    /// Set fastrace span
    pub fn with_span(mut self, span: fastrace::Span) -> Self {
        self.span = span;
        self
    }

    /// Set IO mode
    pub fn with_io_mode(mut self, io_mode: IoMode) -> Self {
        self.io_mode = io_mode;
        self
    }

    /// Set enable shredding
    pub fn with_eager_shredding(mut self, eager_shredding: bool) -> Self {
        self.eager_shredding = eager_shredding;
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
        config.options_mut().execution.parquet.skip_arrow_metadata = false;
        config.options_mut().execution.parquet.skip_metadata = false;
        config.options_mut().execution.batch_size = self.batch_size;

        let cache = LiquidCache::new(
            self.batch_size,
            self.max_cache_bytes,
            self.cache_dir,
            self.cache_policy,
            self.squeeze_policy,
            self.io_mode,
        );
        let cache_ref = Arc::new(cache);

        let date_extract_optimizer = Arc::new(LineageOptimizer::new());

        let optimizer = LocalModeOptimizer::new(cache_ref.clone(), self.eager_shredding);

        let state = datafusion::execution::SessionStateBuilder::new()
            .with_config(config)
            .with_default_features()
            .with_optimizer_rule(date_extract_optimizer)
            .with_physical_optimizer_rule(Arc::new(optimizer))
            .build();

        let ctx = SessionContext::new_with_state(state);
        ctx.register_udf(ScalarUDF::new_from_impl(VariantGetUdf::default()));
        ctx.register_udf(ScalarUDF::new_from_impl(VariantPretty::default()));
        ctx.register_udf(ScalarUDF::new_from_impl(VariantToJsonUdf::default()));
        Ok((ctx, cache_ref))
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
        let (ctx, _) = LiquidCacheLocalBuilder::new().build(SessionConfig::new())?;
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
        let (ctx, _) = LiquidCacheLocalBuilder::new().build(SessionConfig::new())?;

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
}
