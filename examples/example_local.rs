use datafusion::prelude::SessionConfig;
use liquid_cache_local::storage::cache::{CacheAdvice, CacheEntryID};
use liquid_cache_local::storage::policies::{CachePolicy, DiscardPolicy};
use liquid_cache_local::{LiquidCacheLocalBuilder, common::LiquidCacheMode};
use tempfile::TempDir;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let temp_dir = TempDir::new().unwrap();

    let (ctx, _) = LiquidCacheLocalBuilder::new()
        .with_max_cache_bytes(1024 * 1024 * 1024) // 1GB
        .with_cache_dir(temp_dir.path().to_path_buf())
        .with_cache_mode(LiquidCacheMode::Liquid)
        .with_cache_strategy(Box::new(DiscardPolicy))
        .build(SessionConfig::new())?;

    ctx.register_parquet("hits", "examples/nano_hits.parquet", Default::default())
        .await?;

    ctx.sql("SELECT COUNT(*) FROM hits").await?.show().await?;
    Ok(())
}

#[derive(Debug, Default)]
pub struct CustomPolicy;

impl CachePolicy for CustomPolicy {
    fn advise(&self, _entry_id: &CacheEntryID, _cache_mode: &LiquidCacheMode) -> CacheAdvice {
        CacheAdvice::Discard
    }
}
