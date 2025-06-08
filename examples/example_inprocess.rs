use datafusion::prelude::SessionConfig;
use liquid_cache_parquet::{
    LiquidCacheInProcessBuilder,
    common::{CacheEvictionStrategy, LiquidCacheMode},
};
use tempfile::TempDir;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let temp_dir = TempDir::new().unwrap();

    let (ctx, _) = LiquidCacheInProcessBuilder::new()
        .with_max_cache_bytes(1024 * 1024 * 1024) // 1GB
        .with_cache_dir(temp_dir.path().to_path_buf())
        .with_cache_mode(LiquidCacheMode::Liquid {
            transcode_in_background: true,
        })
        .with_cache_strategy(CacheEvictionStrategy::Discard)
        .build(SessionConfig::new())?;

    ctx.register_parquet("hits", "examples/nano_hits.parquet", Default::default())
        .await?;

    ctx.sql("SELECT COUNT(*) FROM hits").await?.show().await?;
    Ok(())
}
