use datafusion::prelude::SessionConfig;
use liquid_cache_datafusion_local::LiquidCacheLocalBuilder;
use liquid_cache_datafusion_local::storage::cache::squeeze_policies::TranscodeSqueezeEvict;
use liquid_cache_datafusion_local::storage::cache_policies::LiquidPolicy;
use tempfile::TempDir;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let temp_dir = TempDir::new().unwrap();

    let (ctx, _) = LiquidCacheLocalBuilder::new()
        .with_max_cache_bytes(1024 * 1024 * 1024) // 1GB
        .with_cache_dir(temp_dir.path().to_path_buf())
        .with_squeeze_policy(Box::new(TranscodeSqueezeEvict))
        .with_cache_policy(Box::new(LiquidPolicy::new()))
        .build(SessionConfig::new())?;

    ctx.register_parquet("hits", "examples/nano_hits.parquet", Default::default())
        .await?;

    ctx.sql("SELECT COUNT(*) FROM hits").await?.show().await?;
    Ok(())
}
