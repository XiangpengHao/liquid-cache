# liquid-cache-local

Local LiquidCache for Apache DataFusion.

This crate provides an in-process version of LiquidCache that doesn't require a separate server. 

## Usage

```rust
use liquid_cache_local::{
    common::LiquidCacheMode,
    LiquidCacheLocalBuilder,
    storage::policies::FiloPolicy,
};
use datafusion::prelude::SessionConfig;
use tempfile::TempDir;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let temp_dir = TempDir::new().unwrap();

    let (ctx, _cache) = LiquidCacheLocalBuilder::new()
        .with_max_cache_bytes(1024 * 1024 * 1024) // 1GB
        .with_cache_dir(temp_dir.path().to_path_buf())
        .with_cache_mode(LiquidCacheMode::Liquid { transcode_in_background: false })
        .with_cache_strategy(Box::new(FiloPolicy::new()))
        .build(SessionConfig::new())?;

    // Register the test parquet file
    ctx.register_parquet("hits", "../../examples/nano_hits.parquet", Default::default())
        .await?;

    ctx.sql("SELECT COUNT(*) FROM hits").await?.show().await?;
    Ok(())
}
```