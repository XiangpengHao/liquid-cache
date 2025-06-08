# LiquidCache Parquet

This crate provides a in-process LiquidCache for Parquet files.

Learn more about LiquidCache in the [README](https://github.com/XiangpengHao/liquid-cache/blob/main/README.md) of the main repository.

## Usage

```rust
use datafusion::prelude::SessionConfig;
use liquid_cache_parquet::{
    LiquidCacheInProcessBuilder,
    common::{CacheEvictionStrategy, LiquidCacheMode},
};
use tempfile::TempDir;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let temp_dir = TempDir::new().unwrap();

    let ctx = LiquidCacheInProcessBuilder::new()
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
```
