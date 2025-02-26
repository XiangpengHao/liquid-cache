<p align="center"> <img src="/dev/doc/liquid_cache.png" alt="liquid_cache_logo" width="450"/> </p>


<p align="center"> Cache that understands your data and cuts your S3 bill by 10x. </p>

[![Rust CI](https://github.com/XiangpengHao/liquid-cache/actions/workflows/ci.yml/badge.svg)](https://github.com/XiangpengHao/liquid-cache/actions/workflows/ci.yml)

## Architecture

![architecture](/dev/doc/arch.svg)


## Try in 5 minutes!
Checkout the `examples` folder for more details.

### 1. Add dependency
```toml
[dependencies]
liquid-cache-common = "0.1.0"
liquid-cache-client = "0.1.0"
liquid-cache-server = "0.1.0"
```

### 2. Start a cache server:
```rust
use arrow_flight::flight_service_server::FlightServiceServer;
use liquid_cache_server::LiquidCacheService;
use tonic::transport::Server;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let addr = "0.0.0.0:50051".parse()?;

    let liquid_cache = LiquidCacheService::try_new()?;
    let flight = FlightServiceServer::new(liquid_cache);

    info!("LiquidCache server listening on {addr:?}");

    Server::builder().add_service(flight).serve(addr).await?;

    Ok(())
}
```

### 3. Connect to the cache server:
```rust
use std::sync::Arc;
use datafusion::{
    error::Result,
    prelude::{SessionConfig, SessionContext},
};
use liquid_cache_client::LiquidCacheTableFactory;
use liquid_common::ParquetMode;
use url::Url;

#[tokio::main]
pub async fn main() -> Result<()> {
    let mut session_config = SessionConfig::from_env()?;
    session_config
        .options_mut()
        .execution
        .parquet
        .pushdown_filters = true;
    let ctx = Arc::new(SessionContext::new_with_config(session_config));

    let entry_point = "http://localhost:50051";
    let sql = "SELECT COUNT(*) FROM nano_hits WHERE \"URL\" <> '';";
    let table_url = Url::parse("file:///examples/nano_hits.parquet").unwrap();

    let table = LiquidCacheTableFactory::open_table(
        entry_point,
        "nano_hits",
        table_url,
        ParquetMode::Liquid,
    )
    .await?;
    ctx.register_table("nano_hits", Arc::new(table))?;

    ctx.sql(sql).await?.show().await?;

    Ok(())
}
```

### 4. Enjoy!


## Development

See [dev/README.md](./dev/README.md)

## Benchmark

See [benchmark/README.md](./benchmark/README.md)