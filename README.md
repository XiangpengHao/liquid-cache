<p align="center"> <img src="/dev/doc/logo.png" alt="liquid_cache_logo" width="450"/> </p>


[![Rust CI](https://github.com/XiangpengHao/liquid-cache/actions/workflows/ci.yml/badge.svg)](https://github.com/XiangpengHao/liquid-cache/actions/workflows/ci.yml)

## Architecture

Welcome to LiquidCache's disaggregated architecture! 🚀

Our design is beautifully simple yet powerful - we run both cache and compute nodes on standard cloud servers, but each is specially configured for what it does best:

- Cache servers come packed with **high memory** and **modest CPU power**
- Compute servers feature **powerful CPUs** with just enough memory for their tasks

Everything stays connected through speedy modern networks, making it easy for multiple compute servers to share a single cache server. 
The best part? You can **scale each component independently** as your workload grows. Need more compute power? Add compute nodes! Need more caching? Expand your cache servers! It's that simple.

<img src="/dev/doc/arch.png" alt="architecture" width="400"/>


## Run ClickBench to feel the performance

#### 1. Setup repo
```bash
git clone https://github.com/XiangpengHao/liquid-cache.git
cd liquid-cache
```

#### 2. Run a LiquidCache server
```bash
cargo run --bin bench_server --release
```

#### 3. Run a ClickBench client
In a different terminal, run the ClickBench client.
```bash
cargo run --bin clickbench_client --release -- --query-path benchmark/queries.sql --file examples/nano_hits.parquet
```

## Try LiquidCache
Checkout the `examples` folder for more details. We are working on a crates.io release, stay tuned!

#### 1. Start a cache server:
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

#### 2. Connect to the cache server:
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

#### 3. Enjoy!


## Development

See [dev/README.md](./dev/README.md)

## Benchmark

See [benchmark/README.md](./benchmark/README.md)

## FAQ

**Can I use LiquidCache in production today?**

No. While production-ready is our goal, we are still working on implementing more features and polishing it.
LiquidCache starts with a research project, mainly to demonstrate a new approach to build cost-effective caching systems. Like most research projects, it takes time to mature, we welcome your help!

**Nightly Rust, seriously?**

We will use stable Rust once we believe the project is ready for production.

**How LiquidCache works?**

We are currently submitting a paper to VLDB (as of Mar 1, 2025), in the meanwhile, we are happy to share our paper on request.
We are also working on a tech blog to introduce LiquidCache in a more techy way.

**How can I get involved?**

We are always looking for contributors, any feedback/improvement is welcome! Feel free to take a look at the issue list and contribute to the project.
If you want to get involved in the research process, feel free to [reach out](https://haoxp.xyz/work-with-me/).

**Who made this?**

LiquidCache is a research project funded by:
- [InfluxData](https://www.influxdata.com/)
- Taxpayers of the state of Wisconsin and federal government. 

As such, LiquidCache is and will always be open source and free to use.

Your support to science is greatly appreciated!

## License

This project is licensed under the [Apache License 2.0](./LICENSE).
