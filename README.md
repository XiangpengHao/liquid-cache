<p align="center"> <img src="https://raw.githubusercontent.com/XiangpengHao/liquid-cache/main/dev/doc/logo.png" alt="liquid_cache_logo" width="450"/> </p>

<div align="center">

[![Crates.io Version](https://img.shields.io/crates/v/liquid-cache-client?label=liquid-cache-client)](https://crates.io/crates/liquid-cache-client)
[![Crates.io Version](https://img.shields.io/crates/v/liquid-cache-server?label=liquid-cache-server)](https://crates.io/crates/liquid-cache-server)
[![docs.rs](https://img.shields.io/docsrs/liquid-cache-client?style=flat&label=client-doc)](https://docs.rs/liquid-cache-client/latest/liquid_cache_client/)
[![docs.rs](https://img.shields.io/docsrs/liquid-cache-server?style=flat&label=server-doc)](https://docs.rs/liquid-cache-server/latest/liquid_cache_server/)

</div>
<div align="center">

[![Rust CI](https://github.com/XiangpengHao/liquid-cache/actions/workflows/ci.yml/badge.svg)](https://github.com/XiangpengHao/liquid-cache/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/XiangpengHao/liquid-cache/graph/badge.svg?token=yTeQR2lVnd)](https://codecov.io/gh/XiangpengHao/liquid-cache)
[![Codacy Badge](https://app.codacy.com/project/badge/Grade/1a23a108cd2b4d2b9ffd2c2258599dfa)](https://app.codacy.com/gh/XiangpengHao/liquid-cache/dashboard?utm_source=gh&utm_medium=referral&utm_content=&utm_campaign=Badge_grade)

</div>

LiquidCache is a pushdown cache for S3.
Projections, filters, and aggregations are pushed down to cache server before sending data to [DataFusion](https://github.com/apache/datafusion).

## Features
LiquidCache is a radical redesign of caching: it **caches logical data** rather than its physical representations.

This means that:
- LiquidCache transcodes S3 data (e.g., JSON, CSV, Parquet) into an in-house format -- more compressed, more NVMe friendly, more efficient for DataFusion operations. 
- LiquidCache returns filtered/aggregated data to DataFusion, significantly reducing network IO.

Cons:
- LiquidCache is not a transparent cache (consider [Foyer](https://github.com/foyer-rs/foyer) instead), it leverages query semantics to optimize caching. 
## Architecture

Both LiquidCache and DataFusion run on cloud servers within the same region, but are configured differently:

- LiquidCache often has a memory/CPU ratio of 16:1 (e.g., 64GB memory and 4 cores)
- DataFusion often has a memory/CPU ratio of 2:1 (e.g., 32GB memory and 16 cores)

Multiple DataFusion nodes share the same LiquidCache instance through network connections. 
Each component can be scaled independently as the workload grows. 

<img src="https://raw.githubusercontent.com/XiangpengHao/liquid-cache/main/dev/doc/arch.png" alt="architecture" width="400"/>


## Integrate LiquidCache in 5 Minutes
Check out the [examples](https://github.com/XiangpengHao/liquid-cache/tree/main/examples) folder for more details. 



#### 1. Start a Cache Server:
```rust
#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let liquid_cache = LiquidCacheService::new(
        SessionContext::new(),
        Some(1024 * 1024 * 1024),               // max memory cache size 1GB
        Some(tempfile::tempdir()?.into_path()), // disk cache dir
    );

    let flight = FlightServiceServer::new(liquid_cache);

    Server::builder()
        .add_service(flight)
        .serve("0.0.0.0:50051".parse()?)
        .await?;

    Ok(())
}
```

Or use our pre-built docker image:
```bash
docker run -p 50051:50051 -v ~/liquid_cache:/cache \
  ghcr.io/xiangpenghao/liquid-cache/liquid-cache-server:latest \
  /app/bench_server \
  --address 0.0.0.0:50051 \
  --disk-cache-dir /cache
```

#### 2. Connect to the cache server:
Add the following dependency to your existing DataFusion project:
```toml
[dependencies]
liquid-cache-client = "0.1.0"
```

Then, create a new DataFusion context with LiquidCache:
```rust
#[tokio::main]
pub async fn main() -> Result<()> {
/*==========================LiquidCache============================*/
    let ctx = LiquidCacheBuilder::new(cache_server)
        .with_object_store(ObjectStoreUrl::parse(object_store_url.as_str())?, None)
        .with_cache_mode(CacheMode::Liquid)
        .build(SessionConfig::from_env()?)?;
/*=================================================================*/

    let ctx: Arc<SessionContext> = Arc::new(ctx);
    ctx.register_table(table_name, ...)
        .await?;
    ctx.sql(&sql).await?.show().await?;
    Ok(())
}
```

## Community server

We run a community server for LiquidCache at <https://hex.tail0766e4.ts.net:50051> (hosted on Xiangpeng's NAS, use at your own risk).

You can try it out by running:
```bash
cargo run --bin example_client --release -- \
    --cache-server https://hex.tail0766e4.ts.net:50051 \
    --file "https://huggingface.co/datasets/HuggingFaceFW/fineweb/resolve/main/data/CC-MAIN-2024-51/000_00042.parquet" \
    --query "SELECT COUNT(*) FROM \"000_00042\" WHERE \"token_count\" < 100"
```

Expected output (within a second):
```
+----------+
| count(*) |
+----------+
| 44805    |
+----------+
```


## Run ClickBench 

#### 1. Setup the Repository
```bash
git clone https://github.com/XiangpengHao/liquid-cache.git
cd liquid-cache
```

#### 2. Run a LiquidCache Server
```bash
cargo run --bin bench_server --release
```

#### 3. Run a ClickBench Client
In a different terminal, run the ClickBench client:
```bash
cargo run --bin clickbench_client --release -- --query-path benchmark/clickbench/queries.sql --file examples/nano_hits.parquet
```
(Note: replace `nano_hits.parquet` with the [real ClickBench dataset](https://github.com/ClickHouse/ClickBench) for full benchmarking)


## Development

See [dev/README.md](./dev/README.md)

## Benchmark

See [benchmark/README.md](./benchmark/README.md)

## FAQ

#### Can I use LiquidCache in production today?

Not yet. While production readiness is our goal, we are still implementing features and polishing the system.
LiquidCache began as a research project exploring new approaches to build cost-effective caching systems. Like most research projects, it takes time to mature, and we welcome your help!

#### Does LiquidCache cache data or results?

LiquidCache is a data cache. It caches logically equivalent but physically different data from object storage.

LiquidCache does not cache query results - it only caches data, allowing the same cache to be used for different queries.

#### Nightly Rust, seriously?

We will transition to stable Rust once we believe the project is ready for production.

#### How does LiquidCache work?

Check out our [paper](/dev/doc/liquid-cache-vldb.pdf) (under submission to VLDB) for more details. Meanwhile, we are working on a technical blog to introduce LiquidCache in a more accessible way.

#### How can I get involved?

We are always looking for contributors! Any feedback or improvements are welcome. Feel free to explore the issue list and contribute to the project.
If you want to get involved in the research process, feel free to [reach out](https://xiangpeng.systems/work-with-me/).

#### Who is behind LiquidCache?

LiquidCache is a research project funded by:
- [InfluxData](https://www.influxdata.com/)
- Taxpayers of the state of Wisconsin and the federal government. 

As such, LiquidCache is and will always be open source and free to use.

Your support for science is greatly appreciated!

## License

[Apache License 2.0](./LICENSE)