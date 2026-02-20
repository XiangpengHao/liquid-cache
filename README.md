<p align="center"> <img src="https://raw.githubusercontent.com/XiangpengHao/liquid-cache/main/dev/doc/logo.png" alt="liquid_cache_logo" width="450"/> </p>

<div align="center">

[![Crates.io Version](https://img.shields.io/crates/v/liquid-cache?label=liquid-cache)](https://crates.io/crates/liquid-cache)
[![docs.rs](https://img.shields.io/docsrs/liquid-cache?style=flat&label=docs)](https://docs.rs/liquid-cache/latest/liquid_cache/)

</div>
<div align="center">

[![Rust CI](https://github.com/XiangpengHao/liquid-cache/actions/workflows/ci.yml/badge.svg)](https://github.com/XiangpengHao/liquid-cache/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/XiangpengHao/liquid-cache/graph/badge.svg?token=yTeQR2lVnd)](https://codecov.io/gh/XiangpengHao/liquid-cache)
[![Codacy Badge](https://app.codacy.com/project/badge/Grade/1a23a108cd2b4d2b9ffd2c2258599dfa)](https://app.codacy.com/gh/XiangpengHao/liquid-cache/dashboard?utm_source=gh&utm_medium=referral&utm_content=&utm_campaign=Badge_grade)
[![ClickBench](https://img.shields.io/badge/ClickBench-passing-brightgreen)](https://github.com/XiangpengHao/liquid-cache/actions/workflows/ci.yml)
[![TPC-H](https://img.shields.io/badge/TPC--H-passing-brightgreen)](https://github.com/XiangpengHao/liquid-cache/actions/workflows/ci.yml)
[![TPC-DS](https://img.shields.io/badge/TPC--DS-passing-brightgreen)](https://github.com/XiangpengHao/liquid-cache/actions/workflows/ci.yml)
</div>

LiquidCache understands both your _data_ and your _query_.
- It transcodes storage data into into an optimized, cache-only format, so you can continue using your favorite formats without worrying about performance.
- It keeps truly important data in memory and makes efficient use of modern SSDs. For example, if your query group by `year`, LiquidCache store only year in memory, and keeps the full timestamp on disk.

LiquidCache is a research project [funded](https://xiangpeng.systems/fund/) by [InfluxData](https://www.influxdata.com/), [SpiralDB](https://spiraldb.com/), and [Bauplan](https://www.bauplanlabs.com).

## Features
LiquidCache is a radical redesign of caching: it **caches logical data** rather than its physical representations.

This means that:
- LiquidCache transcodes S3 data (e.g., JSON, CSV, Parquet) into an in-house format -- more compressed, more NVMe friendly, more efficient for DataFusion operations. 
- LiquidCache returns filtered/aggregated data to DataFusion, significantly reducing network IO.

Cons:
- LiquidCache is not a transparent cache (consider [Foyer](https://github.com/foyer-rs/foyer) instead), it leverages query semantics to optimize caching. 

## Architecture

Multiple DataFusion nodes share the same LiquidCache instance through network connections. 
Each component can be scaled independently as the workload grows. 

<img src="https://raw.githubusercontent.com/XiangpengHao/liquid-cache/main/dev/doc/arch.png" alt="architecture" width="400"/>


## Start a LiquidCache Server in 5 Minutes
Check out the [examples](https://github.com/XiangpengHao/liquid-cache/tree/main/examples) folder for more details. 


#### 1. Create a Cache Server:
```rust
#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let liquid_cache = LiquidCacheService::new(
        SessionContext::new(),
        Some(1024 * 1024 * 1024),               // max memory cache size 1GB
        Some(tempfile::tempdir()?.into_path()), // disk cache dir
    )?;

    let flight = FlightServiceServer::new(liquid_cache);

    Server::builder()
        .add_service(flight)
        .serve("0.0.0.0:15214".parse()?)
        .await?;

    Ok(())
}
```

#### 2. Connect to the cache server:
Add the following dependency to your existing DataFusion project:
```toml
[dependencies]
liquid-cache-datafusion-client = "0.1.0"
```

Then, create a new DataFusion context with LiquidCache:
```rust
#[tokio::main]
pub async fn main() -> Result<()> {
    let ctx = LiquidCacheBuilder::new(cache_server)
        .with_object_store(ObjectStoreUrl::parse(object_store_url.as_str())?, None)
        .build(SessionConfig::from_env()?)?;

    let ctx: Arc<SessionContext> = Arc::new(ctx);
    ctx.register_table(table_name, ...)
        .await?;
    ctx.sql(&sql).await?.show().await?;
    Ok(())
}
```

## In-process mode 

If you are uncomfortable with a dedicated server, LiquidCache also provides an in-process mode via the
`liquid-cache-datafusion-local` crate.

```rust
use datafusion::prelude::SessionConfig;
use liquid_cache_datafusion_local::storage::cache::squeeze_policies::TranscodeSqueezeEvict;
use liquid_cache_datafusion_local::storage::cache_policies::FiloPolicy;
use liquid_cache_datafusion_local::LiquidCacheLocalBuilder;
use tempfile::TempDir;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let temp_dir = TempDir::new().unwrap();

    let (ctx, _cache) = LiquidCacheLocalBuilder::new()
        .with_max_cache_bytes(1024 * 1024 * 1024) // 1GB
        .with_cache_dir(temp_dir.path().to_path_buf())
        .with_squeeze_policy(Box::new(TranscodeSqueezeEvict))
        .with_cache_policy(Box::new(FiloPolicy::new()))
        .build(SessionConfig::new())?;

    ctx.register_parquet("hits", "examples/nano_hits.parquet", Default::default())
        .await?;

    ctx.sql("SELECT COUNT(*) FROM hits").await?.show().await?;
    Ok(())
}

```


## Development

See [dev/README.md](./dev/README.md)

## Benchmark

See [benchmark/README.md](./benchmark/README.md)

## Performance troubleshooting

### Inherit LiquidCache configurations

LiquidCache requires a few non-default DataFusion configurations:

**ListingTable:**
```rust
let (ctx, _) = LiquidCacheLocalBuilder::new().build(config)?;

let listing_options = ParquetReadOptions::default()
    .to_listing_options(&ctx.copied_config(), ctx.copied_table_options());
ctx.register_listing_table("default", &table_path, listing_options, None, None)
    .await?;
```

**Or register Parquet directly:**
```rust
let (ctx, _) = LiquidCacheLocalBuilder::new().build(config)?;
ctx.register_parquet("default", "examples/nano_hits.parquet", Default::default())
    .await?;
```

### Disable background transcoding

For performance testing, disable background transcoding:

```rust
let (ctx, _) = LiquidCacheLocalBuilder::new()
    .with_squeeze_policy(Box::new(
        squeeze_policies::Evict,
    ))
    .build(config)?;
```

### x86-64 optimization

LiquidCache is optimized for x86-64 with specific [instructions](https://github.com/XiangpengHao/liquid-cache/blob/f8d5b77829fa7996a56c031eb25503f7b0b0428d/src/liquid_parquet/src/utils.rs#L229-L327). ARM chips (e.g., Apple Silicon) use fallback implementations. Contributions welcome!


## FAQ

#### Can I use LiquidCache in production today?

Not yet. While production readiness is our goal, we are still implementing features and polishing the system.
LiquidCache began as a research project exploring new approaches to build cost-effective caching systems. Like most research projects, it takes time to mature, and we welcome your help!

#### How does LiquidCache work?

Check out our [paper](/dev/doc/liquid-cache-vldb.pdf) (to appear in VLDB 2026) for more details. Meanwhile, we are working on a technical blog to introduce LiquidCache in a more accessible way.

#### How can I get involved?

We are always looking for contributors! Any feedback or improvements are welcome. Feel free to explore the issue list and contribute to the project.
If you want to get involved in the research process, feel free to [reach out](https://xiangpeng.systems/work-with-me/).

#### Who is behind LiquidCache?

LiquidCache is a research project funded by:
- [SpiralDB](https://spiraldb.com/)
- [InfluxData](https://www.influxdata.com/)
- [Bauplan](https://www.bauplanlabs.com)
- Taxpayers of the state of Wisconsin and the federal government. 

As such, LiquidCache is and will always be open source and free to use.

Your support for science is greatly appreciated!

## License

[Apache License 2.0](./LICENSE)
