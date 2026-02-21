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

LiquidCache understands both your **data** and your **query**.
- It transcodes storage **data** into an optimized, cache-only format, so you can keep using your favorite formats without worrying about performance.
- It keeps the data that matters in memory and uses modern SSDs efficiently. For example, if your **query** groups by `year`, LiquidCache stores only the year in memory and keeps the full timestamp on disk.

LiquidCache is a research project [funded](https://xiangpeng.systems/fund/) by [InfluxData](https://www.influxdata.com/), [SpiralDB](https://spiraldb.com/), and [Bauplan](https://www.bauplanlabs.com).

You may want to consider [Foyer](https://github.com/foyer-rs/foyer) if you're looking for a black-box cache: easier to setup, but not as "smart" as LiquidCache.

## Quick start

This quick start uses the core cache API from `src/core`.
Add these dependencies to your project: `liquid-cache`, `arrow`, and `datafusion`.
The example below shows insert, get, get with selection, and get with predicate pushdown.

```rust
use arrow::array::{BooleanArray, UInt64Array};
use arrow::buffer::BooleanBuffer;
use datafusion::logical_expr::Operator;
use datafusion::physical_plan::PhysicalExpr;
use datafusion::physical_plan::expressions::{BinaryExpr, Column, Literal};
use datafusion::scalar::ScalarValue;
use liquid_cache::cache::{EntryID, LiquidCacheBuilder};
use std::sync::Arc;

tokio_test::block_on(async {
    let cache = LiquidCacheBuilder::new().build();
    let entry_id = EntryID::from(1);
    let values = Arc::new(UInt64Array::from(vec![10, 11, 12, 13, 14, 15]));

    // 1) insert
    cache.insert(entry_id, values.clone()).await;

    // 2) get
    let all_rows = cache.get(&entry_id).await.expect("entry should exist");

    // 3) get filtered (selection pushdown): keep rows 0, 2, 4
    let selection = BooleanBuffer::from(vec![true, false, true, false, true, false]);
    let selected_rows = cache
        .get(&entry_id)
        .with_selection(&selection)
        .await
        .expect("entry should exist");

    // 4) get with predicate pushdown: col > 12
    let predicate: Arc<dyn PhysicalExpr> = Arc::new(BinaryExpr::new(
        Arc::new(Column::new("col", 0)),
        Operator::Gt,
        Arc::new(Literal::new(ScalarValue::UInt64(Some(12)))),
    ));
    let predicate_mask = cache
        .eval_predicate(&entry_id, &predicate)
        .await
        .expect("entry should exist")
        .expect("predicate should be evaluated in cache");

    // Conceptual expectations:
    assert_eq!(all_rows.as_ref(), values.as_ref()); // [10, 11, 12, 13, 14, 15]
    assert_eq!(selected_rows.as_ref(), &UInt64Array::from(vec![10, 12, 14]));
    assert_eq!(
        predicate_mask,
        BooleanArray::from(vec![
        Some(false),
        Some(false),
        Some(false),
        Some(true),
        Some(true),
        Some(true),
        ]),
    );
});
```

## Development

See [dev/README.md](./dev/README.md)

## Benchmark

See [benchmark/README.md](./benchmark/README.md)

## Performance troubleshooting

### Use LiquidCache with DataFusion

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

LiquidCache is optimized for x86-64 with specific [instructions](https://github.com/XiangpengHao/liquid-cache/blob/f8d5b77829fa7996a56c031eb25503f7b0b0428d/src/liquid_parquet/src/utils.rs#L229-L327). On ARM (e.g., Apple Silicon), fallback implementations are used. Contributions are welcome.


## FAQ

#### Can I use LiquidCache in production today?

Not yet. Production readiness is our goal, but we are still implementing features and polishing the system.
LiquidCache began as a research project exploring new approaches to cost-effective caching. Like most research projects, it takes time to mature—we welcome your help.

#### How does LiquidCache work?

See our [paper](/dev/doc/liquid-cache-vldb.pdf) for details. We are also working on a technical blog to introduce LiquidCache in a more accessible way.

#### How can I get involved?

We are always looking for contributors. Feedback and improvements are welcome—explore the issue list and contribute where you can.
If you want to get involved in the research side, [reach out](https://xiangpeng.systems/work-with-me/).

#### Who is behind LiquidCache?

LiquidCache is a research project funded by:
- [SpiralDB](https://spiraldb.com/)
- [InfluxData](https://www.influxdata.com/)
- [Bauplan](https://www.bauplanlabs.com)
- Taxpayers of the state of Wisconsin and the federal government. 

LiquidCache is and will remain open source and free to use.

Your support for science is greatly appreciated!

## License

[Apache License 2.0](./LICENSE)
