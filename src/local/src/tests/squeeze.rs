use arrow::{array::AsArray, datatypes::Int64Type, util::pretty::pretty_format_batches};
use datafusion::prelude::SessionConfig;
use tempfile::TempDir;

use crate::LiquidCacheLocalBuilder;

const TEST_FILE: &str = "../../examples/nano_hits.parquet";

#[tokio::test]
async fn basic_squeeze() {
    let cache_dir = TempDir::new().unwrap();
    let (ctx, cache) = LiquidCacheLocalBuilder::new()
        .with_max_cache_bytes(1024 * 128)
        .with_cache_dir(cache_dir.path().to_path_buf())
        .build(SessionConfig::new())
        .unwrap();
    ctx.register_parquet("hits", TEST_FILE, Default::default())
        .await
        .unwrap();

    let plan = ctx
        .sql("SELECT COUNT(DISTINCT(\"EventTime\")) FROM hits WHERE \"WatchID\" <> 0")
        .await
        .unwrap();
    let result = plan.collect().await.unwrap();
    assert_eq!(result.len(), 1);
    assert_eq!(
        result[0].column(0).as_primitive::<Int64Type>().value(0),
        12385
    );
    let trace = cache.consume_event_trace();
    insta::assert_snapshot!(trace);
}

#[tokio::test]
async fn squeeze_strings() {
    let cache_dir = TempDir::new().unwrap();
    let (ctx, cache) = LiquidCacheLocalBuilder::new()
        .with_max_cache_bytes(1024 * 1024)
        .with_cache_dir(cache_dir.path().to_path_buf())
        .build(SessionConfig::new())
        .unwrap();
    ctx.register_parquet("hits", TEST_FILE, Default::default())
        .await
        .unwrap();

    let plan = ctx
        .sql("SELECT COUNT(DISTINCT(\"URL\")) FROM hits WHERE \"Referer\" <> 0")
        .await
        .unwrap();
    let result = plan.collect().await.unwrap();
    assert_eq!(result.len(), 1);
    assert_eq!(
        result[0].column(0).as_primitive::<Int64Type>().value(0),
        5639
    );
    let trace = cache.consume_event_trace();
    insta::assert_snapshot!(trace);
}

#[tokio::test]
async fn squeeze_substrings_search() {
    let cache_dir = TempDir::new().unwrap();
    let (ctx, cache) = LiquidCacheLocalBuilder::new()
        .with_max_cache_bytes(1024 * 256)
        .with_cache_dir(cache_dir.path().to_path_buf())
        .build(SessionConfig::new())
        .unwrap();
    ctx.register_parquet("hits", TEST_FILE, Default::default())
        .await
        .unwrap();

    let plan = ctx
        .sql("SELECT COUNT(*) FROM hits WHERE \"SearchPhrase\" LIKE '%abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789%'")
        .await
        .unwrap();
    let result = plan.collect().await.unwrap();
    assert_eq!(result.len(), 1);
    assert_eq!(result[0].column(0).as_primitive::<Int64Type>().value(0), 0);
    let trace = cache.consume_event_trace();
    insta::assert_snapshot!(trace);
}

#[tokio::test]
async fn squeeze_substrings_search_title() {
    let cache_dir = TempDir::new().unwrap();
    let (ctx, cache) = LiquidCacheLocalBuilder::new()
        .with_max_cache_bytes(1024 * 1024 * 4)
        .with_cache_dir(cache_dir.path().to_path_buf())
        .build(SessionConfig::new())
        .unwrap();
    ctx.register_parquet("hits", TEST_FILE, Default::default())
        .await
        .unwrap();

    let plan = ctx
        .sql("SELECT COUNT(*) FROM hits WHERE \"Title\" LIKE '%Cosplay%'")
        .await
        .unwrap();
    let result = plan.collect().await.unwrap();
    assert_eq!(result.len(), 1);
    assert_eq!(result[0].column(0).as_primitive::<Int64Type>().value(0), 6);
    let trace = cache.consume_event_trace();
    println!("{:?}", cache.storage().stats());
    insta::assert_snapshot!(trace);
}

#[tokio::test]
async fn squeeze_distinct_search_phase() {
    let cache_dir = TempDir::new().unwrap();
    let (ctx, cache) = LiquidCacheLocalBuilder::new()
        .with_max_cache_bytes(1024 * 256)
        .with_cache_dir(cache_dir.path().to_path_buf())
        .build(SessionConfig::new())
        .unwrap();
    ctx.register_parquet("hits", TEST_FILE, Default::default())
        .await
        .unwrap();

    let plan = ctx
        .sql("SELECT DISTINCT(\"SearchPhrase\") FROM hits ORDER BY \"SearchPhrase\" LIMIT 10")
        .await
        .unwrap();
    let result = plan.collect().await.unwrap();
    println!("{}", pretty_format_batches(result.as_ref()).unwrap());
    assert_eq!(result.len(), 1);
    ctx.sql("SELECT DISTINCT(\"SearchPhrase\") FROM hits ORDER BY \"SearchPhrase\" LIMIT 10")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    let trace = cache.consume_event_trace();
    println!("{:?}", cache.storage().stats());
    insta::assert_snapshot!(trace);
}
