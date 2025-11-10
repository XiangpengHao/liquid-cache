use std::{
    fs::File,
    path::{Path, PathBuf},
    sync::Arc,
};

use arrow::{
    array::{ArrayRef, RecordBatch, StringArray},
    util::pretty::pretty_format_batches,
};
use arrow_schema::Schema;
use datafusion::prelude::{ParquetReadOptions, SessionConfig, SessionContext};
use liquid_cache_storage::cache::squeeze_policies::TranscodeSqueezeEvict;
use parquet::{
    arrow::ArrowWriter,
    variant::{VariantArray, json_to_variant},
};
use tempfile::TempDir;

use crate::LiquidCacheLocalBuilder;

fn make_variant_array() -> VariantArray {
    let values = StringArray::from(vec![
        Some(r#"{"name": "Alice", "age": 30}"#),
        Some(r#"{"name": "Bob", "age": 25, "address": {"city": "New York"}}"#),
        Some(r#"{"name": "Charlie", "age": 30, "address": {"zipcode": 90001}}"#),
        None,
        Some("{}"),
    ]);
    let input_array: ArrayRef = Arc::new(values);
    json_to_variant(&input_array).expect("variant conversion")
}

fn write_variant_parquet_file(dir: &Path) -> PathBuf {
    let file_path = dir.join("variant_rows.parquet");
    let variant = make_variant_array();
    let schema = Arc::new(Schema::new(vec![variant.field("data")]));
    let batch = RecordBatch::try_new(schema, vec![ArrayRef::from(variant)]).expect("variant batch");

    let file = File::create(&file_path).expect("create variant parquet file");
    let mut writer =
        ArrowWriter::try_new(file, batch.schema(), None).expect("create variant writer");
    writer.write(&batch).expect("write variant batch");
    writer.close().expect("close variant writer");

    file_path
}

#[tokio::test]
async fn test_variant_parquet_is_not_supported() {
    let cache_dir = TempDir::new().unwrap();
    let parquet_dir = TempDir::new().unwrap();
    let parquet_path = write_variant_parquet_file(parquet_dir.path());
    let parquet_path_str = parquet_path.to_str().expect("unicode path");

    let baseline_ctx = SessionContext::new();
    baseline_ctx
        .register_parquet(
            "variants_baseline",
            parquet_path_str,
            ParquetReadOptions::default(),
        )
        .await
        .unwrap();
    let baseline_batches = baseline_ctx
        .sql("SELECT data FROM variants_baseline")
        .await
        .unwrap()
        .collect()
        .await
        .expect("DataFusion should read variant parquet");
    assert!(
        !baseline_batches.is_empty(),
        "baseline run should yield at least one record batch"
    );

    println!(
        "baseline_batches: \n{}",
        pretty_format_batches(&baseline_batches).unwrap()
    );

    let (ctx, _cache) = LiquidCacheLocalBuilder::new()
        .with_cache_dir(cache_dir.path().to_path_buf())
        .with_squeeze_policy(Box::new(TranscodeSqueezeEvict))
        .build(SessionConfig::new())
        .unwrap();
    ctx.register_parquet("variants", parquet_path_str, ParquetReadOptions::default())
        .await
        .unwrap();

    let liquid_batches = ctx
        .sql("SELECT data FROM variants")
        .await
        .unwrap()
        .collect()
        .await
        .expect("Liquid Cache should read variant parquet files");

    let baseline_str = pretty_format_batches(&baseline_batches)
        .unwrap()
        .to_string();
    let liquid_str = pretty_format_batches(&liquid_batches).unwrap().to_string();
    assert_eq!(
        baseline_str, liquid_str,
        "variant results should match baseline DataFusion output"
    );
}

#[tokio::test]
async fn test_variant_transcoding_falls_back_to_disk_arrow() {
    let cache_dir = TempDir::new().unwrap();
    let parquet_dir = TempDir::new().unwrap();
    let parquet_path = write_variant_parquet_file(parquet_dir.path());
    let parquet_path_str = parquet_path.to_str().expect("unicode path");

    let (ctx, cache) = LiquidCacheLocalBuilder::new()
        .with_batch_size(1)
        .with_max_cache_bytes(64)
        .with_cache_dir(cache_dir.path().to_path_buf())
        .with_squeeze_policy(Box::new(TranscodeSqueezeEvict))
        .build(SessionConfig::new())
        .unwrap();

    ctx.register_parquet(
        "variants_small_cache",
        parquet_path_str,
        ParquetReadOptions::default(),
    )
    .await
    .unwrap();

    let batches = ctx
        .sql("SELECT data FROM variants_small_cache")
        .await
        .unwrap()
        .collect()
        .await
        .expect("query should succeed with small cache");
    assert!(!batches.is_empty());

    let stats = cache.storage().stats();
    assert!(
        stats.disk_arrow_entries > 0,
        "expected struct columns to fall back to disk arrow, stats: {stats:?}"
    );
}
