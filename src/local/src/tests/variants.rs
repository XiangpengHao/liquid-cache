use std::{
    fs::File,
    path::{Path, PathBuf},
    sync::Arc,
};

use arrow::{
    array::{Array, ArrayRef, RecordBatch, StringArray, StructArray},
    util::pretty::pretty_format_batches,
};
use arrow_schema::{Field, Fields, Schema};
use datafusion::prelude::{ParquetReadOptions, SessionConfig, SessionContext};
use liquid_cache_storage::cache::squeeze_policies::TranscodeSqueezeEvict;
use parquet::{
    arrow::ArrowWriter,
    variant::{VariantArray, VariantType, json_to_variant},
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

fn write_variant_non_nullable_value_file(dir: &Path) -> PathBuf {
    let file_path = dir.join("variant_value_not_nullable.parquet");
    let values = StringArray::from(vec![
        Some(r#"{"name": "Alice", "age": 30}"#),
        Some(r#"{"name": "Bob", "age": 25}"#),
        None,
    ]);
    let input_array: ArrayRef = Arc::new(values);
    let variant = json_to_variant(&input_array).expect("variant conversion");
    let inner = variant.inner();
    let fields = inner.fields();
    let metadata_field = fields
        .iter()
        .find(|f| f.name() == "metadata")
        .expect("metadata field");
    let value_field = fields
        .iter()
        .find(|f| f.name() == "value")
        .expect("value field");
    let metadata_array = inner
        .column_by_name("metadata")
        .expect("metadata column")
        .clone();
    let value_array = inner.column_by_name("value").expect("value column").clone();

    let struct_array = StructArray::new(
        Fields::from(vec![
            Arc::new(metadata_field.as_ref().clone()),
            Arc::new(Field::new(
                value_field.name(),
                value_field.data_type().clone(),
                false,
            )),
        ]),
        vec![metadata_array, value_array],
        inner.nulls().cloned(),
    );

    let data_field = Arc::new(
        Field::new("data", struct_array.data_type().clone(), true).with_extension_type(VariantType),
    );
    let schema = Arc::new(Schema::new(vec![data_field]));
    let batch = RecordBatch::try_new(schema, vec![Arc::new(struct_array) as ArrayRef])
        .expect("variant batch");

    let file = File::create(&file_path).expect("create variant parquet file");
    let mut writer =
        ArrowWriter::try_new(file, batch.schema(), None).expect("create variant writer");
    writer.write(&batch).expect("write variant batch");
    writer.close().expect("close variant writer");

    file_path
}

#[tokio::test]
async fn test_variant_parquet_naive_read() {
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

#[tokio::test]
async fn test_variant_preserved() {
    let parquet_dir = TempDir::new().unwrap();
    let parquet_path = write_variant_parquet_file(parquet_dir.path());
    let parquet_path_str = parquet_path.to_str().expect("unicode path");

    let ctx = SessionContext::new();
    ctx.register_parquet(
        "variants_test",
        parquet_path_str,
        ParquetReadOptions::default().skip_metadata(false),
    )
    .await
    .unwrap();

    let batches = ctx
        .sql("SELECT data FROM variants_test")
        .await
        .unwrap()
        .collect()
        .await
        .expect("query should succeed with small cache");
    let field = batches[0].schema().field_with_name("data").unwrap().clone();
    assert!(field.try_extension_type::<VariantType>().is_ok());
}

#[tokio::test]
async fn test_variant_get() {
    let cache_dir = TempDir::new().unwrap();
    let parquet_dir = TempDir::new().unwrap();
    let parquet_path = write_variant_parquet_file(parquet_dir.path());
    let parquet_path_str = parquet_path.to_str().expect("unicode path");

    let (ctx, _cache) = LiquidCacheLocalBuilder::new()
        .with_cache_dir(cache_dir.path().to_path_buf())
        .with_squeeze_policy(Box::new(TranscodeSqueezeEvict))
        .build(SessionConfig::new())
        .unwrap();

    ctx.register_parquet(
        "variants_test",
        parquet_path_str,
        ParquetReadOptions::default().skip_metadata(false),
    )
    .await
    .unwrap();

    let batches = ctx
        .sql("SELECT variant_to_json(variant_get(data, 'name')) FROM variants_test")
        .await
        .unwrap()
        .collect()
        .await
        .expect("query should succeed with small cache");
    insta::assert_snapshot!(pretty_format_batches(&batches).unwrap());
}

#[tokio::test]
async fn test_variant_predicate() {
    let cache_dir = TempDir::new().unwrap();
    let parquet_dir = TempDir::new().unwrap();
    let parquet_path = write_variant_parquet_file(parquet_dir.path());
    let parquet_path_str = parquet_path.to_str().expect("unicode path");

    let (ctx, _cache) = LiquidCacheLocalBuilder::new()
        .with_cache_dir(cache_dir.path().to_path_buf())
        .with_squeeze_policy(Box::new(TranscodeSqueezeEvict))
        .build(SessionConfig::new())
        .unwrap();

    ctx.register_parquet(
        "variants_test",
        parquet_path_str,
        ParquetReadOptions::default().skip_metadata(false),
    )
    .await
    .unwrap();

    let batches = ctx
        .sql("SELECT variant_to_json(variant_get(data, 'name')) FROM variants_test WHERE variant_get(data, 'name', 'Utf8') = 'Bob'")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    insta::assert_snapshot!(pretty_format_batches(&batches).unwrap());
}

#[tokio::test]
async fn test_variant_get_fails_when_value_field_not_nullable() {
    let cache_dir = TempDir::new().unwrap();
    let parquet_dir = TempDir::new().unwrap();
    let parquet_path = write_variant_non_nullable_value_file(parquet_dir.path());
    let parquet_path_str = parquet_path.to_str().expect("unicode path");

    let (ctx, _cache) = LiquidCacheLocalBuilder::new()
        .with_cache_dir(cache_dir.path().to_path_buf())
        .with_squeeze_policy(Box::new(TranscodeSqueezeEvict))
        .build(SessionConfig::new())
        .unwrap();

    ctx.register_parquet(
        "variants_value_not_nullable",
        parquet_path_str,
        ParquetReadOptions::default().skip_metadata(false),
    )
    .await
    .unwrap();

    let batches = ctx
        .sql("SELECT variant_get(data, 'name', 'Utf8') AS name FROM variants_value_not_nullable")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();

    insta::assert_snapshot!(pretty_format_batches(&batches).unwrap());
}

fn write_large_variant_parquet_file(dir: &Path, num_rows: usize) -> PathBuf {
    let file_path = dir.join("large_variant_rows.parquet");

    let values: Vec<Option<String>> = (0..num_rows)
        .map(|i| {
            if i % 3 == 0 {
                Some(format!(
                    r#"{{"name": "Person_{}", "age": {}, "tags": ["a", "b"], "details": {{"info": "info_{}"}}}}"#,
                    i,
                    20 + (i % 50),
                    i
                ))
            } else if i % 3 == 1 {
                Some(format!(
                    r#"{{"name": "Person_{}", "city": "City_{}", "active": true, "details": {{"info": "info_{}", "extra": "extra_{}"}}}}"#,
                    i,
                    i % 100,
                    i,
                    i
                ))
            } else {
                Some(format!(r#"{{"name": "Person_{}", "score": {}.5}}"#, i, i))
            }
        })
        .collect();

    let input_array: ArrayRef =
        Arc::new(StringArray::from_iter(values.iter().map(|x| x.as_deref())));
    let variant = json_to_variant(&input_array).expect("variant conversion");

    let schema = Arc::new(Schema::new(vec![variant.field("data")]));
    let batch = RecordBatch::try_new(schema, vec![ArrayRef::from(variant)]).expect("variant batch");

    let file = File::create(&file_path).expect("create large variant parquet file");
    let mut writer =
        ArrowWriter::try_new(file, batch.schema(), None).expect("create variant writer");
    writer.write(&batch).expect("write variant batch");
    writer.close().expect("close variant writer");

    file_path
}

#[tokio::test]
async fn test_large_variant_squeeze() {
    let cache_dir = TempDir::new().unwrap();
    let parquet_dir = TempDir::new().unwrap();
    let num_rows = 1_000;
    let parquet_path = write_large_variant_parquet_file(parquet_dir.path(), num_rows);
    let parquet_path_str = parquet_path.to_str().unwrap();

    let (ctx, _cache) = LiquidCacheLocalBuilder::new()
        .with_cache_dir(cache_dir.path().to_path_buf())
        .with_max_cache_bytes(1024)
        .with_squeeze_policy(Box::new(TranscodeSqueezeEvict))
        .build(SessionConfig::new())
        .unwrap();

    ctx.register_parquet(
        "large_variants",
        parquet_path_str,
        ParquetReadOptions::default(),
    )
    .await
    .unwrap();

    let batches = ctx
        .sql("SELECT Count(Distinct(variant_get(data, 'details.info', 'Utf8'))) FROM large_variants LIMIT 5")
        .await
        .unwrap()
        .collect()
        .await
        .expect("should query variant field");
    let count = batches[0]
        .column(0)
        .as_any()
        .downcast_ref::<arrow::array::Int64Array>()
        .unwrap()
        .value(0);
    assert_eq!(count, 667_i64);
}

#[tokio::test]
async fn variant_multi_queries() {
    let cache_dir = TempDir::new().unwrap();
    let parquet_dir = TempDir::new().unwrap();
    let num_rows = 1_000;
    let parquet_path = write_large_variant_parquet_file(parquet_dir.path(), num_rows);
    let parquet_path_str = parquet_path.to_str().unwrap();

    let (ctx, _cache) = LiquidCacheLocalBuilder::new()
        .with_cache_dir(cache_dir.path().to_path_buf())
        .with_max_cache_bytes(1024)
        .with_squeeze_policy(Box::new(TranscodeSqueezeEvict))
        .build(SessionConfig::new())
        .unwrap();

    ctx.register_parquet(
        "large_variants",
        parquet_path_str,
        ParquetReadOptions::default(),
    )
    .await
    .unwrap();

    ctx.sql(
        "SELECT Count(Distinct(variant_get(data, 'name', 'Utf8'))) FROM large_variants LIMIT 5",
    )
    .await
    .unwrap()
    .collect()
    .await
    .unwrap();

    let batches = ctx
        .sql(
            "SELECT Count(Distinct(variant_get(data, 'details.info', 'Utf8'))) FROM large_variants LIMIT 5",
        )
        .await
        .unwrap();
    let batches = batches.collect().await.unwrap();
    let count = batches[0]
        .column(0)
        .as_any()
        .downcast_ref::<arrow::array::Int64Array>()
        .unwrap()
        .value(0);
    assert_eq!(count, 667_i64);
}

#[tokio::test]
async fn variant_multi_queries_complex() {
    let cache_dir = TempDir::new().unwrap();
    let parquet_dir = TempDir::new().unwrap();
    let num_rows = 1_000;
    let parquet_path = write_large_variant_parquet_file(parquet_dir.path(), num_rows);
    let parquet_path_str = parquet_path.to_str().unwrap();

    let (ctx, cache) = LiquidCacheLocalBuilder::new()
        .with_cache_dir(cache_dir.path().to_path_buf())
        .with_max_cache_bytes(1024 * 1000)
        .with_batch_size(4)
        .with_squeeze_policy(Box::new(TranscodeSqueezeEvict))
        .build(SessionConfig::new())
        .unwrap();

    ctx.register_parquet(
        "large_variants",
        parquet_path_str,
        ParquetReadOptions::default(),
    )
    .await
    .unwrap();

    ctx.sql(
        "SELECT Count(Distinct(variant_get(data, 'details.info', 'Utf8'))) FROM large_variants LIMIT 5",
    )
    .await
    .unwrap()
    .collect()
    .await
    .unwrap();

    let batches = ctx
        .sql(
            "SELECT Count(Distinct(variant_get(data, 'details.extra', 'Utf8'))) FROM large_variants WHERE variant_get(data, 'details.info', 'Utf8') = 'info_1'",
        )
        .await
        .unwrap();
    let logical_plan = batches.logical_plan();
    println!("logical_plan: \n{}", logical_plan);
    println!("cache: \n{:?}", cache.storage().stats());
    let batches = batches.collect().await.unwrap();
    let count = batches[0]
        .column(0)
        .as_any()
        .downcast_ref::<arrow::array::Int64Array>()
        .unwrap()
        .value(0);
    assert_eq!(count, 1_i64);
}
