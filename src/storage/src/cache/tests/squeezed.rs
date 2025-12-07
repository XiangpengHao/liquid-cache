use std::sync::Arc;

use arrow::array::{Array, ArrayRef, Date32Array, StringArray};
use arrow_schema::DataType;
use parquet_variant_compute::json_to_variant;

use crate::{
    cache::{
        AlwaysHydrate, CacheExpression, DefaultIoContext, EntryID, LiquidCacheBuilder,
        LiquidPolicy, TranscodeSqueezeEvict,
    },
    liquid_array::Date32Field,
};

fn create_date32_array() -> ArrayRef {
    let date32_array = Date32Array::from_iter_values(0..4096);
    Arc::new(date32_array)
}

#[tokio::test]
async fn read_squeezed_date_time() {
    let temp_dir = tempfile::tempdir().unwrap();
    let array = create_date32_array();
    let array_size = array.get_array_memory_size();

    let cache = LiquidCacheBuilder::new()
        .with_cache_policy(Box::new(LiquidPolicy::new()))
        .with_hydration_policy(Box::new(AlwaysHydrate::new()))
        .with_squeeze_policy(Box::new(TranscodeSqueezeEvict))
        .with_max_cache_bytes(array_size * 2)
        .with_io_context(Arc::new(DefaultIoContext::new(
            temp_dir.path().to_path_buf(),
        )))
        .build();

    let expression = Arc::new(CacheExpression::extract_date32(Date32Field::Year));

    for i in 0..4 {
        let entry_id = EntryID::from(i);
        cache
            .insert(entry_id, array.clone())
            .with_squeeze_hint(expression.clone())
            .await;
    }

    for i in 0..4 {
        let entry_id = EntryID::from(i);
        let array = cache
            .get(&entry_id)
            .with_expression_hint(expression.clone())
            .await
            .unwrap();
        assert_eq!(array.len(), array.len());
    }
    cache
        .get(&EntryID::from(1))
        .with_expression_hint(Arc::new(CacheExpression::extract_date32(
            Date32Field::Month,
        )))
        .await
        .unwrap();
    let trace = cache.consume_trace();
    insta::assert_snapshot!(trace);
}

fn create_variant_array() -> ArrayRef {
    let mut values = Vec::new();
    for i in 0..64 {
        if i % 2 == 0 {
            values.push(Some(r#"{"name":"Ada", "address": {"zipcode": 90001}}"#));
        } else {
            values.push(Some(
                r#"{"name":"Bob", "age": 29, "address": {"city": "New York"}}"#,
            ));
        }
    }
    let json_values: ArrayRef = Arc::new(StringArray::from(values));
    let variant = json_to_variant(&json_values).expect("variant from json");
    ArrayRef::from(variant)
}

#[tokio::test]
async fn read_squeezed_variant_path() {
    let temp_dir = tempfile::tempdir().unwrap();
    let variant_array = create_variant_array();
    let array_size = variant_array.get_array_memory_size();

    let cache = LiquidCacheBuilder::new()
        .with_cache_policy(Box::new(LiquidPolicy::new()))
        .with_hydration_policy(Box::new(AlwaysHydrate::new()))
        .with_squeeze_policy(Box::new(TranscodeSqueezeEvict))
        .with_max_cache_bytes(array_size * 2)
        .with_io_context(Arc::new(DefaultIoContext::new(
            temp_dir.path().to_path_buf(),
        )))
        .build();

    let name_expr = Arc::new(CacheExpression::variant_get("name", DataType::Utf8));
    let age_expr = Arc::new(CacheExpression::variant_get("age", DataType::Int64));
    let zipcode_expr = Arc::new(CacheExpression::variant_get(
        "address.zipcode",
        DataType::Int64,
    ));
    for i in 0..3 {
        let entry_id = EntryID::from(i);
        cache
            .insert(entry_id, variant_array.clone())
            .with_squeeze_hint(name_expr.clone())
            .await;
    }

    let squeezed = cache
        .get(&EntryID::from(0))
        .with_expression_hint(name_expr.clone())
        .read()
        .await
        .unwrap();
    assert_eq!(squeezed.len(), variant_array.len());

    cache
        .get(&EntryID::from(0))
        .with_expression_hint(age_expr.clone())
        .read()
        .await
        .unwrap();
    cache
        .get(&EntryID::from(1))
        .with_expression_hint(zipcode_expr.clone())
        .read()
        .await
        .unwrap();
    let trace = cache.consume_trace();
    insta::assert_snapshot!(trace);
}
