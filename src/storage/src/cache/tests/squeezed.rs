use std::sync::Arc;

use arrow::array::{ArrayRef, Date32Array};

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
