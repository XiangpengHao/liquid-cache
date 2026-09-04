use crate::cache::{
    AlwaysHydrate, CachedBatchType, EntryID, LiquidCacheBuilder, LiquidExpr, LiquidPolicy,
    TranscodeEvict, utils::create_test_arrow_array,
};
use arrow::array::{Array, ArrayRef, Int64Array, StringViewArray};
use arrow::buffer::BooleanBuffer;
use arrow_schema::DataType;
use datafusion_common::ScalarValue;
use datafusion_expr_common::operator::Operator;
use datafusion_physical_expr::PhysicalExpr;
use datafusion_physical_expr::expressions::{BinaryExpr, Column, Literal};
use std::sync::Arc;

#[tokio::test]
async fn default_policies() {
    let test_array = create_test_arrow_array(1024);

    let capacity = test_array.get_array_memory_size() * 2;
    let cache = LiquidCacheBuilder::new()
        .with_cache_policy(Box::new(LiquidPolicy::new()))
        .with_hydration_policy(Box::new(AlwaysHydrate::new()))
        .with_eviction_policy(Box::new(TranscodeEvict))
        .with_max_memory_bytes(capacity)
        .build()
        .await;

    for i in 0..5 {
        let entry_id = EntryID::from(i);
        cache.insert(entry_id, test_array.clone()).await.unwrap();
    }

    for i in 0..5 {
        let entry_id = EntryID::from(i);
        let array = cache.get(&entry_id).read().await.unwrap();
        assert_eq!(array.len(), test_array.len());
    }

    let trace = cache.consume_event_trace();
    insta::with_settings!({ filters => vec![(r"bytes=\d+", "bytes=[bytes]")] }, {
        insta::assert_snapshot!(trace);
    });
}

#[tokio::test]
async fn insert_wont_fit_cache() {
    let test_array = create_test_arrow_array(1024);

    let capacity = test_array.get_array_memory_size() * 2;
    let cache = LiquidCacheBuilder::new()
        .with_cache_policy(Box::new(LiquidPolicy::new()))
        .with_hydration_policy(Box::new(AlwaysHydrate::new()))
        .with_eviction_policy(Box::new(TranscodeEvict))
        .with_max_memory_bytes(capacity)
        .build()
        .await;
    cache
        .insert(EntryID::from(0), test_array.clone())
        .await
        .unwrap();
    let array_3x = arrow::compute::concat(&[&test_array, &test_array, &test_array]).unwrap();
    let array_9x = arrow::compute::concat(&[&array_3x, &array_3x, &array_3x]).unwrap();
    let array_27x = arrow::compute::concat(&[&array_9x, &array_9x, &array_9x]).unwrap();
    cache
        .insert(EntryID::from(1), array_27x.clone())
        .await
        .unwrap();
    cache.get(&EntryID::from(1)).read().await.unwrap();

    let trace = cache.consume_event_trace();
    let json_trace = serde_json::to_string(&trace).unwrap();
    println!("{}", json_trace);
    insta::with_settings!({ filters => vec![(r"bytes=\d+", "bytes=[bytes]")] }, {
        insta::assert_snapshot!(trace);
    });
}

#[tokio::test]
async fn liquid_eviction_reads_memory_and_disk() {
    let integers: ArrayRef = Arc::new(Int64Array::from_iter(
        (0..2_048).map(|value| (value % 13 != 0).then_some(value)),
    ));
    let strings: ArrayRef =
        Arc::new(StringViewArray::from_iter((0..2_048).map(|value| {
            (value % 11 != 0).then(|| format!("value-{value}"))
        })));
    let capacity = integers.get_array_memory_size() + strings.get_array_memory_size();
    let cache = LiquidCacheBuilder::new()
        .with_max_memory_bytes(capacity)
        .with_eviction_policy(Box::new(TranscodeEvict))
        .with_hydration_policy(Box::new(AlwaysHydrate::new()))
        .build()
        .await;

    for (id, array) in [
        integers.clone(),
        strings.clone(),
        integers.clone(),
        strings.clone(),
    ]
    .into_iter()
    .enumerate()
    {
        cache.insert(EntryID::from(id), array).await.unwrap();
    }

    let mut states = Vec::new();
    cache.for_each_entry(|_, entry| states.push(CachedBatchType::from(entry)));
    assert!(
        states
            .iter()
            .any(|state| *state != CachedBatchType::MemoryArrow)
    );

    let id = EntryID::from(0);
    assert_eq!(
        cache.get(&id).read().await.unwrap().as_ref(),
        integers.as_ref()
    );

    let selection = BooleanBuffer::from_iter((0..integers.len()).map(|index| index % 3 == 0));
    let selected = cache
        .get(&id)
        .with_selection(&selection)
        .read()
        .await
        .unwrap();
    let expected = arrow::compute::filter(
        &integers,
        &arrow::array::BooleanArray::new(selection.clone(), None),
    )
    .unwrap();
    assert_eq!(selected.as_ref(), expected.as_ref());

    let physical: Arc<dyn PhysicalExpr> = Arc::new(BinaryExpr::new(
        Arc::new(Column::new("c", 0)),
        Operator::GtEq,
        Arc::new(Literal::new(ScalarValue::Int64(Some(1_000)))),
    ));
    let predicate = LiquidExpr::try_new(physical, &DataType::Int64).unwrap();
    let actual = cache.eval_predicate(&id, &predicate).read().await.unwrap();
    let expected = arrow::array::BooleanArray::from_iter(
        integers
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap()
            .iter()
            .map(|value| value.map(|value| value >= 1_000)),
    );
    assert_eq!(actual, expected);

    cache.flush_all_to_disk().await.unwrap();
    let mut disk_states = Vec::new();
    cache.for_each_entry(|_, entry| disk_states.push(CachedBatchType::from(entry)));
    assert!(disk_states.iter().all(|state| matches!(
        state,
        CachedBatchType::DiskLiquid | CachedBatchType::DiskArrow
    )));
    assert!(disk_states.contains(&CachedBatchType::DiskLiquid));
    assert_eq!(
        cache.get(&id).read().await.unwrap().as_ref(),
        integers.as_ref()
    );
}
