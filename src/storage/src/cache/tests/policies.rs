use crate::cache::{
    AlwaysHydrate, EntryID, LiquidCacheBuilder, LiquidPolicy, TranscodeSqueezeEvict,
    utils::create_test_arrow_array,
};

#[tokio::test]
async fn default_policies() {
    let test_array = create_test_arrow_array(1024);

    let capacity = test_array.get_array_memory_size() * 2;
    let cache = LiquidCacheBuilder::new()
        .with_cache_policy(Box::new(LiquidPolicy::new()))
        .with_hydration_policy(Box::new(AlwaysHydrate::new()))
        .with_squeeze_policy(Box::new(TranscodeSqueezeEvict))
        .with_max_cache_bytes(capacity)
        .build();

    for i in 0..5 {
        let entry_id = EntryID::from(i);
        cache.insert(entry_id, test_array.clone()).await;
    }

    for i in 0..5 {
        let entry_id = EntryID::from(i);
        let array = cache.get(&entry_id).read().await.unwrap();
        assert_eq!(array.len(), test_array.len());
    }

    let trace = cache.consume_trace();
    insta::assert_snapshot!(trace);
}

#[tokio::test]
async fn insert_wont_fit_cache() {
    let test_array = create_test_arrow_array(1024);

    let capacity = test_array.get_array_memory_size() * 2;
    let cache = LiquidCacheBuilder::new()
        .with_cache_policy(Box::new(LiquidPolicy::new()))
        .with_hydration_policy(Box::new(AlwaysHydrate::new()))
        .with_squeeze_policy(Box::new(TranscodeSqueezeEvict))
        .with_max_cache_bytes(capacity)
        .build();
    cache.insert(EntryID::from(0), test_array.clone()).await;
    let array_3x = arrow::compute::concat(&[&test_array, &test_array, &test_array]).unwrap();
    let array_9x = arrow::compute::concat(&[&array_3x, &array_3x, &array_3x]).unwrap();
    let array_27x = arrow::compute::concat(&[&array_9x, &array_9x, &array_9x]).unwrap();
    cache.insert(EntryID::from(1), array_27x.clone()).await;
    cache.get(&EntryID::from(1)).read().await.unwrap();

    let trace = cache.consume_trace();
    insta::assert_snapshot!(trace);
}
