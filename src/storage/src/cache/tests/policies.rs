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
