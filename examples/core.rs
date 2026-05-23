use std::sync::Arc;

use arrow::array::UInt64Array;
use liquid_cache::cache::{
    AlwaysHydrate, EntryID, LiquidCacheBuilder, LiquidPolicy, TranscodeSqueezeEvict,
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let storage = LiquidCacheBuilder::new()
        .with_max_memory_bytes(1024 * 1024 * 1024) // 1GB
        .with_max_disk_bytes(1024 * 1024 * 1024 * 10) // 10GB
        .with_batch_size(8192)
        .with_cache_policy(Box::new(LiquidPolicy::new()))
        .with_squeeze_policy(Box::new(TranscodeSqueezeEvict))
        .with_hydration_policy(Box::new(AlwaysHydrate::new()))
        .build()
        .await;

    let entry_id = EntryID::from(7);
    let arrow_array = Arc::new(UInt64Array::from_iter_values(0..16));
    storage.insert(entry_id, arrow_array.clone()).await.unwrap();

    // Move data to disk so the read demonstrates async I/O
    storage.flush_all_to_disk().await.unwrap();

    let retrieved = storage.get(&entry_id).await.unwrap();
    assert_eq!(retrieved.as_ref(), arrow_array.as_ref());

    Ok(())
}
