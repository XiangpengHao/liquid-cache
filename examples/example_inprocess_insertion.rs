use datafusion::{arrow::array::UInt64Array, prelude::SessionConfig};
use liquid_cache_local::{
    LiquidCacheLocalBuilder,
    storage::{
        cache::{EntryID, squeeze_policies::TranscodeSqueezeEvict},
        cache_policies::LiquidPolicy,
    },
};
use std::sync::Arc;
use tempfile::TempDir;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let temp_dir = TempDir::new().unwrap();

    let (_ctx, storage) = LiquidCacheLocalBuilder::new()
        .with_max_cache_bytes(1024 * 1024) // 1MB
        .with_cache_dir(temp_dir.path().to_path_buf())
        .with_squeeze_policy(Box::new(TranscodeSqueezeEvict))
        .with_cache_policy(Box::new(LiquidPolicy::new()))
        .build(SessionConfig::new())?;

    let entry_id = EntryID::from(42);
    let arrow_array = Arc::new(UInt64Array::from_iter_values(0..1000));

    // Insert once; replacement/placement is handled by the cache policy
    storage
        .storage()
        .insert(entry_id, arrow_array.clone())
        .await;

    assert!(storage.storage().is_cached(&entry_id));

    Ok(())
}
