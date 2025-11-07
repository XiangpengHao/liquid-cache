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

    println!("{:?}", temp_dir);

    let (_ctx, storage) = LiquidCacheLocalBuilder::new()
        .with_max_cache_bytes(1024 * 1024) // 1MB
        .with_cache_dir(temp_dir.path().to_path_buf())
        .with_squeeze_policy(Box::new(TranscodeSqueezeEvict))
        .with_cache_policy(Box::new(LiquidPolicy::new()))
        .build(SessionConfig::new())?;

    for i in 1..730 {
        let entry_id = EntryID::from(i);
        let arrow_array = Arc::new(UInt64Array::from_iter_values(0..1000));
        // Insert once; replacement/placement is handled by the cache policy
        storage
            .storage()
            .insert(entry_id, arrow_array.clone())
            .await;
        let _ = storage.storage().get_arrow_array(&entry_id).await.unwrap();
    }
    // Move data to disk so the read will demonstrate async I/O
    storage.storage().flush_all_to_disk();

    Ok(())
}
