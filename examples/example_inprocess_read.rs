use std::sync::Arc;

use datafusion::arrow::array::UInt64Array;
use liquid_cache_datafusion_local::storage::cache::{EntryID, LiquidCacheBuilder};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let temp_dir = tempfile::tempdir()?;
    let storage = LiquidCacheBuilder::new()
        .with_cache_dir(temp_dir.path().to_path_buf())
        .build();

    let entry_id = EntryID::from(7);
    let arrow_array = Arc::new(UInt64Array::from_iter_values(0..16));
    storage.insert(entry_id, arrow_array.clone()).await;

    // Move data to disk so the read will demonstrate async I/O
    storage.flush_all_to_disk().await;

    // Read asynchronously
    let retrieved = storage.get(&entry_id).await.unwrap();
    assert_eq!(retrieved.as_ref(), arrow_array.as_ref());

    Ok(())
}
