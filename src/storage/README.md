# liquid-cache-storage

Storage layer providing byte caching and liquid array data structures.


```rust
use liquid_cache_storage::cache::{CacheStorage, EntryID, CachedBatch, DefaultIoWorker};
use liquid_cache_storage::policies::ToDiskPolicy;
use liquid_cache_storage::common::LiquidCacheMode;
use arrow::array::UInt64Array;
use std::sync::Arc;

let batch_size = 8192;
let max_cache_bytes = 1024 * 1024 * 1024; // 1GB
let temp_dir = tempfile::tempdir().unwrap();
let policy = Box::new(ToDiskPolicy::new());

let storage = Arc::new(CacheStorage::new(
        batch_size,
        max_cache_bytes,
        temp_dir.keep(),
        LiquidCacheMode::Liquid,
        policy,
        Arc::new(DefaultIoWorker::new()),
));

let entry_id = EntryID::from(0);
let arrow_array = UInt64Array::from_iter_values(0..1000);
storage.insert(entry_id, Arc::new(arrow_array));

let batch = storage.get(&entry_id).unwrap();
assert!(matches!(batch, CachedBatch::MemoryArrow(_)));
```
