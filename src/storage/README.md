# liquid-cache-storage

Storage layer providing byte caching and liquid array data structures.

This library provides one way to insert into the cache and three ways to read from it:
- read as Arrow array
- read as Arrow array with a Boolean filter
- evaluate a predicate against the cached data

Below are four concise, runnable examples showcasing these core operations.

## 1) Insert

```rust
use liquid_cache_storage::cache::{CacheStorageBuilder, EntryID};
use arrow::array::UInt64Array;
use std::sync::Arc;

let storage = CacheStorageBuilder::new().build();

let entry_id = EntryID::from(42);
let arrow_array = Arc::new(UInt64Array::from_iter_values(0..1000));

// Insert once; replacement/placement is handled by the cache policy
storage.insert(entry_id, arrow_array.clone());

assert!(storage.is_cached(&entry_id));
```

## 2) Read as Arrow

```rust
use liquid_cache_storage::cache::{CacheStorageBuilder, EntryID};
use arrow::array::UInt64Array;
use std::sync::Arc;

let storage = CacheStorageBuilder::new().build();

let entry_id = EntryID::from(7);
let arrow_array = Arc::new(UInt64Array::from_iter_values(0..16));
storage.insert(entry_id, arrow_array.clone());

let cached = storage.get(&entry_id).unwrap();
let out = cached.get_arrow_array();
assert_eq!(out.as_ref(), arrow_array.as_ref());
```

## 3) Read with a Boolean filter

```rust
use liquid_cache_storage::cache::{CacheStorageBuilder, EntryID};
use arrow::array::{UInt64Array, BooleanArray};
use std::sync::Arc;

let storage = CacheStorageBuilder::new().build();

let entry_id = EntryID::from(8);
let data = Arc::new(UInt64Array::from_iter_values(0..10));
storage.insert(entry_id, data.clone());

// Keep even indices
let filter = BooleanArray::from((0..10).map(|i| i % 2 == 0).collect::<Vec<_>>());

let cached = storage.get(&entry_id).unwrap();
let filtered = cached.get_arrow_with_filter(&filter).unwrap();

let expected = Arc::new(UInt64Array::from_iter_values((0..10).filter(|i| i % 2 == 0)));
assert_eq!(filtered.as_ref(), expected.as_ref());
```

## 4) Read using a predicate plus a pre-filter mask

TODO: add example
