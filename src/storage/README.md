# liquid-cache-storage

Storage layer providing byte caching and liquid array data structures.

This library provides one way to insert into the cache and three ways to read from it:
- read as Arrow array
- read with selection pushdown
- read with predicate pushdown

All reads use async APIs that handle I/O internally.

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

tokio_test::block_on(async {
let storage = CacheStorageBuilder::new().build();

let entry_id = EntryID::from(7);
let arrow_array = Arc::new(UInt64Array::from_iter_values(0..16));
storage.insert(entry_id, arrow_array.clone());

// Move data to disk so the read will demonstrate async I/O
storage.flush_all_to_disk();

// Read asynchronously
let retrieved = storage.get_arrow_array(&entry_id).await.unwrap();
assert_eq!(retrieved.as_ref(), arrow_array.as_ref());
});
```

## 3) Read with selection pushdown

```rust
use liquid_cache_storage::cache::{CacheStorageBuilder, EntryID};
use arrow::array::UInt64Array;
use arrow::buffer::BooleanBuffer;
use std::sync::Arc;

tokio_test::block_on(async {
let storage = CacheStorageBuilder::new().build();

let entry_id = EntryID::from(8);
let data = Arc::new(UInt64Array::from_iter_values(0..10));
storage.insert(entry_id, data.clone());

// Move data to disk so the read will demonstrate async I/O
storage.flush_all_to_disk();

// Keep even indices
let filter = BooleanBuffer::from((0..10).map(|i| i % 2 == 0).collect::<Vec<_>>());

// Read with selection pushdown
let filtered = storage.get_with_selection(&entry_id, &filter).await.unwrap().unwrap();
let expected = Arc::new(UInt64Array::from_iter_values((0..10).filter(|i| i % 2 == 0)));
assert_eq!(filtered.as_ref(), expected.as_ref());
});
```

## 4) Read with predicate pushdown

```rust
use liquid_cache_storage::cache::{CacheStorageBuilder, EntryID, GetWithPredicateResult};
use arrow::array::{ArrayRef, StringArray};
use arrow::buffer::BooleanBuffer;
use datafusion::logical_expr::Operator;
use datafusion::physical_plan::expressions::{BinaryExpr, Column, Literal};
use datafusion::physical_plan::PhysicalExpr;
use datafusion::scalar::ScalarValue;
use std::sync::Arc;

tokio_test::block_on(async {
let storage = CacheStorageBuilder::new().build();

let entry_id = EntryID::from(9);
let data = Arc::new(StringArray::from(vec![
    Some("apple"), Some("banana"), None, Some("apple"), Some("cherry"),
]));
storage.insert(entry_id, data.clone());

// Move data to disk so the read will demonstrate async I/O
storage.flush_all_to_disk();

let selection = BooleanBuffer::from(vec![true, true, false, true, true]);
let expr: Arc<dyn PhysicalExpr> = Arc::new(BinaryExpr::new(
    Arc::new(Column::new("col", 0)),
    Operator::Eq,
    Arc::new(Literal::new(ScalarValue::Utf8(Some("apple".to_string())))),
));

let expected_filtered: ArrayRef = Arc::new(StringArray::from(vec![
    Some("apple"),
    Some("banana"),
    Some("apple"),
    Some("cherry"),
])) as ArrayRef;

// Read with predicate pushdown
let result = storage.get_with_predicate(&entry_id, &selection, &expr).await.unwrap();
assert_eq!(result, GetWithPredicateResult::Filtered(expected_filtered));
});
```
