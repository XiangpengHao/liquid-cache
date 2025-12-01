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
use liquid_cache_storage::cache::{LiquidCacheBuilder, EntryID};
use arrow::array::UInt64Array;
use std::sync::Arc;

tokio_test::block_on(async {
let storage = LiquidCacheBuilder::new().build();

let entry_id = EntryID::from(42);
let arrow_array = Arc::new(UInt64Array::from_iter_values(0..1000));

// Insert once; replacement/placement is handled by the cache policy
storage.insert(entry_id, arrow_array.clone()).await;

assert!(storage.is_cached(&entry_id));
});
```

## 2) Read as Arrow

```rust
use liquid_cache_storage::cache::{LiquidCacheBuilder, EntryID};
use arrow::array::UInt64Array;
use std::sync::Arc;

tokio_test::block_on(async {
let storage = LiquidCacheBuilder::new().build();

let entry_id = EntryID::from(7);
let arrow_array = Arc::new(UInt64Array::from_iter_values(0..16));
storage.insert(entry_id, arrow_array.clone()).await;

// Move data to disk so the read will demonstrate async I/O
storage.flush_all_to_disk();

// Read asynchronously
let retrieved = storage.get(&entry_id).await.unwrap();
assert_eq!(retrieved.as_ref(), arrow_array.as_ref());
});
```

## 3) Read with selection & predicate pushdown

```rust
use liquid_cache_storage::cache::{LiquidCacheBuilder, EntryID};
use arrow::array::{BooleanArray, StringArray};
use arrow::buffer::BooleanBuffer;
use datafusion::logical_expr::Operator;
use datafusion::physical_plan::expressions::{BinaryExpr, Column, Literal};
use datafusion::physical_plan::PhysicalExpr;
use datafusion::scalar::ScalarValue;
use std::sync::Arc;

tokio_test::block_on(async {
let storage = LiquidCacheBuilder::new().build();

let entry_id = EntryID::from(8);
let data = Arc::new(StringArray::from(vec![
    Some("apple"), Some("banana"), None, Some("apple"), Some("cherry"),
]));
storage.insert(entry_id, data.clone()).await;

// Move data to disk so the read will demonstrate async I/O
storage.flush_all_to_disk();

let selection = BooleanBuffer::from(vec![true, true, false, true, true]);
let expr: Arc<dyn PhysicalExpr> = Arc::new(BinaryExpr::new(
    Arc::new(Column::new("col", 0)),
    Operator::Eq,
    Arc::new(Literal::new(ScalarValue::Utf8(Some("apple".to_string())))),
));

// Read with predicate pushdown
let result = storage
    .eval_predicate(&entry_id, &expr)
    .with_selection(&selection)
    .await
    .unwrap();
let mask = match result {
    Ok(mask) => mask,
    Err(filtered) => {
        // Fallback path when the predicate cannot be evaluated inside the cache.
        BooleanArray::from(
            filtered
                .as_any()
                .downcast_ref::<StringArray>()
                .unwrap()
                .iter()
                .map(|value| Some(value == Some("apple")))
                .collect::<Vec<_>>(),
        )
    }
};
let expected_mask = BooleanArray::from(vec![Some(true), Some(false), Some(true), Some(false)]);
assert_eq!(mask, expected_mask);
});
```
