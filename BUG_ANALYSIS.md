# Liquid Cache Duplicate Values Bug Analysis

## Issue Summary

**Bug Report**: [Issue #312](https://github.com/XiangpengHao/liquid-cache/issues/312) - Getting duplicate values when using liquid cache with filters and projections.

**Symptoms**: 
- When `is_cache = true`, the query returns 4804 duplicate row IDs out of 10000 total rows
- When `is_cache = false`, the query returns 0 duplicates
- The issue only occurs when the liquid cache optimizer is applied

## Root Cause Analysis

### The Problem

The bug is located in the `LiquidBatchReader::next()` method in `src/liquid_parquet/src/reader/runtime/liquid_batch_reader.rs`. The issue occurs in the batch processing logic when using the cache.

### Code Flow Analysis

1. **Batch Processing Path**: The `next()` method has two code paths:
   - `can_optimize_single_column_filter_projection = true`: Uses `read_and_filter_single_column`
   - `can_optimize_single_column_filter_projection = false`: Uses `build_predicate_filter` + `read_selection`

2. **The Buggy Path**: In the `false` path (lines 235-250), the following sequence occurs:
   ```rust
   while let Some(selection) = take_next_batch(&mut self.selection, self.batch_size) {
       match self.build_predicate_filter(selection) {
           Ok(filtered_selection) => {
               match self.read_selection(filtered_selection) {
                   Ok(Some(record_batch)) => {
                       self.current_batch_id.inc(); // ← BUG: Incremented AFTER cache read
                       return Some(Ok(record_batch));
                   }
                   // ...
               }
           }
       }
   }
   ```

3. **Cache Interaction**: The `read_selection` method calls `try_read_from_cache`, which uses `self.current_batch_id` as the cache key:
   ```rust
   column.get_arrow_array_with_filter(self.current_batch_id, &filter)
   ```

### The Root Cause

**The `current_batch_id` is incremented AFTER the cache read attempt, but the cache lookup uses the current `batch_id`.**

This creates a race condition where:
1. Multiple iterations of the `while` loop can use the same `batch_id` for cache lookups
2. The same cached data can be returned multiple times
3. This results in duplicate rows in the final result

### Why It Only Happens With Cache

- Without cache: Data is read directly from the parquet file, so no cache key conflicts occur
- With cache: The cache lookup uses `batch_id` as part of the cache key, and the incorrect timing of `batch_id` increment causes cache key collisions

## The Fix

### Solution

Move the `current_batch_id.inc()` call to **before** the cache read attempt:

```rust
while let Some(selection) = take_next_batch(&mut self.selection, self.batch_size) {
    match self.build_predicate_filter(selection) {
        Ok(filtered_selection) => {
            // Increment batch_id BEFORE reading from cache to ensure unique cache keys
            self.current_batch_id.inc();
            match self.read_selection(filtered_selection) {
                Ok(Some(record_batch)) => {
                    return Some(Ok(record_batch));
                }
                Ok(None) => {
                    continue; // No rows to read, try next batch
                }
                Err(e) => return Some(Err(e)),
            }
        }
        Err(e) => return Some(Err(e)),
    }
}
```

### Why This Fixes the Issue

1. **Unique Cache Keys**: Each cache lookup now uses a unique `batch_id`
2. **No Cache Collisions**: Different batches will have different cache keys
3. **Correct Data Flow**: Each batch is processed exactly once

## Testing

### Test Case

The test case creates a parquet file with:
- 10,000 rows with unique row IDs (0-9999)
- A filter condition that selects specific rows
- Multiple query iterations to trigger cache behavior

### Expected Results After Fix

- Liquid cache should return 0 duplicates (same as DataFusion)
- Results should be consistent across multiple query iterations
- Cache should work correctly without producing duplicate data

## Impact

### Severity: High
- **Data Integrity**: The bug causes incorrect query results
- **Performance**: Duplicate processing wastes resources
- **Reliability**: Users cannot trust query results when cache is enabled

### Affected Components
- `LiquidBatchReader::next()` method
- Cache-based query execution path
- Any query using filters with liquid cache enabled

## Prevention

### Code Review Guidelines
1. Always verify cache key uniqueness when using batch IDs
2. Ensure cache operations happen with the correct batch ID
3. Test cache behavior with multiple query iterations

### Testing Recommendations
1. Add unit tests for cache key uniqueness
2. Test with multiple query iterations
3. Verify no duplicates in query results when cache is enabled