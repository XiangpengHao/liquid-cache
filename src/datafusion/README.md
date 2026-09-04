# liquid-cache-datafusion

Parquet reader with liquid array caching and optimized data formats.

## Lineage expression pushdown

LiquidCache analyzes physical plans to record how each file column is consumed,
including date extraction, variant paths, predicates, and substring searches.
These expressions continue to flow through local and distributed execution even
though the cache currently retains complete Arrow or Liquid arrays. Keeping the
analysis separate preserves the information needed for future representation
work without enabling partial-data storage today.
