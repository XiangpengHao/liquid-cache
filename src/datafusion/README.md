# liquid-cache-datafusion

Parquet reader with a Vortex-backed LiquidCache tier.

## Lineage expression pushdown

LiquidCache analyzes physical plans to record how each file column is consumed,
including date extraction, variant paths, and predicates.
These expressions continue to flow through local and distributed execution even
though the cache currently retains complete Arrow or Liquid arrays. Keeping the
analysis separate preserves the information needed for future representation
work without enabling partial-data storage today.
