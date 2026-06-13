# liquid-cache-datafusion

Parquet reader with liquid array caching and optimized data formats.

## Squeeze-hint (lineage) pushdown

A *squeeze hint* tells the cache how a column is used, so under memory pressure it
keeps only what the query needs instead of the whole column — e.g. just the `YEAR`
of a date read via `EXTRACT(YEAR FROM d)`, the paths of a `variant_get`, or a
substring fingerprint for `LIKE '%foo%'`.

A physical optimizer rule drives the whole flow:

```text
   physical plan
        |
        v
   LocalModeOptimizer  (physical optimizer rule)
        |    1. analyze the plan  ->  one CacheExpression per scan column
        |    2. find the parquet scan (ParquetSource)
        |    3. replace it with a LiquidParquetSource carrying the hints
        v
   LiquidParquetSource
        |    on open, passes the hints down as squeeze hints
        v
   liquid cache
        |    squeezes under memory pressure
        v
   keeps only the hinted form (e.g. YEAR); full data stays on disk
```

The analysis is conservative: a column used in a way the analyzer doesn't model
gets no hint, so the cache never drops data a query still needs.

**Flight mode** is the same flow split across the wire: the server only sees the
pushed-down fragment (which may lack the lineage), so the client derives the hints
from the full plan and ships them with the plan, and the server attaches them to
the `LiquidParquetSource` it builds.
