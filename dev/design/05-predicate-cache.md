Predicate cache is a new approach to cache the cache expression along with its predicate evaluation result.

Let's consider this query:
```sql
SELECT COUNT(*) FROM table WHERE Url like '%google%';
```

There are a few ways to cache data:
1. Cache the `url` column - data cache
2. Cache the result of the sql, e.g., 15211 - result cache
3. Cache a subquery: `SELECT * FROM table WHERE Url like '%google%'`, next time we rewrite the query to `SELECT COUNT(url) FROM view` - materialized view 

Predicate cache is a new approach:
1. It caches the bit mask of the predicate evaluation result, i.e., pair of (`'url like %google%'`, `[1, 0, 1, ..., 0, 1, 0]`).
2. Next time we see this predicate again, we can directly use the cache bit mask without evaluating the predicate again.

## Implementation

To start simple, we only consider predicate cache for string columns (ByteViewArray).

We will have a new SqueezedArray type: PredicateSqueezedArray, which holds a hashmap of `predicate` -> `bit mask`.

At squeeze time, we have two choices:
1. squeeze the string array by dropping the fsst buffers, i.e., becoming a `LiquidByteViewArray<DiskBuffer>` 
2. squeeze the string array by becoming a `PredicateSqueezedArray`, which evaluates the predicates and stores the bit mask.

The intuition is that: 1. if the predicate is comparison, we do first, 2. if the predicate is substring search (e.g., like), we do later.

In order for the squeeze policy to know what expression it previously evaluated, we need to register it at `CachedColumn` creation time.
CachedColumn knows the expression from the lineage analysis in `lineage_opt.rs`, which tells us all the expressions applied to a given column.

Then in the `eval_predicate` method of LiquidCache, we will check if the predicate is already in the `PredicateSqueezedArray`, if so, we can directly return the bit mask.
