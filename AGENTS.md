## Code structure
- `src/storage`, this is the LiquidCache core.
- `src/parquet`, Parquet and DataFusion integration, this allows datafusion/parquet users to use LiquidCache with minimal effort.
- `src/client` and `src/server`, Client/Server library, this enables distributed LiquidCache.
- `src/local`, this is a in-process LiquidCache, used for local DataFusion instances. 


### Lineage-based cache expression
1. The lineage_opt.rs analyze the input query's column usage, and passes it down to LiquidCache.
2. When creating a CachedColumn, it will register the expression to the cache; This info is used to determine the squeeze behavior only.
3. During cache read, a expression is passed in, the cache will decide whether the cached (squeezed) data can satisfy the expression, if so, it will return the cached data directly; otherwise, it will hydrate the data from disk.

## Testing
1. All test cases are intended for human to read. They are part of the code and documents.
2. Test cases should be short, concise, yet have high coverage. Writing test cases almost requires highest level of thinking.
