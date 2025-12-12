## Code structure
- `src/storage`, this is the LiquidCache core.
- `src/parquet`, Parquet and DataFusion integration, this allows datafusion/parquet users to use LiquidCache with minimal effort.
- `src/client` and `src/server`, Client/Server library, this enables distributed LiquidCache.
- `src/local`, this is a in-process LiquidCache, used for local DataFusion instances. 

