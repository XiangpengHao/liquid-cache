# SplitSQL Architecture

SplitSQL consists of three parts: 
- Cache: a server that caches data and evaluates the predicates.
- Compute: the DataFusion instance that executes user queries. 
- LiquidParquet: the cache-specific file format used by the server. 

## Cache

## Compute
The compute node is stateless.

## LiquidParquet