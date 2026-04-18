## Engineering principles

1. Think about minimal changes to complete the task.
2. Always cargo check after coding.
3. Consider refactoring when a function needs more than 3 args.
4. Whenever writing a if or edge case condition, (1) think why it is needed here, whether it is a real edge case, (2) think whether we can move the condition upper to the caller, i.e., is this the best place to handle edge case?
5. Never need to consider backward compatibility.

## Code structure

- `src/core`, this is the LiquidCache core.
- `src/datafusion`, Parquet and DataFusion integration, this allows datafusion/parquet users to use LiquidCache with minimal effort.
- `src/datafusion-client` and `src/datafusion-server`, Client/Server library, this enables distributed LiquidCache.
- `src/datafusion-local`, this is a in-process LiquidCache, used for local DataFusion instances.

## Study guide

This repo has many studies, they are not for production use, but only for research purposes to understand how the system behaves.
The coding guidelines for study are different:

1. Focus on concise, minimal intrusive, easy to understand code.
2. No error handling, no robust edge case handling, just one shot code.
3. Ok to hard code if it can simplify the implementation.

### Lineage-based cache expression

1. The lineage_opt.rs analyze the input query's column usage, and passes it down to LiquidCache.
2. When creating a CachedColumn, it will register the expression to the cache; This info is used to determine the squeeze behavior only.
3. During cache read, a expression is passed in, the cache will decide whether the cached (squeezed) data can satisfy the expression, if so, it will return the cached data directly; otherwise, it will hydrate the data from disk.

## Testing

1. All test cases are intended for human to read. They are part of the code and documents.
2. Test cases should be short, concise, yet have high coverage. Writing test cases almost requires highest level of thinking.
