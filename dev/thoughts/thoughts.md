### Inventing a new file format is not a good idea.
One of the reason is that a file format needs to be well supported by the query engine.
Parquet format itself is only x line of code, but DataFusion has xxx lines of code to support Parquet. The effort to support parquet is x% of the effort to implement a new file format, where x is a very large number.
If we simply invent a new file format, we are likely have worse performance than Parquet.

### Predicate pushdown can be slower
Because the output of predicate pushdown is in CSV/JSON format, meaning that the data is not compressed.
It can result in much larger network traffic if the filter is not selective enough.

Why we can't compress the output or re-encode the data in Parquet? 
That's a lot of CPU cost. (is that true?)


### TableProvider vs LiquidParquetExec 
We currently pack our system as a `TableProvider`. 
This does not work for people who already have their own `TableProvider`.
In that case, they might want to re-implement many parts of our system. 

### In-process mode
While SplitSQL is a one-stop comprehensive solution for disaggregated cache, many people might want to use part of our components.
For example, some people might want disaggregated cache, others might want to use only `LiquidParquetExec`.

### Rigorous testing
We need a lot of systematic testing to ensure our system is correct.
Especially the `LiquidArray` part. Fuzzing is planned.
