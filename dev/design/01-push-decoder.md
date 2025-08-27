Unlike pull-based decoder (also known as "sans-io"), where the IO is fetched on demand, push-based decode fetches IO in advance and send the IO to the compute sink.

Prior/concurrent art: 
- https://github.com/apache/arrow-rs/issues/8000
- https://github.com/apache/arrow-rs/issues/7983


Sans-io:
- https://fasterthanli.me/articles/the-case-for-sans-io
- https://sans-io.readthedocs.io


## Problem 1

Modern SSDs needs 1000 inflight io requests to saturate bandwidth, two approaches:
1. Create 1000 threads to fetch IO. This can easily thrash the CPU, cause excessive context switching, and excessive memory usage (each thread needs non-trivial amount of stack space).
2. Thread-per-core, within each thread, we run an async runtime (user space scheduling) to fetch IO. Problems: user space scheduling (cooperative scheduling) has no fairness guarantee, caused numerous weird bugs and errors.


## Problem 2

PCIe 5 SSDs are very different from S3, in terms of pricing models, bandwidth, and latency. 
For example:
- S3 is billed by requests not traffic, i.e., one 10KB request costs the same as one 10 GB request.  
- PCIe 5 SSDs usually have deep queue depth, requiring one big request to be split into multiple small requests to saturate bandwidth.

There're two existing approaches:
1. Treat them as the same. This is the most common approach in the industry.
2. Rewrite different code for different storage types, for example, [zip2](https://github.com/zip-rs/zip2) and [async-zip](https://github.com/majored/rs-async-zip). Which means duplicate efforts, more bugs, and less maintainability. This is almost impossible for larger projects like Parquet.




## Measure

We first write a benchmark to measure the performance of existing cache.

1. We take the existing CacheStorage and use the FILO policy to cache the data. Set the cache size to be small, e.g., 100 MB. 

2. Then we keep inserting string data into the cache. The string data is from hits.parquet (benchmark/clickbench/data/hits.parquet), we need to read with projections. Some example string columns are: URL, Referer, Title, etc. For simplicity, we only read Referer.

3. Once the data is inserted, we start to read from the cache. For simplicity, we only exercise the `get_with_predicate` method, where the selection is a boolean array with all true, and the predicate is whether this string is empty.

4. We will read all previously inserted data (simulate a scan), and measure the time.

5. The benchmark file will be in src/storage/study/cache_storage.rs. 
