# Benchmark Guide

## [ClickBench](https://github.com/ClickHouse/ClickBench) 

### Download dataset
To download partitioned dataset (~100MB):
```bash
wget https://datasets.clickhouse.com/hits_compatible/athena_partitioned/hits_0.parquet -O benchmark/data/hits_0.parquet
```

To download the entire dataset (~15GB):

```bash
wget https://datasets.clickhouse.com/hits_compatible/athena/hits.parquet -O benchmark/clickbench/data/hits.parquet
```

To download the partitioned dataset (100 files, ~150MB each):
```bash
for i in (seq 0 99)
    wget https://datasets.clickhouse.com/hits_compatible/athena_partitioned/hits_$i.parquet -O benchmark/clickbench/data/partitioned/hits_$i.parquet
end
```
Or bash :
```bash
for i in {0..99}; do
    wget https://datasets.clickhouse.com/hits_compatible/athena_partitioned/hits_$i.parquet -O benchmark/clickbench/data/partitioned/hits_$i.parquet
done
```

### Run benchmarks

#### Minimal 

```bash
cargo run --release --bin bench_server
cargo run --release --bin clickbench_client -- --manifest benchmark/clickbench/manifest.json
```

#### Advanced

```bash
env RUST_LOG=info RUST_BACKTRACE=1 RUSTFLAGS='-C target-cpu=native' cargo run --release --bin bench_server -- --cache-mode liquid_eager_transcode
env RUST_LOG=info RUST_BACKTRACE=1 RUSTFLAGS='-C target-cpu=native' cargo run --release --bin clickbench_client -- --manifest benchmark/clickbench/manifest.json --query 42
```

## TPCH

### Generate data

(make sure you have [uv](https://docs.astral.sh/uv/getting-started/installation/) installed)

```bash
cd benchmark/tpch
uvx --from duckdb python tpch_gen.py --scale 0.01
```

In NixOS, you want to set `env LD_LIBRARY_PATH=$NIX_LD_LIBRARY_PATH`




### Run server (same as ClickBench)

```bash
cargo run --release --bin bench_server -- --cache-mode liquid_eager_transcode
```

### Run client

```bash
env RUST_LOG=info,clickbench_client=debug RUSTFLAGS='-C target-cpu=native' cargo run --release --bin tpch_client -- --query-dir benchmark/tpch/queries/ --data-dir benchmark/tpch/data/sf0.1  --iteration 3 --answer-dir benchmark/tpch/answers/sf0.1
```

## In process mode
The benchmark uses a JSON manifest file to describe the data tables and queries to run.

### JSON Format

```json
{
  "name": "My Benchmark Suite",
  "description": "Description of what this benchmark tests",
  "tables": {
    "table1": "data/table1.parquet",
    "table2": "data/table2.parquet"
  },
  "queries": [
    "queries/aggregation.sql",
    "SELECT COUNT(*) FROM table1 WHERE col > 100",
    "queries/complex_join.sql",
    "SELECT t1.id, t2.name FROM table1 t1 JOIN table2 t2 ON t1.id = t2.id LIMIT 10"
  ],
  "object_stores": [
    {
      "url": "s3://my-bucket",
      "options": {
        "aws_access_key_id": "my-access-key",
        "aws_secret_access_key": "my-secret-key",
        "aws_region": "us-west-2"
      }
    },
    {
      "url": "s3://my-bucket-2",
      "options": {
        "aws_region": "eu-central-1",
        "skip_signature": "true"
      }
    }
  ]
}
```


## Profile

### Flamegraph

To collect flamegraph from server side, simply add `--flamegraph-dir benchmark/data/flamegraph` to the server command, for example:
```bash
cargo run --release --bin bench_server -- --flamegraph-dir benchmark/data/flamegraph
```
It will generate flamegraph for each query that the server executed.

### Cache stats

To collect cache stats, simply add `--stats-dir benchmark/data/cache_stats` to the server command, for example:
```bash
cargo run --release --bin bench_server -- --stats-dir benchmark/data/cache_stats
```
It will generate a parquet file that contains the cache stats for each query that the server executed.
You can use [`parquet-viewer`](https://parquet-viewer.xiangpeng.systems) to view the stats in the browser.

### Collect cache trace

To collect cache trace, simply add `--cache-trace-dir benchmark/data/cache_trace` to the client command, for example:
```bash
env RUST_LOG=info cargo run --bin clickbench_client --release -- --manifest benchmark/clickbench/manifest.json --query 20 --iteration 2 --partitions 8 --cache-trace-dir benchmark/data/
```
It will generate a parquet file that contains the cache trace for each query that the server executed.


### Run encoding benchmarks

```bash
RUST_LOG=info RUSTFLAGS='-C target-cpu=native' cargo run --release --bin encoding -- --file benchmark/clickbench/data/hits.parquet --column 2
```
This will benchmark the encoding time of the `URL` column.
