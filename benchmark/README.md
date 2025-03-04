# Benchmark Guide

## [ClickBench](https://github.com/ClickHouse/ClickBench) 

### Download dataset
To download partitioned dataset (~100MB):
```bash
wget https://datasets.clickhouse.com/hits_compatible/athena_partitioned/hits_0.parquet -O benchmark/data/hits_0.parquet
```

To download the entire dataset (~15GB):

```bash
wget https://datasets.clickhouse.com/hits_compatible/athena/hits.parquet -O benchmark/data/hits.parquet
```

To download the partitioned dataset (100 files, ~150MB each):
```bash
for i in (seq 0 99)
    wget https://datasets.clickhouse.com/hits_compatible/athena_partitioned/hits_$i.parquet -O benchmark/data/partitioned/hits_$i.parquet
end
```
Or bash :
```bash
for i in {0..99}; do
    wget https://datasets.clickhouse.com/hits_compatible/athena_partitioned/hits_$i.parquet -O benchmark/data/partitioned/hits_$i.parquet
done
```

### Run benchmarks

#### Minimal 

```bash
cargo run --release --bin bench_server
cargo run --release --bin clickbench_client -- --query-path benchmark/queries.sql --file benchmark/data/hits.parquet
```

#### Advanced

```bash
env RUST_LOG=info RUST_BACKTRACE=1 RUSTFLAGS='-C target-cpu=native' cargo run --release --bin bench_server
env RUST_LOG=info RUST_BACKTRACE=1 RUSTFLAGS='-C target-cpu=native' cargo run --release --bin clickbench_client -- --query-path benchmark/queries.sql --file benchmark/data/hits.parquet --query 42
```

## TPCH

### Generate data

(make sure you have [docker](https://github.com/docker/docker-install) and [uv](https://docs.astral.sh/uv/getting-started/installation/) installed)

```bash
cd benchmark/tpch
bash tpch_gen.sh 0.1 # Generate data at scale factor 0.1
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


### Run encoding benchmarks

```bash
RUST_LOG=info RUSTFLAGS='-C target-cpu=native' cargo run --release --bin encoding -- --file benchmark/data/hits.parquet --column 2
```
This will benchmark the encoding time of the `URL` column.
