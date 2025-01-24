# Benchmark Guide

## Download dataset

We currently support [ClickBench](https://github.com/ClickHouse/ClickBench).

To download partitioned dataset (~100MB):
```bash
wget https://datasets.clickhouse.com/hits_compatible/athena_partitioned/hits_0.parquet -O benchmark/data/hits_0.parquet
```

To download the entire dataset (~15GB):

```bash
wget https://datasets.clickhouse.com/hits_compatible/athena/hits.parquet -O benchmark/data/hits.parquet
```

## Run benchmarks

### Minimal 

```bash
cargo run --release --bin bench_server
cargo run --release --bin bench_client -- --query-path benchmark/queries.sql --file benchmark/data/hits.parquet
```

### Advanced

```bash
env RUST_LOG=info cargo run --release --bin bench_server
env RUST_LOG=info cargo run --release --bin bench_client -- --query-path benchmark/queries.sql --file benchmark/data/hits_0.parquet --query 42
```
