LiquidCache examples

## DataFusion (in-process)

Query a local parquet file through `LiquidCacheLocalBuilder`.

```bash
cargo run --bin datafusion-local
```

## DataFusion (client/server)

A single binary that runs as either the Flight cache server or a DataFusion client.

Start the server:

```bash
cargo run --bin datafusion-client-server -- --mode server
```

In another shell, run the client:

```bash
cargo run --bin datafusion-client-server -- --mode client
```

Use `--query` and `--file` to customize the SQL and remote parquet URL.

## Core (no DataFusion)

Use the raw `LiquidCacheBuilder` storage API to insert, flush, and read an Arrow array.

```bash
cargo run --bin core
```
