#!/bin/bash
#
# Script to launch the bench_server, disk_monitor, and clickbench_client for benchmarking.
# Cleans up the tmp directory, starts the server and monitor in the background,
# and runs the client benchmark.
#
# Usage: ./run_server.sh NUM_PARTITIONS

if [ $# -lt 1 ]; then
    echo "Error: NUM_PARTITIONS argument required."
    echo "Usage: $0 NUM_PARTITIONS"
    exit 1
fi

NUM_PARTITIONS=$1

# Create log directory if it doesn't exist
mkdir -p log

if [ -d tmp ]; then
    rm -rf tmp/*
fi

echo "Compiling all binaries..."
cargo build --release

echo "Starting bench_server..."
# Launch server and log output
env RUST_LOG=info RUST_BACKTRACE=full cargo run --release --bin bench_server \
    -- --max-cache-mb=10 --disk-cache-dir=tmp > log/server.log 2>&1 &
CARGO_PID=$!

echo "Running clickbench_client..."
# Run client and log output
env RUST_LOG=info cargo run --release --bin clickbench_client \
    -- --query-path clickbench/queries/queries.sql \
    --file clickbench/data/hits.parquet \
    --query 20 \
    --partitions $NUM_PARTITIONS \
    --iteration 100 > log/client.log 2>&1

echo "Killing bench_server (PID $CARGO_PID)..."
# Kill the server process
kill $CARGO_PID
wait $CARGO_PID 2>/dev/null

python3 parse_client_logs.py log/client.log