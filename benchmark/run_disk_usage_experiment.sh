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

# if [ -d tmp ]; then
#     rm -rf tmp/*
# fi

log_dir=log/$(date +"%Y-%m-%d_%H-%M")
mkdir -p $log_dir

echo "Compiling all binaries..."
cargo build --release

echo "Starting bench_server..."
# Launch server and log output
env RUST_LOG=info RUST_BACKTRACE=full cargo run --release --bin bench_server \
    -- --max-cache-mb=10 --disk-cache-dir=tmp > $log_dir/server.log 2>&1 &
CARGO_PID=$!

sleep 20

echo "Starting disk_monitor..."
# Launch monitor and log output (with partition suffix)
cargo run --release --bin disk_monitor -- $CARGO_PID 200 > $log_dir/disk_monitor_${NUM_PARTITIONS}.log &
DISK_MONITOR_PID=$!

echo "Running clickbench_client..."
# Run client and log output
env RUST_LOG=info cargo run --release --bin clickbench_client \
    -- --query-path clickbench/queries/queries.sql \
    --file clickbench/data/hits.parquet \
    --query 20 \
    --partitions $NUM_PARTITIONS \
    --iteration 5 --flamegraph > $log_dir/client.log 2>&1

echo "Killing bench_server (PID $CARGO_PID)..."
# Kill the server process
kill $CARGO_PID
wait $CARGO_PID 2>/dev/null

wait $DISK_MONITOR_PID 2>/dev/null

tail -n 1 $log_dir/disk_monitor_${NUM_PARTITIONS}.log