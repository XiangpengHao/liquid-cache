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

# Create log and cache directory if it doesn't exist
mkdir -p log
mkdir -p tmp
mkdir -p disk_usage_stats

log_dir=log/$(date +"%Y-%m-%d_%H-%M")
mkdir -p $log_dir
ln -sfn $log_dir latest

echo "Compiling all binaries..."
cargo build --release

echo "Starting bench_server..."
# Launch server and log output
env RUST_LOG=info RUST_BACKTRACE=full cargo run --release --bin bench_server \
    -- --max-cache-mb=10 --disk-cache-dir=tmp > $log_dir/server.log 2>&1 &
SERVER_PID=$!

sleep 20

echo "Running clickbench_client..."
# Run client and log output
env RUST_LOG=info RUST_BACKTRACE=full cargo run --release --bin clickbench_client \
    -- --query-path clickbench/queries/queries.sql \
    --file clickbench/data/hits.parquet \
    --query 20 \
    --partitions $NUM_PARTITIONS \
    --iteration 10 \
    --disk-usage-histogram-dir disk_usage_stats > $log_dir/client.log 2>&1

echo "Killing bench_server (PID $SERVER_PID)..."
# Kill the server process
kill $SERVER_PID
wait $SERVER_PID 2>/dev/null

python3 parse_latency_bw.py $log_dir/server.log