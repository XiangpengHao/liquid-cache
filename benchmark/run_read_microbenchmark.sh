#!/bin/bash

# set -e

# Usage message
usage() {
  echo "Usage: $0 --engine uring|posix --num-files N --file-size SIZE --num-threads N --chunk-size SIZE --dir DIR [--extra-args '...']"
  echo "Example: $0 --engine uring --num-files 4 --file-size 1G --num-threads 4 --chunk-size 4096 --dir /tmp/bench"
  exit 1
}

# Default values
CHUNK_SIZE=4096
DIR="tmp/bench"
EXTRA_ARGS=""

# Parse arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --engine) ENGINE="$2"; shift 2 ;;
    --num-files) NUM_FILES="$2"; shift 2 ;;
    --file-size) FILE_SIZE="$2"; shift 2 ;;
    --num-threads) NUM_THREADS="$2"; shift 2 ;;
    --chunk-size) CHUNK_SIZE="$2"; shift 2 ;;
    --dir) DIR="$2"; shift 2 ;;
    --extra-args) EXTRA_ARGS="$2"; shift 2 ;;
    *) usage ;;
  esac
done

# Check required args
if [[ -z "$ENGINE" || -z "$NUM_FILES" || -z "$FILE_SIZE" || -z "$NUM_THREADS" ]]; then
  usage
fi

# Create directory for files
mkdir -p "$DIR"

# Create files using fio
echo "Creating $NUM_FILES files of size $FILE_SIZE in $DIR ..."
for i in $(seq 0 $((NUM_FILES-1))); do
  FILE="$DIR/file${i}.dat"
  if [[ ! -f "$FILE" ]]; then
    echo "  Creating $FILE ..."
    fio --name=prep --filename="$FILE" --size="$FILE_SIZE" --rw=write --bs=1M --direct=1 --iodepth=1 --refill_buffers --randrepeat=0 --output=/dev/null &
  else
    echo "  $FILE already exists, skipping."
  fi
done

wait
sync

# Build file list for microbenchmark
FILE_LIST=""
for i in $(seq 0 $((NUM_FILES-1))); do
  FILE_LIST="$FILE_LIST $DIR/file${i}.dat"
done

sudo /usr/sbin/biosnoop-bpfcc -Q > biosnoop.log 2>&1 &
BIO_PID=$!
# Wait for 10 seconds to let the bio latency tool jit compilation finish
sleep 10

# Run the microbenchmark
echo "Running microbenchmark..."
env RUST_LOG=info RUST_BACKTRACE=full cargo run --release --bin microbench_sequential_read -- \
  --engine "$ENGINE" \
  --num-threads "$NUM_THREADS" \
  --file-size "$(( $(numfmt --from=iec $FILE_SIZE) ))" \
  --chunk-size "$(( $(numfmt --from=iec $CHUNK_SIZE) ))" \
  --files "$FILE_LIST" \
  $EXTRA_ARGS &
MICROBENCH_PID=$!

wait $MICROBENCH_PID
sudo kill $BIO_PID
sync

python3 parse_bpfcc.py --pid $MICROBENCH_PID --engine $ENGINE --file biosnoop.log
echo "Done."
# rm -rf "$DIR"/
