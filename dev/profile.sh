#!/usr/bin/env bash
# usage: sudo ./profile.sh ./target/release/in_process [args...]
# Environment variables:
#   PERF_MODE:   "record" (default) or "stat"
#   START_ITER:  Iteration number to start profiling (default: 2)
#   PERF_OUT:    Output file for perf record mode (default: perf.data)
set -euo pipefail
BIN="$1"; shift || true
"$BIN" "$@" & TARGET_PID=$!
BIN_REAL="$(readlink -f "$BIN")"
START_ITER="${START_ITER:-2}"
PERF_MODE="${PERF_MODE:-record}"
PERF_OUT="${PERF_OUT:-perf.data}"

# Build the appropriate perf command based on mode
if [ "$PERF_MODE" = "stat" ]; then
  PERF_CMD="sudo perf stat -p $TARGET_PID -a -d"
  PERF_STOP_CMD="kill -INT \$(cat /tmp/perfpid)"
else
  # Default to record mode
  PERF_CMD="sudo perf record -F 5999 -p $TARGET_PID -g -a -k CLOCK_MONOTONIC --call-graph dwarf -o $PERF_OUT"
  PERF_STOP_CMD="kill -INT \$(cat /tmp/perfpid)"
fi

sudo bpftrace --unsafe -e '
usdt:'"$BIN_REAL"':liquid_benchmark:iteration_start /arg1 == '"$START_ITER"'/ {
  system("sh -c '\'''"$PERF_CMD"' & echo $! > /tmp/perfpid'\''");
}
usdt:'"$BIN_REAL"':liquid_benchmark:iteration_start /arg1 > '"$START_ITER"'/ {
  system("sh -c '\'''"$PERF_STOP_CMD"''\''");
}
'