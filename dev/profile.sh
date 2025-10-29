#!/usr/bin/env bash
# usage: sudo ./profile.sh ./target/release/in_process [args...]
# Environment variables:
#   PERF_MODE:   "record" (default), "stat", "sched", "offcpu", "sched-verbose"
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
case "$PERF_MODE" in
  "stat")
    # Hardware/software counters
    PERF_CMD="sudo perf stat -p $TARGET_PID -a -d \
      -e tlb:tlb_flush \
      -e page-faults \
      -e context-switches \
      -e dTLB-loads,dTLB-load-misses \
      -e iTLB-loads,iTLB-load-misses \
      -e cache-misses,cache-references"
    PERF_STOP_CMD="kill -INT \$(cat /tmp/perfpid)"
    ;;
    
  "sched")
    # Context switch tracing with call stacks
    # Shows which functions cause context switches
    PERF_CMD="sudo perf record -p $TARGET_PID -a -k CLOCK_MONOTONIC \
      -e sched:sched_switch \
      -e sched:sched_process_exit \
      --call-graph dwarf -o $PERF_OUT"
    PERF_STOP_CMD="kill -INT \$(cat /tmp/perfpid)"
    ;;
    
  "offcpu")
    # Off-CPU profiling: shows where threads spend time blocked
    # Captures both the blocking point and time spent waiting
    PERF_CMD="sudo perf record -p $TARGET_PID -a -k CLOCK_MONOTONIC \
      -e sched:sched_switch \
      -e sched:sched_stat_sleep \
      -e sched:sched_stat_blocked \
      -e sched:sched_stat_iowait \
      --call-graph dwarf -o $PERF_OUT"
    PERF_STOP_CMD="kill -INT \$(cat /tmp/perfpid)"
    ;;
    
  "sched-verbose")
    # Detailed scheduler events including wakeup sources
    PERF_CMD="sudo perf record -p $TARGET_PID -a -k CLOCK_MONOTONIC \
      -e 'sched:*' \
      --call-graph dwarf -o $PERF_OUT"
    PERF_STOP_CMD="kill -INT \$(cat /tmp/perfpid)"
    ;;
    
  *)
    # Default: CPU profiling
    PERF_CMD="sudo perf record -F 5999 -p $TARGET_PID -g -a -k CLOCK_MONOTONIC --call-graph dwarf -o $PERF_OUT"
    PERF_STOP_CMD="kill -INT \$(cat /tmp/perfpid)"
    ;;
esac

sudo bpftrace --unsafe -e '
usdt:'"$BIN_REAL"':liquid_benchmark:iteration_start /arg1 == '"$START_ITER"'/ {
  system("sh -c '\'''"$PERF_CMD"' & echo $! > /tmp/perfpid'\''");
}
usdt:'"$BIN_REAL"':liquid_benchmark:iteration_start /arg1 > '"$START_ITER"'/ {
  system("sh -c '\'''"$PERF_STOP_CMD"''\''");
}
'