import argparse
import sys
import numpy as np

def parse_biosnoop(file, pid, engine, timing_type):
    latencies = []
    for line in file:
        line = line.strip()
        if not line or line.startswith("TIME(") or line.startswith("--"):
            continue
        parts = line.split()
        if len(parts) < 8:
            continue
        try:
            line_pid = int(parts[2])
            comm = parts[1]
            que_ms = float(parts[-2])  # QUE column
            lat_ms = float(parts[-1])  # LAT column
        except (ValueError, IndexError):
            continue
        if line_pid == pid:
            cond = comm.startswith("iou-sqp") if engine == "uring" else comm.startswith("microbench_seq")
            if cond:
                if timing_type == "queued":
                    latencies.append(que_ms)
                elif timing_type == "block_device":
                    latencies.append(lat_ms)
                elif timing_type == "total":
                    latencies.append(lat_ms + que_ms)
    return latencies

def print_percentiles(latencies, engine, timing_type):
    if not latencies:
        print(f"No I/O events found for PID {args.pid} and engine {engine}.")
        return
    percentiles = [50, 70, 90, 99]
    values = np.percentile(latencies, percentiles)
    print(f"{timing_type.replace('_', ' ').title()} latency percentiles for PID {args.pid} (engine: {engine}):")
    for p, v in zip(percentiles, values):
        print(f"  p{p}: {v:.3f} ms")
    print(f"  count: {len(latencies)} events")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parse biosnoop-bpfcc output and print latency percentiles.")
    parser.add_argument("--pid", type=int, required=True, help="PID to filter")
    parser.add_argument("--engine", type=str, required=True, choices=["posix", "uring"], help="Engine type")
    parser.add_argument("--timing-type", type=str, required=True, choices=["queued", "block_device", "total"], 
                       help="Timing type to report: queued (QUE), block_device (LAT+QUE), or total (LAT)")
    parser.add_argument("--file", type=str, help="Path to biosnoop output file (default: stdin)")
    args = parser.parse_args()

    if args.file:
        with open(args.file, "r") as f:
            latencies = parse_biosnoop(f, args.pid, args.engine, args.timing_type)
    else:
        latencies = parse_biosnoop(sys.stdin, args.pid, args.engine, args.timing_type)

    print_percentiles(latencies, args.engine, args.timing_type)
