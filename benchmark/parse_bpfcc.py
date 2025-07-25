import argparse
import sys
import numpy as np

def parse_biosnoop(file, pid, engine):
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
            lat_ms = float(parts[-1])
        except (ValueError, IndexError):
            continue
        if line_pid == pid:
            if engine == "uring" and comm.startswith("iou-sqp"):
                latencies.append(lat_ms)
            elif engine == "posix" and comm.startswith("microbench_seq"):
                latencies.append(lat_ms)
    return latencies

def print_percentiles(latencies, engine):
    if not latencies:
        print(f"No I/O events found for PID {args.pid} and engine {engine}.")
        return
    percentiles = [50, 70, 90, 99]
    values = np.percentile(latencies, percentiles)
    print(f"Latency percentiles for PID {args.pid} (engine: {engine}):")
    for p, v in zip(percentiles, values):
        print(f"  p{p}: {v:.3f} ms")
    print(f"  count: {len(latencies)} events")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parse biosnoop-bpfcc output and print latency percentiles.")
    parser.add_argument("--pid", type=int, required=True, help="PID to filter")
    parser.add_argument("--engine", type=str, required=True, choices=["posix", "uring"], help="Engine type")
    parser.add_argument("--file", type=str, help="Path to biosnoop output file (default: stdin)")
    args = parser.parse_args()

    if args.file:
        with open(args.file, "r") as f:
            latencies = parse_biosnoop(f, args.pid, args.engine)
    else:
        latencies = parse_biosnoop(sys.stdin, args.pid, args.engine)

    print_percentiles(latencies, args.engine)
