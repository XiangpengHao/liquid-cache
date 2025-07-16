# Parse latency and bandwidth from server log file
import sys
import re
import numpy as np

def parse_bandwidth(filename):
    # Pattern to match: p<number> disk usage: <number>
    pattern = re.compile(r'\b(p\d+) disk usage: (\d+)\b')
    bandwidths = {}

    with open(filename, 'r') as f:
        for line in f:
            match = pattern.search(line)
            if match:
                percentile = match.group(1)  # e.g., 'p75'
                value = int(match.group(2))  # disk usage value
                if percentile not in bandwidths:
                    bandwidths[percentile] = []
                # Ignore write bandwidth's for now
                if value > 1500:
                    bandwidths[percentile].append(value)

    return bandwidths


def parse_latency(filename):
    # Pattern to match: read_liquid_from_disk took <number> us
    pattern = re.compile(r'read_liquid_from_disk took (\d+) us')
    latencies = []

    with open(filename, 'r') as f:
        for line in f:
            match = pattern.search(line)
            if match:
                value = int(match.group(1))
                latencies.append(value)

    if not latencies:
        return {}

    arr = np.array(latencies)
    percentiles = {}
    for p in range(50, 100, 10):  # p50, p60, ..., p90
        percentiles[f"p{p}"] = float(np.percentile(arr, p))
    return percentiles

def parse_inflight(filename):
    # Pattern to match: p<number> inflight requests: <number>
    pattern = re.compile(r'\b(p\d+) inflight requests: (\d+)')
    queue_depth = {}

    with open(filename, 'r') as f:
        for line in f:
            match = pattern.search(line)
            if match:
                percentile = match.group(1)  # e.g., 'p75'
                value = int(match.group(2))  # disk usage value
                if percentile not in queue_depth:
                    queue_depth[percentile] = []
                queue_depth[percentile].append(value)

    return queue_depth


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python parse_bandwidth.py <log_filename>")
        sys.exit(1)

    filename = sys.argv[1]
    values = parse_bandwidth(filename)

    if not values:
        print("No bandwidth values found.")
    else:
        print("Parsed bandwidth values (e.g., p75):")
        for percentile, vals in sorted(values.items()):
            print(f"{percentile}: {int(sum(vals)/len(vals))} MBps")

    latency_values = parse_latency(filename)
    if not latency_values:
        print("No latency values found.")
    else:
        print("Parsed latency values (e.g., p50):")
        for percentile, val in sorted(latency_values.items()):
            print(f"{percentile}: {int(val)} us")

    queue_depth_values = parse_inflight(filename)

    if not queue_depth_values:
        print("No queue depth values found.")
    else:
        print("Parsed inflight requests values (e.g., p75):")
        for percentile, vals in sorted(queue_depth_values.items()):
            print(f"{percentile}: {int(sum(vals)/len(vals))}")