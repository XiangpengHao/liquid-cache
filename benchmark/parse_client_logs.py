import re
import sys

def parse_query_times(filename):
    # Regex to match 'Query: <number> ms'
    pattern = re.compile(r'Query:\s+(\d+)\s+ms')
    with open(filename, 'r') as f:
        for line in f:
            match = pattern.search(line)
            if match:
                latencies.append(float(match.group(1)))
    if latencies:
        arr = np.array(latencies)
        p50 = np.percentile(arr, 50)
        p99 = np.percentile(arr, 99)
        print(f"P50 latency: {p50} ms")
        print(f"P99 latency: {p99} ms")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 parse_client_logs.py <logfile>")
        sys.exit(1)
    parse_query_times(sys.argv[1])