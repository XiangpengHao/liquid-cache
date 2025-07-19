import re
import sys
from collections import defaultdict

def main():
    if len(sys.argv) != 2:
        print(f"Usage: python {sys.argv[0]} <log_file_path>")
        sys.exit(1)

    log_file_path = sys.argv[1]
    percentile_re = re.compile(r'p(\d+):\s+(\d+)')
    percentiles = defaultdict(list)

    try:
        with open(log_file_path, 'r') as f:
            for line in f:
                match = percentile_re.search(line)
                if match:
                    percentile = int(match.group(1))
                    value = int(match.group(2))
                    if value > 1200:
                        percentiles[percentile].append(value)
    except FileNotFoundError:
        print(f"Error: File '{log_file_path}' not found.")
        sys.exit(1)

    if not percentiles:
        print("No percentile data found in the log.")
        return

    # Aggregate: average
    aggregated = {p: sum(vals) / len(vals) for p, vals in percentiles.items()}

    print("Aggregated percentile values:")
    for p in sorted(aggregated):
        print(f"p{p}: {aggregated[p]:.2f}")

if __name__ == "__main__":
    main()