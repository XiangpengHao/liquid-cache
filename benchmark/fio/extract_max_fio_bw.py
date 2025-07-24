import sys
import re

def extract_max_bw(filename):
    pattern = re.compile(r'bw \(  KiB/s\):.*?max=(\d+)')
    max_values = []

    with open(filename, 'r') as f:
        for line in f:
            match = pattern.search(line)
            if match:
                max_bw = int(match.group(1))
                max_values.append(max_bw)

    return max_values

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python extract_max_bw.py <log_filename>")
        sys.exit(1)

    filename = sys.argv[1]
    max_bw_list = extract_max_bw(filename)

    if not max_bw_list:
        print("No matching max bandwidth values found.")
    else:
        print("Extracted max bandwidth values (KiB/s):")
        for bw in max_bw_list:
            print(bw)

        print(f"\nMaximum of max values: {max(max_bw_list)} KiB/s")
        print(f"Maximum possible bw: {sum(max_bw_list)/1024} MB/s")
