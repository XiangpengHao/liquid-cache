#!/usr/bin/env python3
import os
import sys
import glob
import pyarrow.csv as csv
import pyarrow.parquet as pq

def convert_tbl_to_parquet(input_file):
    output_file = input_file.replace('.tbl', '.parquet')
    print(f"Converting {input_file} to {output_file}")
    read_options = csv.ReadOptions(
        use_threads=True,
        block_size=1024 * 1024  # 1MB chunks
    )
    parse_options = csv.ParseOptions(
        delimiter='|', 
        quote_char=False, 
        escape_char=False,
        newlines_in_values=False
    )
    
    table = csv.read_csv(input_file, read_options=read_options, parse_options=parse_options)
    pq.write_table(table, output_file, compression='snappy')
    os.remove(input_file)
    return output_file

def main():
    if len(sys.argv) < 2:
        print("Usage: csv_to_parquet.py <directory>")
        sys.exit(1)
    
    directory = sys.argv[1]
    tbl_files = glob.glob(os.path.join(directory, "*.tbl"))
   
    for tbl_file in tbl_files:
        convert_tbl_to_parquet(tbl_file)

if __name__ == "__main__":
    main()