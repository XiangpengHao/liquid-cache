#!/usr/bin/env python3

import os
import argparse
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.orc as orc
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

def convert_file(input_path, output_path):
    """Convert a single parquet file to ORC with memory optimization"""
    print(f"Converting {input_path} to {output_path}")
    input_size = os.path.getsize(input_path) / (1024 * 1024)  # MB
    
    table = pq.read_table(
        input_path,
        memory_map=True,        # Use memory mapping for better memory efficiency
        use_threads=True,       # Enable multi-threading for faster reads
    )
    
    orc.write_table(table, output_path, compression='snappy')
    
    output_size = os.path.getsize(output_path) / (1024 * 1024)  # MB
    
    print(f"✓ {os.path.basename(input_path)}: {input_size:.1f} MB → {output_size:.1f} MB")

def main():
    parser = argparse.ArgumentParser(description='Convert parquet files to ORC format')
    parser.add_argument('--input-dir', required=True, help='Directory with parquet files')
    parser.add_argument('--output-dir', required=True, help='Output directory for ORC files')
    parser.add_argument('--workers', type=int, default=2, help='Number of parallel workers')
    
    args = parser.parse_args()
    
    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Find parquet files
    parquet_files = [f for f in os.listdir(args.input_dir) if f.endswith('.parquet')]
    print(f"Converting {len(parquet_files)} files using {args.workers} workers...")
    
    # Convert files in parallel
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = []
        for file in parquet_files:
            input_path = os.path.join(args.input_dir, file)
            output_path = os.path.join(args.output_dir, file.replace('.parquet', '.orc'))
            futures.append(executor.submit(convert_file, input_path, output_path))
        
        # Wait for all to complete
        for future in futures:
            future.result()
    
    print("Done!")

if __name__ == "__main__":
    main() 