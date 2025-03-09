#!/usr/bin/env python3

import os
import sys
import argparse
import duckdb
from pathlib import Path

def ensure_dir(dir_path):
    """Ensure directory exists"""
    Path(dir_path).mkdir(parents=True, exist_ok=True)

def generate_tpch_data(scale_factor, data_dir, answers_dir):
    """Generate TPCH data at specified scale factor and save to parquet files"""
    print(f"Generating TPCH data with scale factor {scale_factor}")
    
    ensure_dir(data_dir)
    ensure_dir(answers_dir)
    
    conn = duckdb.connect(database=':memory:')
    
    try:
        conn.execute("INSTALL tpch")
        conn.execute("LOAD tpch")
        
        conn.execute(f"CALL dbgen(sf={scale_factor})")
        
        tables = ['lineitem', 'orders', 'customer', 'part', 'partsupp', 'supplier', 'nation', 'region']
        
        for table in tables:
            output_path = os.path.join(data_dir, f"{table}.parquet")
            print(f"Saving {table} to {output_path}")
            conn.execute(f"COPY {table} TO '{output_path}' (FORMAT 'PARQUET')")
        
        print(f"Retrieving answers for scale factor {scale_factor}")
        
        conn.execute(f"CREATE TEMPORARY TABLE tpch_answers_temp AS SELECT * FROM tpch_answers() WHERE scale_factor = {scale_factor}")
        
        result = conn.execute("SELECT query_nr FROM tpch_answers_temp ORDER BY query_nr").fetchall()
        
        for row in result:
            query_nr = row[0]
            
            answer_csv_result = conn.execute(f"SELECT answer FROM tpch_answers_temp WHERE query_nr = {query_nr}").fetchone()
            if not answer_csv_result:
                print(f"Warning: No answer found for query {query_nr}, skipping")
                continue
                
            answer_csv = answer_csv_result[0]
            if not answer_csv or answer_csv.isspace():
                print(f"Warning: Empty answer for query {query_nr}, skipping")
                continue
                
            output_path = os.path.join(answers_dir, f"q{query_nr}.parquet")
            print(f"Processing answer for query {query_nr} and saving to {output_path}")
            
            temp_csv = os.path.join(answers_dir, f"q{query_nr}_temp.csv")
            with open(temp_csv, 'w') as f:
                f.write(answer_csv)
            
            conn.execute(f"CREATE OR REPLACE TABLE q{query_nr}_temp AS SELECT * FROM read_csv('{temp_csv}', delim='|', header=true)")
            conn.execute(f"COPY q{query_nr}_temp TO '{output_path}' (FORMAT 'PARQUET')")
            
            os.remove(temp_csv)
    
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
    finally:
        conn.close()
    
    print("TPCH data and query answers generation completed successfully")

def main():
    parser = argparse.ArgumentParser(description='Generate TPCH data and query answers using DuckDB')
    parser.add_argument('--scale', type=float, default=0.01, help='Scale factor (default: 0.01)')
    parser.add_argument('--data-dir', type=str, default='data', help='Directory to store data parquet files')
    parser.add_argument('--answers-dir', type=str, default='answers', help='Directory to store query answers parquet files')
    
    args = parser.parse_args()
    data_dir = os.path.join(args.data_dir, f"sf{args.scale}")
    answers_dir = os.path.join(args.answers_dir, f"sf{args.scale}")

    generate_tpch_data(args.scale, data_dir, answers_dir)

if __name__ == "__main__":
    main()
