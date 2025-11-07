import duckdb
import os
import glob
import bz2
import shutil

REPO_DIR = ""  # path to your local repo
RAW_BZ2_DIR = os.path.join(REPO_DIR, "raw_bz2")   
CSV_DIR = os.path.join(REPO_DIR, "csv")           
TABLE_SQL_DIR = os.path.join(REPO_DIR, "tables")
PARQUET_DIR = os.path.join(REPO_DIR, "parquet_data")

# Create output folder
os.makedirs(PARQUET_DIR, exist_ok=True)
if not os.path.exists(CSV_DIR):
    os.makedirs(CSV_DIR, exist_ok=True)
    print(f"Created CSV directory {CSV_DIR} and extracting .bz2 files to CSV here.")
    for bz2_file in glob.glob(os.path.join(RAW_BZ2_DIR, "*.bz2")):
        csv_file = os.path.join(CSV_DIR, os.path.basename(bz2_file).replace("filename.bz2", "filename"))
        decompress_bz2(bz2_file, csv_file)
        print(f"Extracted {bz2_file} to {csv_file}")

# Connect to DuckDB
con = duckdb.connect()


def decompress_bz2(bz2_path, csv_path):
    print(f"Decompressing {bz2_path} -> {csv_path}")
    with bz2.open(bz2_path, "rb") as f_in:
        with open(csv_path, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)

# Main function
def process_table(table_name=None):
    sql_files = glob.glob(os.path.join(TABLE_SQL_DIR, "*.table.sql"))
    
    for sql_file in sql_files:
        base_name = os.path.basename(sql_file).replace(".table.sql", "")
        
        # Skip if table_name is specified and doesn't match
        if table_name and table_name != base_name:
            continue
        
        # Load table schema
        with open(sql_file, "r") as f:
            sql = f.read()
        con.execute(sql)
        
        # Determine CSV file
        csv_file = os.path.join(CSV_DIR, f"{base_name}.csv")
        if not os.path.exists(csv_file):
            bz2_file = os.path.join(RAW_BZ2_DIR, f"{base_name}.csv.bz2")
            if os.path.exists(bz2_file):
                decompress_bz2(bz2_file, csv_file)
            else:
                print(f"Skipping {base_name}: no CSV or BZ2 found")
                continue
        
        # Load CSV into DuckDB
        print(f"Loading {csv_file} into DuckDB table {base_name}...")
        con.execute(f"""
            COPY {base_name} FROM '{csv_file}' (DELIMITER '|', HEADER FALSE)
        """)
        
        # Write to Parquet
        parquet_file = os.path.join(PARQUET_DIR, f"{base_name}.parquet")
        print(f"Writing {base_name} to {parquet_file}...")
        con.execute(f"""
            COPY {base_name} TO '{parquet_file}' (FORMAT PARQUET)
        """)
    
    print("All done!")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Load CSVs into DuckDB and write Parquet")
    parser.add_argument("--table", type=str, help="Optional: process only a single table")
    args = parser.parse_args()
    
    process_table(args.table)
