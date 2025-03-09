#!/usr/bin/env python3
import os
import sys
import glob
import pyarrow.csv as csv
import pyarrow.parquet as pq
import pyarrow as pa
import os.path

def get_schema_for_table(table_name):
    """Return the column names for a given TPCH table."""
    schemas = {
        "part": [
            "p_partkey", "p_name", "p_mfgr", "p_brand", "p_type", "p_size", 
            "p_container", "p_retailprice", "p_comment"
        ],
        "supplier": [
            "s_suppkey", "s_name", "s_address", "s_nationkey", "s_phone", 
            "s_acctbal", "s_comment"
        ],
        "partsupp": [
            "ps_partkey", "ps_suppkey", "ps_availqty", "ps_supplycost", "ps_comment"
        ],
        "customer": [
            "c_custkey", "c_name", "c_address", "c_nationkey", "c_phone", 
            "c_acctbal", "c_mktsegment", "c_comment"
        ],
        "orders": [
            "o_orderkey", "o_custkey", "o_orderstatus", "o_totalprice", "o_orderdate", 
            "o_orderpriority", "o_clerk", "o_shippriority", "o_comment"
        ],
        "lineitem": [
            "l_orderkey", "l_partkey", "l_suppkey", "l_linenumber", "l_quantity", 
            "l_extendedprice", "l_discount", "l_tax", "l_returnflag", "l_linestatus", 
            "l_shipdate", "l_commitdate", "l_receiptdate", "l_shipinstruct", 
            "l_shipmode", "l_comment"
        ],
        "nation": [
            "n_nationkey", "n_name", "n_regionkey", "n_comment"
        ],
        "region": [
            "r_regionkey", "r_name", "r_comment"
        ],
    }
    return schemas.get(table_name, [])

def extract_table_name(filepath):
    """Extract the table name from the file path."""
    filename = os.path.basename(filepath)
    # Handle various naming conventions
    if filename.startswith("lineitem"):
        return "lineitem"
    elif filename.startswith("customer"):
        return "customer"
    elif filename.startswith("orders"):
        return "orders"
    elif filename.startswith("part"):
        if filename.startswith("partsupp"):
            return "partsupp"
        else:
            return "part"
    elif filename.startswith("supplier"):
        return "supplier"
    elif filename.startswith("nation"):
        return "nation"
    elif filename.startswith("region"):
        return "region"
    else:
        # Try to extract from the first part of the filename before any dots or underscores
        base_name = filename.split('.')[0].split('_')[0]
        return base_name

def convert_tbl_to_parquet(input_file):
    output_file = input_file.replace('.tbl', '.parquet')
    print(f"Converting {input_file} to {output_file}")
    
    # Extract table name and get column headers
    table_name = extract_table_name(input_file)
    column_names = get_schema_for_table(table_name)
    
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
    
    if column_names and len(column_names) > 0:
        if len(table.column_names) == len(column_names) + 1: 
            renamed_columns = {table.column_names[i]: column_names[i] for i in range(len(column_names))}
            table = table.rename_columns([renamed_columns.get(col, col) for col in table.column_names])
            table = table.drop([table.column_names[-1]])
        else:
            print(f"Warning: Column count mismatch for {table_name}. Expected {len(column_names)+1}, got {len(table.column_names)}. Proceeding without renaming.")
    
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