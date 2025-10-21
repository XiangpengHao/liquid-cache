import os
import gzip
import pandas as pd


INPUT_FOLDER = "raw_gz/"  
if not os.path.exists(INPUT_FOLDER):
    print(f"Input folder {INPUT_FOLDER} does not exist. Please run download_job_dataset.sh first.")
    exit(1)
OUTPUT_FOLDER = "parquet_data/"
if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    print(f"Created output folder {OUTPUT_FOLDER}")

tsv_files = [f for f in os.listdir(INPUT_FOLDER) if f.endswith(".tsv.gz")]

# Convert each .tsv.gz to Parquet
for file_name in tsv_files:
    gz_path = os.path.join(INPUT_FOLDER, file_name)
    parquet_path = os.path.join(OUTPUT_FOLDER, file_name.replace(".tsv.gz", ".parquet"))
    
    print(f"Converting {file_name} to Parquet...")
    with gzip.open(gz_path, 'rt', encoding='utf-8') as f:
        df = pd.read_csv(f, sep='\t', low_memory=False)
        df.to_parquet(parquet_path, engine='pyarrow', index=False)
    
    print(f"Saved {parquet_path}\n")

print("All files converted to Parquet successfully!")
