#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

BENCHMARK="${1:-all}"

setup_job() {
    echo "Setting up JOB benchmark..."
    cd job
    
    # Download data
    if [ ! -d "raw_gz" ] || [ -z "$(ls -A raw_gz 2>/dev/null)" ]; then
        bash download_job_dataset.sh
    fi
    
    # Convert to Parquet
    if [ ! -d "parquet_data" ] || [ -z "$(ls -A parquet_data 2>/dev/null)" ]; then
        python3 convert_to_parquet.py
    fi
    
    # Create manifest
    cargo run --bin create_manifest -- --benchmark job
    
    cd ..
    echo "JOB benchmark setup complete"
}

setup_publicbi() {
    echo "Setting up PublicBI benchmark..."
    cd publicBI
    
    # Download and extract data
    if [ ! -d "csv" ] || [ -z "$(ls -A csv 2>/dev/null)" ]; then
        bash download_and_extract_publicBI.sh
    fi
    
    # Convert to Parquet
    if [ ! -d "parquet_data" ] || [ -z "$(ls -A parquet_data 2>/dev/null)" ]; then
        python3 convert_to_parquet.py
    fi
    
    cd ..
    
    # Copy parquet files to publicbi/data
    if [ -d "publicBI/parquet_data" ] && [ -n "$(ls -A publicBI/parquet_data 2>/dev/null)" ]; then
        mkdir -p publicbi/data
        cp publicBI/parquet_data/*.parquet publicbi/data/ 2>/dev/null || true
    fi
    
    # Copy queries from public_bi_benchmark if available
    if [ -d "../public_bi_benchmark/benchmark" ]; then
        mkdir -p publicbi/queries
        find ../public_bi_benchmark/benchmark -name "*.sql" -path "*/queries/*" | while read -r query_file; do
            workbook=$(echo "$query_file" | sed -n 's|.*benchmark/\([^/]*\)/queries/\(.*\)|\1_\2|p')
            if [ -n "$workbook" ]; then
                dest_file="publicbi/queries/public_bi_benchmark_benchmark_${workbook}"
                cp "$query_file" "$dest_file" 2>/dev/null || true
            fi
        done
    fi
    
    # Create manifest
    cargo run --bin create_manifest -- --benchmark publicbi
    
    echo "PublicBI benchmark setup complete"
}

case "$BENCHMARK" in
    job)
        setup_job
        ;;
    publicbi)
        setup_publicbi
        ;;
    all)
        setup_job
        setup_publicbi
        ;;
    *)
        echo "Usage: $0 [job|publicbi|all]"
        exit 1
        ;;
esac
