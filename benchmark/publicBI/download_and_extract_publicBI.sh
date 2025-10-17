#!/bin/bash

BASE_URL="https://event.cwi.nl/da/PublicBIbenchmark/"
RAW_DIR="raw_bz2"
CSV_DIR="csv"


# Create folders
mkdir -p "$RAW_DIR"
mkdir -p "$CSV_DIR"


# Download all .bz2 files recursively
echo "Downloading .bz2 files from $BASE_URL ..."
wget -r -np -nd -A ".bz2" -P "$RAW_DIR" "$BASE_URL"

# Extract all .bz2 files to CSV - MAKE SURE BUNZIP2 IS INSTALLED
echo "Extracting .bz2 files to CSV ..."
for f in "$RAW_DIR"/*.bz2; do
    [ -e "$f" ] || continue   # skip if no .bz2 files
    filename=$(basename "$f")
    bunzip2 -c "$f" > "$CSV_DIR/${filename%.bz2}"
    echo "Extracted $filename -> ${filename%.bz2}"
done

echo "All downloads and extractions completed!"