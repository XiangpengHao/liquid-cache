# Creates TPCH data at a certain scale factor, if it doesn't already exist
#
# Usage: data_tpch <scale_factor>
#
# Creates data in tpch_sf<scale_factor> directory
data_tpch() {
    SCALE_FACTOR=$1
    if [ -z "$SCALE_FACTOR" ] ; then
        echo "Internal error: Scale factor not specified"
        exit 1
    fi

    TPCH_DIR="tpch_sf${SCALE_FACTOR}"
    echo "Creating tpch dataset at Scale Factor ${SCALE_FACTOR} in ${TPCH_DIR}..."

    # Generate data files if they don't exist
	if [ ! -f "${TPCH_DIR}/supplier.tbl" ]; then
        echo " Creating tbl files with tpch_dbgen..."
        mkdir -p "data/${TPCH_DIR}"
        docker run -v "$(pwd)/data/${TPCH_DIR}":/data -it --rm ghcr.io/scalytics/tpch-docker:main -vf -s "${SCALE_FACTOR}"
        # Fix permissions on created files
        sudo chown -R $(id -u):$(id -g) "data/${TPCH_DIR}"
    else
        echo " tbl files already exist."
    fi
 
    # Copy expected answers if they don't exist
    if [ ! -f "answers/${TPCH_DIR}/q1.out" ]; then
        echo " Copying answers to answers/${TPCH_DIR}"
        mkdir -p "answers/${TPCH_DIR}"
        docker run -v "$(pwd)/answers/${TPCH_DIR}":/answers -it --entrypoint /bin/bash --rm \
            ghcr.io/scalytics/tpch-docker:main \
            -c "cp -f /opt/tpch/2.18.0_rc2/dbgen/answers/* /answers/ && chown -R $(id -u):$(id -g) /answers/*"
        echo " Answers copied to answers/${TPCH_DIR}"
    else
        echo " Expected answers already exist."
    fi
}

# Convert TBL files to Parquet format
convert_to_parquet() {
    SCALE_FACTOR=$1
    if [ -z "$SCALE_FACTOR" ] ; then
        echo "Internal error: Scale factor not specified"
        exit 1
    fi

    TPCH_DIR="tpch_sf${SCALE_FACTOR}"
    FULL_DATA_PATH="$(pwd)/data/${TPCH_DIR}"
    
    echo "Converting TBL files to Parquet format in ${FULL_DATA_PATH}..."
    
    # Check if any .tbl files exist
    if ls "${FULL_DATA_PATH}"/*.tbl 1> /dev/null 2>&1; then
        echo " Running conversion script..."
        # Use uvx to run the Python script with pyarrow
        uvx --from pyarrow python "$(pwd)/csv_to_parquet.py" "${FULL_DATA_PATH}"
        
        if [ $? -eq 0 ]; then
            echo " Successfully converted TBL files to Parquet format"
        else
            echo " Error converting TBL files to Parquet format"
            exit 1
        fi
    else
        echo " No TBL files found to convert in ${FULL_DATA_PATH}"
    fi
}

# Run the data_tpch function with the provided scale factor argument
# If no argument is provided, default to 0.01
SCALE_FACTOR=${1:-0.01}
data_tpch $SCALE_FACTOR
convert_to_parquet $SCALE_FACTOR
