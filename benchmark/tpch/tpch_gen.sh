mkdir -p data

# Creates TPCH data at a certain scale factor, if it doesn't already
# exist
#
# call like: data_tpch($scale_factor)
#
# Creates data in $DATA_DIR/tpch_sf1 for scale factor 1
# Creates data in $DATA_DIR/tpch_sf10 for scale factor 10
# etc
data_tpch() {
    SCALE_FACTOR=$1
    if [ -z "$SCALE_FACTOR" ] ; then
        echo "Internal error: Scale factor not specified"
        exit 1
    fi

    TPCH_DIR="${DATA_DIR}/tpch_sf${SCALE_FACTOR}"
    echo "Creating tpch dataset at Scale Factor ${SCALE_FACTOR} in ${TPCH_DIR}..."

    # Ensure the target data directory exists
    mkdir -p "${TPCH_DIR}"

    # Create 'tbl' (CSV format) data into $DATA_DIR if it does not already exist
    FILE="${TPCH_DIR}/supplier.tbl"
    if test -f "${FILE}"; then
        echo " tbl files exist ($FILE exists)."
    else
        echo " creating tbl files with tpch_dbgen..."
        docker run -v "${TPCH_DIR}":/data -it --rm ghcr.io/scalytics/tpch-docker:main -vf -s "${SCALE_FACTOR}"
    fi

    # Copy expected answers into the ./data/answers directory if it does not already exist
    FILE="${TPCH_DIR}/answers/q1.out"
    if test -f "${FILE}"; then
        echo " Expected answers exist (${FILE} exists)."
    else
        echo " Copying answers to ${TPCH_DIR}/answers"
        mkdir -p "${TPCH_DIR}/answers"
        docker run -v "${TPCH_DIR}":/data -it --entrypoint /bin/bash --rm ghcr.io/scalytics/tpch-docker:main  -c "cp -f /opt/tpch/2.18.0_rc2/dbgen/answers/* /data/answers/"
    fi
}

data_tpch 1
