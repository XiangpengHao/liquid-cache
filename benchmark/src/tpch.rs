use crate::utils::assert_batch_eq;
use datafusion::{
    arrow::array::RecordBatch, parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder,
};
use std::{fs::File, path::Path};

/// Check query results against expected answers stored in parquet files
pub fn check_result_against_answer(results: &[RecordBatch], answer_dir: &Path, query_id: u32) {
    let baseline_path = format!("{}/q{}.parquet", answer_dir.display(), query_id);
    let baseline_file = File::open(baseline_path).unwrap();
    let mut baseline_batches = Vec::new();
    let reader = ParquetRecordBatchReaderBuilder::try_new(baseline_file)
        .unwrap()
        .build()
        .unwrap();
    for batch in reader {
        baseline_batches.push(batch.unwrap());
    }

    // Compare answers and result
    let result_batch =
        datafusion::arrow::compute::concat_batches(&results[0].schema(), results).unwrap();
    let answer_batch = datafusion::arrow::compute::concat_batches(
        &baseline_batches[0].schema(),
        &baseline_batches,
    )
    .unwrap();
    assert_batch_eq(&answer_batch, &result_batch);
}
