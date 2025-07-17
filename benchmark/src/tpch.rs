use crate::{Query, utils::assert_batch_eq};
use datafusion::{
    arrow::array::RecordBatch, error::Result,
    parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder,
};
use std::{fs::File, path::Path};

/// One query file can contain multiple queries, separated by `;`
pub fn get_query_by_id(query_dir: impl AsRef<Path>, query_id: u32) -> Result<Vec<String>> {
    let query_dir = query_dir.as_ref();
    let mut path = query_dir.to_owned();
    path.push(format!("q{query_id}.sql"));
    let content = std::fs::read_to_string(&path)?;
    Ok(content
        .split(';')
        .map(|s| s.trim())
        .filter(|s| !s.is_empty())
        .map(|s| s.to_string())
        .collect())
}

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

/// Get all TPCH queries (1-22) from the query directory
pub fn get_all_queries(query_dir: impl AsRef<Path>) -> Result<Vec<Query>> {
    let query_ids: Vec<u32> = (1..=22).collect();

    let mut queries = Vec::new();
    for id in query_ids {
        let query_strings = get_query_by_id(&query_dir, id)?;

        queries.push(Query::new(id, query_strings));
    }
    Ok(queries)
}
