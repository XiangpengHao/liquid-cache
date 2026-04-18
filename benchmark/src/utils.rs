use std::fs::File;
use std::path::Path;
use std::sync::Arc;

use datafusion::arrow;
use datafusion::arrow::array::Array;
use datafusion::arrow::compute::kernels::numeric::{div, sub_wrapping};
use datafusion::arrow::datatypes::Float64Type;
use datafusion::arrow::{
    array::{AsArray, RecordBatch},
    datatypes::DataType,
};
use datafusion::common::tree_node::TreeNode;
use datafusion::physical_plan::ExecutionPlan;
use liquid_cache_datafusion_client::LiquidCacheClientExec;
use log::warn;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use uuid::Uuid;

pub(crate) fn get_plan_uuids(plan: &Arc<dyn ExecutionPlan>) -> Vec<Uuid> {
    let mut uuids = Vec::new();
    plan.apply(|plan| {
        if let Some(plan) = plan.as_any().downcast_ref::<LiquidCacheClientExec>() {
            uuids.push(plan.get_uuid());
        }
        Ok(datafusion::common::tree_node::TreeNodeRecursion::Continue)
    })
    .unwrap();
    uuids
}

fn float_eq_helper(left: &dyn Array, right: &dyn Array, tol: f64) -> bool {
    let diff = sub_wrapping(&left, &right).unwrap();
    let diff = arrow::compute::kernels::cast(&diff, &DataType::Float64).unwrap();
    let diff = diff.as_primitive_opt::<Float64Type>().unwrap();

    // Check if all differences are within tolerance
    if diff.iter().flatten().all(|v| v.abs() <= tol) {
        return true;
    }

    let scale = div(&diff, &left).unwrap();
    let scale = arrow::compute::kernels::cast(&scale, &DataType::Float64).unwrap();
    let scale = scale.as_primitive_opt::<Float64Type>().unwrap();
    for d in scale.iter().flatten() {
        if d.abs() > tol {
            warn!("scale: {scale:?}");
            return false;
        }
    }
    true
}

pub fn assert_batch_eq(expected: &RecordBatch, actual: &RecordBatch) {
    use datafusion::arrow::compute::*;

    if expected.num_rows() != actual.num_rows() {
        panic!(
            "Left (answer) had {} rows, but right (result) had {} rows",
            expected.num_rows(),
            actual.num_rows()
        );
    }
    if expected.columns().len() != actual.columns().len() {
        panic!(
            "Left (answer) had {} cols, but right (result) had {} cols",
            expected.columns().len(),
            actual.columns().len()
        );
    }
    for (i, (c_expected, c_actual)) in expected
        .columns()
        .iter()
        .zip(actual.columns().iter())
        .enumerate()
    {
        let casted_expected = cast(c_expected, c_actual.data_type()).unwrap();
        let sorted_expected = sort(&casted_expected, None).unwrap();
        let sorted_actual = sort(c_actual, None).unwrap();

        let data_type = c_expected.data_type();
        let tol: f64 = 1e-4;
        let ok = match data_type {
            DataType::Float16 => {
                unreachable!()
            }
            DataType::Float32 | DataType::Float64 => {
                float_eq_helper(&sorted_expected, &sorted_actual, tol)
            }
            _ => {
                let eq =
                    arrow::compute::kernels::cmp::eq(&sorted_expected, &sorted_actual).unwrap();
                eq.false_count() == 0
            }
        };
        assert!(
            ok,
            "Column {} answer does not match result\nExpected: {:?}\n Actual: {:?}",
            expected.schema().field(i).name(),
            sorted_expected,
            sorted_actual
        );
    }
}

/// Check query results against expected answers stored in parquet files
pub fn check_tpch_result(results: &[RecordBatch], answer_dir: &Path, query_id: u32) {
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

/// TPC-DS answer checker: same shape as TPCH but tolerant to empty results.
pub fn check_tpcds_result(results: &[RecordBatch], answer_dir: &Path, query_id: u32) {
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

    let result_rows: usize = results.iter().map(|b| b.num_rows()).sum();
    let answer_rows: usize = baseline_batches.iter().map(|b| b.num_rows()).sum();
    if answer_rows == 0 && result_rows == 0 {
        return;
    }
    assert!(
        answer_rows == result_rows,
        "Row count mismatch for q{query_id}: expected {answer_rows}, got {result_rows}"
    );

    let result_batch =
        datafusion::arrow::compute::concat_batches(&results[0].schema(), results).unwrap();
    let answer_batch = datafusion::arrow::compute::concat_batches(
        &baseline_batches[0].schema(),
        &baseline_batches,
    )
    .unwrap();
    assert_batch_eq(&answer_batch, &result_batch);
}

#[cfg(test)]
mod tests {

    use super::*;
    use datafusion::arrow::array::Float64Array;

    #[test]
    fn test_float_eq() {
        let left = Float64Array::from(vec![
            1.948_194_877_894_923e18,
            1.948_194_877_894_111e18,
            1.948_194_877_894_923e18,
            0.00,
        ]);
        let right = Float64Array::from(vec![
            1.948_194_877_894_922e18,
            1.948_194_877_894_222e18,
            1.948_194_877_894_922e18,
            0.00,
        ]);
        assert!(float_eq_helper(&left, &right, 1e-9));

        let left = Float64Array::from(vec![0.00]);
        let right = Float64Array::from(vec![0.00]);
        assert!(float_eq_helper(&left, &right, 1e-9));
    }

    #[should_panic]
    #[test]
    fn test_float_eq_helper() {
        let left = Float64Array::from(vec![0.00]);
        let right = Float64Array::from(vec![1.00]);
        assert!(float_eq_helper(&left, &right, 1e-9));
    }
}
