use datafusion::arrow;
use datafusion::arrow::array::{Array, ArrowPrimitiveType};
use datafusion::arrow::compute::kernels::numeric::{div, sub_wrapping};
use datafusion::arrow::datatypes::{Float32Type, Float64Type};
use datafusion::arrow::{
    array::{AsArray, RecordBatch},
    datatypes::DataType,
};
use log::warn;
use owo_colors::OwoColorize;

fn float_eq_helper<T: ArrowPrimitiveType, F: Fn(T::Native) -> bool>(
    left: &dyn Array,
    right: &dyn Array,
    abs_f: F,
) -> bool {
    let diff = sub_wrapping(&left, &right).unwrap();
    let scale = div(&diff, &left).unwrap();
    let scale = arrow::compute::kernels::cast(&scale, &DataType::Float64).unwrap();
    let scale = scale.as_primitive_opt::<T>().unwrap();
    for d in scale.iter().flatten() {
        if abs_f(d) {
            warn!("scale: {:?}", scale);
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
        let tol: f32 = 1e-4;
        let ok = match data_type {
            DataType::Float16 => {
                unreachable!()
            }
            DataType::Float32 => {
                let abs_f = |d: f32| d.abs() > tol;
                float_eq_helper::<Float32Type, _>(&sorted_expected, &sorted_actual, abs_f)
            }
            DataType::Float64 => {
                let abs_f = |d: f64| d.abs() > tol.into();
                float_eq_helper::<Float64Type, _>(&sorted_expected, &sorted_actual, abs_f)
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
            expected.schema().field(i).name().red(),
            sorted_expected.red(),
            sorted_actual.red()
        );
    }
}

#[cfg(test)]
mod tests {

    use super::*;
    use datafusion::arrow::array::Float64Array;

    #[test]
    fn test_float_eq() {
        let left = Float64Array::from(vec![
            1.9481948778949233e18,
            1.9481948778941111e18,
            1.9481948778949233e18,
        ]);
        let right = Float64Array::from(vec![
            1.948194877894922e18,
            1.9481948778942222e18,
            1.948194877894922e18,
        ]);
        assert!(float_eq_helper::<Float64Type, _>(
            &left,
            &right,
            |d: f64| d.abs() > 1e-9
        ));
    }
}
