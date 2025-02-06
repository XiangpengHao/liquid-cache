use datafusion::arrow::array::{Array, ArrowPrimitiveType};
use datafusion::arrow::compute::kernels::numeric::{div, sub_wrapping};
use datafusion::arrow::datatypes::{Float32Type, Float64Type};
use datafusion::arrow::{
    array::{AsArray, RecordBatch},
    datatypes::DataType,
};
use std::sync::Arc;

fn float_eq_helper<T: ArrowPrimitiveType, F: Fn(T::Native) -> bool>(
    left: Arc<dyn Array>,
    right: Arc<dyn Array>,
    abs_f: F,
) -> bool {
    let diff = sub_wrapping(&left, &right).unwrap();
    let scale = div(&diff, &left).unwrap();
    let scale = scale.as_primitive_opt::<T>().unwrap();
    for d in scale.iter().flatten() {
        if abs_f(d) {
            return false;
        }
    }
    true
}

pub fn assert_batch_eq(left: &RecordBatch, right: &RecordBatch) -> bool {
    use datafusion::arrow::compute::*;

    if left.num_rows() != right.num_rows() {
        return false;
    }
    if left.columns().len() != right.columns().len() {
        return false;
    }
    for (c_l, c_r) in left.columns().iter().zip(right.columns().iter()) {
        let casted = cast(c_l, c_r.data_type()).unwrap();
        let sorted_c_l = sort(&casted, None).unwrap();
        let sorted_c_r = sort(c_r, None).unwrap();

        let data_type = c_l.data_type();
        let tol: f32 = 1e-9;
        match data_type {
            DataType::Float16 => {
                unreachable!()
            }
            DataType::Float32 => {
                let abs_f = |d: f32| d.abs() > tol;
                if !float_eq_helper::<Float32Type, _>(sorted_c_l, sorted_c_r, abs_f) {
                    return false;
                }
            }
            DataType::Float64 => {
                let abs_f = |d: f64| d.abs() > tol.into();
                if !float_eq_helper::<Float64Type, _>(sorted_c_l, sorted_c_r, abs_f) {
                    return false;
                }
            }
            _ => {
                if sorted_c_l != sorted_c_r {
                    return false;
                }
            }
        }
    }
    true
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
            Arc::new(left),
            Arc::new(right),
            |d: f64| d.abs() > 1e-9
        ));
    }
}
