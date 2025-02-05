use datafusion::arrow::{
    array::{AsArray, RecordBatch},
    datatypes::{DataType, Float32Type, Float64Type},
};

macro_rules! float_eq {
    ($left:expr, $right:expr, $type:ty, $abs_f:expr) => {{
        use datafusion::arrow::compute::kernels::numeric::sub_wrapping;
        let diff = sub_wrapping(&$left, &$right).unwrap();
        let diff = diff.as_primitive_opt::<$type>().unwrap();
        for d in diff.iter() {
            if let Some(d) = d {
                if $abs_f(d) {
                    return false;
                }
            }
        }
    }};
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
                float_eq!(sorted_c_l, sorted_c_r, Float32Type, abs_f)
            }
            DataType::Float64 => {
                let abs_f = |d: f64| d.abs() > tol.into();
                float_eq!(sorted_c_l, sorted_c_r, Float64Type, abs_f)
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
