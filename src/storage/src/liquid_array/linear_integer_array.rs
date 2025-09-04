use std::any::Any;
use std::fmt::Debug;
use std::sync::Arc;

use super::{LiquidArray, LiquidArrayRef, LiquidDataType};
use arrow::array::{
    Array, ArrayRef, BooleanArray, PrimitiveArray, cast::AsArray, types::Int32Type,
};
use arrow::buffer::{BooleanBuffer, ScalarBuffer};
use arrow::compute::kernels::filter;

/// A linear-regression based integer array (first iteration for i32).
///
/// Model: value[i] ≈ intercept + round(slope * i) + residual[i]
/// We store residuals directly (no bit-packing or IPC yet).
#[derive(Debug, Clone)]
pub struct LiquidLinearI32Array {
    // Residuals as a primitive Int32 array (with same nulls as the input array).
    residuals: PrimitiveArray<Int32Type>,
    // Intercept term of the linear model.
    intercept: i32,
    // Slope term of the linear model.
    slope: f64,
}

impl LiquidLinearI32Array {
    /// Build from an Arrow `PrimitiveArray<Int32Type>` by training a linear regression model
    /// (least squares over non-null points) and storing residuals (no bit-packing yet).
    pub fn from_arrow_array(arrow_array: PrimitiveArray<Int32Type>) -> Self {
        let len = arrow_array.len();

        // Compute slope and intercept using linear regression over non-null points.
        // Accumulate sums for least-squares fit: m = (n*sum(xy) - sum(x)sum(y)) / (n*sum(x^2) - (sum(x))^2)
        // b = (sum(y) - m*sum(x)) / n
        let mut n: i64 = 0;
        let mut sum_x: f64 = 0.0;
        let mut sum_y: f64 = 0.0;
        let mut sum_x2: f64 = 0.0;
        let mut sum_xy: f64 = 0.0;
        for (i, oy) in arrow_array.iter().enumerate() {
            if let Some(y) = oy {
                let x = i as f64;
                n += 1;
                sum_x += x;
                sum_y += y as f64;
                sum_x2 += x * x;
                sum_xy += x * (y as f64);
            }
        }

        if n == 0 {
            // All nulls
            return Self {
                residuals: PrimitiveArray::<Int32Type>::new_null(len),
                intercept: 0,
                slope: 0.0,
            };
        }

        let n_f = n as f64;
        let denom = n_f * sum_x2 - (sum_x * sum_x);
        let slope = if denom.abs() < f64::EPSILON {
            0.0
        } else {
            (n_f * sum_xy - sum_x * sum_y) / denom
        };
        // Store intercept as i32 (rounded) for reconstruction math.
        let intercept_f = (sum_y - slope * sum_x) / n_f;
        let intercept = intercept_f.round().clamp(i32::MIN as f64, i32::MAX as f64) as i32;

        // Compute residuals in one pass (keep nulls, fill zero for null slots).
        let mut residuals: Vec<i32> = Vec::with_capacity(len);
        for (i, ov) in arrow_array.iter().enumerate() {
            if let Some(v) = ov {
                let predicted = predict_i32(intercept, slope, i as u32);
                residuals.push(v.wrapping_sub(predicted));
            } else {
                residuals.push(0);
            }
        }
        let residuals_buf: ScalarBuffer<i32> = ScalarBuffer::from(residuals);
        let nulls = arrow_array.nulls().cloned();
        let residuals = PrimitiveArray::<Int32Type>::new(residuals_buf, nulls);

        Self {
            residuals,
            intercept,
            slope,
        }
    }

    fn len(&self) -> usize {
        self.residuals.len()
    }
}

impl LiquidArray for LiquidLinearI32Array {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn get_array_memory_size(&self) -> usize {
        self.residuals.get_array_memory_size()
            + std::mem::size_of::<i32>() // intercept
            + std::mem::size_of::<f64>() // slope
    }

    fn len(&self) -> usize {
        self.len()
    }

    fn to_arrow_array(&self) -> ArrayRef {
        let (_dt, residuals, nulls) = self.residuals.clone().into_parts();

        // Reconstruct final values: predicted(i) + residual_i
        let mut final_values = Vec::<i32>::with_capacity(self.len());
        for (i, &e) in residuals.iter().enumerate() {
            let predicted = predict_i32(self.intercept, self.slope, i as u32);
            final_values.push(predicted.wrapping_add(e));
        }

        let values_buf: ScalarBuffer<i32> = ScalarBuffer::from(final_values);
        Arc::new(PrimitiveArray::<Int32Type>::new(values_buf, nulls))
    }

    fn filter(&self, selection: &BooleanBuffer) -> LiquidArrayRef {
        // Simpler and correct: materialize to Arrow, filter, and retrain a new linear array.
        let arr = self.to_arrow_array();
        let selection = BooleanArray::new(selection.clone(), None);
        let filtered = filter::filter(&arr, &selection).unwrap();
        let filtered = filtered.as_primitive::<Int32Type>().clone();
        Arc::new(Self::from_arrow_array(filtered))
    }

    fn filter_to_arrow(&self, selection: &BooleanBuffer) -> ArrayRef {
        let arr = self.to_arrow_array();
        let selection = BooleanArray::new(selection.clone(), None);
        filter::filter(&arr, &selection).unwrap()
    }

    fn try_eval_predicate(
        &self,
        _predicate: &Arc<dyn datafusion::physical_plan::PhysicalExpr>,
        _filter: &BooleanBuffer,
    ) -> Option<BooleanArray> {
        // No special predicate pushdown here.
        None
    }

    fn to_bytes(&self) -> Vec<u8> {
        unimplemented!("LinearInteger: IPC not implemented yet")
    }

    fn data_type(&self) -> LiquidDataType {
        LiquidDataType::Integer
    }
}

#[inline]
fn predict_i32(intercept: i32, slope: f64, index: u32) -> i32 {
    // Round to nearest integer, clamped to i32 range for safety.
    let pred = (slope * index as f64) + (intercept as f64);
    let r = pred.round();
    if r < i32::MIN as f64 {
        i32::MIN
    } else if r > i32::MAX as f64 {
        i32::MAX
    } else {
        r as i32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn roundtrip_eq(values: Vec<Option<i32>>) {
        let arr = PrimitiveArray::<Int32Type>::from(values.clone());
        let linear = LiquidLinearI32Array::from_arrow_array(arr.clone());
        let decoded = linear.to_arrow_array();
        assert_eq!(decoded.as_ref(), &arr);
    }

    #[test]
    fn test_roundtrip_basic() {
        // Non-monotonic values to ensure we don't rely on simple increasing sequences
        roundtrip_eq(vec![
            Some(10),
            Some(15),
            Some(14),
            Some(20),
            Some(18),
            Some(25),
            Some(24),
        ]);
    }

    #[test]
    fn test_roundtrip_with_nulls() {
        roundtrip_eq(vec![Some(10), None, Some(30), None, Some(50), Some(70)]);
    }

    #[test]
    fn test_all_nulls() {
        roundtrip_eq(vec![None, None, None, None]);
    }

    #[test]
    fn test_single_value() {
        roundtrip_eq(vec![Some(42)]);
    }

    #[test]
    fn test_empty() {
        roundtrip_eq(vec![]);
    }

    #[test]
    fn test_negative_values() {
        roundtrip_eq(vec![
            Some(-100),
            Some(-50),
            Some(0),
            Some(50),
            Some(25),
            None,
            Some(-25),
        ]);
    }

    #[test]
    fn test_roundtrip_bytes() {}

    #[test]
    fn test_filter_basic() {
        let original: Vec<Option<i32>> = vec![Some(1), Some(2), Some(3), None, Some(5), Some(8)];
        let arr = PrimitiveArray::<Int32Type>::from(original.clone());
        let linear = LiquidLinearI32Array::from_arrow_array(arr);
        let selection = BooleanBuffer::from(vec![true, false, true, false, true, false]);
        let filtered = linear.filter(&selection);
        let result = filtered.to_arrow_array();
        let expected = PrimitiveArray::<Int32Type>::from(vec![Some(1), Some(3), Some(5)]);
        assert_eq!(result.as_ref(), &expected);
    }
}
