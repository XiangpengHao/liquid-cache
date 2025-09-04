use std::any::Any;
use std::fmt::Debug;
use std::sync::Arc;

use super::{LiquidArray, LiquidArrayRef, LiquidDataType, LiquidPrimitiveType};
use crate::liquid_array::LiquidPrimitiveArray;
use crate::liquid_array::ipc::{LiquidIPCHeader, get_physical_type_id};
use arrow::array::{
    Array, ArrayRef, ArrowPrimitiveType, BooleanArray, PrimitiveArray, ArrowNativeTypeOp,
    cast::AsArray, types::{Int8Type, Int16Type, Int32Type, Int64Type, UInt8Type, UInt16Type, UInt32Type, UInt64Type, Date32Type, Date64Type},
};
use arrow::buffer::{BooleanBuffer, ScalarBuffer};
use arrow::compute::kernels::filter;
use bytes::Bytes;
use num_traits::{AsPrimitive, FromPrimitive, Bounded};

/// A linear-regression based integer array, generic over Arrow integer-like types.
///
/// Model: value[i] ≈ intercept + round(slope * i) + residual[i]
/// Residuals are stored using LiquidPrimitiveArray encoding of the same primitive type.
#[derive(Debug, Clone)]
pub struct LiquidLinearArray<T: LiquidPrimitiveType>
where
    T::Native: AsPrimitive<f64> + FromPrimitive + Bounded,
{
    // Residuals as a primitive array (with same nulls as the input array).
    residuals: PrimitiveArray<T>,
    // Intercept term of the linear model (rounded to native type domain).
    intercept: T::Native,
    // Slope term of the linear model.
    slope: f64,
}

/// Backward-compatible alias for i32.
pub type LiquidLinearI32Array = LiquidLinearArray<Int32Type>;
/// Linear-regression array for `i8`.
pub type LiquidLinearI8Array = LiquidLinearArray<Int8Type>;
/// Linear-regression array for `i16`.
pub type LiquidLinearI16Array = LiquidLinearArray<Int16Type>;
/// Linear-regression array for `i64`.
pub type LiquidLinearI64Array = LiquidLinearArray<Int64Type>;
/// Linear-regression array for `u8`.
pub type LiquidLinearU8Array = LiquidLinearArray<UInt8Type>;
/// Linear-regression array for `u16`.
pub type LiquidLinearU16Array = LiquidLinearArray<UInt16Type>;
/// Linear-regression array for `u32`.
pub type LiquidLinearU32Array = LiquidLinearArray<UInt32Type>;
/// Linear-regression array for `u64`.
pub type LiquidLinearU64Array = LiquidLinearArray<UInt64Type>;
/// Linear-regression array for `Date32` (days since epoch).
pub type LiquidLinearDate32Array = LiquidLinearArray<Date32Type>;
/// Linear-regression array for `Date64` (ms since epoch).
pub type LiquidLinearDate64Array = LiquidLinearArray<Date64Type>;

impl<T> LiquidLinearArray<T>
where
    T: LiquidPrimitiveType,
    T::Native: AsPrimitive<f64> + FromPrimitive + Bounded,
{
    /// Build from an Arrow `PrimitiveArray<T>` by training a linear regression model
    /// (least squares over non-null points) and storing residuals.
    pub fn from_arrow_array(arrow_array: PrimitiveArray<T>) -> Self {
        let len = arrow_array.len();

        // Compute slope and intercept using linear regression over non-null points.
        let mut n: i64 = 0;
        let mut sum_x: f64 = 0.0;
        let mut sum_y: f64 = 0.0;
        let mut sum_x2: f64 = 0.0;
        let mut sum_xy: f64 = 0.0;
        for (i, oy) in arrow_array.iter().enumerate() {
            if let Some(y) = oy {
                let x = i as f64;
                let yf: f64 = y.as_();
                n += 1;
                sum_x += x;
                sum_y += yf;
                sum_x2 += x * x;
                sum_xy += x * yf;
            }
        }

        if n == 0 {
            // All nulls
            return Self {
                residuals: PrimitiveArray::<T>::new_null(len),
                intercept: T::Native::min_value(), // arbitrary, unused since all nulls
                slope: 0.0,
            };
        }

        let n_f = n as f64;
        let denom = n_f * sum_x2 - (sum_x * sum_x);
        let slope = if denom.abs() < f64::EPSILON { 0.0 } else { (n_f * sum_xy - sum_x * sum_y) / denom };
        // Compute and clamp intercept to native range, rounding to nearest.
        let intercept_f = (sum_y - slope * sum_x) / n_f;
        let min_f: f64 = T::Native::min_value().as_();
        let max_f: f64 = T::Native::max_value().as_();
        let clamped = intercept_f.round().clamp(min_f, max_f);
        let intercept = T::Native::from_f64(clamped).unwrap();

        // Compute residuals in one pass (keep nulls, fill zero for null slots).
        let mut residuals: Vec<T::Native> = Vec::with_capacity(len);
        for (i, ov) in arrow_array.iter().enumerate() {
            if let Some(v) = ov {
                let predicted = predict_value::<T>(intercept, slope, i as u32);
                residuals.push(v.sub_wrapping(predicted));
            } else {
                residuals.push(T::Native::ZERO);
            }
        }
        let residuals_buf: ScalarBuffer<T::Native> = ScalarBuffer::from(residuals);
        let nulls = arrow_array.nulls().cloned();
        let residuals = PrimitiveArray::<T>::new(residuals_buf, nulls);

        Self { residuals, intercept, slope }
    }

    fn len(&self) -> usize {
        self.residuals.len()
    }

    fn residual_starting_loc() -> usize {
        // Header + intercept(native) + slope(f64), aligned to 8 bytes boundary
        let header_size = LiquidIPCHeader::size()
            + std::mem::size_of::<T::Native>()
            + std::mem::size_of::<f64>();
        (header_size + 7) & !7
    }

    fn to_bytes_inner(&self) -> Vec<u8> {
        let header = LiquidIPCHeader::new(
            LiquidDataType::LinearInteger as u16,
            get_physical_type_id::<T>(),
        );
        let start = Self::residual_starting_loc();
        let mut out = Vec::with_capacity(start + 256);

        // Header
        out.extend_from_slice(&header.to_bytes());
        // Model params
        let intercept_bytes = unsafe {
            std::slice::from_raw_parts(
                &self.intercept as *const T::Native as *const u8,
                std::mem::size_of::<T::Native>(),
            )
        };
        out.extend_from_slice(intercept_bytes);
        out.extend_from_slice(&self.slope.to_le_bytes());

        while out.len() < start {
            out.push(0);
        }

        // Encode residuals using primitive array encoding
        let lp = LiquidPrimitiveArray::<T>::from_arrow_array(self.residuals.clone());
        out.extend_from_slice(&lp.to_bytes_inner());
        out
    }

    /// Decode a `LiquidLinearArray<T>` from bytes.
    pub fn from_bytes(bytes: Bytes) -> Self {
        let _hdr = LiquidIPCHeader::from_bytes(&bytes);

        // Read intercept of native size
        let intercept_off = LiquidIPCHeader::size();
        let intercept = unsafe {
            (bytes[intercept_off..intercept_off + std::mem::size_of::<T::Native>()].as_ptr()
                as *const T::Native)
                .read_unaligned()
        };

        // Read slope
        let slope_off = intercept_off + std::mem::size_of::<T::Native>();
        let slope = f64::from_le_bytes(bytes[slope_off..slope_off + 8].try_into().unwrap());

        // Decode residuals
        let start = Self::residual_starting_loc();
        let res_bytes = bytes.slice(start..);
        let lp = LiquidPrimitiveArray::<T>::from_bytes(res_bytes);
        let arr = lp.to_arrow_array().as_primitive::<T>().clone();

        Self { residuals: arr, intercept, slope }
    }
}

impl<T> LiquidArray for LiquidLinearArray<T>
where
    T: LiquidPrimitiveType,
    T::Native: AsPrimitive<f64> + FromPrimitive + Bounded,
{
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn get_array_memory_size(&self) -> usize {
        self.residuals.get_array_memory_size()
            + std::mem::size_of::<T::Native>() // intercept
            + std::mem::size_of::<f64>() // slope
    }

    fn len(&self) -> usize {
        self.len()
    }

    fn to_arrow_array(&self) -> ArrayRef {
        let (_dt, residuals, nulls) = self.residuals.clone().into_parts();

        // Reconstruct final values: predicted(i) + residual_i
        let mut final_values = Vec::<T::Native>::with_capacity(self.len());
        for (i, &e) in residuals.iter().enumerate() {
            let predicted = predict_value::<T>(self.intercept, self.slope, i as u32);
            final_values.push(predicted.add_wrapping(e));
        }

        let values_buf: ScalarBuffer<T::Native> = ScalarBuffer::from(final_values);
        Arc::new(PrimitiveArray::<T>::new(values_buf, nulls))
    }

    fn filter(&self, selection: &BooleanBuffer) -> LiquidArrayRef {
        // Materialize to Arrow, filter, and retrain a new linear array.
        let arr = self.to_arrow_array();
        let selection = BooleanArray::new(selection.clone(), None);
        let filtered = filter::filter(&arr, &selection).unwrap();
        let filtered = filtered.as_primitive::<T>().clone();
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
        self.to_bytes_inner()
    }

    fn data_type(&self) -> LiquidDataType {
        LiquidDataType::LinearInteger
    }
}

#[inline]
fn predict_value<T>(intercept: <T as ArrowPrimitiveType>::Native, slope: f64, index: u32) -> <T as ArrowPrimitiveType>::Native
where
    T: ArrowPrimitiveType,
    T::Native: AsPrimitive<f64> + FromPrimitive + Bounded,
{
    let base: f64 = intercept.as_();
    let pred = slope * index as f64 + base;
    let min_f: f64 = T::Native::min_value().as_();
    let max_f: f64 = T::Native::max_value().as_();
    let r = pred.round().clamp(min_f, max_f);
    T::Native::from_f64(r).unwrap()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn roundtrip_eq(values: Vec<Option<i32>>) {
        let arr = PrimitiveArray::<Int32Type>::from(values.clone());
        let linear = LiquidLinearI32Array::from_arrow_array(arr.clone());
        let decoded = linear.to_arrow_array();
        assert_eq!(decoded.as_ref(), &arr);

        let bytes = Bytes::from(linear.to_bytes());
        let decoded = LiquidLinearI32Array::from_bytes(bytes);
        let round = decoded.to_arrow_array();
        assert_eq!(round.as_ref(), &arr);
    }

    macro_rules! roundtrip_eq_t {
        ($T:ty, $values:expr) => {{
            let arr = PrimitiveArray::<$T>::from(($values).clone());
            let linear = LiquidLinearArray::<$T>::from_arrow_array(arr.clone());
            let decoded = linear.to_arrow_array();
            assert_eq!(decoded.as_ref(), &arr);

            let bytes = Bytes::from(linear.to_bytes());
            let decoded = LiquidLinearArray::<$T>::from_bytes(bytes);
            let round = decoded.to_arrow_array();
            assert_eq!(round.as_ref(), &arr);
        }};
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

    #[test]
    fn test_roundtrip_i8() {
        roundtrip_eq_t!(Int8Type, vec![Some(-10), Some(0), Some(10), None, Some(20)]);
    }

    #[test]
    fn test_roundtrip_i16() {
        roundtrip_eq_t!(Int16Type, vec![Some(-1000), Some(0), Some(1000), None, Some(2000)]);
    }

    #[test]
    fn test_roundtrip_i64() {
        roundtrip_eq_t!(Int64Type, vec![
            Some(-10_000_000_000),
            Some(0),
            Some(10_000_000_000),
            None,
            Some(20_000_000_000),
        ]);
    }

    #[test]
    fn test_roundtrip_u8() {
        roundtrip_eq_t!(UInt8Type, vec![Some(0), Some(10), Some(200), None, Some(255)]);
    }

    #[test]
    fn test_roundtrip_u16() {
        roundtrip_eq_t!(UInt16Type, vec![Some(0), Some(1000), Some(60000), None, Some(500)]);
    }

    #[test]
    fn test_roundtrip_u32() {
        roundtrip_eq_t!(UInt32Type, vec![
            Some(0),
            Some(1_000_000),
            Some(3_000_000_000),
            None,
            Some(123_456_789),
        ]);
    }

    #[test]
    fn test_roundtrip_u64() {
        roundtrip_eq_t!(UInt64Type, vec![
            Some(0),
            Some(10_000_000_000),
            Some(18_000_000_000_000_000_000u64),
            None,
            Some(42),
        ]);
    }

    #[test]
    fn test_roundtrip_date32() {
        roundtrip_eq_t!(Date32Type, vec![Some(-365), Some(0), Some(365), None, Some(18262)]);
    }

    #[test]
    fn test_roundtrip_date64() {
        roundtrip_eq_t!(Date64Type, vec![
            Some(-86_400_000),
            Some(0),
            Some(86_400_000),
            None,
            Some(1_000_000_000_000),
        ]);
    }
}
