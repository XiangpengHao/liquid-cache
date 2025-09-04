use std::any::Any;
use std::fmt::Debug;
use std::num::NonZero;
use std::sync::Arc;

use arrow::array::{
    ArrayRef, BooleanArray, PrimitiveArray,
    cast::AsArray,
    types::{Int32Type, UInt32Type},
};
use arrow::buffer::{BooleanBuffer, ScalarBuffer};
use arrow::compute::kernels::{aggregate, filter};
use bytes::Bytes;

use super::{LiquidArray, LiquidArrayRef, LiquidDataType};
use crate::liquid_array::ipc::LiquidIPCHeader;
use crate::liquid_array::raw::BitPackedArray;
use crate::utils::get_bit_width;

/// A linear-regression based integer array (first iteration for i32).
///
/// Model: value[i] ≈ intercept + round(slope * i) + error[i]
/// We bit-pack the error terms as unsigned with an offset (`error_reference_value`).
#[derive(Debug, Clone)]
pub struct LiquidLinearI32Array {
    // Bit-packed unsigned representation of (error_i - error_min).
    errors: BitPackedArray<UInt32Type>,
    // The minimum error value used as the reference to reconstruct signed errors.
    error_reference_value: i32,
    // Intercept term of the linear model (chosen as min of the original values).
    intercept: i32,
    // Slope term of the linear model: (max - min) / (len - 1) as f64.
    slope: f64,
}

impl LiquidLinearI32Array {
    /// Build from an Arrow `PrimitiveArray<Int32Type>` by training a simple linear model
    /// and packing the residuals.
    pub fn from_arrow_array(arrow_array: PrimitiveArray<Int32Type>) -> Self {
        let len = arrow_array.len();
        if len == 0 {
            return Self {
                errors: BitPackedArray::new_null_array(0),
                error_reference_value: 0,
                intercept: 0,
                slope: 0.0,
            };
        }

        // All nulls case
        let maybe_min = aggregate::min(&arrow_array);
        if maybe_min.is_none() {
            return Self {
                errors: BitPackedArray::new_null_array(len),
                error_reference_value: 0,
                intercept: 0,
                slope: 0.0,
            };
        }

        let min_v = maybe_min.unwrap();
        let max_v = aggregate::max(&arrow_array).unwrap();
        let intercept = min_v;
        let slope = if len > 1 {
            (max_v as f64 - min_v as f64) / ((len - 1) as f64)
        } else {
            0.0
        };

        // Compute error terms and their min/max (ignore nulls).
        let (_dt, values, nulls) = arrow_array.into_parts();
        let mut error_min: i32 = i32::MAX;
        let mut error_max: i32 = i32::MIN;
        for (i, &v) in values.iter().enumerate() {
            if nulls.as_ref().map_or(true, |n| n.is_valid(i)) {
                let predicted = predict_i32(intercept, slope, i as u32);
                let err = v.wrapping_sub(predicted);
                if err < error_min {
                    error_min = err;
                }
                if err > error_max {
                    error_max = err;
                }
            }
        }

        // If all entries were null, we already returned earlier via maybe_min.is_none().
        // Otherwise we have at least one valid error and error_min/max are meaningful.
        let error_reference_value = error_min;

        // Prepare unsigned packed values = (error_i - error_min) as u32
        let sub_range: u32 = (error_max as i64 - error_min as i64) as u32;
        let bit_width: NonZero<u8> = get_bit_width(sub_range as u64);

        let packed_scalar: ScalarBuffer<u32> = ScalarBuffer::from_iter(values.iter().enumerate().map(|(i, &v)| {
            if nulls.as_ref().map_or(true, |n| n.is_valid(i)) {
                let predicted = predict_i32(intercept, slope, i as u32);
                let err = v.wrapping_sub(predicted);
                (err.wrapping_sub(error_reference_value)) as u32
            } else {
                // Placeholder for nulls; will be ignored by null bitmap
                0_u32
            }
        }));

        let unsigned_array = PrimitiveArray::<UInt32Type>::new(packed_scalar, nulls);
        let errors = BitPackedArray::from_primitive(unsigned_array, bit_width);

        Self {
            errors,
            error_reference_value,
            intercept,
            slope,
        }
    }

    fn len(&self) -> usize {
        self.errors.len()
    }

    fn bit_pack_starting_loc() -> usize {
        // Header + intercept(i32) + error_ref(i32) + slope(f64), aligned to 8 bytes boundary
        let header_size = LiquidIPCHeader::size()
            + std::mem::size_of::<i32>()
            + std::mem::size_of::<i32>()
            + std::mem::size_of::<f64>();
        (header_size + 7) & !7
    }

    /// Serialize to bytes. Custom logical type id is used so that generic IPC reader won't
    /// accidentally interpret this as a primitive integer array.
    pub(crate) fn to_bytes_inner(&self) -> Vec<u8> {
        // Choose a custom logical type id for linear-integer (not part of LiquidDataType enum).
        const LOGICAL_TYPE_LINEAR_I32: u16 = 0x8001; // private/custom
        const PHYSICAL_TYPE_LINEAR_I32: u16 = 0x8002; // private/custom

        let header = LiquidIPCHeader::new(LOGICAL_TYPE_LINEAR_I32, PHYSICAL_TYPE_LINEAR_I32);
        let start = Self::bit_pack_starting_loc();
        let mut out = Vec::with_capacity(start + 256);

        // Header
        out.extend_from_slice(&header.to_bytes());

        // intercept (i32)
        out.extend_from_slice(&self.intercept.to_le_bytes());
        // error_reference_value (i32)
        out.extend_from_slice(&self.error_reference_value.to_le_bytes());
        // slope (f64)
        out.extend_from_slice(&self.slope.to_le_bytes());

        // padding to alignment
        while out.len() < start {
            out.push(0);
        }

        // Append bit-packed error payload
        self.errors.to_bytes(&mut out);

        out
    }

    /// Deserialize from bytes produced by `to_bytes_inner`.
    pub fn from_bytes(bytes: Bytes) -> Self {
        let _header = LiquidIPCHeader::from_bytes(&bytes);
        let intercept = i32::from_le_bytes(bytes[LiquidIPCHeader::size()..LiquidIPCHeader::size() + 4].try_into().unwrap());
        let error_reference_value = i32::from_le_bytes(bytes[LiquidIPCHeader::size() + 4..LiquidIPCHeader::size() + 8].try_into().unwrap());
        let slope_off = LiquidIPCHeader::size() + 8;
        let slope = f64::from_le_bytes(bytes[slope_off..slope_off + 8].try_into().unwrap());

        let bit_packed_offset = Self::bit_pack_starting_loc();
        let bit_packed_bytes = bytes.slice(bit_packed_offset..);
        let errors = BitPackedArray::<UInt32Type>::from_bytes(bit_packed_bytes);

        Self {
            errors,
            error_reference_value,
            intercept,
            slope,
        }
    }
}

impl LiquidArray for LiquidLinearI32Array {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn get_array_memory_size(&self) -> usize {
        self.errors.get_array_memory_size()
            + std::mem::size_of::<i32>() // intercept
            + std::mem::size_of::<i32>() // error_reference_value
            + std::mem::size_of::<f64>() // slope
    }

    fn len(&self) -> usize {
        self.len()
    }

    fn to_arrow_array(&self) -> ArrayRef {
        // Unpack unsigned errors
        let unsigned = self.errors.to_primitive();
        let (_dt, packed_values, nulls) = unsigned.into_parts();

        // Convert to signed errors by adding back the reference value
        let errors_i32: ScalarBuffer<i32> = if self.error_reference_value != 0 {
            ScalarBuffer::from_iter(packed_values.iter().map(|&v| {
                (v as i32).wrapping_add(self.error_reference_value)
            }))
        } else {
            // No offset, just cast
            ScalarBuffer::from_iter(packed_values.iter().map(|&v| v as i32))
        };

        // Reconstruct final values: predicted(i) + error_i, rounding using the same model.
        let mut final_values = Vec::<i32>::with_capacity(self.len());
        for (i, &e) in errors_i32.iter().enumerate() {
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
        self.to_bytes_inner()
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
    use arrow::array::Array;

    fn roundtrip_eq(values: Vec<Option<i32>>) {
        let arr = PrimitiveArray::<Int32Type>::from(values.clone());
        let linear = LiquidLinearI32Array::from_arrow_array(arr.clone());
        let decoded = linear.to_arrow_array();
        assert_eq!(decoded.as_ref(), &arr);
    }

    fn bytes_roundtrip_eq(values: Vec<Option<i32>>) {
        let arr = PrimitiveArray::<Int32Type>::from(values.clone());
        let linear = LiquidLinearI32Array::from_arrow_array(arr.clone());
        let bytes = Bytes::from(linear.to_bytes());
        let decoded_linear = LiquidLinearI32Array::from_bytes(bytes);
        let decoded = decoded_linear.to_arrow_array();
        assert_eq!(decoded.as_ref(), &arr);
    }

    #[test]
    fn test_roundtrip_basic() {
        roundtrip_eq(vec![
            Some(0),
            Some(12),
            Some(32),
            Some(48),
            Some(64),
            Some(85),
            Some(90),
        ]);
    }

    #[test]
    fn test_roundtrip_with_nulls() {
        roundtrip_eq(vec![
            Some(10),
            None,
            Some(30),
            None,
            Some(50),
            Some(70),
        ]);
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
    fn test_roundtrip_bytes() {
        bytes_roundtrip_eq(
            (0..1000)
                .map(|i| if i % 17 == 0 { None } else { Some((i as i32) * 3 - 200) })
                .collect(),
        );
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
}
