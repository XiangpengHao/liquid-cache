use std::any::Any;
use std::fmt::Debug;
use std::num::NonZero;
use std::sync::Arc;

use arrow::array::ArrowNativeTypeOp;
use arrow::buffer::ScalarBuffer;
use arrow::{
    array::{
        ArrayRef, ArrowPrimitiveType, BooleanArray, PrimitiveArray, RecordBatch,
        cast::AsArray,
        types::{
            Int8Type, Int16Type, Int32Type, Int64Type, UInt8Type, UInt16Type, UInt32Type,
            UInt64Type,
        },
    },
    datatypes::ArrowNativeType,
};
use arrow_schema::{Field, Schema};
use fastlanes::BitPacking;
use num_traits::{AsPrimitive, FromPrimitive};

use super::BitPackedArray;
use crate::liquid_array::{LiquidArray, LiquidArrayRef, get_bit_width};

mod private {
    pub trait Sealed {}
}

/// LiquidPrimitiveType is a sealed trait that represents the primitive types supported by Liquid.
/// Implementors are: Int8Type, Int16Type, Int32Type, Int64Type, UInt8Type, UInt16Type, UInt32Type, UInt64Type
///
/// I have to admit this trait is super complicated.
/// Luckily users never have to worry about it, they can just use the types that are already implemented.
/// We could have implemented this as a macro, but macro is ugly.
/// Type is spec, code is proof.
pub trait LiquidPrimitiveType:
    ArrowPrimitiveType<
        Native: AsPrimitive<<Self::UnSignedType as ArrowPrimitiveType>::Native>
                    + AsPrimitive<i64>
                    + FromPrimitive,
    > + Debug
    + private::Sealed
{
    type UnSignedType: ArrowPrimitiveType<Native: AsPrimitive<Self::Native> + AsPrimitive<u64> + BitPacking>
        + Debug;
}

macro_rules! impl_has_unsigned_type {
    ($($signed:ty => $unsigned:ty),*) => {
        $(
            impl private::Sealed for $signed {}
            impl LiquidPrimitiveType for $signed {
                type UnSignedType = $unsigned;
            }
        )*
    }
}

impl_has_unsigned_type! {
    Int32Type => UInt32Type,
    Int64Type => UInt64Type,
    Int16Type => UInt16Type,
    Int8Type => UInt8Type,
    UInt32Type => UInt32Type,
    UInt64Type => UInt64Type,
    UInt16Type => UInt16Type,
    UInt8Type => UInt8Type
}

pub type LiquidU8Array = LiquidPrimitiveArray<UInt8Type>;
pub type LiquidU16Array = LiquidPrimitiveArray<UInt16Type>;
pub type LiquidU32Array = LiquidPrimitiveArray<UInt32Type>;
pub type LiquidU64Array = LiquidPrimitiveArray<UInt64Type>;
pub type LiquidI8Array = LiquidPrimitiveArray<Int8Type>;
pub type LiquidI16Array = LiquidPrimitiveArray<Int16Type>;
pub type LiquidI32Array = LiquidPrimitiveArray<Int32Type>;
pub type LiquidI64Array = LiquidPrimitiveArray<Int64Type>;

/// The metadata for an ETC primitive array.
#[derive(Debug, Clone)]
pub struct LiquidPrimitiveMetadata {
    reference_value: i64,
    bit_width: NonZero<u8>,
    original_len: usize,
}

/// ETC's primitive array
#[derive(Debug, Clone)]
pub struct LiquidPrimitiveArray<T: LiquidPrimitiveType> {
    values: BitPackedArray<T::UnSignedType>,
    reference_value: T::Native,
}

impl<T> LiquidPrimitiveArray<T>
where
    T: LiquidPrimitiveType,
{
    /// Get the memory size of the Liquid primitive array.
    pub fn get_array_memory_size(&self) -> usize {
        self.values.get_array_memory_size() + std::mem::size_of::<T::Native>()
    }

    /// Get the length of the Liquid primitive array.
    pub fn len(&self) -> usize {
        self.values.original_len
    }

    /// Check if the Liquid primitive array is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Get the metadata for a Liquid primitive array.
    pub fn metadata(&self) -> LiquidPrimitiveMetadata {
        LiquidPrimitiveMetadata {
            reference_value: self.reference_value.to_i64().unwrap(),
            bit_width: self.values.bit_width,
            original_len: self.values.original_len,
        }
    }

    /// Convert a Liquid primitive array to a record batch.
    pub fn to_record_batch(&self) -> (RecordBatch, LiquidPrimitiveMetadata) {
        let schema = Schema::new(vec![Field::new(
            "values",
            <T as ArrowPrimitiveType>::DATA_TYPE,
            false,
        )]);

        let batch =
            RecordBatch::try_new(Arc::new(schema), vec![Arc::new(self.values.values.clone())])
                .unwrap();

        (batch, self.metadata())
    }

    /// Create a Liquid primitive array from a record batch.
    pub fn from_record_batch(batch: RecordBatch, metadata: &LiquidPrimitiveMetadata) -> Self {
        let values = batch
            .column(0)
            .as_primitive::<<T as LiquidPrimitiveType>::UnSignedType>()
            .clone();
        let values = BitPackedArray::from_parts(values, metadata.bit_width, metadata.original_len);
        Self {
            values,
            reference_value: <<T as ArrowPrimitiveType>::Native as FromPrimitive>::from_i64(
                metadata.reference_value,
            )
            .unwrap(),
        }
    }

    pub fn from_arrow_array(arrow_array: PrimitiveArray<T>) -> LiquidPrimitiveArray<T> {
        let min = match arrow::compute::kernels::aggregate::min(&arrow_array) {
            Some(v) => v,
            None => {
                // entire array is null
                return Self {
                    values: BitPackedArray::new_null_array(arrow_array.len()),
                    reference_value: T::Native::ZERO,
                };
            }
        };
        let max = arrow::compute::kernels::aggregate::max(&arrow_array).unwrap();

        // be careful of overflow:
        // Want: 127i8 - (-128i8) -> 255u64,
        // but we get -1i8
        // (-1i8) as u8 as u64 -> 255u64
        let sub = max.sub_wrapping(min) as <T as ArrowPrimitiveType>::Native;
        let sub: <<T as LiquidPrimitiveType>::UnSignedType as ArrowPrimitiveType>::Native =
            sub.as_();
        let bit_width = get_bit_width(sub.as_());

        let (_data_type, values, nulls) = arrow_array.clone().into_parts();
        let values = if min != T::Native::ZERO {
            ScalarBuffer::from_iter(values.iter().map(|v| {
                let k: <<T as LiquidPrimitiveType>::UnSignedType as ArrowPrimitiveType>::Native =
                    v.sub_wrapping(min).as_();
                k
            }))
        } else {
            #[allow(clippy::missing_transmute_annotations)]
            unsafe {
                std::mem::transmute(values)
            }
        };

        let unsigned_array =
            PrimitiveArray::<<T as LiquidPrimitiveType>::UnSignedType>::new(values, nulls);

        let bit_packed_array = BitPackedArray::from_primitive(unsigned_array, bit_width);

        Self {
            values: bit_packed_array,
            reference_value: min,
        }
    }
}

impl<T> LiquidArray for LiquidPrimitiveArray<T>
where
    T: LiquidPrimitiveType,
{
    fn get_array_memory_size(&self) -> usize {
        self.get_array_memory_size()
    }

    fn len(&self) -> usize {
        self.len()
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    #[inline]
    fn to_arrow_array(&self) -> ArrayRef {
        let unsigned_array = self.values.to_primitive();
        let (_data_type, values, nulls) = unsigned_array.into_parts();
        let values = if self.reference_value != T::Native::ZERO {
            let reference_v = self.reference_value.as_();
            ScalarBuffer::from_iter(values.iter().map(|v| {
                let k: <T as ArrowPrimitiveType>::Native = (*v).add_wrapping(reference_v).as_();
                k
            }))
        } else {
            #[allow(clippy::missing_transmute_annotations)]
            unsafe {
                std::mem::transmute(values)
            }
        };

        Arc::new(PrimitiveArray::<T>::new(values, nulls))
    }

    fn filter(&self, selection: &BooleanArray) -> LiquidArrayRef {
        let values = self.to_arrow_array();
        let filtered_values = arrow::compute::kernels::filter::filter(&values, selection).unwrap();
        let primitive_values = filtered_values.as_primitive::<T>().clone();
        let bit_packed = Self::from_arrow_array(primitive_values);
        Arc::new(bit_packed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    macro_rules! test_roundtrip {
        ($test_name:ident, $type:ty, $values:expr) => {
            #[test]
            fn $test_name() {
                // Create the original array
                let original: Vec<Option<<$type as ArrowPrimitiveType>::Native>> = $values;
                let array = PrimitiveArray::<$type>::from(original.clone());

                // Convert to Liquid array and back
                let etc_array = LiquidPrimitiveArray::<$type>::from_arrow_array(array.clone());
                let result_array = etc_array.to_arrow_array();

                assert_eq!(result_array.as_ref(), &array);
            }
        };
    }

    // Test cases for Int8Type
    test_roundtrip!(test_int8_roundtrip_basic, Int8Type, vec![
        Some(1),
        Some(2),
        Some(3),
        None,
        Some(5)
    ]);
    test_roundtrip!(test_int8_roundtrip_negative, Int8Type, vec![
        Some(-128),
        Some(-64),
        Some(0),
        Some(63),
        Some(127)
    ]);

    // Test cases for Int16Type
    test_roundtrip!(test_int16_roundtrip_basic, Int16Type, vec![
        Some(1),
        Some(2),
        Some(3),
        None,
        Some(5)
    ]);
    test_roundtrip!(test_int16_roundtrip_negative, Int16Type, vec![
        Some(-32768),
        Some(-16384),
        Some(0),
        Some(16383),
        Some(32767)
    ]);

    // Test cases for Int32Type
    test_roundtrip!(test_int32_roundtrip_basic, Int32Type, vec![
        Some(1),
        Some(2),
        Some(3),
        None,
        Some(5)
    ]);
    test_roundtrip!(test_int32_roundtrip_negative, Int32Type, vec![
        Some(-2147483648),
        Some(-1073741824),
        Some(0),
        Some(1073741823),
        Some(2147483647)
    ]);

    // Test cases for Int64Type
    test_roundtrip!(test_int64_roundtrip_basic, Int64Type, vec![
        Some(1),
        Some(2),
        Some(3),
        None,
        Some(5)
    ]);
    test_roundtrip!(test_int64_roundtrip_negative, Int64Type, vec![
        Some(-9223372036854775808),
        Some(-4611686018427387904),
        Some(0),
        Some(4611686018427387903),
        Some(9223372036854775807)
    ]);

    // Test cases for unsigned types
    test_roundtrip!(test_uint8_roundtrip, UInt8Type, vec![
        Some(0),
        Some(128),
        Some(255),
        None,
        Some(64)
    ]);
    test_roundtrip!(test_uint16_roundtrip, UInt16Type, vec![
        Some(0),
        Some(32768),
        Some(65535),
        None,
        Some(16384)
    ]);
    test_roundtrip!(test_uint32_roundtrip, UInt32Type, vec![
        Some(0),
        Some(2147483648),
        Some(4294967295),
        None,
        Some(1073741824)
    ]);
    test_roundtrip!(test_uint64_roundtrip, UInt64Type, vec![
        Some(0),
        Some(9223372036854775808),
        Some(18446744073709551615),
        None,
        Some(4611686018427387904)
    ]);

    // Edge cases
    #[test]
    fn test_all_nulls() {
        let original: Vec<Option<i32>> = vec![None, None, None];
        let array = PrimitiveArray::<Int32Type>::from(original.clone());
        let etc_array = LiquidPrimitiveArray::<Int32Type>::from_arrow_array(array);
        let result_array = etc_array.to_arrow_array();

        assert_eq!(result_array.len(), original.len());
        assert_eq!(result_array.null_count(), original.len());
    }

    #[test]
    fn test_zero_reference_value() {
        let original: Vec<Option<i32>> = vec![Some(0), Some(1), Some(2), None, Some(4)];
        let array = PrimitiveArray::<Int32Type>::from(original.clone());
        let etc_array = LiquidPrimitiveArray::<Int32Type>::from_arrow_array(array.clone());
        let result_array = etc_array.to_arrow_array();

        assert_eq!(etc_array.reference_value, 0);
        assert_eq!(result_array.as_ref(), &array);
    }

    #[test]
    fn test_single_value() {
        let original: Vec<Option<i32>> = vec![Some(42)];
        let array = PrimitiveArray::<Int32Type>::from(original.clone());
        let etc_array = LiquidPrimitiveArray::<Int32Type>::from_arrow_array(array.clone());
        let result_array = etc_array.to_arrow_array();

        assert_eq!(result_array.as_ref(), &array);
    }

    #[test]
    fn test_filter_basic() {
        // Create original array with some values
        let original = vec![Some(1), Some(2), Some(3), None, Some(5)];
        let array = PrimitiveArray::<Int32Type>::from(original);
        let etc_array = LiquidPrimitiveArray::<Int32Type>::from_arrow_array(array);

        // Create selection mask: keep indices 0, 2, and 4
        let selection = BooleanArray::from(vec![true, false, true, false, true]);

        // Apply filter
        let filtered = etc_array.filter(&selection);
        let result_array = filtered.to_arrow_array();

        // Expected result after filtering
        let expected = PrimitiveArray::<Int32Type>::from(vec![Some(1), Some(3), Some(5)]);

        assert_eq!(result_array.as_ref(), &expected);
    }

    #[test]
    fn test_filter_all_nulls() {
        // Create array with all nulls
        let original = vec![None, None, None, None];
        let array = PrimitiveArray::<Int32Type>::from(original);
        let etc_array = LiquidPrimitiveArray::<Int32Type>::from_arrow_array(array);

        // Keep first and last elements
        let selection = BooleanArray::from(vec![true, false, false, true]);

        let filtered = etc_array.filter(&selection);
        let result_array = filtered.to_arrow_array();

        let expected = PrimitiveArray::<Int32Type>::from(vec![None, None]);

        assert_eq!(result_array.as_ref(), &expected);
    }

    #[test]
    fn test_filter_empty_result() {
        let original = vec![Some(1), Some(2), Some(3)];
        let array = PrimitiveArray::<Int32Type>::from(original);
        let etc_array = LiquidPrimitiveArray::<Int32Type>::from_arrow_array(array);

        // Filter out all elements
        let selection = BooleanArray::from(vec![false, false, false]);

        let filtered = etc_array.filter(&selection);
        let result_array = filtered.to_arrow_array();

        assert_eq!(result_array.len(), 0);
    }
}
