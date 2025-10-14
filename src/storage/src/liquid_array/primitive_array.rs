use std::any::Any;
use std::fmt::{Debug, Display};
use std::sync::Arc;

use arrow::array::{
    ArrayRef, ArrowNativeTypeOp, ArrowPrimitiveType, BooleanArray, PrimitiveArray,
    cast::AsArray,
    types::{
        Date32Type, Date64Type, Int8Type, Int16Type, Int32Type, Int64Type, UInt8Type, UInt16Type,
        UInt32Type, UInt64Type,
    },
};
use arrow::buffer::{BooleanBuffer, ScalarBuffer};
use arrow_schema::DataType;
use datafusion::physical_plan::PhysicalExpr;
use fastlanes::BitPacking;
use num_traits::{AsPrimitive, FromPrimitive};

use super::LiquidDataType;
use super::squeezed_date32_array::{Date32Field, SqueezedDate32Array};
use crate::liquid_array::hybrid_primitive_array::{
    LiquidPrimitiveClampedArray, LiquidPrimitiveQuantizedArray,
};
use crate::liquid_array::ipc::LiquidIPCHeader;
use crate::liquid_array::ipc::get_physical_type_id;
use crate::liquid_array::raw::BitPackedArray;
use crate::liquid_array::{LiquidArray, LiquidArrayRef, LiquidHybridArrayRef, PrimitiveKind};
use crate::utils::get_bit_width;
use arrow::datatypes::ArrowNativeType;
use bytes::Bytes;

/// Squeeze policy for primitive integer arrays.
/// Users can choose whether to clamp or quantize when squeezing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum IntegerSqueezePolicy {
    /// Clamp values above the squeezed range to a sentinel (recoverable for non-clamped rows).
    #[default]
    Clamp = 0,
    /// Quantize values into buckets (good for coarse filtering; requires disk to recover values).
    Quantize = 1,
}

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
                    + FromPrimitive
                    + Display,
    > + Debug
    + Send
    + Sync
    + private::Sealed
    + PrimitiveKind
{
    /// The unsigned type that can be used to represent the signed type.
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
    UInt8Type => UInt8Type,
    Date64Type => UInt64Type,
    Date32Type => UInt32Type
}

/// Liquid's unsigned 8-bit integer array.
pub type LiquidU8Array = LiquidPrimitiveArray<UInt8Type>;
/// Liquid's unsigned 16-bit integer array.
pub type LiquidU16Array = LiquidPrimitiveArray<UInt16Type>;
/// Liquid's unsigned 32-bit integer array.
pub type LiquidU32Array = LiquidPrimitiveArray<UInt32Type>;
/// Liquid's unsigned 64-bit integer array.
pub type LiquidU64Array = LiquidPrimitiveArray<UInt64Type>;
/// Liquid's signed 8-bit integer array.
pub type LiquidI8Array = LiquidPrimitiveArray<Int8Type>;
/// Liquid's signed 16-bit integer array.
pub type LiquidI16Array = LiquidPrimitiveArray<Int16Type>;
/// Liquid's signed 32-bit integer array.
pub type LiquidI32Array = LiquidPrimitiveArray<Int32Type>;
/// Liquid's signed 64-bit integer array.
pub type LiquidI64Array = LiquidPrimitiveArray<Int64Type>;
/// Liquid's 32-bit date array.
pub type LiquidDate32Array = LiquidPrimitiveArray<Date32Type>;
/// Liquid's 64-bit date array.
pub type LiquidDate64Array = LiquidPrimitiveArray<Date64Type>;

/// Liquid's primitive array
#[derive(Debug, Clone)]
pub struct LiquidPrimitiveArray<T: LiquidPrimitiveType> {
    bit_packed: BitPackedArray<T::UnSignedType>,
    reference_value: T::Native,
    squeeze_policy: IntegerSqueezePolicy,
}

impl<T> LiquidPrimitiveArray<T>
where
    T: LiquidPrimitiveType,
{
    /// Get the memory size of the Liquid primitive array.
    pub fn get_array_memory_size(&self) -> usize {
        self.bit_packed.get_array_memory_size()
            + std::mem::size_of::<T::Native>()
            + std::mem::size_of::<IntegerSqueezePolicy>()
    }

    /// Get the length of the Liquid primitive array.
    pub fn len(&self) -> usize {
        self.bit_packed.len()
    }

    /// Check if the Liquid primitive array is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Create a Liquid primitive array from an Arrow primitive array.
    pub fn from_arrow_array(arrow_array: PrimitiveArray<T>) -> LiquidPrimitiveArray<T> {
        let min = match arrow::compute::kernels::aggregate::min(&arrow_array) {
            Some(v) => v,
            None => {
                // entire array is null
                return Self {
                    bit_packed: BitPackedArray::new_null_array(arrow_array.len()),
                    reference_value: T::Native::ZERO,
                    squeeze_policy: IntegerSqueezePolicy::default(),
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
            bit_packed: bit_packed_array,
            reference_value: min,
            squeeze_policy: IntegerSqueezePolicy::default(),
        }
    }

    /// Get the current squeeze policy for this array.
    pub fn squeeze_policy(&self) -> IntegerSqueezePolicy {
        self.squeeze_policy
    }

    /// Set the squeeze policy for this array.
    pub fn set_squeeze_policy(&mut self, policy: IntegerSqueezePolicy) {
        self.squeeze_policy = policy;
    }

    /// Set the squeeze policy, returning self for chaining.
    pub fn with_squeeze_policy(mut self, policy: IntegerSqueezePolicy) -> Self {
        self.squeeze_policy = policy;
        self
    }
}

impl<T> LiquidArray for LiquidPrimitiveArray<T>
where
    T: LiquidPrimitiveType + super::PrimitiveKind,
{
    fn get_array_memory_size(&self) -> usize {
        self.get_array_memory_size()
    }

    fn len(&self) -> usize {
        self.len()
    }

    fn original_arrow_data_type(&self) -> DataType {
        T::DATA_TYPE.clone()
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    #[inline]
    fn to_arrow_array(&self) -> ArrayRef {
        let unsigned_array = self.bit_packed.to_primitive();
        let (_data_type, values, _nulls) = unsigned_array.into_parts();
        let nulls = self.bit_packed.nulls();
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

        Arc::new(PrimitiveArray::<T>::new(values, nulls.cloned()))
    }

    fn filter(&self, selection: &BooleanBuffer) -> LiquidArrayRef {
        let unsigned_array: PrimitiveArray<T::UnSignedType> = self.bit_packed.to_primitive();
        let selection = BooleanArray::new(selection.clone(), None);
        let filtered_values =
            arrow::compute::kernels::filter::filter(&unsigned_array, &selection).unwrap();
        let filtered_values = filtered_values.as_primitive::<T::UnSignedType>().clone();
        let Some(bit_width) = self.bit_packed.bit_width() else {
            return Arc::new(LiquidPrimitiveArray::<T> {
                bit_packed: BitPackedArray::new_null_array(filtered_values.len()),
                reference_value: self.reference_value,
                squeeze_policy: self.squeeze_policy,
            });
        };
        let bit_packed = BitPackedArray::from_primitive(filtered_values, bit_width);
        Arc::new(LiquidPrimitiveArray::<T> {
            bit_packed,
            reference_value: self.reference_value,
            squeeze_policy: self.squeeze_policy,
        })
    }

    fn filter_to_arrow(&self, selection: &BooleanBuffer) -> ArrayRef {
        let arrow_array = self.to_arrow_array();
        let selection = BooleanArray::new(selection.clone(), None);
        arrow::compute::kernels::filter::filter(&arrow_array, &selection).unwrap()
    }

    fn try_eval_predicate(
        &self,
        _predicate: &Arc<dyn PhysicalExpr>,
        _filter: &BooleanBuffer,
    ) -> Option<BooleanArray> {
        // primitive array is not supported for liquid predicate
        None
    }

    fn to_bytes(&self) -> Vec<u8> {
        self.to_bytes_inner()
    }

    fn data_type(&self) -> LiquidDataType {
        LiquidDataType::Integer
    }

    fn squeeze(&self) -> Option<(LiquidHybridArrayRef, Bytes)> {
        // Only squeeze if we have a concrete bit width and it is large enough
        let orig_bw = self.bit_packed.bit_width()?;
        if orig_bw.get() < 8 {
            return None;
        }

        // New squeezed bit width is half of the original
        let new_bw_u8 = std::num::NonZero::new((orig_bw.get() / 2).max(1)).unwrap();

        // Decode original unsigned offsets
        let unsigned_array = self.bit_packed.to_primitive();
        let (_dt, values, nulls) = unsigned_array.into_parts();

        // Full bytes (original format) are what we store to disk
        let full_bytes = Bytes::from(self.to_bytes_inner());
        let disk_range = 0u64..(full_bytes.len() as u64);

        match self.squeeze_policy {
            IntegerSqueezePolicy::Clamp => {
                // Sentinel is the max representable value with new_bw bits
                type U<TT> =
                    <<TT as LiquidPrimitiveType>::UnSignedType as ArrowPrimitiveType>::Native;
                let sentinel: U<T> = U::<T>::usize_as((1usize << new_bw_u8.get()) - 1);

                // Clamp values to the squeezed width; values >= sentinel become sentinel
                let squeezed_values: ScalarBuffer<U<T>> = ScalarBuffer::from_iter(
                    values
                        .iter()
                        .map(|&v| if v >= sentinel { sentinel } else { v }),
                );
                let squeezed_unsigned =
                    PrimitiveArray::<<T as LiquidPrimitiveType>::UnSignedType>::new(
                        squeezed_values,
                        nulls,
                    );
                let squeezed_bitpacked =
                    BitPackedArray::from_primitive(squeezed_unsigned, new_bw_u8);

                let hybrid = LiquidPrimitiveClampedArray::<T> {
                    squeezed: squeezed_bitpacked,
                    reference_value: self.reference_value,
                    disk_range,
                };
                Some((Arc::new(hybrid) as LiquidHybridArrayRef, full_bytes))
            }
            IntegerSqueezePolicy::Quantize => {
                // Quantize value offsets into buckets of width W.
                // Determine actual max offset value.
                type U<TT> =
                    <<TT as LiquidPrimitiveType>::UnSignedType as ArrowPrimitiveType>::Native;
                let max_offset: U<T> = if let Some(m) = values.iter().copied().max() {
                    m
                } else {
                    U::<T>::ZERO
                };

                // Compute bucket count and width: ceil((max_offset+1)/bucket_count)
                let bucket_count_u64 = 1u64 << (new_bw_u8.get() as u64);
                let max_off_u64: u64 = num_traits::AsPrimitive::<u64>::as_(max_offset);
                let range_size = max_off_u64.saturating_add(1);
                let bucket_width_u64 = (range_size.div_ceil(bucket_count_u64)).max(1);

                let quantized_values: ScalarBuffer<U<T>> =
                    ScalarBuffer::from_iter(values.iter().map(|&v| {
                        // v / bucket_width, clamped to last bucket
                        let v_u64: u64 = num_traits::AsPrimitive::<u64>::as_(v);
                        let mut idx_u64 = v_u64 / bucket_width_u64;
                        if idx_u64 >= bucket_count_u64 {
                            idx_u64 = bucket_count_u64 - 1;
                        }
                        U::<T>::usize_as(idx_u64 as usize)
                    }));
                let quantized_unsigned =
                    PrimitiveArray::<<T as LiquidPrimitiveType>::UnSignedType>::new(
                        quantized_values,
                        nulls,
                    );
                let quantized_bitpacked =
                    BitPackedArray::from_primitive(quantized_unsigned, new_bw_u8);

                let hybrid = LiquidPrimitiveQuantizedArray::<T> {
                    quantized: quantized_bitpacked,
                    reference_value: self.reference_value,
                    bucket_width: bucket_width_u64,
                    disk_range,
                };
                Some((Arc::new(hybrid) as LiquidHybridArrayRef, full_bytes))
            }
        }
    }
}

impl LiquidPrimitiveArray<Date32Type> {
    /// Convenience: build a `SqueezedDate32Array` by extracting the specified field.
    pub fn squeeze_date32_component(&self, field: Date32Field) -> SqueezedDate32Array {
        SqueezedDate32Array::from_liquid_date32(self, field)
    }
}

impl<T> LiquidPrimitiveArray<T>
where
    T: LiquidPrimitiveType,
{
    fn bit_pack_starting_loc() -> usize {
        let header_size = LiquidIPCHeader::size() + std::mem::size_of::<T::Native>();
        (header_size + 7) & !7
    }

    /*
    Serialized LiquidPrimitiveArray Memory Layout:
    +--------------------------------------------------+
    | LiquidIPCHeader (16 bytes)                       |
    +--------------------------------------------------+

    +--------------------------------------------------+
    | reference_value (size_of::<T::Native> bytes)     |  // The reference value (e.g. minimum value)
    +--------------------------------------------------+
    | Padding (to 8-byte alignment)                    |  // Padding to ensure 8-byte alignment
    +--------------------------------------------------+

    +--------------------------------------------------+
    | BitPackedArray Data                              |
    +--------------------------------------------------+
    | [BitPackedArray Header & Bit-Packed Values]      |  // Written by self.bit_packed.to_bytes()
    +--------------------------------------------------+
    */
    pub(crate) fn to_bytes_inner(&self) -> Vec<u8> {
        // Determine type ID based on the type
        let physical_type_id = get_physical_type_id::<T>();
        let logical_type_id = super::LiquidDataType::Integer as u16;
        let header = LiquidIPCHeader::new(logical_type_id, physical_type_id);

        let bit_pack_starting_loc = Self::bit_pack_starting_loc();
        let mut result = Vec::with_capacity(bit_pack_starting_loc + 256); // Pre-allocate a reasonable size

        // Write header
        result.extend_from_slice(&header.to_bytes());

        // Write reference value
        let ref_value_bytes = unsafe {
            std::slice::from_raw_parts(
                &self.reference_value as *const T::Native as *const u8,
                std::mem::size_of::<T::Native>(),
            )
        };
        result.extend_from_slice(ref_value_bytes);
        while result.len() < bit_pack_starting_loc {
            result.push(0);
        }

        // Let BitPackedArray write the rest of the data
        self.bit_packed.to_bytes(&mut result);

        result
    }

    /// Deserialize a LiquidPrimitiveArray from bytes
    pub fn from_bytes(bytes: Bytes) -> Self {
        let header = LiquidIPCHeader::from_bytes(&bytes);

        let physical_id = header.physical_type_id;
        assert_eq!(physical_id, get_physical_type_id::<T>());
        let logical_id = header.logical_type_id;
        assert_eq!(logical_id, super::LiquidDataType::Integer as u16);

        // Get the reference value
        let ref_value_ptr = &bytes[LiquidIPCHeader::size()];
        let reference_value =
            unsafe { (ref_value_ptr as *const u8 as *const T::Native).read_unaligned() };

        // Skip ahead to the BitPackedArray data
        let bit_packed_data = bytes.slice(Self::bit_pack_starting_loc()..);
        let bit_packed = BitPackedArray::<T::UnSignedType>::from_bytes(bit_packed_data);

        Self {
            bit_packed,
            reference_value,
            squeeze_policy: IntegerSqueezePolicy::default(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::Array;

    macro_rules! test_roundtrip {
        ($test_name:ident, $type:ty, $values:expr) => {
            #[test]
            fn $test_name() {
                // Create the original array
                let original: Vec<Option<<$type as ArrowPrimitiveType>::Native>> = $values;
                let array = PrimitiveArray::<$type>::from(original.clone());

                // Convert to Liquid array and back
                let liquid_array = LiquidPrimitiveArray::<$type>::from_arrow_array(array.clone());
                let result_array = liquid_array.to_arrow_array();
                let bytes_array =
                    LiquidPrimitiveArray::<$type>::from_bytes(liquid_array.to_bytes().into());

                assert_eq!(result_array.as_ref(), &array);
                assert_eq!(bytes_array.to_arrow_array().as_ref(), &array);
            }
        };
    }

    // Test cases for Int8Type
    test_roundtrip!(
        test_int8_roundtrip_basic,
        Int8Type,
        vec![Some(1), Some(2), Some(3), None, Some(5)]
    );
    test_roundtrip!(
        test_int8_roundtrip_negative,
        Int8Type,
        vec![Some(-128), Some(-64), Some(0), Some(63), Some(127)]
    );

    // Test cases for Int16Type
    test_roundtrip!(
        test_int16_roundtrip_basic,
        Int16Type,
        vec![Some(1), Some(2), Some(3), None, Some(5)]
    );
    test_roundtrip!(
        test_int16_roundtrip_negative,
        Int16Type,
        vec![
            Some(-32768),
            Some(-16384),
            Some(0),
            Some(16383),
            Some(32767)
        ]
    );

    // Test cases for Int32Type
    test_roundtrip!(
        test_int32_roundtrip_basic,
        Int32Type,
        vec![Some(1), Some(2), Some(3), None, Some(5)]
    );
    test_roundtrip!(
        test_int32_roundtrip_negative,
        Int32Type,
        vec![
            Some(-2147483648),
            Some(-1073741824),
            Some(0),
            Some(1073741823),
            Some(2147483647)
        ]
    );

    // Test cases for Int64Type
    test_roundtrip!(
        test_int64_roundtrip_basic,
        Int64Type,
        vec![Some(1), Some(2), Some(3), None, Some(5)]
    );
    test_roundtrip!(
        test_int64_roundtrip_negative,
        Int64Type,
        vec![
            Some(-9223372036854775808),
            Some(-4611686018427387904),
            Some(0),
            Some(4611686018427387903),
            Some(9223372036854775807)
        ]
    );

    // Test cases for unsigned types
    test_roundtrip!(
        test_uint8_roundtrip,
        UInt8Type,
        vec![Some(0), Some(128), Some(255), None, Some(64)]
    );
    test_roundtrip!(
        test_uint16_roundtrip,
        UInt16Type,
        vec![Some(0), Some(32768), Some(65535), None, Some(16384)]
    );
    test_roundtrip!(
        test_uint32_roundtrip,
        UInt32Type,
        vec![
            Some(0),
            Some(2147483648),
            Some(4294967295),
            None,
            Some(1073741824)
        ]
    );
    test_roundtrip!(
        test_uint64_roundtrip,
        UInt64Type,
        vec![
            Some(0),
            Some(9223372036854775808),
            Some(18446744073709551615),
            None,
            Some(4611686018427387904)
        ]
    );

    test_roundtrip!(
        test_date32_roundtrip,
        Date32Type,
        vec![Some(-365), Some(0), Some(365), None, Some(18262)]
    );

    test_roundtrip!(
        test_date64_roundtrip,
        Date64Type,
        vec![Some(-365), Some(0), Some(365), None, Some(18262)]
    );

    // Edge cases
    #[test]
    fn test_all_nulls() {
        let original: Vec<Option<i32>> = vec![None, None, None];
        let array = PrimitiveArray::<Int32Type>::from(original.clone());
        let liquid_array = LiquidPrimitiveArray::<Int32Type>::from_arrow_array(array);
        let result_array = liquid_array.to_arrow_array();

        assert_eq!(result_array.len(), original.len());
        assert_eq!(result_array.null_count(), original.len());
    }

    #[test]
    fn test_all_nulls_filter() {
        let original: Vec<Option<i32>> = vec![None, None, None];
        let array = PrimitiveArray::<Int32Type>::from(original.clone());
        let liquid_array = LiquidPrimitiveArray::<Int32Type>::from_arrow_array(array);
        let result_array = liquid_array.filter(&BooleanBuffer::from(vec![true, false, true]));
        let result_array = result_array.to_arrow_array();

        assert_eq!(result_array.len(), 2);
        assert_eq!(result_array.null_count(), 2);
    }

    #[test]
    fn test_zero_reference_value() {
        let original: Vec<Option<i32>> = vec![Some(0), Some(1), Some(2), None, Some(4)];
        let array = PrimitiveArray::<Int32Type>::from(original.clone());
        let liquid_array = LiquidPrimitiveArray::<Int32Type>::from_arrow_array(array.clone());
        let result_array = liquid_array.to_arrow_array();

        assert_eq!(liquid_array.reference_value, 0);
        assert_eq!(result_array.as_ref(), &array);
    }

    #[test]
    fn test_single_value() {
        let original: Vec<Option<i32>> = vec![Some(42)];
        let array = PrimitiveArray::<Int32Type>::from(original.clone());
        let liquid_array = LiquidPrimitiveArray::<Int32Type>::from_arrow_array(array.clone());
        let result_array = liquid_array.to_arrow_array();

        assert_eq!(result_array.as_ref(), &array);
    }

    #[test]
    fn test_filter_basic() {
        // Create original array with some values
        let original = vec![Some(1), Some(2), Some(3), None, Some(5)];
        let array = PrimitiveArray::<Int32Type>::from(original);
        let liquid_array = LiquidPrimitiveArray::<Int32Type>::from_arrow_array(array);

        // Create selection mask: keep indices 0, 2, and 4
        let selection = BooleanBuffer::from(vec![true, false, true, false, true]);

        // Apply filter
        let filtered = liquid_array.filter(&selection);
        let result_array = filtered.to_arrow_array();

        // Expected result after filtering
        let expected = PrimitiveArray::<Int32Type>::from(vec![Some(1), Some(3), Some(5)]);

        assert_eq!(result_array.as_ref(), &expected);
    }

    #[test]
    fn test_original_arrow_data_type_returns_int32() {
        let array = PrimitiveArray::<Int32Type>::from(vec![Some(1), Some(2)]);
        let liquid = LiquidPrimitiveArray::<Int32Type>::from_arrow_array(array);
        assert_eq!(liquid.original_arrow_data_type(), DataType::Int32);
    }

    #[test]
    fn test_filter_all_nulls() {
        // Create array with all nulls
        let original = vec![None, None, None, None];
        let array = PrimitiveArray::<Int32Type>::from(original);
        let liquid_array = LiquidPrimitiveArray::<Int32Type>::from_arrow_array(array);

        // Keep first and last elements
        let selection = BooleanBuffer::from(vec![true, false, false, true]);

        let filtered = liquid_array.filter(&selection);
        let result_array = filtered.to_arrow_array();

        let expected = PrimitiveArray::<Int32Type>::from(vec![None, None]);

        assert_eq!(result_array.as_ref(), &expected);
    }

    #[test]
    fn test_filter_empty_result() {
        let original = vec![Some(1), Some(2), Some(3)];
        let array = PrimitiveArray::<Int32Type>::from(original);
        let liquid_array = LiquidPrimitiveArray::<Int32Type>::from_arrow_array(array);

        // Filter out all elements
        let selection = BooleanBuffer::from(vec![false, false, false]);

        let filtered = liquid_array.filter(&selection);
        let result_array = filtered.to_arrow_array();

        assert_eq!(result_array.len(), 0);
    }
}
