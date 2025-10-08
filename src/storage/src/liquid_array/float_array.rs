///
/// Acknowledgement:
/// The ALP compression implemented in this file is based on the Rust implementation available at https://github.com/spiraldb/alp
///
use std::{
    any::Any,
    fmt::Debug,
    num::NonZero,
    ops::{Mul, Shl, Shr},
    sync::Arc,
};

use arrow::{
    array::{
        Array, ArrayRef, ArrowNativeTypeOp, ArrowPrimitiveType, AsArray, BooleanArray,
        PrimitiveArray,
    },
    buffer::{BooleanBuffer, ScalarBuffer},
    datatypes::{
        ArrowNativeType, Float32Type, Float64Type, Int32Type, Int64Type, UInt32Type, UInt64Type,
    },
};
use arrow_schema::DataType;
use datafusion::{
    physical_plan::{
        PhysicalExpr,
        expressions::{BinaryExpr, Literal},
    },
    scalar::ScalarValue,
};
use fastlanes::BitPacking;
use num_traits::{AsPrimitive, Float, FromPrimitive};

use super::LiquidDataType;
use crate::liquid_array::ipc::get_physical_type_id;
use crate::liquid_array::raw::BitPackedArray;
use crate::liquid_array::{
    IoRange, LiquidHybridArray, LiquidHybridArrayRef, Operator, ipc::LiquidIPCHeader,
};
use crate::liquid_array::{LiquidArray, LiquidArrayRef};
use crate::utils::get_bit_width;
use bytes::Bytes;

mod private {
    use arrow::{
        array::ArrowNumericType,
        datatypes::{Float32Type, Float64Type},
    };
    use num_traits::AsPrimitive;

    pub trait Sealed: ArrowNumericType<Native: AsPrimitive<f64> + AsPrimitive<f32>> {}

    impl Sealed for Float32Type {}
    impl Sealed for Float64Type {}
}

const NUM_SAMPLES: usize = 1024; // we use FASTLANES to encode array, the sample size needs to be at least 1024 to get a good estimate of the best exponents

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum FloatSqueezePolicy {
    /// Quantize values into buckets (good for coarse filtering; requires disk to recover values).
    #[default]
    Quantize = 0,
}

/// LiquidFloatType is a sealed trait that represents all the float types supported by Liquid.
/// Implementors are Float32Type and Float64Type. TODO(): What about Float16Type, decimal types?
pub trait LiquidFloatType:
        ArrowPrimitiveType<
            Native: AsPrimitive<
                <Self::UnsignedIntType as ArrowPrimitiveType>::Native // Native must be convertible to the Native type of Self::UnSignedType
            >
            + AsPrimitive<<Self::SignedIntType as ArrowPrimitiveType>::Native>
            + FromPrimitive
            + AsPrimitive<<Self as ArrowPrimitiveType>::Native>
            + Mul<<Self as ArrowPrimitiveType>::Native>
            + Float // required for decode_single and encode_single_unchecked
        >
        + private::Sealed + Debug {
    type UnsignedIntType:
        ArrowPrimitiveType<
            Native: BitPacking +
                AsPrimitive<<Self as ArrowPrimitiveType>::Native>
                + AsPrimitive<<Self::SignedIntType as ArrowPrimitiveType>::Native>
                + AsPrimitive<u64>
        >
        + Debug;
    type SignedIntType:
        ArrowPrimitiveType<
            Native: AsPrimitive<<Self as ArrowPrimitiveType>::Native>
                + AsPrimitive<<Self::UnsignedIntType as ArrowPrimitiveType>::Native>
                + Ord
                + Shr<u8, Output = <Self::SignedIntType as ArrowPrimitiveType>::Native>
                + Shl<u8, Output = <Self::SignedIntType as ArrowPrimitiveType>::Native>
                + From<i32>
        >
        + Debug + Sync + Send;

    const SWEET: <Self as ArrowPrimitiveType>::Native;
    const MAX_EXPONENT: u8;
    const FRACTIONAL_BITS: u8;
    const F10: &'static [<Self as ArrowPrimitiveType>::Native];
    const IF10: &'static [<Self as ArrowPrimitiveType>::Native];

    #[inline]
    fn fast_round(val: <Self as ArrowPrimitiveType>::Native) -> <Self::SignedIntType as ArrowPrimitiveType>::Native {
        ((val + Self::SWEET) - Self::SWEET).as_()
    }

    #[inline]
    fn encode_single_unchecked(val: &<Self as ArrowPrimitiveType>::Native, exp: &Exponents) -> <Self::SignedIntType as ArrowPrimitiveType>::Native {
        Self::fast_round(*val * Self::F10[exp.e as usize] * Self::IF10[exp.f as usize])
    }

    #[inline]
    fn decode_single(val: &<Self::SignedIntType as ArrowPrimitiveType>::Native, exp: &Exponents) -> <Self as ArrowPrimitiveType>::Native {
        let decoded_float: <Self as ArrowPrimitiveType>::Native = (*val).as_();
        decoded_float * Self::F10[exp.f as usize] * Self::IF10[exp.e as usize]
    }

}

impl LiquidFloatType for Float32Type {
    type UnsignedIntType = UInt32Type;
    type SignedIntType = Int32Type;
    const FRACTIONAL_BITS: u8 = 23;
    const MAX_EXPONENT: u8 = 10;
    const SWEET: <Self as ArrowPrimitiveType>::Native = (1 << Self::FRACTIONAL_BITS)
        as <Self as ArrowPrimitiveType>::Native
        + (1 << (Self::FRACTIONAL_BITS - 1)) as <Self as ArrowPrimitiveType>::Native;
    const F10: &'static [<Self as ArrowPrimitiveType>::Native] = &[
        1.0,
        10.0,
        100.0,
        1000.0,
        10000.0,
        100000.0,
        1000000.0,
        10000000.0,
        100000000.0,
        1000000000.0,
        10000000000.0, // 10^10
    ];
    const IF10: &'static [<Self as ArrowPrimitiveType>::Native] = &[
        1.0,
        0.1,
        0.01,
        0.001,
        0.0001,
        0.00001,
        0.000001,
        0.0000001,
        0.00000001,
        0.000000001,
        0.0000000001, // 10^-10
    ];
}

impl LiquidFloatType for Float64Type {
    type UnsignedIntType = UInt64Type;
    type SignedIntType = Int64Type;
    const FRACTIONAL_BITS: u8 = 52;
    const MAX_EXPONENT: u8 = 18;
    const SWEET: <Self as ArrowPrimitiveType>::Native = (1u64 << Self::FRACTIONAL_BITS)
        as <Self as ArrowPrimitiveType>::Native
        + (1u64 << (Self::FRACTIONAL_BITS - 1)) as <Self as ArrowPrimitiveType>::Native;
    const F10: &'static [<Self as ArrowPrimitiveType>::Native] = &[
        1.0,
        10.0,
        100.0,
        1000.0,
        10000.0,
        100000.0,
        1000000.0,
        10000000.0,
        100000000.0,
        1000000000.0,
        10000000000.0,
        100000000000.0,
        1000000000000.0,
        10000000000000.0,
        100000000000000.0,
        1000000000000000.0,
        10000000000000000.0,
        100000000000000000.0,
        1000000000000000000.0,
        10000000000000000000.0,
        100000000000000000000.0,
        1000000000000000000000.0,
        10000000000000000000000.0,
        100000000000000000000000.0, // 10^23
    ];

    const IF10: &'static [<Self as ArrowPrimitiveType>::Native] = &[
        1.0,
        0.1,
        0.01,
        0.001,
        0.0001,
        0.00001,
        0.000001,
        0.0000001,
        0.00000001,
        0.000000001,
        0.0000000001,
        0.00000000001,
        0.000000000001,
        0.0000000000001,
        0.00000000000001,
        0.000000000000001,
        0.0000000000000001,
        0.00000000000000001,
        0.000000000000000001,
        0.0000000000000000001,
        0.00000000000000000001,
        0.000000000000000000001,
        0.0000000000000000000001,
        0.00000000000000000000001, // 10^-23
    ];
}

/// Liquid's single-precision floating point array
pub type LiquidFloat32Array = LiquidFloatArray<Float32Type>;
/// Liquid's double precision floating point array
pub type LiquidFloat64Array = LiquidFloatArray<Float64Type>;

/// An array that stores floats in ALP
#[derive(Debug, Clone)]
pub struct LiquidFloatArray<T: LiquidFloatType> {
    exponent: Exponents,
    bit_packed: BitPackedArray<T::UnsignedIntType>,
    patch_indices: Vec<u64>,
    patch_values: Vec<T::Native>,
    reference_value: <T::SignedIntType as ArrowPrimitiveType>::Native,
    squeeze_policy: FloatSqueezePolicy,
}

impl<T> LiquidFloatArray<T>
where
    T: LiquidFloatType,
{
    /// Check if the Liquid float array is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Get the length of the Liquid float array.
    pub fn len(&self) -> usize {
        self.bit_packed.len()
    }

    /// Get the memory size of the Liquid primitive array.
    pub fn get_array_memory_size(&self) -> usize {
        self.bit_packed.get_array_memory_size()
            + size_of::<Exponents>()
            + self.patch_indices.capacity() * size_of::<u64>()
            + self.patch_values.capacity() * size_of::<T::Native>()
            + size_of::<<T::SignedIntType as ArrowPrimitiveType>::Native>()
    }

    /// Create a Liquid primitive array from an Arrow float array.
    pub fn from_arrow_array(arrow_array: arrow::array::PrimitiveArray<T>) -> LiquidFloatArray<T> {
        let best_exponents = get_best_exponents::<T>(&arrow_array);
        encode_arrow_array(&arrow_array, &best_exponents)
    }

    /// Get current squeeze policy for this array
    pub fn squeeze_policy(&self) -> FloatSqueezePolicy {
        self.squeeze_policy
    }
}

impl<T> LiquidArray for LiquidFloatArray<T>
where
    T: LiquidFloatType,
{
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn get_array_memory_size(&self) -> usize {
        self.get_array_memory_size()
    }

    fn len(&self) -> usize {
        self.len()
    }

    #[inline]
    fn to_arrow_array(&self) -> ArrayRef {
        let unsigned_array = self.bit_packed.to_primitive();
        let (_data_type, values, _nulls) = unsigned_array.into_parts();
        let nulls = self.bit_packed.nulls();
        // TODO(): Check if we should align vectors to cache line boundary
        let mut decoded_values = Vec::from_iter(values.iter().map(|v| {
            let mut val: <T::SignedIntType as ArrowPrimitiveType>::Native = (*v).as_();
            val = val.add_wrapping(self.reference_value);
            T::decode_single(&val, &self.exponent)
        }));

        // Patch values
        if !self.patch_indices.is_empty() {
            for i in 0..self.patch_indices.len() {
                decoded_values[self.patch_indices[i].as_usize()] = self.patch_values[i];
            }
        }

        Arc::new(PrimitiveArray::<T>::new(
            ScalarBuffer::<<T as ArrowPrimitiveType>::Native>::from(decoded_values),
            nulls.cloned(),
        ))
    }

    fn original_arrow_data_type(&self) -> DataType {
        T::DATA_TYPE.clone()
    }

    fn data_type(&self) -> LiquidDataType {
        LiquidDataType::Float
    }

    fn to_bytes(&self) -> Vec<u8> {
        self.to_bytes_inner()
    }

    fn filter(&self, selection: &BooleanBuffer) -> LiquidArrayRef {
        let values = self.to_arrow_array();
        let selection = BooleanArray::new(selection.clone(), None);
        let filtered_values = arrow::compute::kernels::filter::filter(&values, &selection).unwrap();
        let primitive_values = filtered_values.as_primitive::<T>().clone();
        let bit_packed = Self::from_arrow_array(primitive_values);
        Arc::new(bit_packed)
    }

    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    fn to_best_arrow_array(&self) -> ArrayRef {
        self.to_arrow_array()
    }

    fn squeeze(&self) -> Option<(super::LiquidHybridArrayRef, bytes::Bytes)> {
        let orig_bw = self.bit_packed.bit_width()?;
        if orig_bw.get() < 8 {
            return None;
        }

        // New squeezed bit width is half of the original
        let new_bw = orig_bw.get() / 2;

        let full_bytes = Bytes::from(self.to_bytes_inner());
        let disk_range = 0u64..(full_bytes.len() as u64);

        let (_dt, values, nulls) = self.bit_packed.to_primitive().into_parts();

        match self.squeeze_policy {
            FloatSqueezePolicy::Quantize => {
                let shift = orig_bw.get() - new_bw;
                let quantized_min = self.reference_value.shr(shift);
                // let quantized_max = values
                let quantized_values: ScalarBuffer<
                    <T::UnsignedIntType as ArrowPrimitiveType>::Native,
                > = ScalarBuffer::from_iter(values.iter().map(|&v| {
                    let signed_val: <T::SignedIntType as ArrowPrimitiveType>::Native = v.as_();
                    let v_signed = self.reference_value.add_wrapping(signed_val);
                    let v_quantized: <T::SignedIntType as ArrowPrimitiveType>::Native =
                        v_signed.shr(shift);
                    v_quantized.sub_wrapping(quantized_min).as_()
                }));
                let quantized_array =
                    PrimitiveArray::<<T as LiquidFloatType>::UnsignedIntType>::new(
                        quantized_values,
                        nulls.clone(),
                    );
                let quantized_bitpacked =
                    BitPackedArray::from_primitive(quantized_array, NonZero::new(new_bw).unwrap());
                let hybrid = LiquidFloatQuantizedArray::<T> {
                    exponent: self.exponent,
                    quantized: quantized_bitpacked,
                    reference_value: self.reference_value,
                    bucket_width: shift,
                    disk_range,
                    patch_indices: self.patch_indices.clone(),
                    patch_values: self.patch_values.clone(),
                };
                Some((Arc::new(hybrid) as LiquidHybridArrayRef, full_bytes))
            }
        }
    }
}

impl<T> LiquidFloatArray<T>
where
    T: LiquidFloatType,
{
    /*
    Serialized LiquidFloatArray Memory Layout:
    +--------------------------------------------------+
    | LiquidIPCHeader (16 bytes)                       |
    +--------------------------------------------------+

    +--------------------------------------------------+
    | reference_value                                  |
    | (size_of::<T::SignedIntType::Native> bytes)      |  // The reference value (e.g. minimum value)
    +--------------------------------------------------+
    | Padding (to 8-byte alignment)                    |  // Padding to ensure 8-byte alignment
    +--------------------------------------------------+

    +--------------------------------------------------+
    | Exponents                                        |
    +--------------------------------------------------+
    | e (1 byte)                                       |
    +--------------------------------------------------+
    | f (1 byte)                                       |
    +--------------------------------------------------+
    | Padding (6 bytes)                                |
    +--------------------------------------------------+

    +--------------------------------------------------+
    | Patch Data                                       |
    +--------------------------------------------------+
    | patch_length (8 bytes)                           |
    +--------------------------------------------------+
    | patch_indices (8 * patch_length btyes)           |
    +--------------------------------------------------+
    | patch_values (length *size_of::<T::Native> btyes;|
    |               8-byte aligned)                    |
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
        let logical_type_id = LiquidDataType::Float as u16;
        let header = LiquidIPCHeader::new(logical_type_id, physical_type_id);

        let mut result = Vec::with_capacity(256); // Pre-allocate a reasonable size

        // Write header
        result.extend_from_slice(&header.to_bytes());

        // Write reference value
        let ref_value_bytes = unsafe {
            std::slice::from_raw_parts(
                &self.reference_value as *const <T::SignedIntType as ArrowPrimitiveType>::Native
                    as *const u8,
                std::mem::size_of::<<T::SignedIntType as ArrowPrimitiveType>::Native>(),
            )
        };
        result.extend_from_slice(ref_value_bytes);

        let exponents_starting_loc = (result.len() + 7) & !7;
        // Insert padding before exponents start
        while result.len() < exponents_starting_loc {
            result.push(0);
        }

        let exponent_e_bytes =
            unsafe { std::slice::from_raw_parts(&self.exponent.e as *const u8, 1) };
        let exponent_f_bytes =
            unsafe { std::slice::from_raw_parts(&self.exponent.f as *const u8, 1) };
        // Write exponents and padding
        result.extend_from_slice(exponent_e_bytes);
        result.extend_from_slice(exponent_f_bytes);
        for _i in 0..6 {
            result.push(0);
        }

        // Number of bytes occupied by usize is target-dependent; use u64 instead
        let patch_length = self.patch_indices.len() as u64;

        let patch_length_bytes = unsafe {
            std::slice::from_raw_parts(
                &patch_length as *const u64 as *const u8,
                std::mem::size_of::<u64>(),
            )
        };

        // Write the patch length
        result.extend_from_slice(patch_length_bytes);

        if !self.patch_indices.is_empty() {
            let patch_indices_bytes = unsafe {
                std::slice::from_raw_parts(
                    self.patch_indices.as_ptr() as *const u8,
                    std::mem::size_of::<u64>() * self.patch_indices.len(),
                )
            };

            // Write the patch indices
            result.extend_from_slice(patch_indices_bytes);

            // Write the patch values
            let patch_values_bytes = unsafe {
                std::slice::from_raw_parts(
                    self.patch_values.as_ptr() as *const u8,
                    std::mem::size_of::<T::Native>() * self.patch_indices.len(),
                )
            };
            result.extend_from_slice(patch_values_bytes);
        }
        let padding = ((result.len() + 7) & !7) - result.len();

        // Add padding before writing bit-packed array
        for _i in 0..padding {
            result.push(0);
        }

        // Serialize bit-packed values
        self.bit_packed.to_bytes(&mut result);

        result
    }

    /// Deserialize a LiquidFloatArray from bytes, using zero-copy where possible.
    pub fn from_bytes(bytes: Bytes) -> Self {
        let header = LiquidIPCHeader::from_bytes(&bytes);

        // Verify the type id
        let physical_id = header.physical_type_id;
        assert_eq!(physical_id, get_physical_type_id::<T>());
        let logical_id = header.logical_type_id;
        assert_eq!(logical_id, LiquidDataType::Float as u16);

        // Get the reference value
        let ref_value_ptr = &bytes[LiquidIPCHeader::size()];
        let reference_value = unsafe {
            (ref_value_ptr as *const u8 as *const <T::SignedIntType as ArrowPrimitiveType>::Native)
                .read_unaligned()
        };

        // Read exponents (e, f) & skip padding
        let mut next = ((LiquidIPCHeader::size()
            + std::mem::size_of::<<T::SignedIntType as ArrowPrimitiveType>::Native>())
            + 7)
            & !7;

        // Read exponent fields (1 byte each) and skip 6 padding bytes
        let exponent_e = bytes[next];
        let exponent_f = bytes[next + 1];
        next += 8;

        // Read patch length (8 bytes)
        let mut patch_length = 0u64;
        patch_length |= bytes[next] as u64;
        patch_length |= (bytes[next + 1] as u64) << 8;
        patch_length |= (bytes[next + 2] as u64) << 16;
        patch_length |= (bytes[next + 3] as u64) << 24;
        patch_length |= (bytes[next + 4] as u64) << 32;
        patch_length |= (bytes[next + 5] as u64) << 40;
        patch_length |= (bytes[next + 6] as u64) << 48;
        patch_length |= (bytes[next + 7] as u64) << 56;
        next += 8;

        // Read patch indices
        let mut patch_indices = Vec::new();
        let mut patch_values = Vec::new();
        if patch_length > 0 {
            let count = patch_length as usize;
            let idx_bytes = count * std::mem::size_of::<u64>();
            let val_bytes = count * std::mem::size_of::<T::Native>();

            let indices_slice = bytes.slice(next..next + idx_bytes);
            next += idx_bytes;
            patch_indices = unsafe {
                let ptr = indices_slice.as_ptr() as *const u64;
                std::slice::from_raw_parts(ptr, count).to_vec()
            };

            let values_slice = bytes.slice(next..next + val_bytes);
            next += val_bytes;
            patch_values = unsafe {
                let ptr = values_slice.as_ptr() as *const T::Native;
                std::slice::from_raw_parts(ptr, count).to_vec()
            };
        }

        // Align up to 8 bytes for bit-packed array
        next = (next + 7) & !7;

        let bit_packed = BitPackedArray::<T::UnsignedIntType>::from_bytes(bytes.slice(next..));

        Self {
            exponent: Exponents {
                e: exponent_e,
                f: exponent_f,
            },
            bit_packed,
            patch_indices,
            patch_values,
            reference_value,
            squeeze_policy: FloatSqueezePolicy::Quantize,
        }
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct Exponents {
    pub(crate) e: u8,
    pub(crate) f: u8,
}

fn encode_arrow_array<T: LiquidFloatType>(
    arrow_array: &PrimitiveArray<T>,
    exp: &Exponents, // fill_value: &mut Option<<T::UnsignedIntType as ArrowPrimitiveType>::Native>
) -> LiquidFloatArray<T> {
    let mut patch_indices: Vec<u64> = Vec::new();
    let mut patch_values: Vec<T::Native> = Vec::new();
    let mut patch_count: usize = 0;
    let mut fill_value: Option<<T::SignedIntType as ArrowPrimitiveType>::Native> = None;
    let values = arrow_array.values();
    let nulls = arrow_array.nulls();

    // All values are null
    if arrow_array.null_count() == arrow_array.len() {
        return LiquidFloatArray::<T> {
            bit_packed: BitPackedArray::new_null_array(arrow_array.len()),
            exponent: Exponents { e: 0, f: 0 },
            patch_indices: Vec::new(),
            patch_values: Vec::new(),
            reference_value: <T::SignedIntType as ArrowPrimitiveType>::Native::ZERO,
            squeeze_policy: FloatSqueezePolicy::Quantize,
        };
    }

    let mut encoded_values = Vec::with_capacity(arrow_array.len());
    for v in values.iter() {
        let encoded = T::encode_single_unchecked(&v.as_(), exp);
        let decoded = T::decode_single(&encoded, exp);
        // TODO(): Check if this is a bitwise comparison
        let neq = !decoded.eq(&v.as_()) as usize;
        patch_count += neq;
        encoded_values.push(encoded);
    }

    if patch_count > 0 {
        patch_indices.resize_with(patch_count + 1, Default::default);
        patch_values.resize_with(patch_count + 1, Default::default);
        let mut patch_index: usize = 0;

        for i in 0..encoded_values.len() {
            let decoded = T::decode_single(&encoded_values[i], exp);
            patch_indices[patch_index] = i.as_();
            patch_values[patch_index] = arrow_array.value(i).as_();
            patch_index += !(decoded.eq(&values[i].as_())) as usize;
        }
        assert_eq!(patch_index, patch_count);
        unsafe {
            patch_indices.set_len(patch_count);
            patch_values.set_len(patch_count);
        }
    }

    // find the first successfully encoded value (i.e., not patched)
    // this is our fill value for missing values
    if patch_count > 0 && patch_count < arrow_array.len() {
        for i in 0..encoded_values.len() {
            if i >= patch_indices.len() || patch_indices[i] != i as u64 {
                fill_value = encoded_values.get(i).copied();
                break;
            }
        }
    }

    // replace the patched values in the encoded array with the fill value
    // for better downstream compression
    if let Some(fill_value) = fill_value {
        // handle the edge case where the first N >= 1 chunks are all patches
        for patch_idx in &patch_indices {
            encoded_values[*patch_idx as usize] = fill_value;
        }
    }

    let min = *encoded_values
        .iter()
        .min()
        .expect("`encoded_values` shouldn't be all nulls");
    let max = *encoded_values
        .iter()
        .max()
        .expect("`encoded_values` shouldn't be all nulls");
    let sub: <T::UnsignedIntType as ArrowPrimitiveType>::Native = max.sub_wrapping(min).as_();

    let unsigned_encoded_values = encoded_values
        .iter()
        .map(|v| {
            let k: <T::UnsignedIntType as ArrowPrimitiveType>::Native = v.sub_wrapping(min).as_();
            k
        })
        .collect::<Vec<_>>();
    let encoded_output = PrimitiveArray::<<T as LiquidFloatType>::UnsignedIntType>::new(
        ScalarBuffer::from(unsigned_encoded_values),
        nulls.cloned(),
    );

    let bit_width = get_bit_width(sub.as_());
    let bit_packed_array = BitPackedArray::from_primitive(encoded_output, bit_width);

    LiquidFloatArray::<T> {
        bit_packed: bit_packed_array,
        exponent: *exp,
        patch_indices,
        patch_values,
        reference_value: min,
        squeeze_policy: FloatSqueezePolicy::Quantize,
    }
}

fn get_best_exponents<T: LiquidFloatType>(arrow_array: &PrimitiveArray<T>) -> Exponents {
    let mut best_exponents = Exponents { e: 0, f: 0 };
    let mut min_encoded_size: usize = usize::MAX;

    let sample_arrow_array: Option<PrimitiveArray<T>> =
        (arrow_array.len() > NUM_SAMPLES).then(|| {
            arrow_array
                .iter()
                .step_by(arrow_array.len() / NUM_SAMPLES)
                .filter(|s| s.is_some())
                .collect()
        });

    for e in 0..T::MAX_EXPONENT {
        for f in 0..e {
            let exp = Exponents { e, f };
            let liquid_array =
                encode_arrow_array(sample_arrow_array.as_ref().unwrap_or(arrow_array), &exp);
            if liquid_array.get_array_memory_size() < min_encoded_size {
                best_exponents = exp;
                min_encoded_size = liquid_array.get_array_memory_size();
            }
        }
    }
    best_exponents
}

#[derive(Debug)]
struct LiquidFloatQuantizedArray<T: LiquidFloatType> {
    exponent: Exponents,
    quantized: BitPackedArray<T::UnsignedIntType>,
    reference_value: <T::SignedIntType as ArrowPrimitiveType>::Native,
    bucket_width: u8, // Width of each bucket (in bits)
    disk_range: std::ops::Range<u64>,
    patch_indices: Vec<u64>,
    patch_values: Vec<T::Native>,
}

impl<T> LiquidFloatQuantizedArray<T>
where
    T: LiquidFloatType,
{
    #[allow(dead_code)]
    fn as_any(&self) -> &dyn Any {
        self
    }

    #[inline]
    fn len(&self) -> usize {
        self.quantized.len()
    }

    fn new_from_filtered(
        &self,
        filtered: PrimitiveArray<<T as LiquidFloatType>::UnsignedIntType>,
    ) -> Self {
        let bit_width = self
            .quantized
            .bit_width()
            .expect("quantized bit width must exist");
        let quantized = BitPackedArray::from_primitive(filtered, bit_width);
        Self {
            exponent: self.exponent,
            quantized,
            reference_value: self.reference_value,
            bucket_width: self.bucket_width,
            patch_indices: self.patch_indices.clone(),
            patch_values: self.patch_values.clone(),
            disk_range: self.disk_range.clone(),
        }
    }

    fn filter_inner(&self, selection: &BooleanBuffer) -> Self {
        let q_prim: PrimitiveArray<T::UnsignedIntType> = self.quantized.to_primitive();
        let selection = BooleanArray::new(selection.clone(), None);
        let filtered = arrow::compute::kernels::filter::filter(&q_prim, &selection).unwrap();
        let filtered = filtered.as_primitive::<T::UnsignedIntType>().clone();
        self.new_from_filtered(filtered)
    }

    #[inline]
    fn handle_eq(lo: T::Native, hi: T::Native, k: T::Native) -> Option<bool> {
        if k < lo || k > hi { Some(false) } else { None }
    }

    #[inline]
    fn handle_neq(lo: T::Native, hi: T::Native, k: T::Native) -> Option<bool> {
        if k < lo || k > hi { Some(true) } else { None }
    }

    #[inline]
    fn handle_lt(lo: T::Native, hi: T::Native, k: T::Native) -> Option<bool> {
        if k <= lo {
            Some(false)
        } else if hi < k {
            Some(true)
        } else {
            None
        }
    }

    #[inline]
    fn handle_lteq(lo: T::Native, hi: T::Native, k: T::Native) -> Option<bool> {
        if k < lo {
            Some(false)
        } else if hi <= k {
            Some(true)
        } else {
            None
        }
    }

    #[inline]
    fn handle_gt(lo: T::Native, hi: T::Native, k: T::Native) -> Option<bool> {
        if k < lo {
            Some(true)
        } else if hi <= k {
            Some(false)
        } else {
            None
        }
    }

    #[inline]
    fn handle_gteq(lo: T::Native, hi: T::Native, k: T::Native) -> Option<bool> {
        if k <= lo {
            Some(true)
        } else if hi < k {
            Some(false)
        } else {
            None
        }
    }

    fn try_eval_predicate_inner(
        &self,
        op: &Operator,
        literal: &Literal,
    ) -> Result<Option<BooleanArray>, IoRange> {
        // Extract scalar value as T::Native
        let k_opt: Option<T::Native> = match literal.value() {
            ScalarValue::Int8(Some(v)) => T::Native::from_i8(*v),
            ScalarValue::Int16(Some(v)) => T::Native::from_i16(*v),
            ScalarValue::Int32(Some(v)) => T::Native::from_i32(*v),
            ScalarValue::Int64(Some(v)) => T::Native::from_i64(*v),
            ScalarValue::UInt8(Some(v)) => T::Native::from_u8(*v),
            ScalarValue::UInt16(Some(v)) => T::Native::from_u16(*v),
            ScalarValue::UInt32(Some(v)) => T::Native::from_u32(*v),
            ScalarValue::UInt64(Some(v)) => T::Native::from_u64(*v),
            ScalarValue::Date32(Some(v)) => T::Native::from_i32(*v),
            ScalarValue::Date64(Some(v)) => T::Native::from_i64(*v),
            ScalarValue::Float32(Some(v)) => T::Native::from_f32(*v),
            ScalarValue::Float64(Some(v)) => T::Native::from_f64(*v),
            _ => None,
        };
        let Some(k) = k_opt else { return Ok(None) };
        let q_prim = self.quantized.to_primitive();
        let (_dt, values, _nulls) = q_prim.into_parts();

        let mut out_vals: Vec<bool> = Vec::with_capacity(values.len());
        let mut next_patch_index = 0;
        let mut ignore_patches = false;
        if self.patch_indices.is_empty() {
            ignore_patches = true;
        }
        let comp_fn = match op {
            Operator::Eq => Self::handle_eq,
            Operator::NotEq => Self::handle_neq,
            Operator::Lt => Self::handle_lt,
            Operator::LtEq => Self::handle_lteq,
            Operator::Gt => Self::handle_gt,
            Operator::GtEq => Self::handle_gteq,
        };
        // TODO(): This might not be very vectorization-friendly right now. Figure out optimizations
        for (i, &b) in values.iter().enumerate() {
            if let Some(nulls) = self.quantized.nulls()
                && !nulls.is_valid(i)
            {
                out_vals.push(false);
                continue;
            }
            if !ignore_patches && i as u64 == self.patch_indices[next_patch_index] {
                next_patch_index += 1;
                if next_patch_index == self.patch_indices.len() {
                    ignore_patches = true;
                }
                out_vals.push(false);
                continue;
            }

            let val: <T::SignedIntType as ArrowPrimitiveType>::Native = b.as_();
            let lo = (val << self.bucket_width).add_wrapping(self.reference_value);
            let hi = ((val.add_wrapping(1i32.into())) << self.bucket_width)
                .add_wrapping(self.reference_value);
            let val_lower = T::decode_single(&lo, &self.exponent);
            let val_higher = T::decode_single(&hi, &self.exponent);

            let decided = comp_fn(val_lower, val_higher, k);
            if let Some(v) = decided {
                out_vals.push(v);
            } else {
                return Err(IoRange {
                    range: self.disk_range.clone(),
                });
            }
        }

        // Handle patches separately
        // TODO(): Vectorize this
        for (idx, patch_idx) in self.patch_indices.iter().enumerate() {
            let patch_value = self.patch_values[idx];
            out_vals[*patch_idx as usize] = match op {
                Operator::Eq => patch_value == k,
                Operator::NotEq => patch_value != k,
                Operator::Lt => patch_value < k,
                Operator::LtEq => patch_value <= k,
                Operator::Gt => patch_value > k,
                Operator::GtEq => patch_value >= k,
            }
        }

        let bool_buf = arrow::buffer::BooleanBuffer::from_iter(out_vals);
        let out = BooleanArray::new(bool_buf, self.quantized.nulls().cloned());
        Ok(Some(out))
    }
}

impl<T> LiquidHybridArray for LiquidFloatQuantizedArray<T>
where
    T: LiquidFloatType,
{
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn get_array_memory_size(&self) -> usize {
        self.quantized.get_array_memory_size()
            + size_of::<Exponents>()
            + self.patch_indices.capacity() * size_of::<u64>()
            + self.patch_values.capacity() * size_of::<T::Native>()
            + size_of::<<T::SignedIntType as ArrowPrimitiveType>::Native>()
    }

    fn len(&self) -> usize {
        LiquidFloatQuantizedArray::<T>::len(self)
    }

    fn to_arrow_array(&self) -> Result<ArrayRef, super::IoRange> {
        Err(IoRange {
            range: self.disk_range.clone(),
        })
    }

    fn data_type(&self) -> LiquidDataType {
        LiquidDataType::Float
    }

    fn original_arrow_data_type(&self) -> DataType {
        T::DATA_TYPE.clone()
    }

    fn to_bytes(&self) -> Result<Vec<u8>, super::IoRange> {
        Err(IoRange {
            range: self.disk_range.clone(),
        })
    }

    fn filter(&self, selection: &BooleanBuffer) -> Result<LiquidHybridArrayRef, super::IoRange> {
        let filtered = self.filter_inner(selection);
        Ok(Arc::new(filtered) as LiquidHybridArrayRef)
    }

    fn soak(&self, data: bytes::Bytes) -> LiquidArrayRef {
        // `data` is the full IPC payload for primitive array
        let arr = LiquidFloatArray::<T>::from_bytes(data);
        Arc::new(arr)
    }

    fn to_liquid(&self) -> super::IoRange {
        IoRange {
            range: self.disk_range.clone(),
        }
    }

    fn try_eval_predicate(
        &self,
        expr: &Arc<dyn PhysicalExpr>,
        filter: &BooleanBuffer,
    ) -> Result<Option<BooleanArray>, IoRange> {
        // Apply selection first to reduce input rows
        let filtered = self.filter_inner(filter);

        if let Some(binary_expr) = expr.as_any().downcast_ref::<BinaryExpr>()
            && let Some(literal) = binary_expr.right().as_any().downcast_ref::<Literal>()
        {
            let op = binary_expr.op();
            let supported_op = Operator::from_datafusion(op);
            if let Some(supported_op) = supported_op {
                return filtered.try_eval_predicate_inner(&supported_op, literal);
            }
        }
        Ok(None)
    }
}

#[cfg(test)]
mod tests {
    use datafusion::logical_expr::Operator;
    use rand::{Rng as _, SeedableRng as _, distr::uniform::SampleUniform, rngs::StdRng};

    use super::*;
    macro_rules! test_roundtrip {
        ($test_name: ident, $type:ty, $values: expr) => {
            #[test]
            fn $test_name() {
                let original: Vec<Option<<$type as ArrowPrimitiveType>::Native>> = $values;
                let array = PrimitiveArray::<$type>::from(original.clone());

                // Convert to Liquid array and back
                let liquid_array = LiquidFloatArray::<$type>::from_arrow_array(array.clone());
                let result_array = liquid_array.to_arrow_array();
                let bytes_array =
                    LiquidFloatArray::<$type>::from_bytes(liquid_array.to_bytes().into());

                assert_eq!(result_array.as_ref(), &array);
                assert_eq!(bytes_array.to_arrow_array().as_ref(), &array);
            }
        };
    }

    // Test cases for Float32
    test_roundtrip!(
        test_float32_roundtrip_basic,
        Float32Type,
        vec![Some(-1.0), Some(1.0), Some(0.0)]
    );

    test_roundtrip!(
        test_float32_roundtrip_with_nones,
        Float32Type,
        vec![Some(-1.0), Some(1.0), Some(0.0), None]
    );

    test_roundtrip!(
        test_float32_roundtrip_all_nones,
        Float32Type,
        vec![None, None, None, None]
    );

    test_roundtrip!(test_float32_roundtrip_empty, Float32Type, vec![]);

    // Test cases for Float64
    test_roundtrip!(
        test_float64_roundtrip_basic,
        Float64Type,
        vec![Some(-1.0), Some(1.0), Some(0.0)]
    );

    test_roundtrip!(
        test_float64_roundtrip_with_nones,
        Float64Type,
        vec![Some(-1.0), Some(1.0), Some(0.0), None]
    );

    test_roundtrip!(
        test_float64_roundtrip_all_nones,
        Float64Type,
        vec![None, None, None, None]
    );

    test_roundtrip!(test_float64_roundtrip_empty, Float64Type, vec![]);

    // Tests with ilters
    #[test]
    fn test_filter_basic() {
        // Create original array with some values
        let original = vec![Some(1.0), Some(2.1), Some(3.2), None, Some(5.5)];
        let array = PrimitiveArray::<Float32Type>::from(original);
        let liquid_array = LiquidFloatArray::<Float32Type>::from_arrow_array(array);

        // Create selection mask: keep indices 0, 2, and 4
        let selection = BooleanBuffer::from(vec![true, false, true, false, true]);

        // Apply filter
        let filtered = liquid_array.filter(&selection);
        let result_array = filtered.to_arrow_array();

        // Expected result after filtering
        let expected = PrimitiveArray::<Float32Type>::from(vec![Some(1.0), Some(3.2), Some(5.5)]);

        assert_eq!(result_array.as_ref(), &expected);
    }

    #[test]
    fn test_original_arrow_data_type_returns_float32() {
        let array = PrimitiveArray::<Float32Type>::from(vec![Some(1.0), Some(2.5)]);
        let liquid = LiquidFloatArray::<Float32Type>::from_arrow_array(array);
        assert_eq!(liquid.original_arrow_data_type(), DataType::Float32);
    }

    #[test]
    fn test_filter_all_nulls() {
        // Create array with all nulls
        let original = vec![None, None, None, None];
        let array = PrimitiveArray::<Float32Type>::from(original);
        let liquid_array = LiquidFloatArray::<Float32Type>::from_arrow_array(array);

        // Keep first and last elements
        let selection = BooleanBuffer::from(vec![true, false, false, true]);

        let filtered = liquid_array.filter(&selection);
        let result_array = filtered.to_arrow_array();

        let expected = PrimitiveArray::<Float32Type>::from(vec![None, None]);

        assert_eq!(result_array.as_ref(), &expected);
    }

    #[test]
    fn test_filter_empty_result() {
        let original = vec![Some(1.0), Some(2.1), Some(3.3)];
        let array = PrimitiveArray::<Float32Type>::from(original);
        let liquid_array = LiquidFloatArray::<Float32Type>::from_arrow_array(array);

        // Filter out all elements
        let selection = BooleanBuffer::from(vec![false, false, false]);

        let filtered = liquid_array.filter(&selection);
        let result_array = filtered.to_arrow_array();

        assert_eq!(result_array.len(), 0);
    }

    #[test]
    fn test_compression_f32_f64() {
        fn run_compression_test<T: LiquidFloatType>(
            type_name: &str,
            data_fn: impl Fn(usize) -> T::Native,
        ) {
            let original: Vec<T::Native> = (0..2000).map(data_fn).collect();
            let array = PrimitiveArray::<T>::from_iter_values(original);
            let uncompressed_size = array.get_array_memory_size();

            let liquid_array = LiquidFloatArray::<T>::from_arrow_array(array);
            let compressed_size = liquid_array.get_array_memory_size();

            println!(
                "Type: {type_name}, uncompressed_size: {uncompressed_size}, compressed_size: {compressed_size}"
            );
            // Assert that compression actually reduced the size
            assert!(
                compressed_size < uncompressed_size,
                "{type_name} compression failed to reduce size"
            );
        }

        // Run for f32
        run_compression_test::<Float32Type>("f32", |i| i as f32);

        // Run for f64
        run_compression_test::<Float64Type>("f64", |i| i as f64);
    }

    //  --------- Hybrid (squeeze) tests ----------
    fn make_f_array_with_range<T>(
        len: usize,
        base_min: T::Native,
        range: T::Native,
        null_prob: f32,
        rng: &mut StdRng,
    ) -> PrimitiveArray<T>
    where
        T: LiquidFloatType,
        <T as arrow::array::ArrowPrimitiveType>::Native: SampleUniform,
        PrimitiveArray<T>: From<Vec<Option<<T as ArrowPrimitiveType>::Native>>>,
    {
        let mut vals: Vec<Option<T::Native>> = Vec::with_capacity(len);
        for _ in 0..len {
            if rng.random_bool(null_prob as f64) {
                vals.push(None);
            } else {
                vals.push(Some(rng.random_range(base_min..(base_min + range))));
            }
        }
        PrimitiveArray::<T>::from(vals)
    }

    #[test]
    fn hybrid_squeeze_unsqueezable_small_range() {
        let mut rng = StdRng::seed_from_u64(0x51_71);
        let arr = make_f_array_with_range::<Float32Type>(64, 10_000.0, 100.0, 0.1, &mut rng);
        let liquid = LiquidFloatArray::<Float32Type>::from_arrow_array(arr);
        assert!(liquid.squeeze().is_none());
    }

    #[test]
    fn hybrid_squeeze_and_soak_roundtrip_f32() {
        let mut rng = StdRng::seed_from_u64(0x51_72);
        let arr = make_f_array_with_range::<Float32Type>(
            2000,
            -50_000.0,
            (1 << 16) as f32,
            0.1,
            &mut rng,
        );
        let liq = LiquidFloatArray::<Float32Type>::from_arrow_array(arr.clone());
        let bytes_baseline = liq.to_bytes();
        let (hybrid, bytes) = liq.squeeze().expect("squeezable");
        // ensure we can recover the original using soak
        let recovered = hybrid.soak(bytes.clone());
        assert_eq!(
            recovered.to_arrow_array().as_primitive::<Float32Type>(),
            &arr
        );
        assert_eq!(bytes_baseline, recovered.to_bytes());

        let min = arrow::compute::kernels::aggregate::min(&arr).unwrap();
        let mask = BooleanBuffer::from(vec![true; arr.len()]);
        let build_expr =
            |op: Operator, k: f32| -> Arc<dyn datafusion::physical_plan::PhysicalExpr> {
                let lit = Arc::new(Literal::new(ScalarValue::Float32(Some(k))));
                Arc::new(BinaryExpr::new(lit.clone(), op, lit))
            };

        // Expect resolvable results without IO
        let resolvable_cases: Vec<(Operator, f32, bool)> = vec![
            (Operator::Eq, min - 1.0, false),   // eq false everywhere
            (Operator::NotEq, min - 1.0, true), // neq true everywhere
            (Operator::Lt, min, false),         // lt false everywhere
            (Operator::LtEq, min - 1.0, false), // lte false everywhere
            (Operator::Gt, min - 1.0, true),    // gt true everywhere
            (Operator::GtEq, min, true),        // gte true everywhere
        ];

        for (op, k, expected_const) in resolvable_cases {
            let expr = build_expr(op, k);
            let got = hybrid.try_eval_predicate(&expr, &mask).expect("no IO");
            let expected = {
                let vals: Vec<Option<bool>> = (0..arr.len())
                    .map(|i| {
                        if arr.is_null(i) {
                            None
                        } else {
                            Some(expected_const)
                        }
                    })
                    .collect();
                BooleanArray::from(vals)
            };
            assert_eq!(got.unwrap(), expected);
        }

        // Unresolvable for Eq: pick a present value (ensures ambiguous bucket)
        let k_present = (0..arr.len())
            .find_map(|i| {
                if arr.is_null(i) {
                    None
                } else {
                    Some(arr.value(i))
                }
            })
            .unwrap();
        let expr_eq_present = build_expr(Operator::Eq, k_present);
        let err = hybrid
            .try_eval_predicate(&expr_eq_present, &mask)
            .expect_err("quantized should request IO on ambiguous eq bucket");
        assert_eq!(err.range(), hybrid.to_liquid().range());
    }

    #[test]
    fn hybrid_squeeze_and_soak_roundtrip_f64() {
        let mut rng = StdRng::seed_from_u64(0x51_72);
        let arr = make_f_array_with_range::<Float64Type>(
            2000,
            -50_000.0f64,
            (1 << 16) as f64,
            0.1,
            &mut rng,
        );
        let liq = LiquidFloatArray::<Float64Type>::from_arrow_array(arr.clone());
        let bytes_baseline = liq.to_bytes();
        let (hybrid, bytes) = liq.squeeze().expect("squeezable");
        // ensure we can recover the original using soak
        let recovered = hybrid.soak(bytes.clone());
        assert_eq!(
            recovered.to_arrow_array().as_primitive::<Float64Type>(),
            &arr
        );
        assert_eq!(bytes_baseline, recovered.to_bytes());

        let min = arrow::compute::kernels::aggregate::min(&arr).unwrap();
        let mask = BooleanBuffer::from(vec![true; arr.len()]);
        let build_expr =
            |op: Operator, k: f64| -> Arc<dyn datafusion::physical_plan::PhysicalExpr> {
                let lit = Arc::new(Literal::new(ScalarValue::Float64(Some(k))));
                Arc::new(BinaryExpr::new(lit.clone(), op, lit))
            };

        // Expect resolvable results without IO
        let resolvable_cases: Vec<(Operator, f64, bool)> = vec![
            (Operator::Eq, min - 1.0, false),   // eq false everywhere
            (Operator::NotEq, min - 1.0, true), // neq true everywhere
            (Operator::Lt, min, false),         // lt false everywhere
            (Operator::LtEq, min - 1.0, false), // lte false everywhere
            (Operator::Gt, min - 1.0, true),    // gt true everywhere
            (Operator::GtEq, min, true),        // gte true everywhere
        ];

        for (op, k, expected_const) in resolvable_cases {
            let expr = build_expr(op, k);
            let got = hybrid.try_eval_predicate(&expr, &mask).expect("no IO");
            let expected = {
                let vals: Vec<Option<bool>> = (0..arr.len())
                    .map(|i| {
                        if arr.is_null(i) {
                            None
                        } else {
                            Some(expected_const)
                        }
                    })
                    .collect();
                BooleanArray::from(vals)
            };
            assert_eq!(got.unwrap(), expected);
        }

        // Unresolvable for Eq: pick a present value (ensures ambiguous bucket)
        let k_present = (0..arr.len())
            .find_map(|i| {
                if arr.is_null(i) {
                    None
                } else {
                    Some(arr.value(i))
                }
            })
            .unwrap();
        let expr_eq_present = build_expr(Operator::Eq, k_present);
        let err = hybrid
            .try_eval_predicate(&expr_eq_present, &mask)
            .expect_err("quantized should request IO on ambiguous eq bucket");
        assert_eq!(err.range(), hybrid.to_liquid().range());
    }
}
