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
use crate::liquid_array::ipc::LiquidIPCHeader;
use crate::liquid_array::ipc::get_physical_type_id;
use crate::liquid_array::raw::BitPackedArray;
use crate::liquid_array::{
    IoRange, LiquidArray, LiquidArrayRef, LiquidHybridArray, LiquidHybridArrayRef, Operator,
    PrimitiveKind,
};
use crate::utils::get_bit_width;
use arrow::datatypes::ArrowNativeType;
use bytes::Bytes;
use datafusion::physical_plan::expressions::{BinaryExpr, Literal};

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

    fn squeeze(&self) -> Option<(crate::liquid_array::LiquidHybridArrayRef, bytes::Bytes)> {
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
                Some((
                    Arc::new(hybrid) as crate::liquid_array::LiquidHybridArrayRef,
                    full_bytes,
                ))
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

#[derive(Debug, Clone)]
struct LiquidPrimitiveClampedArray<T: LiquidPrimitiveType> {
    squeezed: BitPackedArray<T::UnSignedType>,
    reference_value: T::Native,
    // Range in the on-disk payload needed to reconstruct the full array (we use full bytes)
    disk_range: std::ops::Range<u64>,
}

impl<T> LiquidPrimitiveClampedArray<T>
where
    T: LiquidPrimitiveType + super::PrimitiveKind,
{
    #[inline]
    fn len(&self) -> usize {
        self.squeezed.len()
    }

    fn new_from_filtered(
        &self,
        filtered: PrimitiveArray<<T as LiquidPrimitiveType>::UnSignedType>,
    ) -> Self {
        let bit_width = self
            .squeezed
            .bit_width()
            .expect("squeezed bit width must exist");
        let squeezed = BitPackedArray::from_primitive(filtered, bit_width);
        Self {
            squeezed,
            reference_value: self.reference_value,
            disk_range: self.disk_range.clone(),
        }
    }

    fn filter_inner(&self, selection: &BooleanBuffer) -> Self {
        let unsigned_array: PrimitiveArray<T::UnSignedType> = self.squeezed.to_primitive();
        let selection = BooleanArray::new(selection.clone(), None);
        let filtered_values =
            arrow::compute::kernels::filter::filter(&unsigned_array, &selection).unwrap();
        let filtered_values = filtered_values.as_primitive::<T::UnSignedType>().clone();
        self.new_from_filtered(filtered_values)
    }

    fn to_arrow_known_only(&self) -> Option<ArrayRef> {
        // Convert squeezed to primitive and ensure no sentinel exists.
        type U<TT> = <<TT as LiquidPrimitiveType>::UnSignedType as ArrowPrimitiveType>::Native;
        let squeezed_prim = self.squeezed.to_primitive();
        let (_dt, values, nulls) = squeezed_prim.into_parts();
        let bw = self.squeezed.bit_width().expect("bit width").get();
        let sentinel: U<T> = U::<T>::usize_as((1usize << bw) - 1);

        // If any valid value equals sentinel, cannot fully materialize without disk
        if let Some(n) = self.squeezed.nulls() {
            for (i, v) in values.iter().enumerate() {
                if n.is_valid(i) && *v == sentinel {
                    return None;
                }
            }
        } else if values.contains(&sentinel) {
            return None;
        }

        // All values are known; reconstruct to full Arrow by adding reference
        let ref_u: U<T> = self.reference_value.as_();
        let restored_vals: ScalarBuffer<T::Native> =
            ScalarBuffer::from_iter(values.iter().map(|&u| {
                let t_val: T::Native = u.add_wrapping(ref_u).as_();
                t_val
            }));
        let arr = PrimitiveArray::<T>::new(restored_vals, nulls);
        Some(Arc::new(arr))
    }

    // Evaluate a simple comparison if fully decidable without disk; otherwise return Err(IoRange)
    fn try_eval_predicate_inner(
        &self,
        op: &Operator,
        literal: &Literal,
    ) -> Result<Option<BooleanArray>, crate::liquid_array::IoRange> {
        use datafusion::common::ScalarValue;

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
            _ => None,
        };
        let Some(k) = k_opt else { return Ok(None) };

        // Prepare squeezed data and thresholds
        type U<TT> = <<TT as LiquidPrimitiveType>::UnSignedType as ArrowPrimitiveType>::Native;
        let squeezed_prim = self.squeezed.to_primitive();
        let (_dt, values, _nulls) = squeezed_prim.into_parts();
        let bw = self.squeezed.bit_width().expect("bit width").get();
        let sentinel: U<T> = U::<T>::usize_as((1usize << bw) - 1);

        // Precompute whether sentinel rows can be resolved under this operator and literal
        let is_unsigned = <T as super::PrimitiveKind>::IS_UNSIGNED;
        let resolves_on_sentinel: bool = if is_unsigned {
            let ref_u: U<T> = self.reference_value.as_();
            let k_u: U<T> = k.as_();
            let ref_u64: u64 = num_traits::AsPrimitive::<u64>::as_(ref_u);
            let sent_u64: u64 = num_traits::AsPrimitive::<u64>::as_(sentinel);
            let k_u64: u64 = num_traits::AsPrimitive::<u64>::as_(k_u);
            let sent_abs: u64 = ref_u64 + sent_u64;
            match op {
                Operator::Eq | Operator::NotEq | Operator::Gt | Operator::LtEq => k_u64 < sent_abs,
                Operator::Lt | Operator::GtEq => k_u64 <= sent_abs,
            }
        } else {
            // signed types (including Date32/Date64)
            let ref_i: i64 = self.reference_value.as_();
            let k_i: i64 = k.as_();
            let sent_abs: i64 = ref_i + (num_traits::AsPrimitive::<u64>::as_(sentinel) as i64);
            match op {
                Operator::Eq | Operator::NotEq | Operator::Gt | Operator::LtEq => k_i < sent_abs,
                Operator::Lt | Operator::GtEq => k_i <= sent_abs,
            }
        };

        // Build boolean values in a single pass; if an unresolved sentinel is seen, return IO range
        let ref_u: U<T> = self.reference_value.as_();
        let k_t: T::Native = k;
        let mut out_vals: Vec<bool> = Vec::with_capacity(values.len());
        if let Some(n) = self.squeezed.nulls() {
            for (i, &u) in values.iter().enumerate() {
                if !n.is_valid(i) {
                    out_vals.push(false);
                    continue;
                }
                if u == sentinel {
                    if !resolves_on_sentinel {
                        return Err(crate::liquid_array::IoRange {
                            range: self.disk_range.clone(),
                        });
                    }
                    let b = match op {
                        Operator::Eq => false,
                        Operator::NotEq => true,
                        Operator::Lt => false,
                        Operator::LtEq => false,
                        Operator::Gt => true,
                        Operator::GtEq => true,
                    };
                    out_vals.push(b);
                } else {
                    let actual: T::Native = u.add_wrapping(ref_u).as_();
                    let b = match op {
                        Operator::Eq => actual == k_t,
                        Operator::NotEq => actual != k_t,
                        Operator::Lt => actual < k_t,
                        Operator::LtEq => actual <= k_t,
                        Operator::Gt => actual > k_t,
                        Operator::GtEq => actual >= k_t,
                    };
                    out_vals.push(b);
                }
            }
        } else {
            for &u in values.iter() {
                if u == sentinel {
                    if !resolves_on_sentinel {
                        return Err(crate::liquid_array::IoRange {
                            range: self.disk_range.clone(),
                        });
                    }
                    let b = match op {
                        Operator::Eq => false,
                        Operator::NotEq => true,
                        Operator::Lt => false,
                        Operator::LtEq => false,
                        Operator::Gt => true,
                        Operator::GtEq => true,
                    };
                    out_vals.push(b);
                } else {
                    let actual: T::Native = u.add_wrapping(ref_u).as_();
                    let b = match op {
                        Operator::Eq => actual == k_t,
                        Operator::NotEq => actual != k_t,
                        Operator::Lt => actual < k_t,
                        Operator::LtEq => actual <= k_t,
                        Operator::Gt => actual > k_t,
                        Operator::GtEq => actual >= k_t,
                    };
                    out_vals.push(b);
                }
            }
        }

        let bool_buf = arrow::buffer::BooleanBuffer::from_iter(out_vals);
        let out = BooleanArray::new(bool_buf, self.squeezed.nulls().cloned());
        Ok(Some(out))
    }
}

impl<T> LiquidHybridArray for LiquidPrimitiveClampedArray<T>
where
    T: LiquidPrimitiveType,
{
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn get_array_memory_size(&self) -> usize {
        self.squeezed.get_array_memory_size() + std::mem::size_of::<T::Native>()
    }

    fn len(&self) -> usize {
        LiquidPrimitiveClampedArray::<T>::len(self)
    }

    fn to_arrow_array(&self) -> Result<ArrayRef, IoRange> {
        if let Some(arr) = self.to_arrow_known_only() {
            Ok(arr)
        } else {
            Err(IoRange {
                range: self.disk_range.clone(),
            })
        }
    }

    fn data_type(&self) -> LiquidDataType {
        LiquidDataType::Integer
    }

    fn original_arrow_data_type(&self) -> DataType {
        T::DATA_TYPE.clone()
    }

    fn to_bytes(&self) -> Result<Vec<u8>, IoRange> {
        Err(IoRange {
            range: self.disk_range.clone(),
        })
    }

    fn filter(&self, selection: &BooleanBuffer) -> Result<LiquidHybridArrayRef, IoRange> {
        let filtered = self.filter_inner(selection);
        Ok(Arc::new(filtered) as LiquidHybridArrayRef)
    }

    fn filter_to_arrow(&self, selection: &BooleanBuffer) -> Result<ArrayRef, IoRange> {
        let filtered = self.filter_inner(selection);
        filtered.to_arrow_array()
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

    fn soak(&self, data: bytes::Bytes) -> LiquidArrayRef {
        // `data` is the full IPC payload for primitive array
        let arr = LiquidPrimitiveArray::<T>::from_bytes(data);
        Arc::new(arr)
    }

    fn to_liquid(&self) -> IoRange {
        IoRange {
            range: self.disk_range.clone(),
        }
    }
}

// Quantized hybrid array: stores bucket indices of value offsets
#[derive(Debug, Clone)]
struct LiquidPrimitiveQuantizedArray<T: LiquidPrimitiveType> {
    quantized: BitPackedArray<T::UnSignedType>,
    reference_value: T::Native,
    // bucket width in terms of absolute offset units
    bucket_width: u64,
    disk_range: std::ops::Range<u64>,
}

impl<T> LiquidPrimitiveQuantizedArray<T>
where
    T: LiquidPrimitiveType + super::PrimitiveKind,
{
    #[inline]
    fn len(&self) -> usize {
        self.quantized.len()
    }

    fn new_from_filtered(
        &self,
        filtered: PrimitiveArray<<T as LiquidPrimitiveType>::UnSignedType>,
    ) -> Self {
        let bit_width = self
            .quantized
            .bit_width()
            .expect("quantized bit width must exist");
        let quantized = BitPackedArray::from_primitive(filtered, bit_width);
        Self {
            quantized,
            reference_value: self.reference_value,
            bucket_width: self.bucket_width,
            disk_range: self.disk_range.clone(),
        }
    }

    fn filter_inner(&self, selection: &BooleanBuffer) -> Self {
        let q_prim: PrimitiveArray<T::UnSignedType> = self.quantized.to_primitive();
        let selection = BooleanArray::new(selection.clone(), None);
        let filtered = arrow::compute::kernels::filter::filter(&q_prim, &selection).unwrap();
        let filtered = filtered.as_primitive::<T::UnSignedType>().clone();
        self.new_from_filtered(filtered)
    }

    // Evaluate using bucket interval semantics; return Err if any ambiguous bucket is encountered
    fn try_eval_predicate_inner(
        &self,
        op: &Operator,
        literal: &Literal,
    ) -> Result<Option<BooleanArray>, IoRange> {
        use datafusion::common::ScalarValue;
        type U<TT> = <<TT as LiquidPrimitiveType>::UnSignedType as ArrowPrimitiveType>::Native;

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
            _ => None,
        };
        let Some(k) = k_opt else { return Ok(None) };

        let q_prim = self.quantized.to_primitive();
        let (_dt, values, _nulls) = q_prim.into_parts();

        let mut out_vals: Vec<bool> = Vec::with_capacity(values.len());

        if T::IS_UNSIGNED {
            let ref_u_native: U<T> = self.reference_value.as_();
            let ref_u: u64 = num_traits::AsPrimitive::<u64>::as_(ref_u_native);
            let k_u_native: U<T> = k.as_();
            let k_u: u64 = num_traits::AsPrimitive::<u64>::as_(k_u_native);
            for (i, &b) in values.iter().enumerate() {
                if let Some(nulls) = self.quantized.nulls()
                    && !nulls.is_valid(i)
                {
                    out_vals.push(false);
                    continue;
                }
                let b_u64: u64 = num_traits::AsPrimitive::<u64>::as_(b);
                let lo = ref_u.saturating_add(b_u64.saturating_mul(self.bucket_width));
                let hi = ref_u
                    .saturating_add((b_u64 + 1).saturating_mul(self.bucket_width))
                    .saturating_sub(1);
                let decided = match op {
                    Operator::Eq => {
                        if k_u < lo || k_u > hi {
                            Some(false)
                        } else {
                            None
                        }
                    }
                    Operator::NotEq => {
                        if k_u < lo || k_u > hi {
                            Some(true)
                        } else {
                            None
                        }
                    }
                    Operator::Lt => {
                        if hi < k_u {
                            Some(true)
                        } else if lo >= k_u {
                            Some(false)
                        } else {
                            None
                        }
                    }
                    Operator::LtEq => {
                        if hi <= k_u {
                            Some(true)
                        } else if lo > k_u {
                            Some(false)
                        } else {
                            None
                        }
                    }
                    Operator::Gt => {
                        if lo > k_u {
                            Some(true)
                        } else if hi <= k_u {
                            Some(false)
                        } else {
                            None
                        }
                    }
                    Operator::GtEq => {
                        if lo >= k_u {
                            Some(true)
                        } else if hi < k_u {
                            Some(false)
                        } else {
                            None
                        }
                    }
                };
                if let Some(v) = decided {
                    out_vals.push(v);
                } else {
                    return Err(IoRange {
                        range: self.disk_range.clone(),
                    });
                }
            }
        } else {
            let ref_i64: i64 = self.reference_value.as_();
            let ref_i: i128 = ref_i64 as i128;
            let k_i64: i64 = k.as_();
            let k_i: i128 = k_i64 as i128;
            let bw = self.bucket_width as i128;
            for (i, &b) in values.iter().enumerate() {
                if let Some(nulls) = self.quantized.nulls()
                    && !nulls.is_valid(i)
                {
                    out_vals.push(false);
                    continue;
                }
                let b_i: i128 = num_traits::AsPrimitive::<u64>::as_(b) as i128;
                let lo = ref_i + b_i.saturating_mul(bw);
                let hi = ref_i + (b_i + 1).saturating_mul(bw) - 1;
                let decided = match op {
                    Operator::Eq => {
                        if k_i < lo || k_i > hi {
                            Some(false)
                        } else {
                            None
                        }
                    }
                    Operator::NotEq => {
                        if k_i < lo || k_i > hi {
                            Some(true)
                        } else {
                            None
                        }
                    }
                    Operator::Lt => {
                        if hi < k_i {
                            Some(true)
                        } else if lo >= k_i {
                            Some(false)
                        } else {
                            None
                        }
                    }
                    Operator::LtEq => {
                        if hi <= k_i {
                            Some(true)
                        } else if lo > k_i {
                            Some(false)
                        } else {
                            None
                        }
                    }
                    Operator::Gt => {
                        if lo > k_i {
                            Some(true)
                        } else if hi <= k_i {
                            Some(false)
                        } else {
                            None
                        }
                    }
                    Operator::GtEq => {
                        if lo >= k_i {
                            Some(true)
                        } else if hi < k_i {
                            Some(false)
                        } else {
                            None
                        }
                    }
                };
                if let Some(v) = decided {
                    out_vals.push(v);
                } else {
                    return Err(IoRange {
                        range: self.disk_range.clone(),
                    });
                }
            }
        }

        let bool_buf = arrow::buffer::BooleanBuffer::from_iter(out_vals);
        let out = BooleanArray::new(bool_buf, self.quantized.nulls().cloned());
        Ok(Some(out))
    }
}

impl<T> LiquidHybridArray for LiquidPrimitiveQuantizedArray<T>
where
    T: LiquidPrimitiveType + super::PrimitiveKind,
{
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn get_array_memory_size(&self) -> usize {
        self.quantized.get_array_memory_size() + std::mem::size_of::<T::Native>()
    }

    fn len(&self) -> usize {
        LiquidPrimitiveQuantizedArray::<T>::len(self)
    }

    fn to_arrow_array(&self) -> Result<ArrayRef, IoRange> {
        Err(IoRange {
            range: self.disk_range.clone(),
        })
    }

    fn data_type(&self) -> LiquidDataType {
        LiquidDataType::Integer
    }

    fn original_arrow_data_type(&self) -> DataType {
        T::DATA_TYPE.clone()
    }

    fn to_bytes(&self) -> Result<Vec<u8>, IoRange> {
        Err(IoRange {
            range: self.disk_range.clone(),
        })
    }

    fn filter(&self, selection: &BooleanBuffer) -> Result<LiquidHybridArrayRef, IoRange> {
        let filtered = self.filter_inner(selection);
        Ok(Arc::new(filtered) as LiquidHybridArrayRef)
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

    fn soak(&self, data: bytes::Bytes) -> LiquidArrayRef {
        // `data` is the full IPC payload for primitive array
        let arr = LiquidPrimitiveArray::<T>::from_bytes(data);
        Arc::new(arr)
    }

    fn to_liquid(&self) -> IoRange {
        IoRange {
            range: self.disk_range.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::Array;
    use datafusion::logical_expr::Operator;
    use datafusion::physical_plan::expressions::{BinaryExpr, Literal};
    use datafusion::scalar::ScalarValue;
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

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

    // ---------- Hybrid (squeeze) tests ----------

    fn make_i32_array_with_range(
        len: usize,
        base_min: i32,
        range: i32,
        null_prob: f32,
        rng: &mut StdRng,
    ) -> PrimitiveArray<Int32Type> {
        let mut vals: Vec<Option<i32>> = Vec::with_capacity(len);
        for _ in 0..len {
            if rng.random_bool(null_prob as f64) {
                vals.push(None);
            } else {
                let delta = rng.random_range(0..=range);
                vals.push(Some(base_min.saturating_add(delta)));
            }
        }
        PrimitiveArray::<Int32Type>::from(vals)
    }

    fn make_u32_array_with_range(
        len: usize,
        base_min: u32,
        range: u32,
        null_prob: f32,
        rng: &mut StdRng,
    ) -> PrimitiveArray<UInt32Type> {
        let mut vals: Vec<Option<u32>> = Vec::with_capacity(len);
        for _ in 0..len {
            if rng.random_bool(null_prob as f64) {
                vals.push(None);
            } else {
                let delta = rng.random_range(0..=range);
                vals.push(Some(base_min.saturating_add(delta)));
            }
        }
        PrimitiveArray::<UInt32Type>::from(vals)
    }

    fn compute_boundary_i32(arr: &PrimitiveArray<Int32Type>) -> Option<i32> {
        // boundary = min + ((1 << (bit_width(range)/2)) - 1)
        let min = arrow::compute::kernels::aggregate::min(arr)?;
        let max = arrow::compute::kernels::aggregate::max(arr)?;
        let range = (max as i64 - min as i64) as u64;
        let bw = crate::utils::get_bit_width(range);
        let half = (bw.get() / 2) as u32;
        let sentinel = if half == 0 { 0 } else { (1u64 << half) - 1 } as i64;
        (min as i64 + sentinel).try_into().ok()
    }

    fn compute_boundary_u32(arr: &PrimitiveArray<UInt32Type>) -> Option<u32> {
        let min = arrow::compute::kernels::aggregate::min(arr)?;
        let max = arrow::compute::kernels::aggregate::max(arr)?;
        let range = (max as u128 - min as u128) as u64;
        let bw = crate::utils::get_bit_width(range);
        let half = (bw.get() / 2) as u32;
        let sentinel = if half == 0 { 0 } else { (1u128 << half) - 1 } as u128;
        let b = (min as u128 + sentinel) as u64 as u32;
        Some(b)
    }

    #[test]
    fn hybrid_squeeze_unsqueezable_small_range() {
        // range < 512 -> bit width < 10 => None
        let mut rng = StdRng::seed_from_u64(0x51_71);
        let arr = make_i32_array_with_range(64, 10_000, 100, 0.1, &mut rng);
        let liquid = LiquidPrimitiveArray::<Int32Type>::from_arrow_array(arr);
        assert!(liquid.squeeze().is_none());
    }

    #[test]
    fn hybrid_squeeze_and_soak_roundtrip_i32() {
        let mut rng = StdRng::seed_from_u64(0x51_72);
        let arr = make_i32_array_with_range(128, -50_000, 1 << 16, 0.1, &mut rng);
        let liq = LiquidPrimitiveArray::<Int32Type>::from_arrow_array(arr.clone());
        let bytes_baseline = liq.to_bytes();
        let (hybrid, bytes) = liq.squeeze().expect("squeezable");
        // ensure we can recover the original using soak
        let recovered = hybrid.soak(bytes.clone());
        assert_eq!(recovered.to_arrow_array().as_primitive::<Int32Type>(), &arr);
        assert_eq!(bytes_baseline, recovered.to_bytes());

        // If we filter to only known values, hybrid can materialize without IO
        let boundary = compute_boundary_i32(&arr).unwrap();
        let mask_bits: Vec<bool> = (0..arr.len())
            .map(|i| {
                if arr.is_null(i) {
                    true
                } else {
                    arr.value(i) < boundary
                }
            })
            .collect();
        let mask = BooleanBuffer::from_iter(mask_bits.iter().copied());
        let filtered_arrow = hybrid
            .filter_to_arrow(&mask)
            .expect("known-only selection should be materializable");

        let expected = {
            let vals: Vec<Option<i32>> = (0..arr.len())
                .zip(mask_bits.iter())
                .filter(|&(_, &keep)| keep)
                .map(|(i, &_keep)| {
                    if arr.is_null(i) {
                        None
                    } else {
                        Some(arr.value(i))
                    }
                })
                .collect();
            PrimitiveArray::<Int32Type>::from(vals)
        };
        assert_eq!(filtered_arrow.as_primitive::<Int32Type>(), &expected);
    }

    #[test]
    fn hybrid_predicate_eval_i32_resolvable_and_unresolvable() {
        let mut rng = StdRng::seed_from_u64(0x51_73);
        let arr = make_i32_array_with_range(200, -1_000_000, 1 << 16, 0.2, &mut rng);
        let liq = LiquidPrimitiveArray::<Int32Type>::from_arrow_array(arr.clone());
        let (hybrid, _bytes) = liq.squeeze().expect("squeezable");

        let boundary = compute_boundary_i32(&arr).unwrap();
        // selection mask: random subset
        let mask_bits: Vec<bool> = (0..arr.len()).map(|_| rng.random()).collect();
        let mask = BooleanBuffer::from_iter(mask_bits.iter().copied());

        let build_expr =
            |op: Operator, k: i32| -> Arc<dyn datafusion::physical_plan::PhysicalExpr> {
                let lit = Arc::new(Literal::new(ScalarValue::Int32(Some(k))));
                Arc::new(BinaryExpr::new(lit.clone(), op, lit))
            };

        // Helper to compute expected boolean array on selected rows
        let expected_for = |op: Operator, k: i32| -> BooleanArray {
            let vals: Vec<Option<bool>> = (0..arr.len())
                .zip(mask_bits.iter())
                .filter(|&(_, &keep)| keep)
                .map(|(i, &_keep)| {
                    if arr.is_null(i) {
                        None
                    } else {
                        let v = arr.value(i);
                        Some(match op {
                            Operator::Eq => v == k,
                            Operator::NotEq => v != k,
                            Operator::Lt => v < k,
                            Operator::LtEq => v <= k,
                            Operator::Gt => v > k,
                            Operator::GtEq => v >= k,
                            _ => unreachable!(),
                        })
                    }
                })
                .collect();
            BooleanArray::from(vals)
        };

        // Resolvable cases: K strictly less than boundary for Eq,Neq,LtEq,Gt; K <= boundary for Lt,GtEq
        let resolvable_cases: Vec<(Operator, i32)> = vec![
            (Operator::Eq, boundary - 1),
            (Operator::NotEq, boundary - 1),
            (Operator::Lt, boundary),
            (Operator::LtEq, boundary - 1),
            (Operator::Gt, boundary - 1),
            (Operator::GtEq, boundary),
        ];

        for (op, k) in resolvable_cases {
            let expr = build_expr(op, k);
            let got = hybrid.try_eval_predicate(&expr, &mask).expect("no IO");
            let expected = expected_for(op, k);
            assert_eq!(got.unwrap(), expected);
        }

        // Unresolvable: choose constants >= boundary for ops that require disk
        let unresolvable_cases: Vec<(Operator, i32)> = vec![
            (Operator::Eq, boundary),
            (Operator::NotEq, boundary),
            (Operator::Lt, boundary + 1),
            (Operator::LtEq, boundary),
            (Operator::Gt, boundary + 1),
            (Operator::GtEq, boundary + 1),
        ];
        for (op, k) in unresolvable_cases {
            let expr = build_expr(op, k);
            let err = hybrid
                .try_eval_predicate(&expr, &mask)
                .expect_err("should request IO");
            let io = hybrid.to_liquid();
            assert_eq!(err.range(), io.range());
        }
    }

    #[test]
    fn hybrid_predicate_eval_u32_resolvable_and_unresolvable() {
        let mut rng = StdRng::seed_from_u64(0x51_74);
        let arr = make_u32_array_with_range(180, 1_000_000, 1 << 16, 0.15, &mut rng);
        let liq = LiquidPrimitiveArray::<UInt32Type>::from_arrow_array(arr.clone())
            .with_squeeze_policy(IntegerSqueezePolicy::Clamp);
        let (hybrid, _bytes) = liq.squeeze().expect("squeezable");

        let boundary = compute_boundary_u32(&arr).unwrap();
        let mask_bits: Vec<bool> = (0..arr.len()).map(|_| rng.random()).collect();
        let mask = BooleanBuffer::from_iter(mask_bits.iter().copied());

        let build_expr =
            |op: Operator, k: u32| -> Arc<dyn datafusion::physical_plan::PhysicalExpr> {
                let lit = Arc::new(Literal::new(ScalarValue::UInt32(Some(k))));
                Arc::new(BinaryExpr::new(lit.clone(), op, lit))
            };

        let expected_for = |op: Operator, k: u32| -> BooleanArray {
            let vals: Vec<Option<bool>> = (0..arr.len())
                .zip(mask_bits.iter())
                .filter(|&(_, &keep)| keep)
                .map(|(i, &_keep)| {
                    if arr.is_null(i) {
                        None
                    } else {
                        let v = arr.value(i);
                        Some(match op {
                            Operator::Eq => v == k,
                            Operator::NotEq => v != k,
                            Operator::Lt => v < k,
                            Operator::LtEq => v <= k,
                            Operator::Gt => v > k,
                            Operator::GtEq => v >= k,
                            _ => unreachable!(),
                        })
                    }
                })
                .collect();
            BooleanArray::from(vals)
        };

        let resolvable_cases: Vec<(Operator, u32)> = vec![
            (Operator::Eq, boundary - 1),
            (Operator::NotEq, boundary - 1),
            (Operator::Lt, boundary),
            (Operator::LtEq, boundary - 1),
            (Operator::Gt, boundary - 1),
            (Operator::GtEq, boundary),
        ];
        for (op, k) in resolvable_cases {
            let expr = build_expr(op, k);
            let got = hybrid.try_eval_predicate(&expr, &mask).expect("no IO");
            let expected = expected_for(op, k);
            assert_eq!(got.unwrap(), expected);
        }

        let unresolvable_cases: Vec<(Operator, u32)> = vec![
            (Operator::Eq, boundary),
            (Operator::NotEq, boundary),
            (Operator::Lt, boundary + 1),
            (Operator::LtEq, boundary),
            (Operator::Gt, boundary + 1),
            (Operator::GtEq, boundary + 1),
        ];
        for (op, k) in unresolvable_cases {
            let expr = build_expr(op, k);
            let err = hybrid
                .try_eval_predicate(&expr, &mask)
                .expect_err("should request IO");
            assert_eq!(err.range(), hybrid.to_liquid().range());
        }
    }

    #[test]
    fn quantized_predicate_eval_u32_resolvable_and_unresolvable() {
        let mut rng = StdRng::seed_from_u64(0x51_84);
        let arr = make_u32_array_with_range(200, 1_000_000, 1 << 16, 0.2, &mut rng);
        let liq = LiquidPrimitiveArray::<UInt32Type>::from_arrow_array(arr.clone())
            .with_squeeze_policy(IntegerSqueezePolicy::Quantize);
        let (hybrid, _bytes) = liq.squeeze().expect("squeezable");

        let min = arrow::compute::kernels::aggregate::min(&arr).unwrap();

        let mask = BooleanBuffer::from(vec![true; arr.len()]);
        let build_expr =
            |op: Operator, k: u32| -> Arc<dyn datafusion::physical_plan::PhysicalExpr> {
                let lit = Arc::new(Literal::new(ScalarValue::UInt32(Some(k))));
                Arc::new(BinaryExpr::new(lit.clone(), op, lit))
            };

        // Expect resolvable results without IO
        let resolvable_cases: Vec<(Operator, u32, bool)> = vec![
            (Operator::Eq, min.saturating_sub(1), false), // eq false everywhere
            (Operator::NotEq, min.saturating_sub(1), true), // neq true everywhere
            (Operator::Lt, min, false),                   // lt false everywhere
            (Operator::LtEq, min.saturating_sub(1), false), // lte false everywhere
            (Operator::Gt, min.saturating_sub(1), true),  // gt true everywhere
            (Operator::GtEq, min, true),                  // gte true everywhere
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
    fn quantized_predicate_eval_i32_resolvable_and_unresolvable() {
        let mut rng = StdRng::seed_from_u64(0x51_85);
        let arr = make_i32_array_with_range(220, -1_000_000, 1 << 16, 0.2, &mut rng);
        let liq = LiquidPrimitiveArray::<Int32Type>::from_arrow_array(arr.clone())
            .with_squeeze_policy(IntegerSqueezePolicy::Quantize);
        let (hybrid, _bytes) = liq.squeeze().expect("squeezable");

        let min = arrow::compute::kernels::aggregate::min(&arr).unwrap();
        let mask = BooleanBuffer::from(vec![true; arr.len()]);
        let build_expr =
            |op: Operator, k: i32| -> Arc<dyn datafusion::physical_plan::PhysicalExpr> {
                let lit = Arc::new(Literal::new(ScalarValue::Int32(Some(k))));
                Arc::new(BinaryExpr::new(lit.clone(), op, lit))
            };

        let resolvable_cases: Vec<(Operator, i32, bool)> = vec![
            (Operator::Eq, min - 1, false), // eq false everywhere
            (Operator::NotEq, min - 1, true),
            (Operator::Lt, min, false),
            (Operator::LtEq, min - 1, false),
            (Operator::Gt, min - 1, true),
            (Operator::GtEq, min, true),
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

        // Unresolvable for Eq: pick a present value
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
