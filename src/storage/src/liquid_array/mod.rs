//! LiquidArray is the core data structure of LiquidCache.
//! You should not use this module directly.
//! Instead, use `liquid_cache_server` or `liquid_cache_client` to interact with LiquidCache.
mod byte_array;
pub mod byte_view_array;
mod fix_len_byte_array;
mod float_array;
pub mod ipc;
mod linear_integer_array;
mod primitive_array;
pub mod raw;
#[cfg(test)]
mod tests;
pub(crate) mod utils;

use std::{any::Any, ops::Range, sync::Arc};

use arrow::{
    array::{ArrayRef, BooleanArray},
    buffer::BooleanBuffer,
};
pub use byte_array::{LiquidByteArray, get_string_needle};
pub use byte_view_array::LiquidByteViewArray;
use datafusion::physical_plan::PhysicalExpr;
pub use fix_len_byte_array::LiquidFixedLenByteArray;
use float_array::LiquidFloatType;
pub use float_array::{LiquidFloat32Array, LiquidFloat64Array, LiquidFloatArray};
pub use linear_integer_array::{
    LiquidLinearArray, LiquidLinearDate32Array, LiquidLinearDate64Array, LiquidLinearI8Array,
    LiquidLinearI16Array, LiquidLinearI32Array, LiquidLinearI64Array, LiquidLinearU8Array,
    LiquidLinearU16Array, LiquidLinearU32Array, LiquidLinearU64Array,
};
pub use primitive_array::IntegerSqueezePolicy;
pub use primitive_array::{
    LiquidDate32Array, LiquidDate64Array, LiquidI8Array, LiquidI16Array, LiquidI32Array,
    LiquidI64Array, LiquidPrimitiveArray, LiquidPrimitiveType, LiquidU8Array, LiquidU16Array,
    LiquidU32Array, LiquidU64Array,
};

use crate::liquid_array::byte_view_array::MemoryBuffer;

/// Liquid data type is only logical type
#[derive(Debug, Clone, Copy)]
#[repr(u16)]
pub enum LiquidDataType {
    /// A byte array.
    ByteArray = 0,
    /// A byte-view array (dictionary + FSST raw + views).
    ByteViewArray = 4,
    /// An integer.
    Integer = 1,
    /// A float.
    Float = 2,
    /// A fixed length byte array.
    FixedLenByteArray = 3,
    /// A linear-model based integer (signed residuals + model params).
    LinearInteger = 5,
}

impl From<u16> for LiquidDataType {
    fn from(value: u16) -> Self {
        match value {
            0 => LiquidDataType::ByteArray,
            4 => LiquidDataType::ByteViewArray,
            1 => LiquidDataType::Integer,
            2 => LiquidDataType::Float,
            3 => LiquidDataType::FixedLenByteArray,
            5 => LiquidDataType::LinearInteger,
            _ => panic!("Invalid liquid data type: {value}"),
        }
    }
}

/// A trait to access the underlying Liquid array.
pub trait AsLiquidArray {
    /// Get the underlying string array.
    fn as_string_array_opt(&self) -> Option<&LiquidByteArray>;

    /// Get the underlying string array.
    fn as_string(&self) -> &LiquidByteArray {
        self.as_string_array_opt().expect("liquid string array")
    }

    /// Get the underlying binary array.
    fn as_binary_array_opt(&self) -> Option<&LiquidByteArray>;

    /// Get the underlying binary array.
    fn as_binary(&self) -> &LiquidByteArray {
        self.as_binary_array_opt().expect("liquid binary array")
    }

    /// Get the underlying byte view array.
    fn as_byte_view_array_opt(&self) -> Option<&LiquidByteViewArray<MemoryBuffer>>;

    /// Get the underlying byte view array.
    fn as_byte_view(&self) -> &LiquidByteViewArray<MemoryBuffer> {
        self.as_byte_view_array_opt()
            .expect("liquid byte view array")
    }

    /// Get the underlying primitive array.
    fn as_primitive_array_opt<T: LiquidPrimitiveType>(&self) -> Option<&LiquidPrimitiveArray<T>>;

    /// Get the underlying primitive array.
    fn as_primitive<T: LiquidPrimitiveType>(&self) -> &LiquidPrimitiveArray<T> {
        self.as_primitive_array_opt()
            .expect("liquid primitive array")
    }

    /// Get the underlying float array.
    fn as_float_array_opt<T: LiquidFloatType>(&self) -> Option<&LiquidFloatArray<T>>;

    /// Get the underlying float array.
    fn as_float<T: LiquidFloatType>(&self) -> &LiquidFloatArray<T> {
        self.as_float_array_opt().expect("liquid float array")
    }
}

impl AsLiquidArray for dyn LiquidArray + '_ {
    fn as_string_array_opt(&self) -> Option<&LiquidByteArray> {
        self.as_any().downcast_ref()
    }

    fn as_primitive_array_opt<T: LiquidPrimitiveType>(&self) -> Option<&LiquidPrimitiveArray<T>> {
        self.as_any().downcast_ref()
    }

    fn as_binary_array_opt(&self) -> Option<&LiquidByteArray> {
        self.as_any().downcast_ref()
    }

    fn as_byte_view_array_opt(&self) -> Option<&LiquidByteViewArray<MemoryBuffer>> {
        self.as_any().downcast_ref()
    }

    fn as_float_array_opt<T: LiquidFloatType>(&self) -> Option<&LiquidFloatArray<T>> {
        self.as_any().downcast_ref()
    }
}

/// A Liquid array.
pub trait LiquidArray: std::fmt::Debug + Send + Sync {
    /// Get the underlying any type.
    fn as_any(&self) -> &dyn Any;

    /// Get the memory size of the Liquid array.
    fn get_array_memory_size(&self) -> usize;

    /// Get the length of the Liquid array.
    fn len(&self) -> usize;

    /// Check if the Liquid array is empty.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Convert the Liquid array to an Arrow array.
    fn to_arrow_array(&self) -> ArrayRef;

    /// Convert the Liquid array to an Arrow array.
    /// Except that it will pick the best encoding for the arrow array.
    /// Meaning that it may not obey the data type of the original arrow array.
    fn to_best_arrow_array(&self) -> ArrayRef {
        self.to_arrow_array()
    }

    /// Get the logical data type of the Liquid array.
    fn data_type(&self) -> LiquidDataType;

    /// Serialize the Liquid array to a byte array.
    fn to_bytes(&self) -> Vec<u8>;

    /// Filter the Liquid array with a boolean buffer.
    fn filter(&self, selection: &BooleanBuffer) -> LiquidArrayRef;

    /// Filter the Liquid array with a boolean array and return an **arrow array**.
    fn filter_to_arrow(&self, selection: &BooleanBuffer) -> ArrayRef {
        let filtered = self.filter(selection);
        filtered.to_best_arrow_array()
    }

    /// Try to evaluate a predicate on the Liquid array with a filter.
    /// Returns `None` if the predicate is not supported.
    ///
    /// Note that the filter is a boolean buffer, not a boolean array, i.e., filter can't be nullable.
    /// The returned boolean mask is nullable if the the original array is nullable.
    fn try_eval_predicate(
        &self,
        _predicate: &Arc<dyn PhysicalExpr>,
        _filter: &BooleanBuffer,
    ) -> Option<BooleanArray> {
        None
    }

    /// Squeeze the Liquid array to a `LiquidHybridArrayRef` and a `bytes::Bytes`.
    /// Return `None` if the Liquid array cannot be squeezed.
    ///
    /// This is the bridge from in-memory array to hybrid array.
    /// Important: The returned `Bytes` is the data that is stored on disk, it is the same as to_bytes().
    ///
    /// If we `soak` the `LiquidHybridArrayRef` back with the bytes, we should get the same `LiquidArray`.
    fn squeeze(&self) -> Option<(LiquidHybridArrayRef, bytes::Bytes)> {
        None
    }
}

/// A reference to a Liquid array.
pub type LiquidArrayRef = Arc<dyn LiquidArray>;

/// A reference to a Liquid hybrid array.
pub type LiquidHybridArrayRef = Arc<dyn LiquidHybridArray>;

/// A range of bytes on disk.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IoRange {
    range: Range<u64>,
}

impl IoRange {
    /// Get the range of bytes on disk.
    pub fn range(&self) -> &Range<u64> {
        &self.range
    }
}

/// A Liquid hybrid array is a Liquid array that part of its data is stored on disk.
/// `LiquidHybridArray` is more complex than in-memory `LiquidArray` because it needs to handle IO.
pub trait LiquidHybridArray: std::fmt::Debug + Send + Sync {
    /// Get the underlying any type.
    fn as_any(&self) -> &dyn Any;

    /// Get the memory size of the Liquid array.
    fn get_array_memory_size(&self) -> usize;

    /// Get the length of the Liquid array.
    fn len(&self) -> usize;

    /// Check if the Liquid array is empty.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Convert the Liquid array to an Arrow array.
    fn to_arrow_array(&self) -> Result<ArrayRef, IoRange>;

    /// Convert the Liquid array to an Arrow array.
    /// Except that it will pick the best encoding for the arrow array.
    /// Meaning that it may not obey the data type of the original arrow array.
    fn to_best_arrow_array(&self) -> Result<ArrayRef, IoRange> {
        self.to_arrow_array()
    }

    /// Get the logical data type of the Liquid array.
    fn data_type(&self) -> LiquidDataType;

    /// Serialize the Liquid array to a byte array.
    fn to_bytes(&self) -> Result<Vec<u8>, IoRange>;

    /// Filter the Liquid array with a boolean buffer.
    fn filter(&self, selection: &BooleanBuffer) -> Result<LiquidHybridArrayRef, IoRange>;

    /// Filter the Liquid array with a boolean array and return an **arrow array**.
    fn filter_to_arrow(&self, selection: &BooleanBuffer) -> Result<ArrayRef, IoRange> {
        let filtered = self.filter(selection)?;
        filtered.to_best_arrow_array()
    }

    /// Try to evaluate a predicate on the Liquid array with a filter.
    /// Returns `Ok(None)` if the predicate is not supported.
    ///
    /// Note that the filter is a boolean buffer, not a boolean array, i.e., filter can't be nullable.
    /// The returned boolean mask is nullable if the the original array is nullable.
    fn try_eval_predicate(
        &self,
        _predicate: &Arc<dyn PhysicalExpr>,
        _filter: &BooleanBuffer,
    ) -> Result<Option<BooleanArray>, IoRange> {
        Ok(None)
    }

    /// Feed IO data to the `LiquidHybridArray` and return the in-memory `LiquidArray`.
    /// For byte-view arrays, `data` should be the raw FSST buffer bytes.
    fn soak(&self, data: bytes::Bytes) -> LiquidArrayRef;

    /// Get the `IoRequest` to convert the `LiquidHybridArray` to a `LiquidArray`.
    fn to_liquid(&self) -> IoRange;
}

/// Compile-time info about primitive kind (signed vs unsigned) and bounds.
/// Implemented for all Liquid-supported primitive integer and date types.
pub trait PrimitiveKind {
    /// Whether the logical type is unsigned (true for u8/u16/u32/u64).
    const IS_UNSIGNED: bool;
    /// Maximum representable value as u64 for unsigned types (unused for signed).
    const MAX_U64: u64;
    /// Minimum representable value as i64 for signed/date types (unused for unsigned).
    const MIN_I64: i64;
    /// Maximum representable value as i64 for signed/date types (unused for unsigned).
    const MAX_I64: i64;
}

macro_rules! impl_unsigned_kind {
    ($t:ty, $max:expr) => {
        impl PrimitiveKind for $t {
            const IS_UNSIGNED: bool = true;
            const MAX_U64: u64 = $max as u64;
            const MIN_I64: i64 = 0; // unused
            const MAX_I64: i64 = 0; // unused
        }
    };
}

macro_rules! impl_signed_kind {
    ($t:ty, $min:expr, $max:expr) => {
        impl PrimitiveKind for $t {
            const IS_UNSIGNED: bool = false;
            const MAX_U64: u64 = 0; // unused
            const MIN_I64: i64 = $min as i64;
            const MAX_I64: i64 = $max as i64;
        }
    };
}

use arrow::datatypes::{
    Date32Type, Date64Type, Int8Type, Int16Type, Int32Type, Int64Type, UInt8Type, UInt16Type,
    UInt32Type, UInt64Type,
};

impl_unsigned_kind!(UInt8Type, u8::MAX);
impl_unsigned_kind!(UInt16Type, u16::MAX);
impl_unsigned_kind!(UInt32Type, u32::MAX);
impl_unsigned_kind!(UInt64Type, u64::MAX);

impl_signed_kind!(Int8Type, i8::MIN, i8::MAX);
impl_signed_kind!(Int16Type, i16::MIN, i16::MAX);
impl_signed_kind!(Int32Type, i32::MIN, i32::MAX);
impl_signed_kind!(Int64Type, i64::MIN, i64::MAX);

// Dates are logically signed in Arrow (Date32: i32 days, Date64: i64 ms)
impl_signed_kind!(Date32Type, i32::MIN, i32::MAX);
impl_signed_kind!(Date64Type, i64::MIN, i64::MAX);
