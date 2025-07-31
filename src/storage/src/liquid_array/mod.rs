//! LiquidArray is the core data structure of LiquidCache.
//! You should not use this module directly.
//! Instead, use `liquid_cache_server` or `liquid_cache_client` to interact with LiquidCache.
mod byte_array;
mod byte_view_array;
mod fix_len_byte_array;
mod float_array;
pub mod ipc;
mod primitive_array;
pub mod raw;
pub(crate) mod utils;

use std::{any::Any, num::NonZero, sync::Arc};

use arrow::array::{ArrayRef, BooleanArray};
use arrow_schema::ArrowError;
pub use byte_array::{LiquidByteArray, get_string_needle};
pub use byte_view_array::{ByteViewArrayMemoryUsage, LiquidByteViewArray};
use datafusion::physical_plan::PhysicalExpr;
pub use fix_len_byte_array::LiquidFixedLenByteArray;
use float_array::LiquidFloatType;
pub use float_array::{LiquidFloat32Array, LiquidFloat64Array, LiquidFloatArray};
pub use primitive_array::{
    LiquidDate32Array, LiquidDate64Array, LiquidI8Array, LiquidI16Array, LiquidI32Array,
    LiquidI64Array, LiquidPrimitiveArray, LiquidPrimitiveType, LiquidU8Array, LiquidU16Array,
    LiquidU32Array, LiquidU64Array,
};

/// Liquid data type is only logical type
#[derive(Debug, Clone, Copy)]
#[repr(u16)]
pub enum LiquidDataType {
    /// A byte array.
    ByteArray = 0,
    /// An integer.
    Integer = 1,
    /// A float.
    Float = 2,
    /// A fixed length byte array.
    FixedLenByteArray = 3,
}

impl From<u16> for LiquidDataType {
    fn from(value: u16) -> Self {
        match value {
            0 => LiquidDataType::ByteArray,
            1 => LiquidDataType::Integer,
            2 => LiquidDataType::Float,
            3 => LiquidDataType::FixedLenByteArray,
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
    fn as_byte_view_array_opt(&self) -> Option<&LiquidByteViewArray>;

    /// Get the underlying byte view array.
    fn as_byte_view(&self) -> &LiquidByteViewArray {
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

    fn as_byte_view_array_opt(&self) -> Option<&LiquidByteViewArray> {
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

    /// Filter the Liquid array with a boolean array.
    fn filter(&self, selection: &BooleanArray) -> LiquidArrayRef;

    /// Filter the Liquid array with a boolean array and return an **arrow array**.
    fn filter_to_arrow(&self, selection: &BooleanArray) -> ArrayRef {
        let filtered = self.filter(selection);
        filtered.to_best_arrow_array()
    }

    /// Try to evaluate a predicate on the Liquid array with a filter.
    /// Returns `None` if the predicate is not supported.
    fn try_eval_predicate(
        &self,
        _predicate: &Arc<dyn PhysicalExpr>,
        _filter: &BooleanArray,
    ) -> Result<Option<BooleanArray>, ArrowError> {
        Ok(None)
    }
}

/// A reference to a Liquid array.
pub type LiquidArrayRef = Arc<dyn LiquidArray>;

/// Get the bit width for a given max value.
/// Returns 1 if the max value is 0.
/// Returns 64 - max_value.leading_zeros() as u8 otherwise.
pub(crate) fn get_bit_width(max_value: u64) -> NonZero<u8> {
    if max_value == 0 {
        // todo: here we actually should return 0, as we should just use constant encoding.
        // but that's not implemented yet.
        NonZero::new(1).unwrap()
    } else {
        NonZero::new(64 - max_value.leading_zeros() as u8).unwrap()
    }
}
