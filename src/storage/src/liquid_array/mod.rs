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

use std::{any::Any, path::PathBuf, sync::Arc};

use arrow::{
    array::{ArrayRef, BooleanArray},
    buffer::BooleanBuffer,
};
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

    /// Filter the Liquid array with a boolean buffer.
    fn filter(&self, selection: &BooleanBuffer) -> LiquidArrayRef;

    /// Filter the Liquid array with a boolean array and return an **arrow array**.
    fn filter_to_arrow(&self, selection: &BooleanBuffer) -> ArrayRef {
        let filtered = self.filter(selection);
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
    ) -> Option<BooleanArray> {
        None
    }
}

/// A reference to a Liquid array.
pub type LiquidArrayRef = Arc<dyn LiquidArray>;

/// A wrapper around a value that may need IO to be read.
pub enum MaybeIO<T> {
    /// The value is ready.
    Ok(T),
    /// The value is not ready. The second element is the path to the IO.
    NeedIo(PathBuf),
}

impl<T> MaybeIO<T> {
    /// Unwrap the value.
    /// Panics if the value is not ready.
    pub fn unwrap(self) -> T {
        match self {
            MaybeIO::Ok(t) => t,
            MaybeIO::NeedIo(path) => {
                panic!("Need IO: {}", path.display());
            }
        }
    }

    /// Map the value.
    pub fn map<F, R>(self, f: F) -> MaybeIO<R>
    where
        F: FnOnce(T) -> R,
    {
        match self {
            MaybeIO::Ok(t) => MaybeIO::Ok(f(t)),
            MaybeIO::NeedIo(path) => MaybeIO::NeedIo(path),
        }
    }
}

type LiquidHybridArrayRef = Arc<dyn LiquidHybridArray>;

/// A Liquid array.
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
    fn to_arrow_array(&self) -> MaybeIO<ArrayRef>;

    /// Convert the Liquid array to an Arrow array.
    /// Except that it will pick the best encoding for the arrow array.
    /// Meaning that it may not obey the data type of the original arrow array.
    fn to_best_arrow_array(&self) -> MaybeIO<ArrayRef> {
        self.to_arrow_array()
    }

    /// Get the logical data type of the Liquid array.
    fn data_type(&self) -> LiquidDataType;

    /// Serialize the Liquid array to a byte array.
    fn to_bytes(&self) -> MaybeIO<Vec<u8>>;

    /// Filter the Liquid array with a boolean buffer.
    fn filter(&self, selection: &BooleanBuffer) -> MaybeIO<LiquidHybridArrayRef>;

    /// Filter the Liquid array with a boolean array and return an **arrow array**.
    fn filter_to_arrow(&self, selection: &BooleanBuffer) -> MaybeIO<ArrayRef> {
        match self.filter(selection) {
            MaybeIO::Ok(filtered) => match filtered.to_best_arrow_array() {
                MaybeIO::Ok(array) => MaybeIO::Ok(array),
                MaybeIO::NeedIo(path) => MaybeIO::NeedIo(path),
            },
            MaybeIO::NeedIo(path) => MaybeIO::NeedIo(path),
        }
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
    ) -> Result<Option<BooleanArray>, ArrowError> {
        Ok(None)
    }
}
