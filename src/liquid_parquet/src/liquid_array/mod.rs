mod byte_array;
pub mod ipc;
mod primitive_array;
mod raw;

use std::{any::Any, num::NonZero, sync::Arc};

use arrow::array::{ArrayRef, BooleanArray};
pub use byte_array::LiquidByteArray;
pub use primitive_array::{
    LiquidI8Array, LiquidI16Array, LiquidI32Array, LiquidI64Array, LiquidPrimitiveArray,
    LiquidPrimitiveType, LiquidU8Array, LiquidU16Array, LiquidU32Array, LiquidU64Array,
};
pub use raw::bit_pack_array::BitPackedArray;
pub use raw::fsst_array::FsstArray;

/// Liquid data type is only logical type
#[derive(Debug, Clone, Copy)]
#[repr(u16)]
pub enum LiquidDataType {
    ByteArray = 0,
    Integer = 1,
}

impl From<u16> for LiquidDataType {
    fn from(value: u16) -> Self {
        match value {
            0 => LiquidDataType::ByteArray,
            1 => LiquidDataType::Integer,
            _ => panic!("Invalid liquid data type: {}", value),
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

    fn as_binary_array_opt(&self) -> Option<&LiquidByteArray>;

    fn as_binary(&self) -> &LiquidByteArray {
        self.as_binary_array_opt().expect("liquid binary array")
    }

    /// Get the underlying primitive array.
    fn as_primitive_array_opt<T: LiquidPrimitiveType>(&self) -> Option<&LiquidPrimitiveArray<T>>;

    /// Get the underlying primitive array.
    fn as_primitive<T: LiquidPrimitiveType>(&self) -> &LiquidPrimitiveArray<T> {
        self.as_primitive_array_opt()
            .expect("liquid primitive array")
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
}

/// A reference to a Liquid array.
pub type LiquidArrayRef = Arc<dyn LiquidArray>;

pub(crate) fn get_bit_width(max_value: u64) -> NonZero<u8> {
    if max_value <= 1 {
        // todo: here we actually should return 0, as we should just use constant encoding.
        // but that's not implemented yet.
        NonZero::new(1).unwrap()
    } else {
        NonZero::new(64 - max_value.leading_zeros() as u8).unwrap()
    }
}
