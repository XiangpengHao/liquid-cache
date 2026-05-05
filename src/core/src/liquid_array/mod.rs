//! LiquidArray is the core data structure of LiquidCache.
//! You should not use this module directly.
//! Instead, use `liquid_cache_datafusion_server` or `liquid_cache_datafusion_client` to interact with LiquidCache.
pub mod byte_view_array;
mod decimal_array;
mod fix_len_byte_array;
mod float_array;
mod hybrid_primitive_array;
pub mod ipc;
mod linear_integer_array;
mod primitive_array;
pub mod raw;
mod squeezed_date32_array;
#[cfg(test)]
mod tests;
pub(crate) mod utils;
mod variant_array;

use std::{any::Any, ops::Range, sync::Arc};

use arrow::{
    array::{ArrayRef, BooleanArray, cast::AsArray},
    buffer::BooleanBuffer,
    record_batch::RecordBatch,
};
use arrow_schema::{DataType, Field, Schema};
pub use byte_view_array::LiquidByteViewArray;
use bytes::Bytes;
use datafusion_expr_common::operator::Operator as DFOperator;
pub use decimal_array::LiquidDecimalArray;
pub use fix_len_byte_array::LiquidFixedLenByteArray;
pub use float_array::{LiquidFloat32Array, LiquidFloat64Array, LiquidFloatArray};
pub use linear_integer_array::{
    LiquidLinearArray, LiquidLinearDate32Array, LiquidLinearDate64Array, LiquidLinearI8Array,
    LiquidLinearI16Array, LiquidLinearI32Array, LiquidLinearI64Array, LiquidLinearU8Array,
    LiquidLinearU16Array, LiquidLinearU32Array, LiquidLinearU64Array,
};
pub use primitive_array::IntegerSqueezePolicy;
pub use primitive_array::{
    LiquidDate32Array, LiquidDate64Array, LiquidI8Array, LiquidI16Array, LiquidI32Array,
    LiquidI64Array, LiquidPrimitiveArray, LiquidPrimitiveDeltaArray, LiquidPrimitiveType,
    LiquidU8Array, LiquidU16Array, LiquidU32Array, LiquidU64Array,
};
pub use squeezed_date32_array::{Date32Field, SqueezedDate32Array};
pub use variant_array::VariantStructSqueezedArray;

use crate::cache::{CacheExpression, LiquidExpr};

/// Liquid data type is only logical type
#[derive(Debug, Clone, Copy)]
#[repr(u16)]
pub enum LiquidDataType {
    /// A byte-view array (dictionary + FSST raw + views).
    ByteViewArray = 4,
    /// An integer.
    Integer = 1,
    /// A float.
    Float = 2,
    /// A fixed length byte array.
    FixedLenByteArray = 3,
    /// A decimal encoded as a primitive u64 array.
    Decimal = 6,
    /// A linear-model based integer (signed residuals + model params).
    LinearInteger = 5,
}

impl From<u16> for LiquidDataType {
    fn from(value: u16) -> Self {
        match value {
            4 => LiquidDataType::ByteViewArray,
            1 => LiquidDataType::Integer,
            2 => LiquidDataType::Float,
            3 => LiquidDataType::FixedLenByteArray,
            5 => LiquidDataType::LinearInteger,
            6 => LiquidDataType::Decimal,
            _ => panic!("Invalid liquid data type: {value}"),
        }
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

    /// Get the original arrow data type of the Liquid array.
    fn original_arrow_data_type(&self) -> DataType;

    /// Serialize the Liquid array to a byte array.
    fn to_bytes(&self) -> Vec<u8>;

    /// Filter the Liquid array with a boolean array and return an **arrow array**.
    fn filter(&self, selection: &BooleanBuffer) -> ArrayRef {
        let arrow_array = self.to_arrow_array();
        let selection = BooleanArray::new(selection.clone(), None);
        arrow::compute::kernels::filter::filter(&arrow_array, &selection).unwrap()
    }

    /// Evaluate a predicate on the Liquid array with a filter.
    ///
    /// Note that the filter is a boolean buffer, not a boolean array, i.e., filter can't be nullable.
    /// The returned boolean mask is nullable if the the original array is nullable.
    fn try_eval_predicate(&self, predicate: &LiquidExpr, filter: &BooleanBuffer) -> BooleanArray {
        let filtered = self.filter(filter);
        eval_predicate_on_array(filtered, predicate)
    }

    /// Squeeze the Liquid array to a `LiquidHybridArrayRef` and a `bytes::Bytes`.
    /// Return `None` if the Liquid array cannot be squeezed.
    ///
    /// This is the bridge from in-memory array to hybrid array.
    /// Important: The returned `Bytes` is the data that is stored on disk, it is the same as to_bytes().
    ///
    /// Hydrating the hybrid array from the stored bytes should yield the same `LiquidArray`.
    fn squeeze(
        &self,
        _io: Arc<dyn SqueezeIoHandler>,
        _expression_hint: Option<&CacheExpression>,
    ) -> Option<(LiquidSqueezedArrayRef, bytes::Bytes)> {
        None
    }
}

/// A reference to a Liquid array.
pub type LiquidArrayRef = Arc<dyn LiquidArray>;

/// On-disk backing for a squeezed array.
///
/// Each variant carries the byte length of the persisted backing data, so the
/// cache can release the disk budget when the entry is evicted.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SqueezedBacking {
    /// Bytes are stored using the Liquid IPC format.
    Liquid(usize),
    /// Bytes are stored using Arrow IPC (or another Arrow-compatible encoding).
    Arrow(usize),
}

impl SqueezedBacking {
    /// Byte length of the backing data persisted on disk.
    pub fn disk_bytes(&self) -> usize {
        match self {
            Self::Liquid(n) | Self::Arrow(n) => *n,
        }
    }
}

/// A reference to a Liquid squeezed array.
pub type LiquidSqueezedArrayRef = Arc<dyn LiquidSqueezedArray>;

/// Signals that the squeezed representation needs to be hydrated from disk.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NeedsBacking;

/// Result type for squeezed operations that may require disk hydration.
pub type SqueezeResult<T> = Result<T, NeedsBacking>;

enum Operator {
    Eq,
    NotEq,
    Lt,
    LtEq,
    Gt,
    GtEq,
}

impl Operator {
    fn from_datafusion(op: &DFOperator) -> Option<Self> {
        let op = match op {
            DFOperator::Eq => Operator::Eq,
            DFOperator::NotEq => Operator::NotEq,
            DFOperator::Lt => Operator::Lt,
            DFOperator::LtEq => Operator::LtEq,
            DFOperator::Gt => Operator::Gt,
            DFOperator::GtEq => Operator::GtEq,
            _ => return None,
        };
        Some(op)
    }
}

/// A Liquid squeezed array is a Liquid array that part of its data is stored on disk.
/// `LiquidSqueezedArray` is more complex than in-memory `LiquidArray` because it needs to handle IO.
#[async_trait::async_trait]
pub trait LiquidSqueezedArray: std::fmt::Debug + Send + Sync {
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
    async fn to_arrow_array(&self) -> ArrayRef;

    /// Convert the Liquid array to an Arrow array.
    /// Except that it will pick the best encoding for the arrow array.
    /// Meaning that it may not obey the data type of the original arrow array.
    async fn to_best_arrow_array(&self) -> ArrayRef {
        self.to_arrow_array().await
    }

    /// Get the logical data type of the Liquid array.
    fn data_type(&self) -> LiquidDataType;

    /// Get the original arrow data type of the Liquid squeezed array.
    fn original_arrow_data_type(&self) -> DataType;

    /// Filter the Liquid array with a boolean array and return an **arrow array**.
    async fn filter(&self, selection: &BooleanBuffer) -> ArrayRef {
        let arrow_array = self.to_arrow_array().await;
        let selection = BooleanArray::new(selection.clone(), None);
        arrow::compute::kernels::filter::filter(&arrow_array, &selection).unwrap()
    }

    /// Evaluate a predicate on the Liquid array with a filter.
    ///
    /// Note that the filter is a boolean buffer, not a boolean array, i.e., filter can't be nullable.
    /// The returned boolean mask is nullable if the the original array is nullable.
    async fn try_eval_predicate(
        &self,
        predicate: &LiquidExpr,
        filter: &BooleanBuffer,
    ) -> BooleanArray {
        let filtered = self.filter(filter).await;
        eval_predicate_on_array(filtered, predicate)
    }

    /// Describe how the squeezed array persists its backing bytes on disk,
    /// including the byte length of the persisted data.
    fn disk_backing(&self) -> SqueezedBacking;
}

pub(crate) fn eval_predicate_on_array(array: ArrayRef, predicate: &LiquidExpr) -> BooleanArray {
    let schema = Arc::new(Schema::new(vec![Field::new(
        "liquid_predicate_col",
        array.data_type().clone(),
        true,
    )]));
    let record_batch = RecordBatch::try_new(schema, vec![array]).expect("predicate input batch");
    let result = predicate
        .physical_expr()
        .evaluate(&record_batch)
        .expect("validated LiquidExpr must evaluate");
    let boolean_array = result
        .into_array(record_batch.num_rows())
        .expect("predicate output must be an array");
    boolean_array.as_boolean().clone()
}

/// A trait to read the backing bytes of a squeezed array from disk.
#[async_trait::async_trait]
pub trait SqueezeIoHandler: std::fmt::Debug + Send + Sync {
    /// Read the backing bytes of a squeezed array from disk.
    async fn read(&self, range: Option<Range<u64>>) -> std::io::Result<Bytes>;

    /// Trace the number of decompressions performed.
    // TODO: this is ugly.
    fn tracing_decompress_count(&self, _decompress_cnt: usize, _total_cnt: usize) {
        // Do nothing by default
    }

    /// Trace the number of IO saved by squeezing.
    // TODO: this is ugly.
    fn trace_io_saved(&self) {
        // Do nothing by default
    }
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
    Date32Type, Date64Type, Int8Type, Int16Type, Int32Type, Int64Type, TimestampMicrosecondType,
    TimestampMillisecondType, TimestampNanosecondType, TimestampSecondType, UInt8Type, UInt16Type,
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
impl_signed_kind!(TimestampSecondType, i64::MIN, i64::MAX);
impl_signed_kind!(TimestampMillisecondType, i64::MIN, i64::MAX);
impl_signed_kind!(TimestampMicrosecondType, i64::MIN, i64::MAX);
impl_signed_kind!(TimestampNanosecondType, i64::MIN, i64::MAX);
