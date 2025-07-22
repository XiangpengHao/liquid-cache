use arrow::array::{
    Array, ArrayAccessor, ArrayIter, ArrayRef, BinaryArray, BooleanArray, DictionaryArray,
    GenericByteArray, StringArray, StringViewArray, UInt16Array, UInt32Array, cast::AsArray,
    types::UInt16Type,
};
use arrow::array::{BinaryViewArray, BooleanBufferBuilder};
use arrow::buffer::{BooleanBuffer, Buffer, NullBuffer, OffsetBuffer};
use arrow::compute::{cast, kernels, sort_to_indices};
use arrow::datatypes::ByteArrayType;
use arrow_schema::ArrowError;
use bytes;
use datafusion::logical_expr::{ColumnarValue, Operator};
use datafusion::physical_expr_common::datum::apply_cmp;
use datafusion::physical_plan::PhysicalExpr;
use datafusion::physical_plan::expressions::{BinaryExpr, LikeExpr, Literal};
use fsst::{Compressor, Decompressor};
use std::any::Any;
use std::fmt::Display;
use std::mem::MaybeUninit;
use std::path::PathBuf;
use std::sync::{Arc, RwLock};

#[cfg(test)]
use std::cell::Cell;

#[cfg(test)]
thread_local! {
    static DISK_READ_COUNTER: Cell<usize> = const { Cell::new(0)};
    static FULL_DATA_COMPARISON_COUNTER: Cell<usize> = const { Cell::new(0)};
}

#[cfg(test)]
fn increment_disk_read_counter() {
    DISK_READ_COUNTER.with(|counter| {
        counter.set(counter.get() + 1);
    });
}

#[cfg(test)]
fn get_disk_read_counter() -> usize {
    DISK_READ_COUNTER.with(|counter| counter.get())
}

#[cfg(test)]
fn reset_disk_read_counter() {
    DISK_READ_COUNTER.with(|counter| counter.set(0));
}

use super::{
    LiquidArray, LiquidArrayRef, LiquidDataType,
    byte_array::{ArrowByteType, get_string_needle},
};
use crate::liquid_array::raw::fsst_array::{RawFsstBuffer, train_compressor};
use crate::utils::CheckedDictionaryArray;
use std::fs::File;
use std::io::{self, Write};

#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct OffsetView {
    offset: u32,
    prefix: [u8; 8],
}

const _: () = if std::mem::size_of::<OffsetView>() != 12 {
    panic!("OffsetView must be 12 bytes")
};

impl OffsetView {
    pub fn new(offset: u32, prefix: [u8; 8]) -> Self {
        Self { offset, prefix }
    }

    pub fn offset(&self) -> u32 {
        self.offset
    }

    pub fn prefix(&self) -> &[u8; 8] {
        &self.prefix
    }
}

/// Storage options for FSST buffer - can be in memory or on disk
#[derive(Clone)]
pub enum FsstBufferStorage {
    InMemory(Arc<RawFsstBuffer>),
    OnDisk(PathBuf),
}

impl std::fmt::Debug for FsstBufferStorage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FsstBufferStorage::InMemory(raw_buffer) => {
                f.debug_tuple("InMemory").field(raw_buffer).finish()
            }
            FsstBufferStorage::OnDisk(path) => f.debug_tuple("OnDisk").field(path).finish(),
        }
    }
}

impl FsstBufferStorage {
    /// Get the raw FSST buffer, loading from disk if necessary
    pub fn get_raw_buffer(&self) -> Result<Arc<RawFsstBuffer>, io::Error> {
        match self {
            FsstBufferStorage::InMemory(buffer) => Ok(buffer.clone()),
            FsstBufferStorage::OnDisk(path) => {
                // Increment disk read counter for testing
                #[cfg(test)]
                increment_disk_read_counter();

                let bytes = std::fs::read(path)?;
                let bytes = bytes::Bytes::from(bytes);
                let raw_buffer = RawFsstBuffer::from_bytes(bytes);
                Ok(Arc::new(raw_buffer))
            }
        }
    }

    /// Get memory size - returns 0 for on-disk storage
    pub fn get_array_memory_size(&self) -> usize {
        match self {
            FsstBufferStorage::InMemory(buffer) => buffer.get_memory_size(),
            FsstBufferStorage::OnDisk(_) => 0,
        }
    }

    /// Check if the buffer is stored in memory
    pub fn is_in_memory(&self) -> bool {
        matches!(self, FsstBufferStorage::InMemory(_))
    }

    /// Check if the buffer is stored on disk
    pub fn is_on_disk(&self) -> bool {
        matches!(self, FsstBufferStorage::OnDisk(_))
    }
}

impl LiquidArray for LiquidByteViewArray {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn get_array_memory_size(&self) -> usize {
        self.get_detailed_memory_usage().total()
    }

    fn len(&self) -> usize {
        self.dictionary_keys.len()
    }

    #[inline]
    fn to_arrow_array(&self) -> ArrayRef {
        let dict = self.to_arrow_array();
        Arc::new(dict)
    }

    fn to_best_arrow_array(&self) -> ArrayRef {
        let dict = self.to_dict_arrow();
        Arc::new(dict)
    }

    fn filter(&self, selection: &BooleanArray) -> LiquidArrayRef {
        let filtered = filter_inner(self, selection);
        Arc::new(filtered)
    }

    fn try_eval_predicate(
        &self,
        expr: &Arc<dyn PhysicalExpr>,
        filter: &BooleanArray,
    ) -> Result<Option<BooleanArray>, ArrowError> {
        let filtered = filter_inner(self, filter);
        try_eval_predicate_inner(expr, &filtered)
    }

    fn to_bytes(&self) -> Vec<u8> {
        todo!("I need to think more carefully about this")
    }

    fn data_type(&self) -> LiquidDataType {
        LiquidDataType::ByteArray
    }
}

fn filter_inner(array: &LiquidByteViewArray, filter: &BooleanArray) -> LiquidByteViewArray {
    // Only filter the dictionary keys, not the offset views!
    // Offset views reference unique values in FSST buffer and should remain unchanged

    // Filter the dictionary keys using Arrow's built-in filter functionality
    let filtered_keys = arrow::compute::filter(&array.dictionary_keys, filter).unwrap();
    let filtered_keys = filtered_keys.as_primitive::<UInt16Type>().clone();

    LiquidByteViewArray {
        dictionary_keys: filtered_keys,
        offset_views: array.offset_views.clone(), // Keep original offset views - they reference unique values
        fsst_buffer: array.fsst_buffer.clone(),
        original_arrow_type: array.original_arrow_type,
        shared_prefix: array.shared_prefix.clone(),
        compressor: array.compressor.clone(),
    }
}

fn try_eval_predicate_inner(
    expr: &Arc<dyn PhysicalExpr>,
    array: &LiquidByteViewArray,
) -> Result<Option<BooleanArray>, ArrowError> {
    // Handle binary expressions (comparisons)
    if let Some(binary_expr) = expr.as_any().downcast_ref::<BinaryExpr>() {
        if let Some(literal) = binary_expr.right().as_any().downcast_ref::<Literal>() {
            let op = binary_expr.op();

            // Try to use string needle optimization first
            if let Some(needle) = get_string_needle(literal.value()) {
                let needle_bytes = needle.as_bytes();
                let result = array.compare_with(needle_bytes, op)?;
                return Ok(Some(result));
            }

            // Fallback to Arrow operations
            let dict_array = array.to_dict_arrow();
            let lhs = ColumnarValue::Array(Arc::new(dict_array));
            let rhs = ColumnarValue::Scalar(literal.value().clone());

            let result = match op {
                Operator::NotEq => apply_cmp(&lhs, &rhs, kernels::cmp::neq),
                Operator::Eq => apply_cmp(&lhs, &rhs, kernels::cmp::eq),
                Operator::Lt => apply_cmp(&lhs, &rhs, kernels::cmp::lt),
                Operator::LtEq => apply_cmp(&lhs, &rhs, kernels::cmp::lt_eq),
                Operator::Gt => apply_cmp(&lhs, &rhs, kernels::cmp::gt),
                Operator::GtEq => apply_cmp(&lhs, &rhs, kernels::cmp::gt_eq),
                Operator::LikeMatch => apply_cmp(&lhs, &rhs, arrow::compute::like),
                Operator::ILikeMatch => apply_cmp(&lhs, &rhs, arrow::compute::ilike),
                Operator::NotLikeMatch => apply_cmp(&lhs, &rhs, arrow::compute::nlike),
                Operator::NotILikeMatch => apply_cmp(&lhs, &rhs, arrow::compute::nilike),
                _ => return Ok(None),
            };
            if let Ok(result) = result {
                let filtered = result.into_array(array.len())?.as_boolean().clone();
                return Ok(Some(filtered));
            }
        }
    }
    // Handle like expressions
    else if let Some(like_expr) = expr.as_any().downcast_ref::<LikeExpr>()
        && like_expr
            .pattern()
            .as_any()
            .downcast_ref::<Literal>()
            .is_some()
        && let Some(literal) = like_expr.pattern().as_any().downcast_ref::<Literal>()
    {
        let arrow_dict = array.to_dict_arrow();

        let lhs = ColumnarValue::Array(Arc::new(arrow_dict));
        let rhs = ColumnarValue::Scalar(literal.value().clone());

        let result = match (like_expr.negated(), like_expr.case_insensitive()) {
            (false, false) => apply_cmp(&lhs, &rhs, arrow::compute::like),
            (true, false) => apply_cmp(&lhs, &rhs, arrow::compute::nlike),
            (false, true) => apply_cmp(&lhs, &rhs, arrow::compute::ilike),
            (true, true) => apply_cmp(&lhs, &rhs, arrow::compute::nilike),
        };
        if let Ok(result) = result {
            let filtered = result.into_array(array.len())?.as_boolean().clone();
            return Ok(Some(filtered));
        }
    }
    Ok(None)
}

/// An array that stores strings using the FSST view format:
/// - Dictionary keys with 2-byte keys stored in memory
/// - Offset views with 4-byte offsets and 8-byte prefixes stored in memory
/// - FSST buffer can be stored in memory or on disk
///
/// Data access flow:
/// 1. Use dictionary key to index into offset_views buffer
/// 2. Use offset from offset_views to read the corresponding bytes from FSST buffer
/// 3. Use prefix from offset_views for quick comparisons to avoid decompression when possible
/// 4. Decompress bytes from FSST buffer to get the full value when needed
#[derive(Clone)]
pub struct LiquidByteViewArray {
    /// Dictionary keys (u16) - one per array element, using Arrow's UInt16Array for zero-copy
    dictionary_keys: UInt16Array,
    /// Offset views containing offset (u32) and prefix (8 bytes) - one per unique value
    offset_views: Vec<OffsetView>,
    /// FSST-compressed buffer (can be in memory or on disk)
    fsst_buffer: Arc<RwLock<FsstBufferStorage>>,
    /// Used to convert back to the original arrow type
    original_arrow_type: ArrowByteType,
    /// Shared prefix across all strings in the array
    shared_prefix: Vec<u8>,
    /// Compressor for decompression
    compressor: Arc<Compressor>,
}

impl std::fmt::Debug for LiquidByteViewArray {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LiquidByteViewArray")
            .field("dictionary_keys", &self.dictionary_keys)
            .field("offset_views", &self.offset_views)
            .field("fsst_buffer", &self.fsst_buffer)
            .field("original_arrow_type", &self.original_arrow_type)
            .field("shared_prefix", &self.shared_prefix)
            .field("compressor", &"<Compressor>")
            .finish()
    }
}

impl LiquidByteViewArray {
    /// Create a LiquidByteViewArray from an Arrow StringViewArray
    pub fn from_string_view_array(array: &StringViewArray, compressor: Arc<Compressor>) -> Self {
        Self::from_view_array_inner(array, compressor, ArrowByteType::Utf8View)
    }

    /// Create a LiquidByteViewArray from an Arrow BinaryViewArray
    pub fn from_binary_view_array(array: &BinaryViewArray, compressor: Arc<Compressor>) -> Self {
        Self::from_view_array_inner(array, compressor, ArrowByteType::BinaryView)
    }

    /// Create a LiquidByteViewArray from an Arrow StringArray
    pub fn from_string_array(array: &StringArray, compressor: Arc<Compressor>) -> Self {
        Self::from_byte_array_inner(array, compressor, ArrowByteType::Utf8)
    }

    /// Create a LiquidByteViewArray from an Arrow BinaryArray
    pub fn from_binary_array(array: &BinaryArray, compressor: Arc<Compressor>) -> Self {
        Self::from_byte_array_inner(array, compressor, ArrowByteType::Binary)
    }

    /// Train a compressor from an Arrow StringViewArray
    pub fn train_from_string_view(array: &StringViewArray) -> (Arc<Compressor>, Self) {
        let compressor = Self::train_compressor(array.iter());
        (
            compressor.clone(),
            Self::from_view_array_inner(array, compressor, ArrowByteType::Utf8View),
        )
    }

    /// Train a compressor from an Arrow BinaryViewArray
    pub fn train_from_binary_view(array: &BinaryViewArray) -> (Arc<Compressor>, Self) {
        let compressor = Self::train_compressor_bytes(array.iter());
        (
            compressor.clone(),
            Self::from_view_array_inner(array, compressor, ArrowByteType::BinaryView),
        )
    }

    /// Train a compressor from an iterator of strings
    pub fn train_compressor<'a, T: ArrayAccessor<Item = &'a str>>(
        array: ArrayIter<T>,
    ) -> Arc<Compressor> {
        Arc::new(train_compressor(
            array.filter_map(|s| s.as_ref().map(|s| s.as_bytes())),
        ))
    }

    /// Train a compressor from an iterator of byte arrays
    pub fn train_compressor_bytes<'a, T: ArrayAccessor<Item = &'a [u8]>>(
        array: ArrayIter<T>,
    ) -> Arc<Compressor> {
        Arc::new(train_compressor(
            array.filter_map(|s| s.as_ref().map(|s| *s)),
        ))
    }

    /// Generic implementation for view arrays (StringViewArray and BinaryViewArray)
    fn from_view_array_inner<T>(
        array: &T,
        compressor: Arc<Compressor>,
        arrow_type: ArrowByteType,
    ) -> Self
    where
        T: Array + 'static,
    {
        // Convert view array to CheckedDictionaryArray using existing infrastructure
        let dict = if let Some(string_view) = array.as_any().downcast_ref::<StringViewArray>() {
            CheckedDictionaryArray::from_string_view_array(string_view)
        } else if let Some(binary_view) = array.as_any().downcast_ref::<BinaryViewArray>() {
            CheckedDictionaryArray::from_binary_view_array(binary_view)
        } else {
            panic!("Unsupported view array type")
        };

        Self::from_dict_array_inner(dict, compressor, arrow_type)
    }

    fn from_byte_array_inner<T: ByteArrayType>(
        array: &GenericByteArray<T>,
        compressor: Arc<Compressor>,
        arrow_type: ArrowByteType,
    ) -> Self {
        let dict = CheckedDictionaryArray::from_byte_array::<T>(array);
        Self::from_dict_array_inner(dict, compressor, arrow_type)
    }

    /// Core implementation that converts a CheckedDictionaryArray to LiquidByteViewArray
    fn from_dict_array_inner(
        dict: CheckedDictionaryArray,
        compressor: Arc<Compressor>,
        arrow_type: ArrowByteType,
    ) -> Self {
        let (keys, values) = dict.as_ref().clone().into_parts();

        // Calculate shared prefix directly from values array without intermediate allocations
        let shared_prefix = if values.is_empty() {
            Vec::new()
        } else {
            // Get first value as initial candidate for shared prefix
            let first_value_bytes = if let Some(string_values) = values.as_string_opt::<i32>() {
                string_values.value(0).as_bytes()
            } else if let Some(binary_values) = values.as_binary_opt::<i32>() {
                binary_values.value(0)
            } else {
                panic!("Unsupported dictionary value type")
            };

            let mut shared_prefix = first_value_bytes.to_vec();

            // Compare with remaining values and truncate shared prefix
            for i in 1..values.len() {
                let value_bytes = if let Some(string_values) = values.as_string_opt::<i32>() {
                    string_values.value(i).as_bytes()
                } else if let Some(binary_values) = values.as_binary_opt::<i32>() {
                    binary_values.value(i)
                } else {
                    panic!("Unsupported dictionary value type")
                };

                let common_len = shared_prefix
                    .iter()
                    .zip(value_bytes.iter())
                    .take_while(|(a, b)| a == b)
                    .count();
                shared_prefix.truncate(common_len);

                // Early exit if no common prefix
                if shared_prefix.is_empty() {
                    break;
                }
            }

            shared_prefix
        };

        let shared_prefix_len = shared_prefix.len();

        // Create offset views with prefixes - one per unique value in dictionary
        let mut offset_views = Vec::with_capacity(values.len());

        let mut compress_buffer = Vec::with_capacity(1024 * 1024 * 2);

        // Create the raw buffer and get the byte offsets
        let (raw_fsst_buffer, byte_offsets) =
            if let Some(string_values) = values.as_string_opt::<i32>() {
                RawFsstBuffer::from_byte_slices(
                    string_values.iter().map(|s| s.map(|s| s.as_bytes())),
                    compressor.clone(),
                    &mut compress_buffer,
                )
            } else if let Some(binary_values) = values.as_binary_opt::<i32>() {
                RawFsstBuffer::from_byte_slices(
                    binary_values.iter(),
                    compressor.clone(),
                    &mut compress_buffer,
                )
            } else {
                panic!("Unsupported dictionary value type")
            };

        // Calculate offset views for each unique value in the dictionary using actual byte offsets
        for i in 0..values.len() {
            let value_bytes = if let Some(string_values) = values.as_string_opt::<i32>() {
                string_values.value(i).as_bytes()
            } else if let Some(binary_values) = values.as_binary_opt::<i32>() {
                binary_values.value(i)
            } else {
                panic!("Unsupported dictionary value type")
            };

            // Extract 8-byte prefix after removing shared prefix
            let remaining_bytes = if shared_prefix_len < value_bytes.len() {
                &value_bytes[shared_prefix_len..]
            } else {
                &[]
            };

            let mut prefix = [0u8; 8];
            let prefix_len = std::cmp::min(remaining_bytes.len(), 8);
            prefix[..prefix_len].copy_from_slice(&remaining_bytes[..prefix_len]);

            // Use actual byte offset from the byte_offsets vec
            let byte_offset = byte_offsets[i];
            offset_views.push(OffsetView::new(byte_offset, prefix));
        }

        assert_eq!(values.len(), byte_offsets.len() - 1);
        offset_views.push(OffsetView::new(byte_offsets[values.len()], [0u8; 8]));

        Self {
            dictionary_keys: keys,
            offset_views,
            fsst_buffer: Arc::new(RwLock::new(FsstBufferStorage::InMemory(Arc::new(
                raw_fsst_buffer,
            )))),
            original_arrow_type: arrow_type,
            shared_prefix,
            compressor,
        }
    }

    /// Convert to Arrow DictionaryArray
    pub fn to_dict_arrow(&self) -> DictionaryArray<UInt16Type> {
        let keys_array = self.dictionary_keys.clone();

        // Convert raw FSST buffer to values using our offset views
        let storage = self.fsst_buffer.read().unwrap();
        let raw_buffer = storage.get_raw_buffer().unwrap();

        let (values_buffer, offsets_buffer) =
            raw_buffer.to_uncompressed(&self.compressor.decompressor(), &self.offset_views);

        let values = if self.original_arrow_type == ArrowByteType::Utf8
            || self.original_arrow_type == ArrowByteType::Utf8View
            || self.original_arrow_type == ArrowByteType::Dict16Utf8
        {
            let string_array =
                unsafe { StringArray::new_unchecked(offsets_buffer, values_buffer, None) };
            Arc::new(string_array) as ArrayRef
        } else {
            let binary_array =
                unsafe { BinaryArray::new_unchecked(offsets_buffer, values_buffer, None) };
            Arc::new(binary_array) as ArrayRef
        };

        unsafe { DictionaryArray::<UInt16Type>::new_unchecked(keys_array, values) }
    }

    /// Convert to Arrow array with original type
    pub fn to_arrow_array(&self) -> ArrayRef {
        let dict = self.to_dict_arrow();
        cast(&dict, &self.original_arrow_type.to_arrow_type()).unwrap()
    }

    /// Get the nulls buffer
    pub fn nulls(&self) -> Option<&NullBuffer> {
        self.dictionary_keys.nulls()
    }

    /// Compare equality with a byte needle
    fn compare_equals(&self, _needle: &[u8]) -> BooleanArray {
        // strategy:
        // 1. if the fsst buffer is in memory, we directly compress the needle and compare it with the fsst buffer
        // 2. if the fsst buffer is on disk, we first create a bitmask of values whose prefix matches the needle, then use bitmask to only decompress the values that match the bitmask
        todo!()
    }

    /// Compare not equals with a byte needle
    fn compare_not_equals(&self, needle: &[u8]) -> BooleanArray {
        let result = self.compare_equals(needle);
        let (values, nulls) = result.into_parts();
        let values = !&values;
        BooleanArray::new(values, nulls)
    }

    /// Compare with prefix optimization and fallback to Arrow operations
    pub fn compare_with(&self, needle: &[u8], op: &Operator) -> Result<BooleanArray, ArrowError> {
        match op {
            // Handle equality operations with existing optimized methods
            Operator::Eq => Ok(self.compare_equals(needle)),
            Operator::NotEq => Ok(self.compare_not_equals(needle)),

            // Handle ordering operations with prefix optimization
            Operator::Lt | Operator::LtEq | Operator::Gt | Operator::GtEq => {
                self.compare_with_inner(needle, op)
            }

            // For other operations, fall back to Arrow operations
            _ => self.compare_with_arrow_fallback(needle, op),
        }
    }

    /// Check if shared prefix comparison can short-circuit the entire operation
    fn try_shared_prefix_short_circuit(
        &self,
        needle: &[u8],
        op: &Operator,
    ) -> Option<BooleanArray> {
        let shared_prefix_len = self.shared_prefix.len();

        let needle_shared_len = std::cmp::min(needle.len(), shared_prefix_len);
        let shared_cmp = self.shared_prefix[..needle_shared_len].cmp(&needle[..needle_shared_len]);

        let all_true = || {
            let buffer = BooleanBuffer::new_set(self.dictionary_keys.len());
            BooleanArray::new(buffer, self.nulls().cloned())
        };

        let all_false = || {
            let buffer = BooleanBuffer::new_unset(self.dictionary_keys.len());
            BooleanArray::new(buffer, self.nulls().cloned())
        };

        match (op, shared_cmp) {
            (Operator::Lt | Operator::LtEq, std::cmp::Ordering::Less) => Some(all_true()),
            (Operator::Lt | Operator::LtEq, std::cmp::Ordering::Greater) => Some(all_false()),
            (Operator::Gt | Operator::GtEq, std::cmp::Ordering::Greater) => Some(all_true()),
            (Operator::Gt | Operator::GtEq, std::cmp::Ordering::Less) => Some(all_false()),

            // Handle case where compared parts are equal but lengths differ
            (op, std::cmp::Ordering::Equal) => {
                if needle.len() < shared_prefix_len {
                    // All strings start with shared_prefix which is longer than needle
                    // So all strings > needle (for Gt/GtEq) or all strings not < needle (for Lt/LtEq)
                    match op {
                        Operator::Gt | Operator::GtEq => Some(all_true()),
                        Operator::Lt => Some(all_false()),
                        Operator::LtEq => {
                            // Only true if some string equals the needle exactly
                            // Since all strings start with shared_prefix (longer than needle), none can equal needle
                            Some(all_false())
                        }
                        _ => None,
                    }
                } else if needle.len() > shared_prefix_len {
                    // Needle is longer than shared prefix - can't determine from shared prefix alone
                    None
                } else {
                    // needle.len() == shared_prefix_len
                    // All strings start with exactly needle, so need to check if any string equals needle exactly
                    None
                }
            }

            // For all other operators that shouldn't be handled by this function
            _ => None,
        }
    }

    /// Prefix optimization for ordering operations
    fn compare_with_inner(&self, needle: &[u8], op: &Operator) -> Result<BooleanArray, ArrowError> {
        // Try to short-circuit based on shared prefix comparison
        if let Some(result) = self.try_shared_prefix_short_circuit(needle, op) {
            return Ok(result);
        }

        // need to compare the entire string.
        todo!()
    }

    /// Fallback to Arrow operations for unsupported operations
    fn compare_with_arrow_fallback(
        &self,
        needle: &[u8],
        op: &Operator,
    ) -> Result<BooleanArray, ArrowError> {
        let dict_array = self.to_dict_arrow();
        let needle_scalar = datafusion::common::ScalarValue::Binary(Some(needle.to_vec()));
        let lhs = ColumnarValue::Array(Arc::new(dict_array));
        let rhs = ColumnarValue::Scalar(needle_scalar);

        let result = match op {
            Operator::LikeMatch => apply_cmp(&lhs, &rhs, arrow::compute::like),
            Operator::ILikeMatch => apply_cmp(&lhs, &rhs, arrow::compute::ilike),
            Operator::NotLikeMatch => apply_cmp(&lhs, &rhs, arrow::compute::nlike),
            Operator::NotILikeMatch => apply_cmp(&lhs, &rhs, arrow::compute::nilike),
            _ => {
                return Err(ArrowError::NotYetImplemented(format!(
                    "Operator {op:?} not supported in compare_with"
                )));
            }
        }?;

        match result {
            ColumnarValue::Array(arr) => Ok(arr.as_boolean().clone()),
            ColumnarValue::Scalar(_) => Err(ArrowError::ComputeError(
                "Expected array result from comparison".to_string(),
            )),
        }
    }

    /// Evict the FSST buffer to disk
    pub fn evict_to_disk(&self, path: PathBuf) -> Result<(), io::Error> {
        let mut storage = self.fsst_buffer.write().unwrap();

        match &*storage {
            FsstBufferStorage::InMemory(raw_buffer) => {
                let buffer = raw_buffer.to_bytes();

                let mut file = File::create(&path)?;
                file.write_all(&buffer)?;

                *storage = FsstBufferStorage::OnDisk(path);
                Ok(())
            }
            FsstBufferStorage::OnDisk(_) => Ok(()),
        }
    }

    /// Load the FSST buffer from disk into memory
    pub fn load_from_disk(&self) -> Result<(), io::Error> {
        let mut storage = self.fsst_buffer.write().unwrap();

        match &*storage {
            FsstBufferStorage::OnDisk(path) => {
                let bytes = std::fs::read(path)?;
                let bytes = bytes::Bytes::from(bytes);
                let raw_buffer = RawFsstBuffer::from_bytes(bytes);
                *storage = FsstBufferStorage::InMemory(Arc::new(raw_buffer));
                Ok(())
            }
            FsstBufferStorage::InMemory(_) => Ok(()),
        }
    }

    /// Check if the FSST buffer is currently stored on disk
    pub fn is_fsst_buffer_on_disk(&self) -> bool {
        self.fsst_buffer.read().unwrap().is_on_disk()
    }

    /// Check if the FSST buffer is currently stored in memory
    pub fn is_fsst_buffer_in_memory(&self) -> bool {
        self.fsst_buffer.read().unwrap().is_in_memory()
    }

    /// Get disk read count for testing
    #[cfg(test)]
    pub fn get_disk_read_count(&self) -> usize {
        get_disk_read_counter()
    }

    /// Reset disk read count for testing
    #[cfg(test)]
    pub fn reset_disk_read_count(&self) {
        reset_disk_read_counter()
    }

    /// Get detailed memory usage of the byte view array
    pub fn get_detailed_memory_usage(&self) -> ByteViewArrayMemoryUsage {
        ByteViewArrayMemoryUsage {
            dictionary_key: self.dictionary_keys.get_array_memory_size(),
            offsets: self.offset_views.len() * std::mem::size_of::<OffsetView>(),
            fsst_buffer: self.fsst_buffer.read().unwrap().get_array_memory_size(),
            shared_prefix: self.shared_prefix.len(),
            struct_size: std::mem::size_of::<Self>(),
        }
    }

    /// Sort the array and return indices that would sort the array
    pub fn sort_to_indices(&self) -> Result<UInt32Array, ArrowError> {
        let arrow_array = self.to_dict_arrow();
        sort_to_indices(&arrow_array, None, None)
    }
}

/// Detailed memory usage of the byte view array
pub struct ByteViewArrayMemoryUsage {
    /// Memory usage of the dictionary key
    pub dictionary_key: usize,
    /// Memory usage of the offset views
    pub offsets: usize,
    /// Memory usage of the FSST buffer
    pub fsst_buffer: usize,
    /// Memory usage of the shared prefix
    pub shared_prefix: usize,
    /// Memory usage of the struct size
    pub struct_size: usize,
}

impl Display for ByteViewArrayMemoryUsage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ByteViewArrayMemoryUsage")
            .field("dictionary_key", &self.dictionary_key)
            .field("offsets", &self.offsets)
            .field("fsst_buffer", &self.fsst_buffer)
            .field("shared_prefix", &self.shared_prefix)
            .field("struct_size", &self.struct_size)
            .field("total", &self.total())
            .finish()
    }
}

impl ByteViewArrayMemoryUsage {
    /// Get the total memory usage of the byte view array
    pub fn total(&self) -> usize {
        self.dictionary_key
            + self.offsets
            + self.fsst_buffer
            + self.shared_prefix
            + self.struct_size
    }
}

impl std::ops::AddAssign for ByteViewArrayMemoryUsage {
    fn add_assign(&mut self, other: Self) {
        self.dictionary_key += other.dictionary_key;
        self.offsets += other.offsets;
        self.fsst_buffer += other.fsst_buffer;
        self.shared_prefix += other.shared_prefix;
        self.struct_size += other.struct_size;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::Array;

    fn test_string_roundtrip(input: StringArray) {
        let compressor = LiquidByteViewArray::train_compressor(input.iter());
        let liquid_array = LiquidByteViewArray::from_string_array(&input, compressor.clone());
        let output = liquid_array.to_arrow_array();
        assert_eq!(&input, output.as_string::<i32>());

        let dict_output = liquid_array.to_dict_arrow();
        assert_eq!(
            &input,
            cast(&dict_output, input.data_type())
                .unwrap()
                .as_string::<i32>()
        );
    }

    #[test]
    fn test_roundtrip_with_nulls() {
        let input = StringArray::from(vec![
            Some("hello"),
            None,
            Some("world"),
            None,
            Some("This is a very long string that should be compressed well"),
            Some("hello"),
            Some(""),
            Some("This is a very long string that should be compressed well"),
        ]);
        test_string_roundtrip(input);
    }
    #[test]
    fn test_string_view_roundtrip() {
        let input = StringViewArray::from(vec![
            Some("hello"),
            Some("world"),
            Some("hello"),
            Some("rust"),
            None,
            Some("This is a very long string that should be compressed well"),
            Some(""),
            Some("This is a very long string that should be compressed well"),
        ]);

        let (_compressor, liquid_array) = LiquidByteViewArray::train_from_string_view(&input);
        let output = liquid_array.to_arrow_array();
        assert_eq!(&input, output.as_string_view());
    }

    #[test]
    fn test_binary_view_roundtrip() {
        let input = BinaryViewArray::from(vec![
            Some(b"hello".as_slice()),
            Some(b"world".as_slice()),
            Some(b"hello".as_slice()),
            Some(b"rust\x00".as_slice()),
            None,
            Some(b"This is a very long string that should be compressed well"),
            Some(b""),
            Some(b"This is a very long string that should be compressed well"),
        ]);

        let (_compressor, liquid_array) = LiquidByteViewArray::train_from_binary_view(&input);
        let output = liquid_array.to_arrow_array();
        assert_eq!(&input, output.as_binary_view());
    }

    #[test]
    fn test_compare_equals_comprehensive() {
        struct TestCase<'a> {
            input: Vec<Option<&'a str>>,
            needle: &'a str,
            expected: Vec<Option<bool>>,
        }

        let test_cases = vec![
            TestCase {
                input: vec![Some("hello"), Some("world"), Some("hello"), Some("rust")],
                needle: "hello",
                expected: vec![Some(true), Some(false), Some(true), Some(false)],
            },
            TestCase {
                input: vec![Some("hello"), Some("world"), Some("hello"), Some("rust")],
                needle: "nonexistent",
                expected: vec![Some(false), Some(false), Some(false), Some(false)],
            },
            TestCase {
                input: vec![Some("hello"), None, Some("hello"), None, Some("world")],
                needle: "hello",
                expected: vec![Some(true), None, Some(true), None, Some(false)],
            },
            TestCase {
                input: vec![Some(""), Some("hello"), Some(""), Some("world")],
                needle: "",
                expected: vec![Some(true), Some(false), Some(true), Some(false)],
            },
            TestCase {
                input: vec![Some("short"), Some("longer"), Some("short"), Some("test")],
                needle: "short",
                expected: vec![Some(true), Some(false), Some(true), Some(false)],
            },
        ];

        for case in test_cases {
            let input_array = StringArray::from(case.input.clone());
            let compressor = LiquidByteViewArray::train_compressor(input_array.iter());
            let liquid_array = LiquidByteViewArray::from_string_array(&input_array, compressor);

            let result = liquid_array.compare_equals(case.needle.as_bytes());
            let expected_array = BooleanArray::from(case.expected.clone());

            assert_eq!(result, expected_array);
        }
    }

    #[test]
    fn test_dictionary_view_structure() {
        // Test OffsetView structure
        let offset_view = OffsetView::new(1024, [1, 2, 3, 4, 5, 6, 7, 8]);
        assert_eq!(offset_view.offset(), 1024);
        assert_eq!(offset_view.prefix(), &[1, 2, 3, 4, 5, 6, 7, 8]);

        // Test UInt16Array creation (dictionary keys are now stored directly in UInt16Array)
        let keys = UInt16Array::from(vec![42, 100, 255]);
        assert_eq!(keys.value(0), 42);
        assert_eq!(keys.value(1), 100);
        assert_eq!(keys.value(2), 255);
    }

    #[test]
    fn test_prefix_extraction() {
        let input = StringArray::from(vec!["hello", "world", "test"]);
        let compressor = LiquidByteViewArray::train_compressor(input.iter());
        let liquid_array = LiquidByteViewArray::from_string_array(&input, compressor);

        // With no shared prefix, the offset view prefixes should be the original strings (truncated to 8 bytes)
        assert_eq!(liquid_array.shared_prefix, Vec::<u8>::new());
        assert_eq!(liquid_array.offset_views[0].prefix(), b"hello\0\0\0");
        assert_eq!(liquid_array.offset_views[1].prefix(), b"world\0\0\0");
        assert_eq!(liquid_array.offset_views[2].prefix(), b"test\0\0\0\0");
    }

    #[test]
    fn test_shared_prefix_functionality() {
        // Test case with shared prefix
        let input = StringArray::from(vec![
            "hello_world",
            "hello_rust",
            "hello_test",
            "hello_code",
        ]);
        let compressor = LiquidByteViewArray::train_compressor(input.iter());
        let liquid_array = LiquidByteViewArray::from_string_array(&input, compressor);

        // Should extract "hello_" as shared prefix
        assert_eq!(liquid_array.shared_prefix, b"hello_");

        // Offset view prefixes should be the remaining parts after shared prefix (8 bytes)
        assert_eq!(liquid_array.offset_views[0].prefix(), b"world\0\0\0");
        assert_eq!(liquid_array.offset_views[1].prefix(), b"rust\0\0\0\0");
        assert_eq!(liquid_array.offset_views[2].prefix(), b"test\0\0\0\0");
        assert_eq!(liquid_array.offset_views[3].prefix(), b"code\0\0\0\0");

        // Test roundtrip - should reconstruct original strings correctly
        let output = liquid_array.to_arrow_array();
        assert_eq!(&input, output.as_string::<i32>());

        // Test comparison with shared prefix optimization
        let result = liquid_array.compare_equals(b"hello_rust");
        let expected = BooleanArray::from(vec![false, true, false, false]);
        assert_eq!(result, expected);

        // Test comparison that doesn't match shared prefix
        let result = liquid_array.compare_equals(b"goodbye_world");
        let expected = BooleanArray::from(vec![false, false, false, false]);
        assert_eq!(result, expected);

        // Test partial shared prefix match
        let result = liquid_array.compare_equals(b"hello_");
        let expected = BooleanArray::from(vec![false, false, false, false]);
        assert_eq!(result, expected);
    }

    #[test]
    fn test_shared_prefix_with_short_strings() {
        // Test case: short strings that fit entirely in the 6-byte prefix
        let input = StringArray::from(vec!["abc", "abcde", "abcdef", "abcdefg"]);
        let compressor = LiquidByteViewArray::train_compressor(input.iter());
        let liquid_array = LiquidByteViewArray::from_string_array(&input, compressor);

        // Should extract "abc" as shared prefix
        assert_eq!(liquid_array.shared_prefix, b"abc");

        // Offset view prefixes should be the remaining parts after shared prefix (8 bytes)
        assert_eq!(liquid_array.offset_views[0].prefix(), &[0u8; 8]); // empty after "abc"
        assert_eq!(liquid_array.offset_views[1].prefix(), b"de\0\0\0\0\0\0"); // "de" after "abc"
        assert_eq!(liquid_array.offset_views[2].prefix(), b"def\0\0\0\0\0"); // "def" after "abc"
        assert_eq!(liquid_array.offset_views[3].prefix(), b"defg\0\0\0\0"); // "defg" after "abc"

        // Test roundtrip
        let output = liquid_array.to_arrow_array();
        assert_eq!(&input, output.as_string::<i32>());

        // Test equality comparisons with short strings
        let result = liquid_array.compare_equals(b"abc");
        let expected = BooleanArray::from(vec![true, false, false, false]);
        assert_eq!(result, expected);

        let result = liquid_array.compare_equals(b"abcde");
        let expected = BooleanArray::from(vec![false, true, false, false]);
        assert_eq!(result, expected);

        // Test ordering comparisons that can be resolved by shared prefix
        let result = liquid_array.compare_with(b"ab", &Operator::Gt).unwrap();
        let expected = BooleanArray::from(vec![true, true, true, true]); // All start with "abc" > "ab"
        assert_eq!(result, expected);

        let result = liquid_array.compare_with(b"abcd", &Operator::Lt).unwrap();
        let expected = BooleanArray::from(vec![true, false, false, false]); // Only "abc" < "abcd"
        assert_eq!(result, expected);
    }

    #[test]
    fn test_shared_prefix_contains_complete_strings() {
        // Test case: shared prefix completely contains some strings
        let input = StringArray::from(vec!["data", "database", "data_entry", "data_", "datatype"]);
        let compressor = LiquidByteViewArray::train_compressor(input.iter());
        let liquid_array = LiquidByteViewArray::from_string_array(&input, compressor);

        // Should extract "data" as shared prefix
        assert_eq!(liquid_array.shared_prefix, b"data");

        // Offset view prefixes should be the remaining parts (8 bytes)
        assert_eq!(liquid_array.offset_views[0].prefix(), &[0u8; 8]); // "data" - empty remainder
        assert_eq!(liquid_array.offset_views[1].prefix(), b"base\0\0\0\0"); // "database" - "base" remainder
        assert_eq!(liquid_array.offset_views[2].prefix(), b"_entry\0\0"); // "data_entry" - "_entry" remainder
        assert_eq!(liquid_array.offset_views[3].prefix(), b"_\0\0\0\0\0\0\0"); // "data_" - "_" remainder
        assert_eq!(liquid_array.offset_views[4].prefix(), b"type\0\0\0\0"); // "datatype" - "type" remainder

        // Test roundtrip
        let output = liquid_array.to_arrow_array();
        assert_eq!(&input, output.as_string::<i32>());

        // Test equality with exact shared prefix
        let result = liquid_array.compare_equals(b"data");
        let expected = BooleanArray::from(vec![true, false, false, false, false]);
        assert_eq!(result, expected);

        // Test comparisons where shared prefix helps
        let result = liquid_array.compare_with(b"dat", &Operator::Gt).unwrap();
        let expected = BooleanArray::from(vec![true, true, true, true, true]); // All > "dat"
        assert_eq!(result, expected);

        let result = liquid_array.compare_with(b"datab", &Operator::Lt).unwrap();
        let expected = BooleanArray::from(vec![true, false, true, true, false]); // "data", "data_entry", and "data_" < "datab"
        assert_eq!(result, expected);

        // Test comparison with needle shorter than shared prefix
        let result = liquid_array.compare_with(b"da", &Operator::Gt).unwrap();
        let expected = BooleanArray::from(vec![true, true, true, true, true]); // All > "da"
        assert_eq!(result, expected);

        // Test comparison with needle equal to shared prefix
        let result = liquid_array.compare_with(b"data", &Operator::GtEq).unwrap();
        let expected = BooleanArray::from(vec![true, true, true, true, true]); // All >= "data"
        assert_eq!(result, expected);

        let result = liquid_array.compare_with(b"data", &Operator::Gt).unwrap();
        let expected = BooleanArray::from(vec![false, true, true, true, true]); // All except exact "data" > "data"
        assert_eq!(result, expected);
    }

    #[test]
    fn test_shared_prefix_edge_cases() {
        // Test case 1: All strings are the same (full shared prefix)
        let input = StringArray::from(vec!["identical", "identical", "identical"]);
        let compressor = LiquidByteViewArray::train_compressor(input.iter());
        let liquid_array = LiquidByteViewArray::from_string_array(&input, compressor);

        assert_eq!(liquid_array.shared_prefix, b"identical");
        // All offset view prefixes should be empty
        for offset_view in &liquid_array.offset_views {
            assert_eq!(offset_view.prefix(), &[0u8; 8]);
        }

        // Test roundtrip
        let output = liquid_array.to_arrow_array();
        assert_eq!(&input, output.as_string::<i32>());

        // Test case 2: One string is a prefix of others
        let input = StringArray::from(vec!["hello", "hello_world", "hello_test"]);
        let compressor = LiquidByteViewArray::train_compressor(input.iter());
        let liquid_array = LiquidByteViewArray::from_string_array(&input, compressor);

        assert_eq!(liquid_array.shared_prefix, b"hello");
        assert_eq!(liquid_array.offset_views[0].prefix(), &[0u8; 8]); // empty after "hello"
        assert_eq!(liquid_array.offset_views[1].prefix(), b"_world\0\0");
        assert_eq!(liquid_array.offset_views[2].prefix(), b"_test\0\0\0");

        // Test roundtrip
        let output = liquid_array.to_arrow_array();
        assert_eq!(&input, output.as_string::<i32>());

        // Test case 3: Empty string in array (should limit shared prefix)
        let input = StringArray::from(vec!["", "hello", "hello_world"]);
        let compressor = LiquidByteViewArray::train_compressor(input.iter());
        let liquid_array = LiquidByteViewArray::from_string_array(&input, compressor);

        assert_eq!(liquid_array.shared_prefix, Vec::<u8>::new()); // empty shared prefix
        assert_eq!(liquid_array.offset_views[0].prefix(), &[0u8; 8]);
        assert_eq!(liquid_array.offset_views[1].prefix(), b"hello\0\0\0");
        assert_eq!(liquid_array.offset_views[2].prefix(), b"hello_wo"); // "hello_world" truncated to 8 bytes

        // Test roundtrip
        let output = liquid_array.to_arrow_array();
        assert_eq!(&input, output.as_string::<i32>());
    }

    #[test]
    fn test_memory_layout() {
        let input = StringArray::from(vec!["hello", "world", "test"]);
        let compressor = LiquidByteViewArray::train_compressor(input.iter());
        let liquid_array = LiquidByteViewArray::from_string_array(&input, compressor);

        // Verify memory layout components
        assert_eq!(liquid_array.dictionary_keys.len(), 3);
        // Offset views - one per unique value
        assert_eq!(liquid_array.offset_views.len(), 3); // 3 unique values = 3 offset views
        assert!(liquid_array.nulls().is_none());
        let _raw_buffer = liquid_array
            .fsst_buffer
            .read()
            .unwrap()
            .get_raw_buffer()
            .unwrap();
    }

    #[test]
    fn test_filter_functionality() {
        let input = StringArray::from(vec!["hello", "test", "test", "test", "rust"]);
        let compressor = LiquidByteViewArray::train_compressor(input.iter());
        let liquid_array = LiquidByteViewArray::from_string_array(&input, compressor);

        let filter = BooleanArray::from(vec![true, false, true, false, true]);
        let filtered = liquid_array.filter(&filter);

        assert_eq!(filtered.len(), 3);
        let output = filtered.to_arrow_array();
        let expected = StringArray::from(vec!["hello", "test", "rust"]);
        assert_eq!(&expected, output.as_string::<i32>());
    }

    #[test]
    fn test_memory_efficiency() {
        let input = StringArray::from(vec!["hello", "world", "hello", "world", "hello"]);
        let compressor = LiquidByteViewArray::train_compressor(input.iter());
        let liquid_array = LiquidByteViewArray::from_string_array(&input, compressor);

        // Verify that dictionary views store unique values efficiently
        assert_eq!(liquid_array.dictionary_keys.len(), 5);

        // Verify that FSST buffer contains unique values
        let dict = liquid_array.to_dict_arrow();
        assert_eq!(dict.values().len(), 2); // Only "hello" and "world"
    }

    #[test]
    fn test_to_best_arrow_array() {
        let input = StringArray::from(vec!["hello", "world", "test"]);
        let compressor = LiquidByteViewArray::train_compressor(input.iter());
        let liquid_array = LiquidByteViewArray::from_string_array(&input, compressor);

        let best_array = liquid_array.to_best_arrow_array();
        let dict_array = best_array.as_dictionary::<UInt16Type>();

        // Should return dictionary array as the best encoding
        assert_eq!(dict_array.len(), 3);
        assert_eq!(dict_array.values().len(), 3); // Three unique values
    }

    #[test]
    fn test_data_type() {
        let input = StringArray::from(vec!["hello", "world"]);
        let compressor = LiquidByteViewArray::train_compressor(input.iter());
        let liquid_array = LiquidByteViewArray::from_string_array(&input, compressor);

        // Just verify we can get the data type without errors
        let data_type = liquid_array.data_type();
        assert!(matches!(data_type, LiquidDataType::ByteArray));
    }

    #[test]
    fn test_compare_with_prefix_optimization_fast_path() {
        // Test case 1: Prefix comparison can decide most results without decompression
        // Uses strings with distinct prefixes to test the fast path
        let input = StringArray::from(vec![
            "apple123",  // prefix: "apple\0"
            "banana456", // prefix: "banana"
            "cherry789", // prefix: "cherry"
            "apple999",  // prefix: "apple\0" (same as first)
            "zebra000",  // prefix: "zebra\0"
        ]);

        let compressor = LiquidByteViewArray::train_compressor(input.iter());
        let liquid_array = LiquidByteViewArray::from_string_array(&input, compressor);

        // Test Lt with needle "car" (prefix: "car\0\0\0")
        // Expected: "apple123" < "car" => true, "banana456" < "car" => true, others false
        let result = liquid_array
            .compare_with_inner(b"car", &Operator::Lt)
            .unwrap();
        let expected = BooleanArray::from(vec![true, true, false, true, false]);
        assert_eq!(result, expected);

        // Test Gt with needle "dog" (prefix: "dog\0\0\0")
        // Expected: only "zebra000" > "dog" => true
        let result = liquid_array
            .compare_with_inner(b"dog", &Operator::Gt)
            .unwrap();
        let expected = BooleanArray::from(vec![false, false, false, false, true]);
        assert_eq!(result, expected);

        // Test GtEq with needle "apple" (prefix: "apple\0")
        // Expected: all except "apple123" and "apple999" need decompression, others by prefix
        let result = liquid_array
            .compare_with_inner(b"apple", &Operator::GtEq)
            .unwrap();
        let expected = BooleanArray::from(vec![true, true, true, true, true]);
        assert_eq!(result, expected);
    }

    #[test]
    fn test_compare_with_prefix_optimization_decompression_path() {
        // Test case 2: Cases where prefix comparison is inconclusive and requires decompression
        // Uses strings with identical prefixes but different suffixes
        let input = StringArray::from(vec![
            "prefix_aaa", // prefix: "prefix"
            "prefix_bbb", // prefix: "prefix" (same prefix)
            "prefix_ccc", // prefix: "prefix" (same prefix)
            "prefix_abc", // prefix: "prefix" (same prefix)
            "different",  // prefix: "differ" (different prefix)
        ]);

        let compressor = LiquidByteViewArray::train_compressor(input.iter());
        let liquid_array = LiquidByteViewArray::from_string_array(&input, compressor);

        // Test Lt with needle "prefix_b" - this will require decompression for prefix matches
        // Expected: "prefix_aaa" < "prefix_b" => true, "prefix_bbb" < "prefix_b" => false, etc.
        let result = liquid_array
            .compare_with_inner(b"prefix_b", &Operator::Lt)
            .unwrap();
        let expected = BooleanArray::from(vec![true, false, false, true, true]);
        assert_eq!(result, expected);

        // Test LtEq with needle "prefix_bbb" - exact match case with decompression
        let result = liquid_array
            .compare_with_inner(b"prefix_bbb", &Operator::LtEq)
            .unwrap();
        let expected = BooleanArray::from(vec![true, true, false, true, true]);
        assert_eq!(result, expected);

        // Test Gt with needle "prefix_abc" - requires decompression for prefix matches
        let result = liquid_array
            .compare_with_inner(b"prefix_abc", &Operator::Gt)
            .unwrap();
        let expected = BooleanArray::from(vec![false, true, true, false, false]);
        assert_eq!(result, expected);
    }

    #[test]
    fn test_compare_with_prefix_optimization_edge_cases_and_nulls() {
        // Test case 3: Edge cases including nulls, empty strings, and boundary conditions
        let input = StringArray::from(vec![
            Some(""),           // Empty string (prefix: "\0\0\0\0\0\0")
            None,               // Null value
            Some("a"),          // Single character (prefix: "a\0\0\0\0\0")
            Some("abcdef"),     // Exactly 6 chars (prefix: "abcdef")
            Some("abcdefghij"), // Longer than 6 chars (prefix: "abcdef")
            Some("abcdeg"),     // Differs at position 5 (prefix: "abcdeg")
        ]);

        let compressor = LiquidByteViewArray::train_compressor(input.iter());
        let liquid_array = LiquidByteViewArray::from_string_array(&input, compressor);

        // Test Lt with empty string needle - should test null handling
        let result = liquid_array.compare_with_inner(b"", &Operator::Lt).unwrap();
        let expected = BooleanArray::from(vec![
            Some(false),
            None,
            Some(false),
            Some(false),
            Some(false),
            Some(false),
        ]);
        assert_eq!(result, expected);

        // Test Gt with needle "abcdef" - tests exact prefix match requiring decompression
        let result = liquid_array
            .compare_with_inner(b"abcdef", &Operator::Gt)
            .unwrap();
        let expected = BooleanArray::from(vec![
            Some(false),
            None,
            Some(false),
            Some(false),
            Some(true),
            Some(true),
        ]);
        assert_eq!(result, expected);

        // Test LtEq with needle "b" - tests single character comparisons
        let result = liquid_array
            .compare_with_inner(b"b", &Operator::LtEq)
            .unwrap();
        let expected = BooleanArray::from(vec![
            Some(true),
            None,
            Some(true),
            Some(true),
            Some(true),
            Some(true),
        ]);
        assert_eq!(result, expected);

        // Test GtEq with needle "abcdeg" - tests decompression when prefix exactly matches needle prefix
        let result = liquid_array
            .compare_with_inner(b"abcdeg", &Operator::GtEq)
            .unwrap();
        // b"" >= b"abcdeg" => false
        // null => null
        // b"a" >= b"abcdeg" => false
        // b"abcdef" >= b"abcdeg" => false (because b"abcdef" < b"abcdeg" since 'f' < 'g')
        // b"abcdefghij" >= b"abcdeg" => false (because b"abcdefghij" < b"abcdeg" since 'f' < 'g')
        // b"abcdeg" >= b"abcdeg" => true (exact match)
        let expected = BooleanArray::from(vec![
            Some(false),
            None,
            Some(false),
            Some(false),
            Some(false),
            Some(true),
        ]);
        assert_eq!(result, expected);
    }

    #[test]
    fn test_compare_with_prefix_optimization_utf8_and_binary() {
        // Test case 4: UTF-8 encoded strings and binary data comparisons
        // This demonstrates the advantage of byte-level comparison
        let input = StringArray::from(vec![
            "café",   // UTF-8: [99, 97, 102, 195, 169]
            "naïve",  // UTF-8: [110, 97, 195, 175, 118, 101]
            "résumé", // UTF-8: [114, 195, 169, 115, 117, 109, 195, 169]
            "hello",  // ASCII: [104, 101, 108, 108, 111]
            "世界",   // UTF-8: [228, 184, 150, 231, 149, 140]
        ]);

        let compressor = LiquidByteViewArray::train_compressor(input.iter());
        let liquid_array = LiquidByteViewArray::from_string_array(&input, compressor);

        // Test Lt with UTF-8 needle "naïve" (UTF-8: [110, 97, 195, 175, 118, 101])
        // Expected: "café" < "naïve" => true (99 < 110), "hello" < "naïve" => true, others false
        let naive_bytes = "naïve".as_bytes(); // [110, 97, 195, 175, 118, 101]
        let result = liquid_array
            .compare_with_inner(naive_bytes, &Operator::Lt)
            .unwrap();
        let expected = BooleanArray::from(vec![true, false, false, true, false]);
        assert_eq!(result, expected);

        // Test Gt with UTF-8 needle "café" (UTF-8: [99, 97, 102, 195, 169])
        // Expected: strings with first byte > 99 should be true
        let cafe_bytes = "café".as_bytes(); // [99, 97, 102, 195, 169]
        let result = liquid_array
            .compare_with_inner(cafe_bytes, &Operator::Gt)
            .unwrap();
        let expected = BooleanArray::from(vec![false, true, true, true, true]);
        assert_eq!(result, expected);

        // Test LtEq with Chinese characters "世界" (UTF-8: [228, 184, 150, 231, 149, 140])
        // Expected: only strings with first byte <= 228 should be true, but since 228 is quite high,
        // most Latin characters will be true
        let world_bytes = "世界".as_bytes(); // [228, 184, 150, 231, 149, 140]
        let result = liquid_array
            .compare_with_inner(world_bytes, &Operator::LtEq)
            .unwrap();
        let expected = BooleanArray::from(vec![true, true, true, true, true]);
        assert_eq!(result, expected);

        // Test exact equality with "résumé" using GtEq and LtEq to verify byte-level precision
        let resume_bytes = "résumé".as_bytes(); // [114, 195, 169, 115, 117, 109, 195, 169]
        let gte_result = liquid_array
            .compare_with_inner(resume_bytes, &Operator::GtEq)
            .unwrap();
        let lte_result = liquid_array
            .compare_with_inner(resume_bytes, &Operator::LtEq)
            .unwrap();

        // Check GtEq and LtEq results separately
        // GtEq: "café"(99) >= "résumé"(114) => false, "naïve"(110) >= "résumé"(114) => false,
        //       "résumé"(114) >= "résumé"(114) => true, "hello"(104) >= "résumé"(114) => false,
        //       "世界"(228) >= "résumé"(114) => true
        let gte_expected = BooleanArray::from(vec![false, false, true, false, true]);
        // LtEq: all strings with first byte <= 114 should be true, "世界"(228) should be false
        let lte_expected = BooleanArray::from(vec![true, true, true, true, false]);
        assert_eq!(gte_result, gte_expected);
        assert_eq!(lte_result, lte_expected);
    }

    #[test]
    fn test_evict_to_disk_functionality() {
        let input = StringArray::from(vec![
            "hello world",
            "fsst compression",
            "evict to disk",
            "hello world", // duplicate
            "test data",
        ]);
        let compressor = LiquidByteViewArray::train_compressor(input.iter());
        let liquid_array = LiquidByteViewArray::from_string_array(&input, compressor);

        // Verify initially in memory
        assert!(liquid_array.is_fsst_buffer_in_memory());
        assert!(!liquid_array.is_fsst_buffer_on_disk());

        // Test behavior before eviction
        let original_arrow = liquid_array.to_arrow_array();
        let original_compare = liquid_array.compare_equals(b"hello world");

        // Create a temporary file for eviction
        let temp_path = std::env::temp_dir().join("test_evict_fsst.bin");

        // Evict to disk
        liquid_array.evict_to_disk(temp_path.clone()).unwrap();

        // Verify now on disk
        assert!(!liquid_array.is_fsst_buffer_in_memory());
        assert!(liquid_array.is_fsst_buffer_on_disk());

        // Test that functionality still works after eviction
        let evicted_arrow = liquid_array.to_arrow_array();
        let evicted_compare = liquid_array.compare_equals(b"hello world");

        // Should be identical
        assert_eq!(original_arrow.as_ref(), evicted_arrow.as_ref());
        assert_eq!(original_compare, evicted_compare);

        // Test load from disk
        liquid_array.load_from_disk().unwrap();

        // Verify back in memory
        assert!(liquid_array.is_fsst_buffer_in_memory());
        assert!(!liquid_array.is_fsst_buffer_on_disk());

        // Test that functionality still works after loading
        let loaded_arrow = liquid_array.to_arrow_array();
        let loaded_compare = liquid_array.compare_equals(b"hello world");

        // Should be identical to original
        assert_eq!(original_arrow.as_ref(), loaded_arrow.as_ref());
        assert_eq!(original_compare, loaded_compare);

        // Test double eviction (should be no-op)
        liquid_array.evict_to_disk(temp_path.clone()).unwrap();
        liquid_array.evict_to_disk(temp_path.clone()).unwrap(); // Should not fail

        // Test double load (should be no-op)
        liquid_array.load_from_disk().unwrap();
        liquid_array.load_from_disk().unwrap(); // Should not fail

        // Cleanup
        let _ = std::fs::remove_file(temp_path);
    }

    fn sort_to_indices_test(input: Vec<Option<&str>>, expected_idx: Vec<u32>) {
        let input = StringArray::from(input);
        let compressor = LiquidByteViewArray::train_compressor(input.iter());
        let liquid_array = LiquidByteViewArray::from_string_array(&input, compressor);
        let indices = liquid_array.sort_to_indices().unwrap();
        assert_eq!(indices, UInt32Array::from(expected_idx));
    }

    #[test]
    fn test_sort_to_indices_basic() {
        sort_to_indices_test(
            vec![Some("zebra"), Some("apple"), Some("banana"), Some("cherry")],
            vec![1, 2, 3, 0],
        );

        // with nulls
        sort_to_indices_test(
            vec![
                Some("zebra"),
                None,
                Some("apple"),
                Some("banana"),
                None,
                Some("cherry"),
            ],
            // Expected order: null(1), null(4), apple(2), banana(3), cherry(5), zebra(0)
            vec![1, 4, 2, 3, 5, 0],
        );

        // shared prefix
        sort_to_indices_test(
            vec![
                Some("prefix_zebra"),
                Some("prefix_apple"),
                Some("prefix_banana"),
                Some("prefix_cherry"),
            ],
            vec![1, 2, 3, 0],
        );

        // with duplicate values
        sort_to_indices_test(
            vec![
                Some("apple"),
                Some("banana"),
                Some("apple"),
                Some("cherry"),
                Some("banana"),
            ],
            vec![0, 2, 1, 4, 3],
        );
    }

    #[test]
    fn test_sort_to_indices_test_full_data_comparison() {
        sort_to_indices_test(
            vec![
                Some("pre_apple_abc"),
                Some("pre_banana_abc"),
                Some("pre_apple_def"),
            ],
            vec![0, 2, 1],
        );

        sort_to_indices_test(
            vec![
                Some("pre_apple_abc"),
                Some("pre_banana_abc"),
                Some("pre_appll_abc"),
            ],
            vec![0, 2, 1],
        );
    }

    #[test]
    fn test_disk_read_counter_instrumentation() {
        // Reset counter at start of test to avoid interference from other tests
        reset_disk_read_counter();

        let input = StringArray::from(vec![
            "apple123", // Different prefixes to test prefix optimization
            "banana456",
            "cherry789",
            "zebra000",
        ]);
        let compressor = LiquidByteViewArray::train_compressor(input.iter());
        let liquid_array = LiquidByteViewArray::from_string_array(&input, compressor);

        // Initial state - in memory, no disk reads
        assert!(liquid_array.is_fsst_buffer_in_memory());
        assert_eq!(liquid_array.get_disk_read_count(), 0);

        // Evict to disk
        let temp_path = std::env::temp_dir().join("test_disk_reads.bin");
        liquid_array.evict_to_disk(temp_path.clone()).unwrap();
        assert!(liquid_array.is_fsst_buffer_on_disk());
        assert_eq!(liquid_array.get_disk_read_count(), 0); // Eviction doesn't count as read

        // Reset counter and test operations that should NOT trigger disk reads
        liquid_array.reset_disk_read_count();

        // 3. Prefix-only comparison (should resolve without disk I/O)
        // "car" < "apple123" is false, "car" < "banana456" is false, etc.
        // All comparisons should be resolved by prefix alone
        let result = liquid_array.compare_with(b"car", &Operator::Lt).unwrap();
        assert_eq!(liquid_array.get_disk_read_count(), 0);
        let expected = BooleanArray::from(vec![true, true, false, false]);
        assert_eq!(result, expected);

        // 4. Another prefix-only comparison
        let result = liquid_array.compare_with(b"zzz", &Operator::Gt).unwrap();
        assert_eq!(liquid_array.get_disk_read_count(), 0);
        let expected = BooleanArray::from(vec![false, false, false, false]);
        assert_eq!(result, expected);

        // Reset counter and test operations that SHOULD trigger disk reads
        liquid_array.reset_disk_read_count();

        // 5. Equality comparison - needs to search through compressed values
        let result = liquid_array.compare_equals(b"apple123");
        assert_eq!(liquid_array.get_disk_read_count(), 1); // Should read once
        let expected = BooleanArray::from(vec![true, false, false, false]);
        assert_eq!(result, expected);

        // 6. Another equality comparison - should read again (no caching)
        let result = liquid_array.compare_equals(b"banana456");
        assert_eq!(liquid_array.get_disk_read_count(), 2); // Should read twice total
        let expected = BooleanArray::from(vec![false, true, false, false]);
        assert_eq!(result, expected);

        // 7. Arrow conversion - needs full data
        let _arrow_array = liquid_array.to_arrow_array();
        assert_eq!(liquid_array.get_disk_read_count(), 3); // Should read third time

        // 8. Comparison with equal prefixes (needs decompression)
        let input_equal_prefixes =
            StringArray::from(vec!["prefix_aaa", "prefix_bbb", "prefix_ccc", "different"]);
        let compressor2 = LiquidByteViewArray::train_compressor(input_equal_prefixes.iter());
        let liquid_array2 =
            LiquidByteViewArray::from_string_array(&input_equal_prefixes, compressor2);
        let temp_path2 = std::env::temp_dir().join("test_disk_reads2.bin");
        liquid_array2.evict_to_disk(temp_path2.clone()).unwrap();
        liquid_array2.reset_disk_read_count();

        // This should trigger disk read because prefixes are equal and need decompression
        let result = liquid_array2
            .compare_with(b"prefix_b", &Operator::Lt)
            .unwrap();
        assert_eq!(liquid_array2.get_disk_read_count(), 1); // Should read once for decompression
        let expected = BooleanArray::from(vec![true, false, false, true]);
        assert_eq!(result, expected);

        // Cleanup
        let _ = std::fs::remove_file(temp_path);
        let _ = std::fs::remove_file(temp_path2);
    }
}
