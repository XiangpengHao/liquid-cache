//! LiquidByteViewArray

use arrow::array::BinaryViewArray;
use arrow::array::{
    Array, ArrayAccessor, ArrayIter, ArrayRef, BinaryArray, BooleanArray, BooleanBuilder,
    DictionaryArray, GenericByteArray, StringArray, StringViewArray, UInt16Array, UInt32Array,
    cast::AsArray, types::UInt16Type,
};
use arrow::buffer::{BooleanBuffer, NullBuffer};
use arrow::compute::{cast, kernels, sort_to_indices};
use arrow::datatypes::ByteArrayType;
use datafusion::logical_expr::{ColumnarValue, Operator};
use datafusion::physical_expr_common::datum::apply_cmp;
use datafusion::physical_plan::PhysicalExpr;
use datafusion::physical_plan::expressions::{BinaryExpr, LikeExpr, Literal};
use fsst::Compressor;
use std::any::Any;
use std::fmt::Display;
use std::path::PathBuf;
use std::sync::Arc;

#[cfg(test)]
use std::cell::Cell;

#[cfg(test)]
thread_local! {
    static DISK_READ_COUNTER: Cell<usize> = const { Cell::new(0)};
    static FULL_DATA_COMPARISON_COUNTER: Cell<usize> = const { Cell::new(0)};
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
    LiquidArray, LiquidArrayRef, LiquidDataType, LiquidHybridArray,
    byte_array::{ArrowByteType, get_string_needle},
};
use crate::liquid_array::ipc::{ByteViewArrayHeader, LiquidIPCHeader};
use crate::liquid_array::raw::fsst_array::{RawFsstBuffer, train_compressor};
use crate::liquid_array::{IoRequest, LiquidHybridArrayRef};
use crate::utils::CheckedDictionaryArray;

#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub(crate) struct OffsetView {
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

/// Memory buffer for FSST buffer
#[derive(Debug, Clone)]
pub struct MemoryBuffer {
    buffer: Arc<RawFsstBuffer>,
}

impl MemoryBuffer {
    pub(crate) fn new(raw_buffer: Arc<RawFsstBuffer>) -> Self {
        Self { buffer: raw_buffer }
    }
}

/// Disk buffer for FSST buffer
#[derive(Debug, Clone)]
pub struct DiskBuffer {
    path: PathBuf,
    uncompressed_bytes: usize,
}

impl DiskBuffer {
    pub(crate) fn new(path: PathBuf, uncompressed_bytes: usize) -> Self {
        Self {
            path,
            uncompressed_bytes,
        }
    }
}

mod sealed {
    pub trait Sealed {}
}

/// Trait for FSST buffer - can be in memory or on disk
pub trait FsstBuffer: std::fmt::Debug + Clone + sealed::Sealed {
    /// Get the raw FSST buffer, loading from disk if necessary
    fn get_fsst_buffer(&self) -> Result<Arc<RawFsstBuffer>, IoRequest>;

    /// Get the memory size of the FSST buffer
    fn get_array_memory_size(&self) -> usize;

    /// Get the uncompressed bytes of the FSST buffer
    fn uncompressed_bytes(&self) -> usize;
}

impl sealed::Sealed for MemoryBuffer {}
impl sealed::Sealed for DiskBuffer {}

impl FsstBuffer for MemoryBuffer {
    fn get_fsst_buffer(&self) -> Result<Arc<RawFsstBuffer>, IoRequest> {
        Ok(self.buffer.clone())
    }

    fn get_array_memory_size(&self) -> usize {
        self.buffer.get_memory_size()
    }

    fn uncompressed_bytes(&self) -> usize {
        self.buffer.uncompressed_bytes()
    }
}

impl FsstBuffer for DiskBuffer {
    fn get_fsst_buffer(&self) -> Result<Arc<RawFsstBuffer>, IoRequest> {
        Err(IoRequest {
            path: self.path.clone(),
        })
    }

    fn get_array_memory_size(&self) -> usize {
        0
    }

    fn uncompressed_bytes(&self) -> usize {
        self.uncompressed_bytes
    }
}

impl LiquidArray for LiquidByteViewArray<MemoryBuffer> {
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
        let dict = self.to_arrow_array().expect("InMemoryFsstBuffer");
        Arc::new(dict)
    }

    fn to_best_arrow_array(&self) -> ArrayRef {
        let dict = self.to_dict_arrow().expect("InMemoryFsstBuffer");
        Arc::new(dict)
    }

    fn filter(&self, selection: &BooleanBuffer) -> LiquidArrayRef {
        let filtered = filter_inner(self, selection);
        Arc::new(filtered)
    }

    fn try_eval_predicate(
        &self,
        expr: &Arc<dyn PhysicalExpr>,
        filter: &BooleanBuffer,
    ) -> Option<BooleanArray> {
        let filtered = filter_inner(self, filter);
        try_eval_predicate_inner(expr, &filtered).expect("InMemoryFsstBuffer")
    }

    fn to_bytes(&self) -> Vec<u8> {
        self.to_bytes_inner().expect("InMemoryFsstBuffer")
    }

    fn data_type(&self) -> LiquidDataType {
        LiquidDataType::ByteArray
    }

    fn squeeze(&self) -> Option<(LiquidHybridArrayRef, (bytes::Bytes, std::ops::Range<u64>))> {
        // Serialize full IPC bytes first
        let bytes = match self.to_bytes_inner() {
            Ok(b) => b,
            Err(_) => return None,
        };

        const IPC_HEADER_SIZE: usize = LiquidIPCHeader::size();
        const VIEW_HEADER_SIZE: usize = ByteViewArrayHeader::size();
        let header_size = IPC_HEADER_SIZE + VIEW_HEADER_SIZE;

        assert!(bytes.len() >= header_size);

        let fsst_size = u32::from_le_bytes(
            bytes[IPC_HEADER_SIZE + 12..IPC_HEADER_SIZE + 16]
                .try_into()
                .unwrap(),
        ) as usize;

        // FSST starts at the first 8-byte boundary after headers
        let fsst_start = (header_size + 7) & !7;
        let fsst_end = fsst_start + fsst_size;

        assert!(fsst_end <= bytes.len());

        // Build the hybrid (disk-backed FSST) view
        let disk = DiskBuffer::new(
            std::path::PathBuf::new(),
            self.fsst_buffer.uncompressed_bytes(),
        );
        let hybrid = LiquidByteViewArray::<DiskBuffer> {
            dictionary_keys: self.dictionary_keys.clone(),
            offset_views: self.offset_views.clone(),
            fsst_buffer: disk,
            original_arrow_type: self.original_arrow_type,
            shared_prefix: self.shared_prefix.clone(),
            compressor: self.compressor.clone(),
        };

        let range = (fsst_start as u64)..(fsst_end as u64);
        let bytes = bytes::Bytes::from(bytes);
        Some((Arc::new(hybrid) as LiquidHybridArrayRef, (bytes, range)))
    }
}

impl LiquidHybridArray for LiquidByteViewArray<DiskBuffer> {
    /// Get the underlying any type.
    fn as_any(&self) -> &dyn Any {
        self
    }

    /// Get the memory size of the Liquid array.
    fn get_array_memory_size(&self) -> usize {
        self.get_detailed_memory_usage().total()
    }

    /// Get the length of the Liquid array.
    fn len(&self) -> usize {
        self.dictionary_keys.len()
    }

    /// Check if the Liquid array is empty.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Convert the Liquid array to an Arrow array.
    fn to_arrow_array(&self) -> Result<ArrayRef, IoRequest> {
        self.to_arrow_array()
    }

    /// Convert the Liquid array to an Arrow array.
    /// Except that it will pick the best encoding for the arrow array.
    /// Meaning that it may not obey the data type of the original arrow array.
    fn to_best_arrow_array(&self) -> Result<ArrayRef, IoRequest> {
        self.to_arrow_array()
    }

    /// Get the logical data type of the Liquid array.
    fn data_type(&self) -> LiquidDataType {
        LiquidDataType::ByteArray
    }

    /// Serialize the Liquid array to a byte array.
    fn to_bytes(&self) -> Result<Vec<u8>, IoRequest> {
        self.to_bytes_inner()
    }

    /// Filter the Liquid array with a boolean buffer.
    fn filter(&self, selection: &BooleanBuffer) -> Result<LiquidHybridArrayRef, IoRequest> {
        Ok(Arc::new(filter_inner(self, selection)))
    }

    /// Filter the Liquid array with a boolean array and return an **arrow array**.
    fn filter_to_arrow(&self, selection: &BooleanBuffer) -> Result<ArrayRef, IoRequest> {
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
    ) -> Result<Option<BooleanArray>, IoRequest> {
        Ok(None)
    }

    /// Feed IO data to the `LiquidHybridArray`.
    /// Returns the in-memory `LiquidArray`.
    /// This is the bridge from hybrid array to in-memory array.
    fn soak(&self, data: bytes::Bytes) -> LiquidArrayRef {
        // `data` is the raw FSST buffer bytes (no IPC header)
        let buffer = MemoryBuffer::new(Arc::new(RawFsstBuffer::from_bytes(data)));

        let in_memory_array = LiquidByteViewArray::<MemoryBuffer> {
            dictionary_keys: self.dictionary_keys.clone(),
            offset_views: self.offset_views.clone(),
            fsst_buffer: buffer,
            original_arrow_type: self.original_arrow_type,
            shared_prefix: self.shared_prefix.clone(),
            compressor: self.compressor.clone(),
        };
        Arc::new(in_memory_array)
    }
}

fn filter_inner<B: FsstBuffer>(
    array: &LiquidByteViewArray<B>,
    filter: &BooleanBuffer,
) -> LiquidByteViewArray<B> {
    // Only filter the dictionary keys, not the offset views!
    // Offset views reference unique values in FSST buffer and should remain unchanged

    // Filter the dictionary keys using Arrow's built-in filter functionality
    let filter = BooleanArray::new(filter.clone(), None);
    let filtered_keys = arrow::compute::filter(&array.dictionary_keys, &filter).unwrap();
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

fn try_eval_predicate_inner<B: FsstBuffer>(
    expr: &Arc<dyn PhysicalExpr>,
    array: &LiquidByteViewArray<B>,
) -> Result<Option<BooleanArray>, IoRequest> {
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
            let dict_array = array.to_dict_arrow()?;
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
                let filtered = result.into_array(array.len()).unwrap().as_boolean().clone();
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
        let arrow_dict = array.to_dict_arrow()?;

        let lhs = ColumnarValue::Array(Arc::new(arrow_dict));
        let rhs = ColumnarValue::Scalar(literal.value().clone());

        let result = match (like_expr.negated(), like_expr.case_insensitive()) {
            (false, false) => apply_cmp(&lhs, &rhs, arrow::compute::like),
            (true, false) => apply_cmp(&lhs, &rhs, arrow::compute::nlike),
            (false, true) => apply_cmp(&lhs, &rhs, arrow::compute::ilike),
            (true, true) => apply_cmp(&lhs, &rhs, arrow::compute::nilike),
        };
        if let Ok(result) = result {
            let filtered = result.into_array(array.len()).unwrap().as_boolean().clone();
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
pub struct LiquidByteViewArray<B: FsstBuffer> {
    /// Dictionary keys (u16) - one per array element, using Arrow's UInt16Array for zero-copy
    pub(crate) dictionary_keys: UInt16Array,
    /// Offset views containing offset (u32) and prefix (8 bytes) - one per unique value
    pub(crate) offset_views: Vec<OffsetView>,
    /// FSST-compressed buffer (can be in memory or on disk)
    pub(crate) fsst_buffer: B,
    /// Used to convert back to the original arrow type
    pub(crate) original_arrow_type: ArrowByteType,
    /// Shared prefix across all strings in the array
    pub(crate) shared_prefix: Vec<u8>,
    /// Compressor for decompression
    pub(crate) compressor: Arc<Compressor>,
}

impl<B: FsstBuffer> std::fmt::Debug for LiquidByteViewArray<B> {
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

impl<B: FsstBuffer> LiquidByteViewArray<B> {
    /// Create a LiquidByteViewArray from an Arrow StringViewArray
    pub fn from_string_view_array(
        array: &StringViewArray,
        compressor: Arc<Compressor>,
    ) -> LiquidByteViewArray<MemoryBuffer> {
        Self::from_view_array_inner(array, compressor, ArrowByteType::Utf8View)
    }

    /// Create a LiquidByteViewArray from an Arrow BinaryViewArray
    pub fn from_binary_view_array(
        array: &BinaryViewArray,
        compressor: Arc<Compressor>,
    ) -> LiquidByteViewArray<MemoryBuffer> {
        Self::from_view_array_inner(array, compressor, ArrowByteType::BinaryView)
    }

    /// Create a LiquidByteViewArray from an Arrow StringArray
    pub fn from_string_array(
        array: &StringArray,
        compressor: Arc<Compressor>,
    ) -> LiquidByteViewArray<MemoryBuffer> {
        Self::from_byte_array_inner(array, compressor, ArrowByteType::Utf8)
    }

    /// Create a LiquidByteViewArray from an Arrow BinaryArray
    pub fn from_binary_array(
        array: &BinaryArray,
        compressor: Arc<Compressor>,
    ) -> LiquidByteViewArray<MemoryBuffer> {
        Self::from_byte_array_inner(array, compressor, ArrowByteType::Binary)
    }

    /// Train a compressor from an Arrow StringViewArray
    pub fn train_from_string_view(
        array: &StringViewArray,
    ) -> (Arc<Compressor>, LiquidByteViewArray<MemoryBuffer>) {
        let compressor = Self::train_compressor(array.iter());
        (
            compressor.clone(),
            Self::from_view_array_inner(array, compressor, ArrowByteType::Utf8View),
        )
    }

    /// Train a compressor from an Arrow BinaryViewArray
    pub fn train_from_binary_view(
        array: &BinaryViewArray,
    ) -> (Arc<Compressor>, LiquidByteViewArray<MemoryBuffer>) {
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

    /// Convert to Arrow DictionaryArray
    pub fn to_dict_arrow(&self) -> Result<DictionaryArray<UInt16Type>, IoRequest> {
        let keys_array = self.dictionary_keys.clone();

        // Convert raw FSST buffer to values using our offset views
        let raw_buffer = self.fsst_buffer.get_fsst_buffer()?;

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

        Ok(unsafe { DictionaryArray::<UInt16Type>::new_unchecked(keys_array, values) })
    }

    /// Convert to Arrow array with original type
    pub fn to_arrow_array(&self) -> Result<ArrayRef, IoRequest> {
        let dict = self.to_dict_arrow()?;
        Ok(cast(&dict, &self.original_arrow_type.to_arrow_type()).unwrap())
    }

    /// Get the nulls buffer
    pub fn nulls(&self) -> Option<&NullBuffer> {
        self.dictionary_keys.nulls()
    }

    /// Compare with prefix optimization and fallback to Arrow operations
    pub fn compare_with(&self, needle: &[u8], op: &Operator) -> Result<BooleanArray, IoRequest> {
        match op {
            // Handle equality operations with existing optimized methods
            Operator::Eq => self.compare_equals(needle),
            Operator::NotEq => self.compare_not_equals(needle),

            // Handle ordering operations with prefix optimization
            Operator::Lt | Operator::LtEq | Operator::Gt | Operator::GtEq => {
                self.compare_with_inner(needle, op)
            }

            // For other operations, fall back to Arrow operations
            _ => self.compare_with_arrow_fallback(needle, op),
        }
    }

    /// Get detailed memory usage of the byte view array
    pub fn get_detailed_memory_usage(&self) -> ByteViewArrayMemoryUsage {
        ByteViewArrayMemoryUsage {
            dictionary_key: self.dictionary_keys.get_array_memory_size(),
            offsets: self.offset_views.len() * std::mem::size_of::<OffsetView>(),
            fsst_buffer: self.fsst_buffer.get_array_memory_size(),
            shared_prefix: self.shared_prefix.len(),
            struct_size: std::mem::size_of::<Self>(),
        }
    }

    /// Sort the array and return indices that would sort the array
    /// This implements efficient sorting as described in the design document:
    /// 1. First sort the dictionary using prefixes to delay decompression
    /// 2. If decompression is needed, decompress the entire array at once
    /// 3. Use dictionary ranks to sort the final keys
    pub fn sort_to_indices(&self) -> Result<UInt32Array, IoRequest> {
        // if distinct ratio is more than 10%, use arrow sort.
        if self.offset_views.len() > (self.dictionary_keys.len() / 10) {
            let array = self.to_dict_arrow()?;
            let sorted_array = sort_to_indices(&array, None, None).unwrap();
            Ok(sorted_array)
        } else {
            self.sort_to_indices_inner()
        }
    }

    /// Check if the FSST buffer is currently stored on disk
    pub fn is_fsst_buffer_on_disk(&self) -> bool {
        self.fsst_buffer.get_fsst_buffer().is_err()
    }

    /// Check if the FSST buffer is currently stored in memory
    pub fn is_fsst_buffer_in_memory(&self) -> bool {
        self.fsst_buffer.get_fsst_buffer().is_ok()
    }

    /// Get the length of the array
    pub fn len(&self) -> usize {
        self.dictionary_keys.len()
    }

    /// Is the array empty?
    pub fn is_empty(&self) -> bool {
        self.dictionary_keys.is_empty()
    }
}

impl<B: FsstBuffer> LiquidByteViewArray<B> {
    /// Generic implementation for view arrays (StringViewArray and BinaryViewArray)
    fn from_view_array_inner<T>(
        array: &T,
        compressor: Arc<Compressor>,
        arrow_type: ArrowByteType,
    ) -> LiquidByteViewArray<MemoryBuffer>
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
    ) -> LiquidByteViewArray<MemoryBuffer> {
        let dict = CheckedDictionaryArray::from_byte_array::<T>(array);
        Self::from_dict_array_inner(dict, compressor, arrow_type)
    }

    /// Core implementation that converts a CheckedDictionaryArray to LiquidByteViewArray
    fn from_dict_array_inner(
        dict: CheckedDictionaryArray,
        compressor: Arc<Compressor>,
        arrow_type: ArrowByteType,
    ) -> LiquidByteViewArray<MemoryBuffer> {
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

        for (i, byte_offset) in byte_offsets.iter().enumerate().take(values.len()) {
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

            offset_views.push(OffsetView::new(*byte_offset, prefix));
        }

        assert_eq!(values.len(), byte_offsets.len() - 1);
        offset_views.push(OffsetView::new(byte_offsets[values.len()], [0u8; 8]));

        LiquidByteViewArray {
            dictionary_keys: keys,
            offset_views,
            fsst_buffer: MemoryBuffer {
                buffer: Arc::new(raw_fsst_buffer),
            },
            original_arrow_type: arrow_type,
            shared_prefix,
            compressor,
        }
    }

    /// Compare equality with a byte needle
    fn compare_equals(&self, needle: &[u8]) -> Result<BooleanArray, IoRequest> {
        // Fast path 1: Check shared prefix
        let shared_prefix_len = self.shared_prefix.len();
        if needle.len() < shared_prefix_len || needle[..shared_prefix_len] != self.shared_prefix {
            return Ok(BooleanArray::new(
                BooleanBuffer::new_unset(self.dictionary_keys.len()),
                self.nulls().cloned(),
            ));
        }

        let raw_buffer = self.fsst_buffer.get_fsst_buffer()?;
        Ok(self.compare_equals_in_memory(needle, &raw_buffer))
    }

    fn compare_equals_with_raw_buffer(
        &self,
        needle: &[u8],
        raw_buffer: &RawFsstBuffer,
    ) -> BooleanArray {
        let compressed_needle = self.compressor.compress(needle);

        // Find the matching dictionary value (early exit since values are unique)
        let num_unique = self.offset_views.len().saturating_sub(1);
        let mut matching_dict_key = None;

        for i in 0..num_unique {
            let start_offset = self.offset_views[i].offset();
            let end_offset = self.offset_views[i + 1].offset();

            let compressed_value = raw_buffer.get_compressed_slice(start_offset, end_offset);
            if compressed_value == compressed_needle.as_slice() {
                matching_dict_key = Some(i as u16);
                break; // Early exit - dictionary values are unique
            }
        }
        let Some(matching_dict_key) = matching_dict_key else {
            return BooleanArray::new(
                BooleanBuffer::new_unset(self.dictionary_keys.len()),
                self.nulls().cloned(),
            );
        };

        let to_compare = UInt16Array::new_scalar(matching_dict_key);
        arrow::compute::kernels::cmp::eq(&self.dictionary_keys, &to_compare).unwrap()
    }

    fn compare_equals_in_memory(
        &self,
        needle: &[u8],
        raw_buffer: &Arc<RawFsstBuffer>,
    ) -> BooleanArray {
        self.compare_equals_with_raw_buffer(needle, raw_buffer)
    }

    /// Compare not equals with a byte needle
    fn compare_not_equals(&self, needle: &[u8]) -> Result<BooleanArray, IoRequest> {
        let result = self.compare_equals(needle)?;
        let (values, nulls) = result.into_parts();
        let values = !&values;
        Ok(BooleanArray::new(values, nulls))
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
    fn compare_with_inner(&self, needle: &[u8], op: &Operator) -> Result<BooleanArray, IoRequest> {
        // Try to short-circuit based on shared prefix comparison
        if let Some(result) = self.try_shared_prefix_short_circuit(needle, op) {
            return Ok(result);
        }

        let needle_suffix = &needle[self.shared_prefix.len()..];
        let num_unique = self.offset_views.len().saturating_sub(1);
        let mut dict_results = Vec::with_capacity(num_unique);
        let mut needs_full_comparison = Vec::new();

        // Try prefix comparison for each unique value
        for i in 0..num_unique {
            let prefix = self.offset_views[i].prefix();

            // Compare prefix with needle_suffix
            let cmp_len = std::cmp::min(8, needle_suffix.len());
            let prefix_slice = &prefix[..cmp_len];
            let needle_slice = &needle_suffix[..cmp_len];

            match prefix_slice.cmp(needle_slice) {
                std::cmp::Ordering::Less => {
                    // Prefix < needle, so full string < needle
                    let result = match op {
                        Operator::Lt | Operator::LtEq => Some(true),
                        Operator::Gt | Operator::GtEq => Some(false),
                        _ => None,
                    };
                    dict_results.push(result);
                }
                std::cmp::Ordering::Greater => {
                    // Prefix > needle, so full string > needle
                    let result = match op {
                        Operator::Lt | Operator::LtEq => Some(false),
                        Operator::Gt | Operator::GtEq => Some(true),
                        _ => None,
                    };
                    dict_results.push(result);
                }
                std::cmp::Ordering::Equal => {
                    dict_results.push(None);
                    needs_full_comparison.push(i);
                }
            }
        }

        // For values needing full comparison, load buffer and decompress
        if !needs_full_comparison.is_empty() {
            let raw_buffer = self.fsst_buffer.get_fsst_buffer()?;

            let mut decompressed_buffer = Vec::with_capacity(1024 * 1024 * 2);
            for &i in &needs_full_comparison {
                let start_offset = self.offset_views[i].offset();
                let end_offset = self.offset_views[i + 1].offset();

                let compressed_value = raw_buffer.get_compressed_slice(start_offset, end_offset);
                decompressed_buffer.clear();
                let decompressed_len = unsafe {
                    let slice = std::slice::from_raw_parts_mut(
                        decompressed_buffer.as_mut_ptr() as *mut std::mem::MaybeUninit<u8>,
                        decompressed_buffer.capacity(),
                    );
                    self.compressor
                        .decompressor()
                        .decompress_into(compressed_value, slice)
                };
                unsafe {
                    decompressed_buffer.set_len(decompressed_len);
                }

                let value_cmp = decompressed_buffer.as_slice().cmp(needle);
                let result = match (op, value_cmp) {
                    (Operator::Lt, std::cmp::Ordering::Less) => Some(true),
                    (Operator::Lt, _) => Some(false),
                    (Operator::LtEq, std::cmp::Ordering::Less | std::cmp::Ordering::Equal) => {
                        Some(true)
                    }
                    (Operator::LtEq, _) => Some(false),
                    (Operator::Gt, std::cmp::Ordering::Greater) => Some(true),
                    (Operator::Gt, _) => Some(false),
                    (Operator::GtEq, std::cmp::Ordering::Greater | std::cmp::Ordering::Equal) => {
                        Some(true)
                    }
                    (Operator::GtEq, _) => Some(false),
                    _ => None,
                };
                dict_results[i] = result;
            }
        }

        // Map dictionary results to array results
        let mut builder = BooleanBuilder::with_capacity(self.dictionary_keys.len());
        for &dict_key in self.dictionary_keys.values().iter() {
            let matches = if dict_key as usize >= dict_results.len() {
                false
            } else {
                dict_results[dict_key as usize].unwrap_or(false)
            };
            builder.append_value(matches);
        }

        let mut result = builder.finish();
        // Preserve nulls from dictionary keys
        if let Some(nulls) = self.nulls() {
            let (values, _) = result.into_parts();
            result = BooleanArray::new(values, Some(nulls.clone()));
        }

        Ok(result)
    }

    /// Fallback to Arrow operations for unsupported operations
    fn compare_with_arrow_fallback(
        &self,
        needle: &[u8],
        op: &Operator,
    ) -> Result<BooleanArray, IoRequest> {
        let dict_array = self.to_dict_arrow()?;
        let needle_scalar = datafusion::common::ScalarValue::Binary(Some(needle.to_vec()));
        let lhs = ColumnarValue::Array(Arc::new(dict_array));
        let rhs = ColumnarValue::Scalar(needle_scalar);

        let result = match op {
            Operator::LikeMatch => apply_cmp(&lhs, &rhs, arrow::compute::like),
            Operator::ILikeMatch => apply_cmp(&lhs, &rhs, arrow::compute::ilike),
            Operator::NotLikeMatch => apply_cmp(&lhs, &rhs, arrow::compute::nlike),
            Operator::NotILikeMatch => apply_cmp(&lhs, &rhs, arrow::compute::nilike),
            _ => {
                unreachable!()
            }
        };

        match result.expect("ArrowError") {
            ColumnarValue::Array(arr) => Ok(arr.as_boolean().clone()),
            ColumnarValue::Scalar(_) => unreachable!(),
        }
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

    fn sort_to_indices_inner(&self) -> Result<UInt32Array, IoRequest> {
        // Step 1: Get dictionary ranks using prefix optimization
        let dict_ranks = self.get_dictionary_ranks()?;

        // Step 2: Partition array indices into nulls and non-nulls, then sort non-nulls
        let mut non_null_indices = Vec::with_capacity(self.dictionary_keys.len());

        let mut array_indices = Vec::with_capacity(self.dictionary_keys.len());
        for array_idx in 0..self.dictionary_keys.len() as u32 {
            if self.dictionary_keys.is_null(array_idx as usize) {
                array_indices.push(array_idx);
            } else {
                non_null_indices.push(array_idx);
            }
        }

        // Sort non-null indices by their dictionary ranks
        non_null_indices.sort_unstable_by_key(|&array_idx| unsafe {
            let dict_key = self.dictionary_keys.value_unchecked(array_idx as usize);
            dict_ranks.get_unchecked(dict_key as usize)
        });

        array_indices.extend(non_null_indices);

        Ok(UInt32Array::from(array_indices))
    }

    /// Get dictionary ranks using prefix optimization and lazy decompression
    /// Returns a mapping from dictionary key to its rank in sorted order
    fn get_dictionary_ranks(&self) -> Result<Vec<u16>, IoRequest> {
        let num_unique = self.offset_views.len().saturating_sub(1);
        let mut dict_indices: Vec<u32> = (0..num_unique as u32).collect();

        let mut decompressed: Option<BinaryArray> = None;
        let raw_buffer = self.fsst_buffer.get_fsst_buffer()?;

        // Sort using prefix optimization first, then full strings when needed
        dict_indices.sort_unstable_by(|&a, &b| unsafe {
            // First try prefix comparison - no need to include shared_prefix since all strings have it
            let prefix_a = self.offset_views.get_unchecked(a as usize).prefix();
            let prefix_b = self.offset_views.get_unchecked(b as usize).prefix();

            let prefix_cmp = prefix_a.cmp(prefix_b);

            if prefix_cmp != std::cmp::Ordering::Equal {
                // Prefix comparison is sufficient
                prefix_cmp
            } else {
                // Prefixes are equal, need full string comparison
                // This will trigger decompression on first call if needed
                match &decompressed {
                    Some(decompressed) => {
                        let string_a = decompressed.value_unchecked(a as usize);
                        let string_b = decompressed.value_unchecked(b as usize);
                        string_a.cmp(string_b)
                    }
                    None => {
                        let (values_buffer, offsets_buffer) = raw_buffer
                            .to_uncompressed(&self.compressor.decompressor(), &self.offset_views);

                        let binary_array =
                            BinaryArray::new_unchecked(offsets_buffer, values_buffer, None);

                        let string_a = binary_array.value(a as usize);
                        let string_b = binary_array.value(b as usize);
                        let rt = string_a.cmp(string_b);
                        decompressed = Some(binary_array);
                        rt
                    }
                }
            }
        });

        // Convert sorted indices to rank mapping
        let mut dict_ranks = vec![0u16; dict_indices.len()];
        for (rank, dict_key) in dict_indices.into_iter().enumerate() {
            dict_ranks[dict_key as usize] = rank as u16;
        }

        Ok(dict_ranks)
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
    use rand::{Rng, SeedableRng};

    fn test_string_roundtrip(input: StringArray) {
        let compressor = LiquidByteViewArray::<MemoryBuffer>::train_compressor(input.iter());
        let liquid_array =
            LiquidByteViewArray::<MemoryBuffer>::from_string_array(&input, compressor.clone());
        let output = liquid_array.to_arrow_array().expect("InMemoryFsstBuffer");
        assert_eq!(&input, output.as_string::<i32>());

        let dict_output = liquid_array.to_dict_arrow().unwrap();
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

        let (_compressor, liquid_array) =
            LiquidByteViewArray::<MemoryBuffer>::train_from_string_view(&input);
        let output = liquid_array.to_arrow_array().unwrap();
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

        let (_compressor, liquid_array) =
            LiquidByteViewArray::<MemoryBuffer>::train_from_binary_view(&input);
        let output = liquid_array.to_arrow_array().unwrap();
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
            let compressor =
                LiquidByteViewArray::<MemoryBuffer>::train_compressor(input_array.iter());
            let liquid_array =
                LiquidByteViewArray::<MemoryBuffer>::from_string_array(&input_array, compressor);

            let result = liquid_array.compare_equals(case.needle.as_bytes()).unwrap();
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
        let compressor = LiquidByteViewArray::<MemoryBuffer>::train_compressor(input.iter());
        let liquid_array =
            LiquidByteViewArray::<MemoryBuffer>::from_string_array(&input, compressor);

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
        let compressor = LiquidByteViewArray::<MemoryBuffer>::train_compressor(input.iter());
        let liquid_array =
            LiquidByteViewArray::<MemoryBuffer>::from_string_array(&input, compressor);

        // Should extract "hello_" as shared prefix
        assert_eq!(liquid_array.shared_prefix, b"hello_");

        // Offset view prefixes should be the remaining parts after shared prefix (8 bytes)
        assert_eq!(liquid_array.offset_views[0].prefix(), b"world\0\0\0");
        assert_eq!(liquid_array.offset_views[1].prefix(), b"rust\0\0\0\0");
        assert_eq!(liquid_array.offset_views[2].prefix(), b"test\0\0\0\0");
        assert_eq!(liquid_array.offset_views[3].prefix(), b"code\0\0\0\0");

        // Test roundtrip - should reconstruct original strings correctly
        let output = liquid_array.to_arrow_array().unwrap();
        assert_eq!(&input, output.as_string::<i32>());

        // Test comparison with shared prefix optimization
        let result = liquid_array.compare_equals(b"hello_rust").unwrap();
        let expected = BooleanArray::from(vec![false, true, false, false]);
        assert_eq!(result, expected);

        // Test comparison that doesn't match shared prefix
        let result = liquid_array.compare_equals(b"goodbye_world").unwrap();
        let expected = BooleanArray::from(vec![false, false, false, false]);
        assert_eq!(result, expected);

        // Test partial shared prefix match
        let result = liquid_array.compare_equals(b"hello_").unwrap();
        let expected = BooleanArray::from(vec![false, false, false, false]);
        assert_eq!(result, expected);
    }

    #[test]
    fn test_shared_prefix_with_short_strings() {
        // Test case: short strings that fit entirely in the 6-byte prefix
        let input = StringArray::from(vec!["abc", "abcde", "abcdef", "abcdefg"]);
        let compressor = LiquidByteViewArray::<MemoryBuffer>::train_compressor(input.iter());
        let liquid_array =
            LiquidByteViewArray::<MemoryBuffer>::from_string_array(&input, compressor);

        // Should extract "abc" as shared prefix
        assert_eq!(liquid_array.shared_prefix, b"abc");

        // Offset view prefixes should be the remaining parts after shared prefix (8 bytes)
        assert_eq!(liquid_array.offset_views[0].prefix(), &[0u8; 8]); // empty after "abc"
        assert_eq!(liquid_array.offset_views[1].prefix(), b"de\0\0\0\0\0\0"); // "de" after "abc"
        assert_eq!(liquid_array.offset_views[2].prefix(), b"def\0\0\0\0\0"); // "def" after "abc"
        assert_eq!(liquid_array.offset_views[3].prefix(), b"defg\0\0\0\0"); // "defg" after "abc"

        // Test roundtrip
        let output = liquid_array.to_arrow_array().unwrap();
        assert_eq!(&input, output.as_string::<i32>());

        // Test equality comparisons with short strings
        let result = liquid_array.compare_equals(b"abc").unwrap();
        let expected = BooleanArray::from(vec![true, false, false, false]);
        assert_eq!(result, expected);

        let result = liquid_array.compare_equals(b"abcde").unwrap();
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
        let compressor = LiquidByteViewArray::<MemoryBuffer>::train_compressor(input.iter());
        let liquid_array =
            LiquidByteViewArray::<MemoryBuffer>::from_string_array(&input, compressor);

        // Should extract "data" as shared prefix
        assert_eq!(liquid_array.shared_prefix, b"data");

        // Offset view prefixes should be the remaining parts (8 bytes)
        assert_eq!(liquid_array.offset_views[0].prefix(), &[0u8; 8]); // "data" - empty remainder
        assert_eq!(liquid_array.offset_views[1].prefix(), b"base\0\0\0\0"); // "database" - "base" remainder
        assert_eq!(liquid_array.offset_views[2].prefix(), b"_entry\0\0"); // "data_entry" - "_entry" remainder
        assert_eq!(liquid_array.offset_views[3].prefix(), b"_\0\0\0\0\0\0\0"); // "data_" - "_" remainder
        assert_eq!(liquid_array.offset_views[4].prefix(), b"type\0\0\0\0"); // "datatype" - "type" remainder

        // Test roundtrip
        let output = liquid_array.to_arrow_array().unwrap();
        assert_eq!(&input, output.as_string::<i32>());

        // Test equality with exact shared prefix
        let result = liquid_array.compare_equals(b"data").unwrap();
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
    fn test_shared_prefix_corner_case() {
        let input = StringArray::from(vec!["data", "database", "data_entry", "data_", "datatype"]);
        let compressor = LiquidByteViewArray::<MemoryBuffer>::train_compressor(input.iter());
        let liquid_array =
            LiquidByteViewArray::<MemoryBuffer>::from_string_array(&input, compressor);
        let result = liquid_array.compare_with(b"data", &Operator::GtEq).unwrap();
        let expected = BooleanArray::from(vec![true, true, true, true, true]); // All >= "data"
        assert_eq!(result, expected);
    }

    #[test]
    fn test_shared_prefix_edge_cases() {
        // Test case 1: All strings are the same (full shared prefix)
        let input = StringArray::from(vec!["identical", "identical", "identical"]);
        let compressor = LiquidByteViewArray::<MemoryBuffer>::train_compressor(input.iter());
        let liquid_array =
            LiquidByteViewArray::<MemoryBuffer>::from_string_array(&input, compressor);

        assert_eq!(liquid_array.shared_prefix, b"identical");
        // All offset view prefixes should be empty
        for offset_view in &liquid_array.offset_views {
            assert_eq!(offset_view.prefix(), &[0u8; 8]);
        }

        // Test roundtrip
        let output = liquid_array.to_arrow_array().unwrap();
        assert_eq!(&input, output.as_string::<i32>());

        // Test case 2: One string is a prefix of others
        let input = StringArray::from(vec!["hello", "hello_world", "hello_test"]);
        let compressor = LiquidByteViewArray::<MemoryBuffer>::train_compressor(input.iter());
        let liquid_array =
            LiquidByteViewArray::<MemoryBuffer>::from_string_array(&input, compressor);

        assert_eq!(liquid_array.shared_prefix, b"hello");
        assert_eq!(liquid_array.offset_views[0].prefix(), &[0u8; 8]); // empty after "hello"
        assert_eq!(liquid_array.offset_views[1].prefix(), b"_world\0\0");
        assert_eq!(liquid_array.offset_views[2].prefix(), b"_test\0\0\0");

        // Test roundtrip
        let output = liquid_array.to_arrow_array().unwrap();
        assert_eq!(&input, output.as_string::<i32>());

        // Test case 3: Empty string in array (should limit shared prefix)
        let input = StringArray::from(vec!["", "hello", "hello_world"]);
        let compressor = LiquidByteViewArray::<MemoryBuffer>::train_compressor(input.iter());
        let liquid_array =
            LiquidByteViewArray::<MemoryBuffer>::from_string_array(&input, compressor);

        assert_eq!(liquid_array.shared_prefix, Vec::<u8>::new()); // empty shared prefix
        assert_eq!(liquid_array.offset_views[0].prefix(), &[0u8; 8]);
        assert_eq!(liquid_array.offset_views[1].prefix(), b"hello\0\0\0");
        assert_eq!(liquid_array.offset_views[2].prefix(), b"hello_wo"); // "hello_world" truncated to 8 bytes

        // Test roundtrip
        let output = liquid_array.to_arrow_array().unwrap();
        assert_eq!(&input, output.as_string::<i32>());
    }

    #[test]
    fn test_memory_layout() {
        let input = StringArray::from(vec!["hello", "world", "test"]);
        let compressor = LiquidByteViewArray::<MemoryBuffer>::train_compressor(input.iter());
        let liquid_array =
            LiquidByteViewArray::<MemoryBuffer>::from_string_array(&input, compressor);

        // Verify memory layout components
        assert_eq!(liquid_array.dictionary_keys.len(), 3);
        assert_eq!(liquid_array.offset_views.len(), 4);
        assert!(liquid_array.nulls().is_none());
        let _raw_buffer = liquid_array.fsst_buffer.get_fsst_buffer().unwrap();
    }

    fn check_filter_result(input: &StringArray, filter: BooleanBuffer) {
        let compressor = LiquidByteViewArray::<MemoryBuffer>::train_compressor(input.iter());
        let liquid_array =
            LiquidByteViewArray::<MemoryBuffer>::from_string_array(input, compressor);
        let filtered = liquid_array.filter(&filter);
        let output = filtered.to_arrow_array();
        let expected = {
            let selection = BooleanArray::new(filter.clone(), None);
            let arrow_filtered = arrow::compute::filter(&input, &selection).unwrap();
            arrow_filtered.as_string::<i32>().clone()
        };
        assert_eq!(output.as_ref(), &expected);
    }

    #[test]
    fn test_filter_functionality() {
        let input = StringArray::from(vec![
            Some("hello"),
            Some("test"),
            None,
            Some("test"),
            None,
            Some("test"),
            Some("rust"),
        ]);
        let mut seeded_rng = rand::rngs::StdRng::seed_from_u64(42);
        for _i in 0..100 {
            let filter =
                BooleanBuffer::from_iter((0..input.len()).map(|_| seeded_rng.random::<bool>()));
            check_filter_result(&input, filter);
        }
    }

    #[test]
    fn test_memory_efficiency() {
        let input = StringArray::from(vec!["hello", "world", "hello", "world", "hello"]);
        let compressor = LiquidByteViewArray::<MemoryBuffer>::train_compressor(input.iter());
        let liquid_array =
            LiquidByteViewArray::<MemoryBuffer>::from_string_array(&input, compressor);

        // Verify that dictionary views store unique values efficiently
        assert_eq!(liquid_array.dictionary_keys.len(), 5);

        // Verify that FSST buffer contains unique values
        let dict = liquid_array.to_dict_arrow().unwrap();
        assert_eq!(dict.values().len(), 2); // Only "hello" and "world"
    }

    #[test]
    fn test_to_best_arrow_array() {
        let input = StringArray::from(vec!["hello", "world", "test"]);
        let compressor = LiquidByteViewArray::<MemoryBuffer>::train_compressor(input.iter());
        let liquid_array =
            LiquidByteViewArray::<MemoryBuffer>::from_string_array(&input, compressor);

        let best_array = liquid_array.to_best_arrow_array();
        let dict_array = best_array.as_dictionary::<UInt16Type>();

        // Should return dictionary array as the best encoding
        assert_eq!(dict_array.len(), 3);
        assert_eq!(dict_array.values().len(), 3); // Three unique values
    }

    #[test]
    fn test_data_type() {
        let input = StringArray::from(vec!["hello", "world"]);
        let compressor = LiquidByteViewArray::<MemoryBuffer>::train_compressor(input.iter());
        let liquid_array =
            LiquidByteViewArray::<MemoryBuffer>::from_string_array(&input, compressor);

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

        let compressor = LiquidByteViewArray::<MemoryBuffer>::train_compressor(input.iter());
        let liquid_array =
            LiquidByteViewArray::<MemoryBuffer>::from_string_array(&input, compressor);

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

        let compressor = LiquidByteViewArray::<MemoryBuffer>::train_compressor(input.iter());
        let liquid_array =
            LiquidByteViewArray::<MemoryBuffer>::from_string_array(&input, compressor);

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

        let compressor = LiquidByteViewArray::<MemoryBuffer>::train_compressor(input.iter());
        let liquid_array =
            LiquidByteViewArray::<MemoryBuffer>::from_string_array(&input, compressor);

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

        let compressor = LiquidByteViewArray::<MemoryBuffer>::train_compressor(input.iter());
        let liquid_array =
            LiquidByteViewArray::<MemoryBuffer>::from_string_array(&input, compressor);

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

    fn sort_to_indices_test(input: Vec<Option<&str>>, expected_idx: Vec<u32>) {
        let input = StringArray::from(input);
        let compressor = LiquidByteViewArray::<MemoryBuffer>::train_compressor(input.iter());
        let liquid_array =
            LiquidByteViewArray::<MemoryBuffer>::from_string_array(&input, compressor);
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

    fn test_compare_equals(input: StringArray, needle: &[u8], expected: BooleanArray) {
        let compressor = LiquidByteViewArray::<MemoryBuffer>::train_compressor(input.iter());
        let liquid_array =
            LiquidByteViewArray::<MemoryBuffer>::from_string_array(&input, compressor);
        let result = liquid_array.compare_equals(needle).unwrap();
        assert_eq!(result, expected);
    }

    #[test]
    fn test_compare_equals_on_disk() {
        let input = StringArray::from(vec![
            Some("apple_orange"),
            None,
            Some("apple_orange_long_string"),
            Some("apple_b"),
            Some("apple_oo_long_string"),
            Some("apple_b"),
            Some("apple"),
        ]);
        test_compare_equals(
            input.clone(),
            b"apple",
            BooleanArray::from(vec![
                Some(false),
                None,
                Some(false),
                Some(false),
                Some(false),
                Some(false),
                Some(true),
            ]),
        );
        test_compare_equals(
            input.clone(),
            b"",
            BooleanArray::from(vec![
                Some(false),
                None,
                Some(false),
                Some(false),
                Some(false),
                Some(false),
                Some(false),
            ]),
        );
        test_compare_equals(
            input.clone(),
            b"apple_b",
            BooleanArray::from(vec![
                Some(false),
                None,
                Some(false),
                Some(true),
                Some(false),
                Some(true),
                Some(false),
            ]),
        );
        test_compare_equals(
            input.clone(),
            b"apple_oo_long_string",
            BooleanArray::from(vec![
                Some(false),
                None,
                Some(false),
                Some(false),
                Some(true),
                Some(false),
                Some(false),
            ]),
        );
    }

    #[test]
    fn test_squeeze_and_soak_roundtrip() {
        // Build a small array
        let input = StringArray::from(vec![
            Some("hello"),
            Some("world"),
            Some("hello"),
            None,
            Some("byteview"),
        ]);
        let compressor = LiquidByteViewArray::<MemoryBuffer>::train_compressor(input.iter());
        let liquid = LiquidByteViewArray::<MemoryBuffer>::from_string_array(&input, compressor);

        // Full IPC bytes as baseline
        let baseline = liquid.to_bytes();

        // Squeeze
        let Some((hybrid, (bytes, range))) = liquid.squeeze() else {
            panic!("squeeze should succeed");
        };

        // Sanity: range bounds are valid
        assert!(range.start < range.end);
        assert!(range.end as usize <= bytes.len());

        // Soak back to memory with raw FSST bytes
        let fsst_bytes = bytes.slice(range.start as usize..range.end as usize);
        let restored = hybrid.soak(fsst_bytes);

        // Arrow equality check
        use crate::liquid_array::LiquidArray as _;
        let a1 = LiquidArray::to_arrow_array(&liquid);
        let a2 = restored.to_arrow_array();
        assert_eq!(a1.as_ref(), a2.as_ref());

        // IPC bytes should match as well
        assert_eq!(baseline, restored.to_bytes());
    }
}
