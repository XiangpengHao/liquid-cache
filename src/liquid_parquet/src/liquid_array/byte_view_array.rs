use arrow::array::{
    Array, ArrayAccessor, ArrayIter, ArrayRef, BinaryArray, BooleanArray, DictionaryArray,
    GenericByteArray, StringArray, StringViewArray, UInt16Array, cast::AsArray, types::UInt16Type,
};
use arrow::array::{BinaryViewArray, BooleanBufferBuilder};
use arrow::buffer::{BooleanBuffer, NullBuffer, OffsetBuffer, ScalarBuffer};
use arrow::compute::{cast, kernels};
use arrow::datatypes::{BinaryType, ByteArrayType, Utf8Type};
use arrow_schema::ArrowError;
use datafusion::logical_expr::{ColumnarValue, Operator};
use datafusion::physical_expr_common::datum::apply_cmp;
use datafusion::physical_plan::PhysicalExpr;
use datafusion::physical_plan::expressions::{BinaryExpr, LikeExpr, Literal};
use fsst::{Compressor, Decompressor};
use std::any::Any;
use std::path::PathBuf;
use std::sync::{Arc, RwLock};

#[cfg(test)]
use std::cell::Cell;

#[cfg(test)]
thread_local! {
    static DISK_READ_COUNTER: Cell<usize> = const { Cell::new(0)};
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
use crate::liquid_array::raw::FsstArray;
use crate::utils::CheckedDictionaryArray;
use std::fs::File;
use std::io::{self, Write};

/// A dictionary view structure that stores dictionary key and a 6-byte prefix
/// Layout: [key: u16][prefix: 6 bytes]
#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct DictionaryView {
    key: u16,
    prefix: [u8; 6],
}

const _: () = if std::mem::size_of::<DictionaryView>() != 8 {
    panic!("DictionaryView must be 8 bytes")
};

impl DictionaryView {
    pub fn new(key: u16, prefix: [u8; 6]) -> Self {
        Self { key, prefix }
    }

    pub fn key(&self) -> u16 {
        self.key
    }

    pub fn prefix(&self) -> &[u8; 6] {
        &self.prefix
    }
}

/// Storage options for FSST buffer - can be in memory or on disk
#[derive(Clone)]
pub enum FsstBufferStorage {
    InMemory(Arc<FsstArray>),
    OnDisk(PathBuf, Arc<Compressor>),
}

impl std::fmt::Debug for FsstBufferStorage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FsstBufferStorage::InMemory(fsst_array) => {
                f.debug_tuple("InMemory").field(fsst_array).finish()
            }
            FsstBufferStorage::OnDisk(path, _) => f
                .debug_tuple("OnDisk")
                .field(path)
                .field(&"<Compressor>")
                .finish(),
        }
    }
}

impl FsstBufferStorage {
    /// Get the FSST array, loading from disk if necessary
    pub fn get_fsst_array(&self) -> Result<Arc<FsstArray>, io::Error> {
        match self {
            FsstBufferStorage::InMemory(array) => Ok(array.clone()),
            FsstBufferStorage::OnDisk(path, compressor) => {
                // Increment disk read counter for testing
                #[cfg(test)]
                increment_disk_read_counter();

                let bytes = std::fs::read(path)?;
                let bytes = bytes::Bytes::from(bytes);
                let fsst_array = FsstArray::from_bytes(bytes, compressor.clone());
                Ok(Arc::new(fsst_array))
            }
        }
    }

    /// Get the compressor without loading the full FSST array
    pub fn get_compressor(&self) -> &Compressor {
        match self {
            FsstBufferStorage::InMemory(array) => array.compressor(),
            FsstBufferStorage::OnDisk(_, compressor) => compressor,
        }
    }

    /// Get the decompressor without loading the full FSST array
    pub fn get_decompressor(&self) -> Decompressor<'_> {
        match self {
            FsstBufferStorage::InMemory(array) => array.decompressor(),
            FsstBufferStorage::OnDisk(_, compressor) => compressor.decompressor(),
        }
    }

    /// Get memory size - returns 0 for on-disk storage
    pub fn get_array_memory_size(&self) -> usize {
        match self {
            FsstBufferStorage::InMemory(array) => array.get_array_memory_size(),
            FsstBufferStorage::OnDisk(_, _) => 0,
        }
    }

    /// Check if the buffer is stored in memory
    pub fn is_in_memory(&self) -> bool {
        matches!(self, FsstBufferStorage::InMemory(_))
    }

    /// Check if the buffer is stored on disk
    pub fn is_on_disk(&self) -> bool {
        matches!(self, FsstBufferStorage::OnDisk(_, _))
    }
}

impl LiquidArray for LiquidByteViewArray {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn get_array_memory_size(&self) -> usize {
        self.dictionary_views.len() * std::mem::size_of::<DictionaryView>()
            + self.offsets.inner().len() * std::mem::size_of::<i32>()
            + self.nulls.as_ref().map_or(0, |n| n.buffer().len())
            + self.fsst_buffer.read().unwrap().get_array_memory_size()
            + std::mem::size_of::<Self>()
    }

    fn len(&self) -> usize {
        self.dictionary_views.len()
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
    // Only filter the dictionary views, not the offsets!
    // Offsets reference unique values in FSST buffer and should remain unchanged
    let filtered_views: Vec<DictionaryView> = array
        .dictionary_views
        .iter()
        .zip(filter.iter())
        .filter_map(|(view, select)| {
            if select.unwrap_or(false) {
                Some(*view)
            } else {
                None
            }
        })
        .collect();

    // Filter nulls to match the filtered views
    let filtered_nulls = if let Some(nulls) = &array.nulls {
        let indices: Vec<usize> = filter
            .iter()
            .enumerate()
            .filter_map(|(i, select)| {
                if select.unwrap_or(false) {
                    Some(i)
                } else {
                    None
                }
            })
            .collect();

        let filtered_len = indices.len();
        let mut filtered_nulls = Vec::with_capacity(filtered_len);
        for idx in indices {
            filtered_nulls.push(nulls.is_null(idx));
        }

        let buffer = BooleanBuffer::from(filtered_nulls);
        Some(NullBuffer::from(buffer))
    } else {
        None
    };

    LiquidByteViewArray {
        dictionary_views: filtered_views,
        offsets: array.offsets.clone(), // Keep original offsets - they reference unique values
        nulls: filtered_nulls,
        fsst_buffer: array.fsst_buffer.clone(),
        original_arrow_type: array.original_arrow_type,
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
/// - Dictionary views with 2-byte keys and 6-byte prefixes stored in memory
/// - Offsets for unique values in FSST buffer stored in memory
/// - Nulls stored in memory
/// - FSST buffer can be stored in memory or on disk
///
/// Data access flow:
/// 1. Use dictionary view key to index into offsets buffer to get start/end positions
/// 2. Use those offsets to read the corresponding bytes from FSST buffer
/// 3. Decompress those bytes to get the full value
/// 4. Use prefix for quick comparisons to avoid decompression when possible
#[derive(Clone)]
pub struct LiquidByteViewArray {
    /// Dictionary views containing key (u16) and prefix (6 bytes)
    dictionary_views: Vec<DictionaryView>,
    /// Offsets into the FSST buffer - one offset per unique value (same length as fsst_buffer)
    offsets: OffsetBuffer<i32>,
    /// Null buffer
    nulls: Option<NullBuffer>,
    /// FSST-compressed buffer (can be in memory or on disk)
    fsst_buffer: Arc<RwLock<FsstBufferStorage>>,
    /// Used to convert back to the original arrow type
    original_arrow_type: ArrowByteType,
}

impl std::fmt::Debug for LiquidByteViewArray {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LiquidByteViewArray")
            .field("dictionary_views", &self.dictionary_views)
            .field("offsets", &self.offsets)
            .field("nulls", &self.nulls)
            .field("fsst_buffer", &self.fsst_buffer)
            .field("original_arrow_type", &self.original_arrow_type)
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
        let strings = array.filter_map(|s| s.as_ref().map(|s| s.as_bytes()));
        Arc::new(FsstArray::train_compressor(strings))
    }

    /// Train a compressor from an iterator of byte arrays
    pub fn train_compressor_bytes<'a, T: ArrayAccessor<Item = &'a [u8]>>(
        array: ArrayIter<T>,
    ) -> Arc<Compressor> {
        let strings = array.filter_map(|s| s.as_ref().map(|s| *s));
        Arc::new(FsstArray::train_compressor(strings))
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

        // Create dictionary views with prefixes - one per original array element
        let mut dictionary_views = Vec::with_capacity(keys.len());

        // Create offsets for unique values - one per unique value in FSST buffer
        let mut offsets = Vec::with_capacity(values.len() + 1);
        let mut current_offset = 0i32;
        offsets.push(current_offset);

        // Calculate offsets for each unique value in the dictionary
        for i in 0..values.len() {
            let value_bytes: &[u8] = if let Some(string_values) = values.as_string_opt::<i32>() {
                string_values.value(i).as_bytes()
            } else if let Some(binary_values) = values.as_binary_opt::<i32>() {
                binary_values.value(i)
            } else {
                panic!("Unsupported dictionary value type")
            };
            current_offset += value_bytes.len() as i32;
            offsets.push(current_offset);
        }

        // Create dictionary views with prefixes for each key
        for key_opt in keys.iter() {
            if let Some(key) = key_opt {
                // Get value bytes for prefix extraction
                let value_bytes: &[u8] = if let Some(string_values) = values.as_string_opt::<i32>()
                {
                    string_values.value(key as usize).as_bytes()
                } else if let Some(binary_values) = values.as_binary_opt::<i32>() {
                    binary_values.value(key as usize)
                } else {
                    panic!("Unsupported dictionary value type")
                };

                // Extract 6-byte prefix
                let mut prefix = [0u8; 6];
                let prefix_len = std::cmp::min(value_bytes.len(), 6);
                prefix[..prefix_len].copy_from_slice(&value_bytes[..prefix_len]);

                dictionary_views.push(DictionaryView::new(key, prefix));
            } else {
                // For null values, use a default view
                dictionary_views.push(DictionaryView::new(0, [0u8; 6]));
            }
        }

        let offsets = OffsetBuffer::new(ScalarBuffer::from(offsets));

        // Create FSST buffer from unique values
        let fsst_buffer = if let Some(string_values) = values.as_string_opt::<i32>() {
            FsstArray::from_byte_array_with_compressor(string_values, compressor.clone())
        } else if let Some(binary_values) = values.as_binary_opt::<i32>() {
            FsstArray::from_byte_array_with_compressor(binary_values, compressor.clone())
        } else {
            panic!("Unsupported dictionary value type")
        };

        Self {
            dictionary_views,
            offsets,
            nulls: keys.nulls().cloned(),
            fsst_buffer: Arc::new(RwLock::new(FsstBufferStorage::InMemory(Arc::new(
                fsst_buffer,
            )))),
            original_arrow_type: arrow_type,
        }
    }

    /// Convert to Arrow DictionaryArray
    pub fn to_dict_arrow(&self) -> DictionaryArray<UInt16Type> {
        // Create keys array from dictionary views
        let keys = self
            .dictionary_views
            .iter()
            .map(|view| view.key())
            .collect::<Vec<_>>();
        let keys_array = if let Some(nulls) = &self.nulls {
            UInt16Array::new(keys.into(), Some(nulls.clone()))
        } else {
            UInt16Array::from(keys)
        };

        // Convert FSST buffer to values
        let storage = self.fsst_buffer.read().unwrap();
        let fsst_array = storage.get_fsst_array().unwrap();
        let values = if self.original_arrow_type == ArrowByteType::Utf8
            || self.original_arrow_type == ArrowByteType::Utf8View
            || self.original_arrow_type == ArrowByteType::Dict16Utf8
        {
            Arc::new(fsst_array.to_arrow_byte_array::<Utf8Type>()) as ArrayRef
        } else {
            Arc::new(fsst_array.to_arrow_byte_array::<BinaryType>()) as ArrayRef
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
        self.nulls.as_ref()
    }

    /// Compare equality with a byte needle
    fn compare_equals(&self, needle: &[u8]) -> BooleanArray {
        let storage = self.fsst_buffer.read().unwrap();
        let compressor = storage.get_compressor();
        let compressed = compressor.compress(needle);

        // We need the full FSST array to search through the compressed values
        let fsst_array = storage.get_fsst_array().unwrap();
        let values = &fsst_array.compressed;
        let idx = values.iter().position(|v| v == Some(compressed.as_ref()));

        if let Some(idx) = idx {
            let target_key = idx as u16;
            let mut buffer_builder = BooleanBufferBuilder::new(self.dictionary_views.len());

            for view in &self.dictionary_views {
                buffer_builder.append(view.key() == target_key);
            }

            let buffer = buffer_builder.finish();
            BooleanArray::new(buffer, self.nulls().cloned())
        } else {
            let buffer = BooleanBuffer::new_unset(self.dictionary_views.len());
            BooleanArray::new(buffer, self.nulls().cloned())
        }
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
                self.compare_with_prefix_optimization(needle, op)
            }

            // For other operations, fall back to Arrow operations
            _ => self.compare_with_arrow_fallback(needle, op),
        }
    }

    /// Prefix optimization for ordering operations
    fn compare_with_prefix_optimization(
        &self,
        needle: &[u8],
        op: &Operator,
    ) -> Result<BooleanArray, ArrowError> {
        // Extract needle prefix (first 6 bytes, padded with zeros if shorter)
        let mut needle_prefix = [0u8; 6];
        let prefix_len = std::cmp::min(needle.len(), 6);
        needle_prefix[..prefix_len].copy_from_slice(&needle[..prefix_len]);

        let mut buffer_builder = BooleanBufferBuilder::new(self.dictionary_views.len());
        let storage = self.fsst_buffer.read().unwrap();
        let decompressor = storage.get_decompressor();

        // Only load the full FSST array if we encounter any equal prefixes
        let mut fsst_array: Option<Arc<FsstArray>> = None;

        for view in &self.dictionary_views {
            let prefix_cmp = view.prefix().cmp(&needle_prefix);

            let result = match (op, prefix_cmp) {
                // For Lt: if prefix < needle_prefix => true, if prefix > needle_prefix => false
                (Operator::Lt, std::cmp::Ordering::Less) => true,
                (Operator::Lt, std::cmp::Ordering::Greater) => false,

                // For LtEq: if prefix < needle_prefix => true, if prefix > needle_prefix => false
                (Operator::LtEq, std::cmp::Ordering::Less) => true,
                (Operator::LtEq, std::cmp::Ordering::Greater) => false,

                // For Gt: if prefix > needle_prefix => true, if prefix < needle_prefix => false
                (Operator::Gt, std::cmp::Ordering::Greater) => true,
                (Operator::Gt, std::cmp::Ordering::Less) => false,

                // For GtEq: if prefix > needle_prefix => true, if prefix < needle_prefix => false
                (Operator::GtEq, std::cmp::Ordering::Greater) => true,
                (Operator::GtEq, std::cmp::Ordering::Less) => false,

                // When prefixes are equal, we need to decompress and compare full values
                (
                    Operator::Lt | Operator::LtEq | Operator::Gt | Operator::GtEq,
                    std::cmp::Ordering::Equal,
                ) => {
                    let key = view.key() as usize;

                    // Lazy load the full FSST array only when we need it
                    if fsst_array.is_none() {
                        fsst_array = Some(storage.get_fsst_array().unwrap());
                    }
                    let fsst_array_ref = fsst_array.as_ref().unwrap();

                    if key < fsst_array_ref.compressed.len()
                        && !fsst_array_ref.compressed.is_null(key)
                    {
                        let compressed_value = fsst_array_ref.compressed.value(key);
                        // Decompress the value - this allocates but only for inconclusive cases
                        let decompressed_bytes = decompressor.decompress(compressed_value);

                        // Compare the decompressed bytes with needle bytes directly
                        match op {
                            Operator::Lt => decompressed_bytes.as_slice() < needle,
                            Operator::LtEq => decompressed_bytes.as_slice() <= needle,
                            Operator::Gt => decompressed_bytes.as_slice() > needle,
                            Operator::GtEq => decompressed_bytes.as_slice() >= needle,
                            _ => unreachable!("Should only be called with comparison operators"),
                        }
                    } else {
                        // Handle null case - nulls are typically considered "less than" any value
                        matches!(op, Operator::Lt | Operator::LtEq)
                    }
                }

                // This should never happen if called correctly
                (_, _) => {
                    unreachable!("Unexpected operator {op:?} in compare_with_prefix_optimization");
                }
            };

            buffer_builder.append(result);
        }

        let buffer = buffer_builder.finish();
        Ok(BooleanArray::new(buffer, self.nulls().cloned()))
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
            FsstBufferStorage::InMemory(fsst_array) => {
                let mut buffer = Vec::new();
                fsst_array.to_bytes(&mut buffer);

                let mut file = File::create(&path)?;
                file.write_all(&buffer)?;

                let compressor = Arc::new(fsst_array.compressor().clone());
                *storage = FsstBufferStorage::OnDisk(path, compressor);
                Ok(())
            }
            FsstBufferStorage::OnDisk(_, _) => Ok(()),
        }
    }

    /// Load the FSST buffer from disk into memory
    pub fn load_from_disk(&self) -> Result<(), io::Error> {
        let mut storage = self.fsst_buffer.write().unwrap();

        match &*storage {
            FsstBufferStorage::OnDisk(path, compressor) => {
                let bytes = std::fs::read(path)?;
                let bytes = bytes::Bytes::from(bytes);
                let fsst_array = FsstArray::from_bytes(bytes, compressor.clone());
                *storage = FsstBufferStorage::InMemory(Arc::new(fsst_array));
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
    fn test_simple_roundtrip() {
        let input = StringArray::from(vec!["hello", "world", "hello", "rust"]);
        test_string_roundtrip(input);
    }

    #[test]
    fn test_roundtrip_with_nulls() {
        let input = StringArray::from(vec![
            Some("hello"),
            None,
            Some("world"),
            None,
            Some("hello"),
        ]);
        test_string_roundtrip(input);
    }

    #[test]
    fn test_roundtrip_with_long_strings() {
        let input = StringArray::from(vec![
            "This is a very long string that should be compressed well",
            "Another long string with some common patterns",
            "This is a very long string that should be compressed well",
            "Some unique text here to mix things up",
            "Another long string with some common patterns",
        ]);
        test_string_roundtrip(input);
    }

    #[test]
    fn test_empty_strings() {
        let input = StringArray::from(vec!["", "", "non-empty", ""]);
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
        let view = DictionaryView::new(42, [1, 2, 3, 4, 5, 6]);
        assert_eq!(view.key(), 42);
        assert_eq!(view.prefix(), &[1, 2, 3, 4, 5, 6]);
    }

    #[test]
    fn test_prefix_extraction() {
        let input = StringArray::from(vec!["hello", "world", "test"]);
        let compressor = LiquidByteViewArray::train_compressor(input.iter());
        let liquid_array = LiquidByteViewArray::from_string_array(&input, compressor);

        // Check that prefixes are extracted correctly
        assert_eq!(liquid_array.dictionary_views[0].prefix(), b"hello\0");
        assert_eq!(liquid_array.dictionary_views[1].prefix(), b"world\0");
        assert_eq!(liquid_array.dictionary_views[2].prefix(), b"test\0\0");
    }

    #[test]
    fn test_memory_layout() {
        let input = StringArray::from(vec!["hello", "world", "test"]);
        let compressor = LiquidByteViewArray::train_compressor(input.iter());
        let liquid_array = LiquidByteViewArray::from_string_array(&input, compressor);

        // Verify memory layout components
        assert_eq!(liquid_array.dictionary_views.len(), 3);
        // Offsets has one more element than unique values (standard offset buffer format)
        assert_eq!(liquid_array.offsets.len(), 4); // 3 unique values + 1 = 4 offsets
        assert!(liquid_array.nulls.is_none());
        let fsst_array = liquid_array
            .fsst_buffer
            .read()
            .unwrap()
            .get_fsst_array()
            .unwrap();
        assert!(!fsst_array.compressed.is_empty());
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
        assert_eq!(liquid_array.dictionary_views.len(), 5);

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
    fn test_decompressor_access() {
        let input = StringArray::from(vec!["hello", "world"]);
        let compressor = LiquidByteViewArray::train_compressor(input.iter());
        let liquid_array = LiquidByteViewArray::from_string_array(&input, compressor);

        // Just verify we can get the decompressor without errors
        let fsst_array = liquid_array
            .fsst_buffer
            .read()
            .unwrap()
            .get_fsst_array()
            .unwrap();
        let _decompressor = fsst_array.decompressor();
        assert_eq!(
            fsst_array.compressor().symbol_table().len(),
            fsst_array.compressor().symbol_table().len()
        );
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
            .compare_with_prefix_optimization(b"car", &Operator::Lt)
            .unwrap();
        let expected = BooleanArray::from(vec![true, true, false, true, false]);
        assert_eq!(result, expected);

        // Test Gt with needle "dog" (prefix: "dog\0\0\0")
        // Expected: only "zebra000" > "dog" => true
        let result = liquid_array
            .compare_with_prefix_optimization(b"dog", &Operator::Gt)
            .unwrap();
        let expected = BooleanArray::from(vec![false, false, false, false, true]);
        assert_eq!(result, expected);

        // Test GtEq with needle "apple" (prefix: "apple\0")
        // Expected: all except "apple123" and "apple999" need decompression, others by prefix
        let result = liquid_array
            .compare_with_prefix_optimization(b"apple", &Operator::GtEq)
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
            .compare_with_prefix_optimization(b"prefix_b", &Operator::Lt)
            .unwrap();
        let expected = BooleanArray::from(vec![true, false, false, true, true]);
        assert_eq!(result, expected);

        // Test LtEq with needle "prefix_bbb" - exact match case with decompression
        let result = liquid_array
            .compare_with_prefix_optimization(b"prefix_bbb", &Operator::LtEq)
            .unwrap();
        let expected = BooleanArray::from(vec![true, true, false, true, true]);
        assert_eq!(result, expected);

        // Test Gt with needle "prefix_abc" - requires decompression for prefix matches
        let result = liquid_array
            .compare_with_prefix_optimization(b"prefix_abc", &Operator::Gt)
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
        let result = liquid_array
            .compare_with_prefix_optimization(b"", &Operator::Lt)
            .unwrap();
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
            .compare_with_prefix_optimization(b"abcdef", &Operator::Gt)
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
            .compare_with_prefix_optimization(b"b", &Operator::LtEq)
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
            .compare_with_prefix_optimization(b"abcdeg", &Operator::GtEq)
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
            .compare_with_prefix_optimization(naive_bytes, &Operator::Lt)
            .unwrap();
        let expected = BooleanArray::from(vec![true, false, false, true, false]);
        assert_eq!(result, expected);

        // Test Gt with UTF-8 needle "café" (UTF-8: [99, 97, 102, 195, 169])
        // Expected: strings with first byte > 99 should be true
        let cafe_bytes = "café".as_bytes(); // [99, 97, 102, 195, 169]
        let result = liquid_array
            .compare_with_prefix_optimization(cafe_bytes, &Operator::Gt)
            .unwrap();
        let expected = BooleanArray::from(vec![false, true, true, true, true]);
        assert_eq!(result, expected);

        // Test LtEq with Chinese characters "世界" (UTF-8: [228, 184, 150, 231, 149, 140])
        // Expected: only strings with first byte <= 228 should be true, but since 228 is quite high,
        // most Latin characters will be true
        let world_bytes = "世界".as_bytes(); // [228, 184, 150, 231, 149, 140]
        let result = liquid_array
            .compare_with_prefix_optimization(world_bytes, &Operator::LtEq)
            .unwrap();
        let expected = BooleanArray::from(vec![true, true, true, true, true]);
        assert_eq!(result, expected);

        // Test exact equality with "résumé" using GtEq and LtEq to verify byte-level precision
        let resume_bytes = "résumé".as_bytes(); // [114, 195, 169, 115, 117, 109, 195, 169]
        let gte_result = liquid_array
            .compare_with_prefix_optimization(resume_bytes, &Operator::GtEq)
            .unwrap();
        let lte_result = liquid_array
            .compare_with_prefix_optimization(resume_bytes, &Operator::LtEq)
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

    #[test]
    fn test_efficient_compressor_decompressor_access() {
        // Reset counter at start of test to avoid interference from other tests
        reset_disk_read_counter();

        let input = StringArray::from(vec![
            "prefix_aaa", // prefix: "prefix"
            "prefix_bbb", // prefix: "prefix" (same prefix)
            "prefix_ccc", // prefix: "prefix" (same prefix)
            "different",  // prefix: "differ" (different prefix)
        ]);
        let compressor = LiquidByteViewArray::train_compressor(input.iter());
        let liquid_array = LiquidByteViewArray::from_string_array(&input, compressor);

        // Evict to disk to test efficiency
        let temp_path = std::env::temp_dir().join("test_efficient_access.bin");
        liquid_array.evict_to_disk(temp_path.clone()).unwrap();

        // Test that we can access compressor and decompressor without loading full array
        let storage = liquid_array.fsst_buffer.read().unwrap();
        let _compressor = storage.get_compressor();
        let _decompressor = storage.get_decompressor();

        // Test prefix optimization - should work efficiently with mostly prefix comparisons
        // "prefix_b" should be able to determine most results via prefix comparison
        let result = liquid_array
            .compare_with(b"prefix_b", &Operator::Lt)
            .unwrap();
        let expected = BooleanArray::from(vec![true, false, false, true]);
        assert_eq!(result, expected);

        // Test that comparison with very different prefix works efficiently
        // "zzz" should be resolvable by prefix comparison alone for most values
        let result = liquid_array.compare_with(b"zzz", &Operator::Lt).unwrap();
        let expected = BooleanArray::from(vec![true, true, true, true]);
        assert_eq!(result, expected);

        // Cleanup
        let _ = std::fs::remove_file(temp_path);
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

        // 1. Get compressor - should not read from disk
        let storage = liquid_array.fsst_buffer.read().unwrap();
        let _compressor = storage.get_compressor();
        assert_eq!(liquid_array.get_disk_read_count(), 0);

        // 2. Get decompressor - should not read from disk
        let _decompressor = storage.get_decompressor();
        assert_eq!(liquid_array.get_disk_read_count(), 0);
        drop(storage);

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
