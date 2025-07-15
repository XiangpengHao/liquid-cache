use arrow::array::{
    Array, ArrayAccessor, ArrayIter, ArrayRef, BinaryArray, BooleanArray, DictionaryArray,
    GenericByteArray, StringArray, StringViewArray, UInt16Array, cast::AsArray, types::UInt16Type,
};
use arrow::array::{BinaryViewArray, BooleanBufferBuilder};
use arrow::buffer::{BooleanBuffer, Buffer, NullBuffer, OffsetBuffer, ScalarBuffer};
use arrow::compute::{cast, kernels};
use arrow::datatypes::{BinaryType, ByteArrayType, Utf8Type};
use arrow_schema::ArrowError;
use datafusion::logical_expr::{ColumnarValue, Operator};
use datafusion::physical_expr_common::datum::apply_cmp;
use datafusion::physical_plan::PhysicalExpr;
use datafusion::physical_plan::expressions::{BinaryExpr, LikeExpr, Literal};
use fsst::{Compressor, Decompressor};
use std::any::Any;
use std::sync::Arc;

use super::{
    LiquidArray, LiquidArrayRef, LiquidDataType,
    byte_array::{ArrowByteType, get_string_needle},
};
use crate::liquid_array::raw::FsstArray;
use crate::utils::CheckedDictionaryArray;

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

impl LiquidArray for LiquidByteViewArray {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn get_array_memory_size(&self) -> usize {
        self.dictionary_views.len() * std::mem::size_of::<DictionaryView>()
            + self.offsets.inner().len() * std::mem::size_of::<i32>()
            + self.nulls.as_ref().map_or(0, |n| n.buffer().len())
            + self.fsst_buffer.get_array_memory_size()
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
        self.to_bytes_inner()
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
/// - FSST buffer stored on disk (currently in memory, on disk not implemented yet)
///
/// Data access flow:
/// 1. Use dictionary view key to index into offsets buffer to get start/end positions
/// 2. Use those offsets to read the corresponding bytes from FSST buffer
/// 3. Decompress those bytes to get the full value
/// 4. Use prefix for quick comparisons to avoid decompression when possible
#[derive(Debug, Clone)]
pub struct LiquidByteViewArray {
    /// Dictionary views containing key (u16) and prefix (6 bytes)
    dictionary_views: Vec<DictionaryView>,
    /// Offsets into the FSST buffer - one offset per unique value (same length as fsst_buffer)
    offsets: OffsetBuffer<i32>,
    /// Null buffer
    nulls: Option<NullBuffer>,
    /// FSST-compressed buffer (stored on disk)
    fsst_buffer: Arc<FsstArray>,
    /// Used to convert back to the original arrow type
    original_arrow_type: ArrowByteType,
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
            FsstArray::from_byte_array_with_compressor(string_values, compressor)
        } else if let Some(binary_values) = values.as_binary_opt::<i32>() {
            FsstArray::from_byte_array_with_compressor(binary_values, compressor)
        } else {
            panic!("Unsupported dictionary value type")
        };

        Self {
            dictionary_views,
            offsets,
            nulls: keys.nulls().cloned(),
            fsst_buffer: Arc::new(fsst_buffer),
            original_arrow_type: arrow_type,
        }
    }

    /// Get the decompressor of the FSST buffer
    pub fn decompressor(&self) -> Decompressor {
        self.fsst_buffer.decompressor()
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
        let values = if self.original_arrow_type == ArrowByteType::Utf8
            || self.original_arrow_type == ArrowByteType::Utf8View
            || self.original_arrow_type == ArrowByteType::Dict16Utf8
        {
            Arc::new(self.fsst_buffer.to_arrow_byte_array::<Utf8Type>()) as ArrayRef
        } else {
            Arc::new(self.fsst_buffer.to_arrow_byte_array::<BinaryType>()) as ArrayRef
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
    pub fn compare_equals(&self, needle: &[u8]) -> BooleanArray {
        let compressor = self.fsst_buffer.compressor();
        let compressed = compressor.compress(needle);

        let values = &self.fsst_buffer.compressed;
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
    pub fn compare_not_equals(&self, needle: &[u8]) -> BooleanArray {
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
        let decompressor = self.fsst_buffer.decompressor();

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

                    // Get the compressed value from fsst_buffer
                    if key < self.fsst_buffer.compressed.len()
                        && !self.fsst_buffer.compressed.is_null(key)
                    {
                        let compressed_value = self.fsst_buffer.compressed.value(key);
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

    /// Serialize to bytes
    pub fn to_bytes_inner(&self) -> Vec<u8> {
        let mut buffer = Vec::new();

        // Write header
        buffer.extend_from_slice(&(self.dictionary_views.len() as u64).to_le_bytes());
        buffer.extend_from_slice(&(self.original_arrow_type as u16).to_le_bytes());

        // Write nulls flag and data
        let has_nulls = self.nulls.is_some();
        buffer.push(has_nulls as u8);
        if let Some(nulls) = &self.nulls {
            let nulls_bytes = nulls.buffer().as_slice();
            buffer.extend_from_slice(&(nulls_bytes.len() as u32).to_le_bytes());
            buffer.extend_from_slice(nulls_bytes);
        }

        // Write dictionary views
        for view in &self.dictionary_views {
            buffer.extend_from_slice(&view.key.to_le_bytes());
            buffer.extend_from_slice(&view.prefix);
        }

        // Write offsets
        for offset in self.offsets.iter() {
            buffer.extend_from_slice(&offset.to_le_bytes());
        }

        // Write FSST buffer
        self.fsst_buffer.to_bytes(&mut buffer);

        buffer
    }

    /// Deserialize from bytes
    pub fn from_bytes(bytes: bytes::Bytes, compressor: Arc<Compressor>) -> Self {
        let mut offset = 0;

        // Read header
        let array_len = u64::from_le_bytes(bytes[offset..offset + 8].try_into().unwrap()) as usize;
        offset += 8;

        let arrow_type = ArrowByteType::from(u16::from_le_bytes(
            bytes[offset..offset + 2].try_into().unwrap(),
        ));
        offset += 2;

        // Read nulls
        let has_nulls = bytes[offset] != 0;
        offset += 1;

        let nulls = if has_nulls {
            let nulls_len =
                u32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap()) as usize;
            offset += 4;

            let nulls_slice = bytes.slice(offset..offset + nulls_len);
            let nulls_buffer = Buffer::from(nulls_slice);
            let boolean_buffer = BooleanBuffer::new(nulls_buffer, 0, array_len);
            offset += nulls_len;

            Some(NullBuffer::from(boolean_buffer))
        } else {
            None
        };

        // Read dictionary views
        let mut dictionary_views = Vec::with_capacity(array_len);
        for _ in 0..array_len {
            let key = u16::from_le_bytes(bytes[offset..offset + 2].try_into().unwrap());
            offset += 2;

            let mut prefix = [0u8; 6];
            prefix.copy_from_slice(&bytes[offset..offset + 6]);
            offset += 6;

            dictionary_views.push(DictionaryView::new(key, prefix));
        }

        // Read offsets
        let mut offsets = Vec::with_capacity(array_len);
        for _ in 0..array_len {
            let offset_val = i32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap());
            offset += 4;
            offsets.push(offset_val);
        }
        let offsets = OffsetBuffer::new(ScalarBuffer::from(offsets));

        // Read FSST buffer
        let fsst_bytes = bytes.slice(offset..);
        let fsst_buffer = FsstArray::from_bytes(fsst_bytes, compressor);

        Self {
            dictionary_views,
            offsets,
            nulls,
            fsst_buffer: Arc::new(fsst_buffer),
            original_arrow_type: arrow_type,
        }
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
        assert!(!liquid_array.fsst_buffer.compressed.is_empty());
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

        let _decompressor = liquid_array.decompressor();
        // Just verify we can get the decompressor without errors
        assert_eq!(
            liquid_array.fsst_buffer.compressor().symbol_table().len(),
            liquid_array.fsst_buffer.compressor().symbol_table().len()
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
}
