use ahash::HashMap;
use arrow::array::{
    Array, ArrayAccessor, ArrayIter, ArrayRef, BinaryArray, BooleanArray, BooleanBufferBuilder,
    BufferBuilder, DictionaryArray, GenericByteArray, StringArray, StringViewArray, UInt16Array,
    cast::AsArray, types::UInt16Type,
};
use arrow::buffer::{BooleanBuffer, Buffer, NullBuffer, OffsetBuffer, ScalarBuffer};
use arrow::compute::{cast, kernels};
use arrow::datatypes::{BinaryType, ByteArrayType, Utf8Type};
use arrow_schema::{ArrowError, DataType};
use datafusion::logical_expr::{ColumnarValue, Operator};
use datafusion::physical_expr_common::datum::apply_cmp;
use datafusion::physical_plan::PhysicalExpr;
use datafusion::physical_plan::expressions::{BinaryExpr, LikeExpr, Literal};
use datafusion::scalar::ScalarValue;
use fsst::{Compressor, Decompressor};
use std::any::Any;
use std::mem::MaybeUninit;
use std::sync::Arc;

use super::{LiquidArray, LiquidArrayRef, LiquidDataType};
use crate::liquid_array::{raw::BitPackedArray, raw::FsstArray};
use crate::utils::CheckedDictionaryArray;

impl LiquidArray for LiquidByteArray {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn get_array_memory_size(&self) -> usize {
        self.keys.get_array_memory_size() + self.values.get_array_memory_size()
    }

    fn len(&self) -> usize {
        self.keys.len()
    }

    #[inline]
    fn to_arrow_array(&self) -> ArrayRef {
        let dict = self.to_arrow_array();
        Arc::new(dict)
    }

    fn to_best_arrow_array(&self) -> ArrayRef {
        // the best arrow string is DictionaryArray<UInt16Type>
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

fn filter_inner(array: &LiquidByteArray, filter: &BooleanArray) -> LiquidByteArray {
    let values = array.values.clone();
    let keys = array.keys.clone();
    let primitive_keys = keys.to_primitive();
    let filtered_keys = arrow::compute::filter(&primitive_keys, filter)
        .unwrap()
        .as_primitive::<UInt16Type>()
        .clone();
    let bit_packed_array = match keys.bit_width() {
        Some(bit_width) => BitPackedArray::from_primitive(filtered_keys, bit_width),
        None => BitPackedArray::new_null_array(filtered_keys.len()),
    };
    LiquidByteArray {
        keys: bit_packed_array,
        values,
        original_arrow_type: array.original_arrow_type,
    }
}

fn try_eval_predicate_inner(
    expr: &Arc<dyn PhysicalExpr>,
    array: &LiquidByteArray,
) -> Result<Option<BooleanArray>, ArrowError> {
    if let Some(binary_expr) = expr.as_any().downcast_ref::<BinaryExpr>() {
        if let Some(literal) = binary_expr.right().as_any().downcast_ref::<Literal>() {
            let op = binary_expr.op();
            if let Some(needle) = get_string_needle(literal.value()) {
                if op == &Operator::Eq {
                    let result = array.compare_equals(needle);
                    return Ok(Some(result));
                } else if op == &Operator::NotEq {
                    let result = array.compare_not_equals(needle);
                    return Ok(Some(result));
                }
            }

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
    } else if let Some(like_expr) = expr.as_any().downcast_ref::<LikeExpr>()
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

/// Extract a string needle from a scalar value for string comparison operations.
///
/// This function handles various string scalar value types including Utf8, Utf8View,
/// LargeUtf8, and Dictionary types. Returns `None` for non-string types or null values.
pub fn get_string_needle(value: &ScalarValue) -> Option<&str> {
    if let ScalarValue::Utf8(Some(v)) = value {
        Some(v)
    } else if let ScalarValue::Utf8View(Some(v)) = value {
        Some(v)
    } else if let ScalarValue::LargeUtf8(Some(v)) = value {
        Some(v)
    } else if let ScalarValue::Dictionary(_, value) = value {
        get_string_needle(value.as_ref())
    } else {
        None
    }
}

#[derive(Debug, Clone, Copy)]
#[repr(u16)]
pub(crate) enum ArrowByteType {
    Utf8 = 0,
    Utf8View = 1,
    Dict16Binary = 2, // DictionaryArray<UInt16Type>
    Dict16Utf8 = 3,   // DictionaryArray<UInt16Type>
    Binary = 4,
    BinaryView = 5,
}

impl From<u16> for ArrowByteType {
    fn from(value: u16) -> Self {
        match value {
            0 => ArrowByteType::Utf8,
            1 => ArrowByteType::Utf8View,
            2 => ArrowByteType::Dict16Binary,
            3 => ArrowByteType::Dict16Utf8,
            4 => ArrowByteType::Binary,
            5 => ArrowByteType::BinaryView,
            _ => panic!("Invalid arrow byte type: {value}"),
        }
    }
}

impl ArrowByteType {
    pub fn to_arrow_type(self) -> DataType {
        match self {
            ArrowByteType::Utf8 => DataType::Utf8,
            ArrowByteType::Utf8View => DataType::Utf8View,
            ArrowByteType::Dict16Binary => {
                DataType::Dictionary(Box::new(DataType::UInt16), Box::new(DataType::Binary))
            }
            ArrowByteType::Dict16Utf8 => {
                DataType::Dictionary(Box::new(DataType::UInt16), Box::new(DataType::Utf8))
            }
            ArrowByteType::Binary => DataType::Binary,
            ArrowByteType::BinaryView => DataType::BinaryView,
        }
    }

    fn is_string(&self) -> bool {
        matches!(
            self,
            ArrowByteType::Utf8 | ArrowByteType::Utf8View | ArrowByteType::Dict16Utf8
        )
    }

    pub fn from_arrow_type(ty: &DataType) -> Self {
        match ty {
            DataType::Utf8 => ArrowByteType::Utf8,
            DataType::Utf8View => ArrowByteType::Utf8View,
            DataType::Binary => ArrowByteType::Binary,
            DataType::BinaryView => ArrowByteType::BinaryView,
            DataType::Dictionary(_, _) => {
                if ty
                    == &DataType::Dictionary(Box::new(DataType::UInt16), Box::new(DataType::Binary))
                {
                    ArrowByteType::Dict16Binary
                } else if ty
                    == &DataType::Dictionary(Box::new(DataType::UInt16), Box::new(DataType::Utf8))
                {
                    ArrowByteType::Dict16Utf8
                } else {
                    panic!("Unsupported arrow type: {ty:?}")
                }
            }
            _ => panic!("Unsupported arrow type: {ty:?}"),
        }
    }
}

/// An array that stores strings in a dictionary format, with a bit-packed array for the keys and a FSST array for the values.
#[derive(Debug, Clone)]
pub struct LiquidByteArray {
    pub(crate) keys: BitPackedArray<UInt16Type>,
    /// TODO: we need to specify that the values in the FsstArray must be unique, this enables us some optimizations.
    pub(crate) values: FsstArray,
    /// Used to convert back to the original arrow type.
    pub(crate) original_arrow_type: ArrowByteType,
}

impl LiquidByteArray {
    /// Create a LiquidByteArray from an Arrow [StringViewArray].
    pub fn from_string_view_array(array: &StringViewArray, compressor: Arc<Compressor>) -> Self {
        let dict = CheckedDictionaryArray::from_string_view_array(array);
        Self::from_dict_array_inner(dict, compressor, ArrowByteType::Utf8View)
    }

    /// Train a compressor from an iterator of byte arrays.
    pub fn train_compressor_bytes<'a, T: ArrayAccessor<Item = &'a [u8]>>(
        array: ArrayIter<T>,
    ) -> Arc<Compressor> {
        let strings = array.filter_map(|s| s.as_ref().map(|s| *s));
        Arc::new(FsstArray::train_compressor(strings))
    }

    /// Train a compressor from an iterator of strings.
    pub fn train_compressor<'a, T: ArrayAccessor<Item = &'a str>>(
        array: ArrayIter<T>,
    ) -> Arc<Compressor> {
        let strings = array.filter_map(|s| s.as_ref().map(|s| s.as_bytes()));
        Arc::new(FsstArray::train_compressor(strings))
    }

    /// Create a LiquidByteArray from an Arrow [StringArray].
    pub fn from_string_array(array: &StringArray, compressor: Arc<Compressor>) -> Self {
        Self::from_byte_array(array, compressor)
    }

    /// Create a LiquidByteArray from an Arrow ByteArray.
    pub fn from_byte_array<T: ByteArrayType>(
        array: &GenericByteArray<T>,
        compressor: Arc<Compressor>,
    ) -> Self {
        let dict = CheckedDictionaryArray::from_byte_array::<T>(array);
        Self::from_dict_array_inner(
            dict,
            compressor,
            ArrowByteType::from_arrow_type(&T::DATA_TYPE),
        )
    }

    /// Train a compressor from an Arrow StringViewArray.
    pub fn train_from_arrow_view(array: &StringViewArray) -> (Arc<Compressor>, Self) {
        let dict = CheckedDictionaryArray::from_string_view_array(array);
        let compressor = Self::train_compressor(dict.as_ref().values().as_string::<i32>().iter());
        (
            compressor.clone(),
            Self::from_dict_array_inner(dict, compressor, ArrowByteType::Utf8View),
        )
    }

    /// Train a compressor from an Arrow ByteArray.
    pub fn train_from_arrow<T: ByteArrayType>(
        array: &GenericByteArray<T>,
    ) -> (Arc<Compressor>, Self) {
        let dict = CheckedDictionaryArray::from_byte_array::<T>(array);
        let value_type = dict.as_ref().values().data_type();

        let compressor = if value_type == &DataType::Utf8 {
            Self::train_compressor(dict.as_ref().values().as_string::<i32>().iter())
        } else {
            Self::train_compressor_bytes(dict.as_ref().values().as_binary::<i32>().iter())
        };
        (
            compressor.clone(),
            Self::from_dict_array_inner(
                dict,
                compressor,
                ArrowByteType::from_arrow_type(&T::DATA_TYPE),
            ),
        )
    }

    /// Train a compressor from an Arrow DictionaryArray.
    pub fn train_from_arrow_dict(array: &DictionaryArray<UInt16Type>) -> (Arc<Compressor>, Self) {
        if array.values().data_type() == &DataType::Utf8 {
            let values = array.values().as_string::<i32>();

            let compressor = Self::train_compressor(values.iter());
            (
                compressor.clone(),
                Self::from_dict_array_inner(
                    CheckedDictionaryArray::new_checked(array),
                    compressor,
                    ArrowByteType::Dict16Utf8,
                ),
            )
        } else if array.values().data_type() == &DataType::Binary {
            let values = array.values().as_binary::<i32>();
            let compressor = Self::train_compressor_bytes(values.iter());
            (
                compressor.clone(),
                Self::from_dict_array_inner(
                    CheckedDictionaryArray::new_checked(array),
                    compressor,
                    ArrowByteType::Dict16Binary,
                ),
            )
        } else {
            panic!("Unsupported dictionary type: {:?}", array.data_type())
        }
    }

    /// Create a LiquidByteArray from an Arrow DictionaryArray.
    fn from_dict_array_inner(
        array: CheckedDictionaryArray,
        compressor: Arc<Compressor>,
        arrow_type: ArrowByteType,
    ) -> Self {
        let bit_width_for_key = array.bit_width_for_key();
        let (keys, values) = array.into_inner().into_parts();

        let bit_packed_array = BitPackedArray::from_primitive(keys, bit_width_for_key);

        let fsst_values = if let Some(values) = values.as_string_opt::<i32>() {
            FsstArray::from_byte_array_with_compressor(values, compressor)
        } else if let Some(values) = values.as_binary_opt::<i32>() {
            FsstArray::from_byte_array_with_compressor(values, compressor)
        } else {
            panic!("Unsupported dictionary type")
        };
        LiquidByteArray {
            keys: bit_packed_array,
            values: fsst_values,
            original_arrow_type: arrow_type,
        }
    }

    /// Only used when the dictionary is read from a trusted parquet reader,
    /// which reads a trusted parquet file, written by a trusted writer.
    ///
    /// # Safety
    /// The caller must ensure that the values in the dictionary are unique.
    pub unsafe fn from_unique_dict_array(
        array: &DictionaryArray<UInt16Type>,
        compressor: Arc<Compressor>,
    ) -> Self {
        Self::from_dict_array_inner(
            unsafe { CheckedDictionaryArray::new_unchecked_i_know_what_i_am_doing(array) },
            compressor,
            ArrowByteType::Dict16Utf8,
        )
    }

    /// Create a LiquidByteArray from an Arrow DictionaryArray.
    pub fn from_dict_array(
        array: &DictionaryArray<UInt16Type>,
        compressor: Arc<Compressor>,
    ) -> Self {
        if array.downcast_dict::<StringArray>().is_some() {
            let dict = CheckedDictionaryArray::new_checked(array);
            Self::from_dict_array_inner(dict, compressor, ArrowByteType::Dict16Utf8)
        } else if array.downcast_dict::<BinaryArray>().is_some() {
            let dict = CheckedDictionaryArray::new_checked(array);
            Self::from_dict_array_inner(dict, compressor, ArrowByteType::Dict16Binary)
        } else {
            panic!("Unsupported dictionary type: {:?}", array.data_type())
        }
    }

    /// Get the decompressor of the LiquidStringArray.
    pub fn decompressor(&self) -> Decompressor {
        self.values.decompressor()
    }

    /// Convert the LiquidStringArray to arrow's DictionaryArray.
    pub fn to_dict_arrow(&self) -> DictionaryArray<UInt16Type> {
        if self.keys.len() < 2048 || self.keys.len() < self.values.compressed.len() {
            // a heuristic to selective decompress.
            self.to_dict_arrow_decompress_keyed()
        } else {
            self.to_dict_arrow_decompress_all()
        }
    }

    fn to_dict_arrow_decompress_all(&self) -> DictionaryArray<UInt16Type> {
        let primitive_key = self.keys.to_primitive();
        if self.original_arrow_type.is_string() {
            let values = self.values.to_arrow_byte_array::<Utf8Type>();
            unsafe { DictionaryArray::<UInt16Type>::new_unchecked(primitive_key, Arc::new(values)) }
        } else {
            let values = self.values.to_arrow_byte_array::<BinaryType>();
            unsafe { DictionaryArray::<UInt16Type>::new_unchecked(primitive_key, Arc::new(values)) }
        }
    }

    fn to_dict_arrow_decompress_keyed(&self) -> DictionaryArray<UInt16Type> {
        let primitive_key = self.keys.to_primitive();
        let mut hit_mask = BooleanBufferBuilder::new(self.values.compressed.len());
        hit_mask.advance(self.values.compressed.len());
        for v in primitive_key.iter().flatten() {
            hit_mask.set_bit(v as usize, true);
        }
        let hit_mask = hit_mask.finish();
        let selected_cnt = hit_mask.count_set_bits();

        let mut key_map =
            HashMap::with_capacity_and_hasher(selected_cnt, ahash::RandomState::new());
        let mut offset = 0;
        for (i, select) in hit_mask.iter().enumerate() {
            if select {
                key_map.insert(i, offset);
                offset += 1;
            }
        }
        let new_keys = UInt16Array::from_iter(
            primitive_key
                .iter()
                .map(|v| v.map(|v| key_map[&(v as usize)])),
        );

        let decompressor = self.values.decompressor();
        let mut value_buffer: Vec<u8> = Vec::with_capacity(self.values.uncompressed_len + 8);
        let mut offsets_builder = BufferBuilder::<i32>::new(selected_cnt + 1);
        offsets_builder.append(0);

        assert_eq!(hit_mask.len(), self.values.compressed.len());
        for (_select, v) in hit_mask
            .iter()
            .zip(self.values.compressed.iter())
            .filter(|(select, _v)| *select)
        {
            let v = v.expect("values array can't be null");
            let slice = unsafe {
                std::slice::from_raw_parts_mut(
                    value_buffer.as_mut_ptr().add(value_buffer.len()) as *mut MaybeUninit<u8>,
                    value_buffer.capacity(),
                )
            };
            let len = decompressor.decompress_into(v, slice);
            let new_len = value_buffer.len() + len;
            debug_assert!(new_len <= value_buffer.capacity());
            unsafe {
                value_buffer.set_len(new_len);
            }
            offsets_builder.append(value_buffer.len() as i32);
        }
        value_buffer.shrink_to_fit();
        let value_buffer = Buffer::from(value_buffer);
        let offsets_buffer: ScalarBuffer<i32> = ScalarBuffer::from(offsets_builder.finish());
        let values = if self.original_arrow_type.is_string() {
            unsafe {
                Arc::new(GenericByteArray::<Utf8Type>::new_unchecked(
                    OffsetBuffer::new_unchecked(offsets_buffer),
                    value_buffer,
                    None,
                )) as ArrayRef
            }
        } else {
            unsafe {
                Arc::new(GenericByteArray::<BinaryType>::new_unchecked(
                    OffsetBuffer::new_unchecked(offsets_buffer),
                    value_buffer,
                    None,
                ))
            }
        };
        unsafe { DictionaryArray::<UInt16Type>::new_unchecked(new_keys, values) }
    }

    /// Convert the LiquidStringArray to a DictionaryArray with a selection.
    pub fn to_dict_arrow_with_selection(
        &self,
        selection: &BooleanArray,
    ) -> DictionaryArray<UInt16Type> {
        let primitive_key = self.keys.to_primitive().clone();
        let filtered_keys = arrow::compute::filter(&primitive_key, selection)
            .unwrap()
            .as_primitive::<UInt16Type>()
            .clone();
        let values: StringArray = StringArray::from(&self.values);
        unsafe { DictionaryArray::<UInt16Type>::new_unchecked(filtered_keys, Arc::new(values)) }
    }

    /// Convert the LiquidStringArray to a StringArray.
    pub fn to_arrow_array(&self) -> ArrayRef {
        let dict = self.to_dict_arrow();
        cast(&dict, &self.original_arrow_type.to_arrow_type()).unwrap()
    }

    /// Compare the values of the LiquidStringArray with a given string and return a BooleanArray of the result.
    pub fn compare_not_equals(&self, needle: &str) -> BooleanArray {
        let result = self.compare_equals(needle);
        let (values, nulls) = result.into_parts();
        let values = !&values;
        BooleanArray::new(values, nulls)
    }

    /// Get the nulls of the LiquidStringArray.
    pub fn nulls(&self) -> Option<&NullBuffer> {
        self.keys.nulls()
    }

    /// Compare the values of the LiquidStringArray with a given string.
    /// Leverage the distinct values to speed up the comparison.
    /// TODO: We can further optimize this by vectorizing the comparison.
    pub fn compare_equals(&self, needle: &str) -> BooleanArray {
        let compressor = self.values.compressor();
        let compressed = compressor.compress(needle.as_bytes());

        let values = &self.values.compressed;
        let keys = self.keys.to_primitive();

        let idx = values.iter().position(|v| v == Some(compressed.as_ref()));

        if let Some(idx) = idx {
            let to_compare = UInt16Array::new_scalar(idx as u16);
            arrow::compute::kernels::cmp::eq(&keys, &to_compare).unwrap()
        } else {
            let buffer = BooleanBuffer::new_unset(keys.len());
            BooleanArray::new(buffer, self.nulls().cloned())
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::liquid_array::LiquidPrimitiveArray;

    use super::*;
    use arrow::{
        array::{Array, Int32Array},
        datatypes::Int32Type,
    };
    use bytes::Bytes;
    use datafusion::physical_plan::expressions::Column;
    use std::num::NonZero;

    fn test_roundtrip(input: StringArray) {
        let compressor = LiquidByteArray::train_compressor(input.iter());
        let liquid_array = LiquidByteArray::from_string_array(&input, compressor.clone());
        let output = liquid_array.to_arrow_array();
        assert_eq!(&input, output.as_string::<i32>());

        let bytes = liquid_array.to_bytes_inner();
        let bytes = Bytes::from(bytes);
        let deserialized = LiquidByteArray::from_bytes(bytes, compressor);
        let output = deserialized.to_arrow_array();
        assert_eq!(&input, output.as_string::<i32>());
    }

    #[test]
    fn test_simple_roundtrip() {
        let input = StringArray::from(vec!["hello", "world", "hello", "rust"]);
        test_roundtrip(input);
    }

    #[test]
    fn test_to_arrow_array_preserve_arrow_type() {
        let input = StringArray::from(vec!["hello", "world", "hello", "rust"]);
        let compressor = LiquidByteArray::train_compressor(input.iter());
        let etc = LiquidByteArray::from_string_array(&input, compressor);
        let output = etc.to_arrow_array();
        assert_eq!(&input, output.as_string::<i32>());

        let input = cast(&input, &DataType::Utf8View)
            .unwrap()
            .as_string_view()
            .clone();
        let compressor = LiquidByteArray::train_compressor(input.iter());
        let etc = LiquidByteArray::from_string_view_array(&input, compressor);
        let output = etc.to_arrow_array();
        assert_eq!(&input, output.as_string_view());

        let input = cast(
            &input,
            &DataType::Dictionary(Box::new(DataType::UInt16), Box::new(DataType::Utf8)),
        )
        .unwrap()
        .as_dictionary()
        .clone();
        let compressor =
            LiquidByteArray::train_compressor(input.values().as_string::<i32>().iter());
        let etc = LiquidByteArray::from_dict_array(&input, compressor);
        let output = etc.to_arrow_array();
        assert_eq!(&input, output.as_dictionary());
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
        test_roundtrip(input);
    }

    #[test]
    fn test_roundtrip_with_many_duplicates() {
        let values = ["a", "b", "c"];
        let input: Vec<&str> = (0..1000).map(|i| values[i % values.len()]).collect();
        let input = StringArray::from(input);
        test_roundtrip(input);
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
        test_roundtrip(input);
    }

    #[test]
    fn test_empty_strings() {
        let input = StringArray::from(vec!["", "", "non-empty", ""]);
        test_roundtrip(input);
    }

    #[test]
    fn test_dictionary_roundtrip() {
        let input = StringArray::from(vec!["hello", "world", "hello", "rust"]);
        let compressor = LiquidByteArray::train_compressor(input.iter());
        let etc = LiquidByteArray::from_string_array(&input, compressor);
        let dict = etc.to_dict_arrow();

        // Check dictionary values are unique
        let dict_values = dict.values();
        let unique_values: std::collections::HashSet<&str> = dict_values
            .as_string::<i32>()
            .into_iter()
            .flatten()
            .collect();

        assert_eq!(unique_values.len(), 3); // "hello", "world", "rust"

        // Convert back to string array and verify
        let output = etc.to_arrow_array();
        let string_array = output.as_string::<i32>();
        assert_eq!(input.len(), string_array.len());
        for i in 0..input.len() {
            assert_eq!(input.value(i), string_array.value(i));
        }
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
                input: vec![Some("same"), Some("same"), Some("same"), Some("same")],
                needle: "same",
                expected: vec![Some(true), Some(true), Some(true), Some(true)],
            },
            TestCase {
                input: vec![
                    Some("apple"),
                    None,
                    Some("banana"),
                    Some("apple"),
                    None,
                    Some("cherry"),
                ],
                needle: "apple",
                expected: vec![Some(true), None, Some(false), Some(true), None, Some(false)],
            },
            TestCase {
                input: vec![Some("Hello"), Some("hello"), Some("HELLO"), Some("HeLLo")],
                needle: "hello",
                expected: vec![Some(false), Some(true), Some(false), Some(false)],
            },
            TestCase {
                input: vec![
                    Some("こんにちは"), // "Hello" in Japanese
                    Some("世界"),       // "World" in Japanese
                    Some("こんにちは"),
                    Some("rust"),
                ],
                needle: "こんにちは",
                expected: vec![Some(true), Some(false), Some(true), Some(false)],
            },
            TestCase {
                input: vec![Some("123"), Some("456"), Some("123"), Some("789")],
                needle: "123",
                expected: vec![Some(true), Some(false), Some(true), Some(false)],
            },
            TestCase {
                input: vec![Some("@home"), Some("#rust"), Some("@home"), Some("$money")],
                needle: "@home",
                expected: vec![Some(true), Some(false), Some(true), Some(false)],
            },
            TestCase {
                input: vec![None, None, None, None, Some("world")],
                needle: "hello",
                expected: vec![None, None, None, None, Some(false)],
            },
            // This cannot pass because the nulls are not handled correctly in `BitPackedArray::from_primitive`
            // TestCase {
            //     input: vec![None, None, None, None],
            //     needle: "hello",
            //     expected: vec![None, None, None, None],
            // },
        ];

        for case in test_cases {
            let input_array: StringArray = StringArray::from(case.input.clone());

            let compressor = LiquidByteArray::train_compressor(input_array.iter());
            let etc = LiquidByteArray::from_string_array(&input_array, compressor);

            let result: BooleanArray = etc.compare_equals(case.needle);

            let expected_array: BooleanArray = BooleanArray::from(case.expected.clone());

            assert_eq!(result, expected_array,);
        }
    }

    #[test]
    fn test_to_dict_arrow_preserves_type() {
        // Test string type preservation
        let input_str = StringArray::from(vec!["hello", "world", "test"]);
        let (_compressor_str, liquid_str) = LiquidByteArray::train_from_arrow(&input_str);
        let dict_str = liquid_str.to_dict_arrow();
        assert_eq!(
            dict_str.values().data_type(),
            &DataType::Utf8,
            "String values should be preserved as Utf8"
        );

        // Test binary type preservation
        let input_bin = cast(&input_str, &DataType::Binary)
            .unwrap()
            .as_binary::<i32>()
            .clone();
        let (_compressor_bin, liquid_bin) = LiquidByteArray::train_from_arrow(&input_bin);
        let dict_bin = liquid_bin.to_dict_arrow();
        assert_eq!(
            dict_bin.values().data_type(),
            &DataType::Binary,
            "Binary values should be preserved as Binary"
        );

        // Test dictionary-string array
        let dict_array = DictionaryArray::<UInt16Type>::from_iter(input_str.iter());
        let (_compressor_dict, liquid_dict) = LiquidByteArray::train_from_arrow_dict(&dict_array);
        let dict_result = liquid_dict.to_dict_arrow();
        assert_eq!(
            dict_result.values().data_type(),
            &DataType::Utf8,
            "Dictionary with binary values should preserve Utf8 type"
        );

        // Test dictionary-binary array
        let dict_array = DictionaryArray::<UInt16Type>::from_iter(input_str.iter());
        let dict_array = cast(
            &dict_array,
            &DataType::Dictionary(Box::new(DataType::UInt16), Box::new(DataType::Binary)),
        )
        .unwrap();
        let (_compressor_dict, liquid_dict) =
            LiquidByteArray::train_from_arrow_dict(dict_array.as_dictionary());
        let dict_result = liquid_dict.to_dict_arrow();
        assert_eq!(
            dict_result.values().data_type(),
            &DataType::Binary,
            "Dictionary with binary values should preserve Binary type"
        );
    }

    #[test]
    fn test_decompress_keyed_all_same_value() {
        // Tests when all keys reference the same compressed value
        let input_values = vec!["repeat"; 8];
        let input_array = StringArray::from(input_values);

        // Create liquid array with all keys pointing to index 0
        let compressor = LiquidByteArray::train_compressor(input_array.iter());
        let mut etc = LiquidByteArray::from_string_array(&input_array, compressor);
        etc.keys = BitPackedArray::from_primitive(
            UInt16Array::from(vec![0; 1000]),
            NonZero::new(1).unwrap(),
        );

        let dict = etc.to_dict_arrow_decompress_keyed();

        // Verify only one unique value exists
        assert_eq!(dict.values().len(), 1);
        assert_eq!(dict.values().as_string::<i32>().value(0), "repeat");

        // Verify all keys are remapped to 0
        let keys = dict.keys();
        assert!(keys.iter().all(|v| v == Some(0)));
    }

    #[test]
    fn test_decompress_keyed_sparse_references() {
        // Tests when only a subset of values are referenced
        let values = vec!["a", "b", "c", "d", "e"];
        let input_keys = UInt16Array::from(vec![0, 2, 4, 2, 0]); // References a, c, e, c, a
        let input_array = StringArray::from(values.clone());

        let compressor = LiquidByteArray::train_compressor(input_array.iter());
        let etc = LiquidByteArray {
            keys: BitPackedArray::from_primitive(input_keys.clone(), NonZero::new(3).unwrap()),
            values: FsstArray::from_byte_array_with_compressor(&input_array, compressor),
            original_arrow_type: ArrowByteType::Dict16Utf8,
        };

        let dict = etc.to_dict_arrow_decompress_keyed();

        // Should only decompress a, c, e (indexes 0,2,4 from original)
        assert_eq!(dict.values().len(), 3);
        let dict_values = dict.values().as_string::<i32>();
        assert_eq!(dict_values.value(0), "a");
        assert_eq!(dict_values.value(1), "c");
        assert_eq!(dict_values.value(2), "e");

        // Verify key remapping: original 0→0, 2→1, 4→2
        let expected_keys = UInt16Array::from(vec![0, 1, 2, 1, 0]);
        assert_eq!(dict.keys(), &expected_keys);
    }

    #[test]
    fn test_decompress_keyed_with_nulls_and_unreferenced() {
        // Tests null handling and unreferenced values
        let values = vec!["a", "b", "c", "d"];
        let input_keys = UInt16Array::from(vec![Some(0), None, Some(3), Some(0), None, Some(2)]);
        let input_array = StringArray::from(values.clone());

        let compressor = LiquidByteArray::train_compressor(input_array.iter());
        let etc = LiquidByteArray {
            keys: BitPackedArray::from_primitive(input_keys.clone(), NonZero::new(2).unwrap()),
            values: FsstArray::from_byte_array_with_compressor(&input_array, compressor),
            original_arrow_type: ArrowByteType::Dict16Utf8,
        };

        let dict = etc.to_dict_arrow_decompress_keyed();

        // Verify values
        assert_eq!(dict.values().len(), 3);
        let dict_values = dict.values().as_string::<i32>();
        assert_eq!(dict_values.value(0), "a");
        assert_eq!(dict_values.value(1), "c");
        assert_eq!(dict_values.value(2), "d");

        // Verify keys and nulls
        let expected_keys = UInt16Array::from(vec![Some(0), None, Some(2), Some(0), None, Some(1)]);
        assert_eq!(dict.keys(), &expected_keys);
        assert_eq!(dict.nulls(), input_keys.nulls());
    }

    #[test]
    fn test_roundtrip_edge_cases() {
        use arrow::array::StringBuilder;

        // Create a string array with various edge cases
        let mut builder = StringBuilder::new();

        // Empty string
        builder.append_value("");

        // Section of nulls
        for _ in 0..10 {
            builder.append_null();
        }

        // Very long string
        let long_string = "a".repeat(10_000);
        builder.append_value(&long_string);

        // Unicode and special characters
        builder.append_value("こんにちは世界"); // Hello world in Japanese
        builder.append_value("🚀🔥🌈⭐"); // Emoji characters
        builder.append_value("Special chars: !@#$%^&*(){}[]|\\/.,<>?`~");

        // Single character strings
        for c in "abcdefghijklmnopqrstuvwxyz".chars() {
            builder.append_value(c.to_string());
        }

        // Highly repetitive string to test compression effectiveness
        builder.append_value("ABABABABABABABABABABABABABABAB");

        // Test the roundtrip
        let string_array = builder.finish();
        test_roundtrip(string_array);
    }

    #[test]
    fn test_filter_all_nulls() {
        let original: Vec<Option<&str>> = vec![None, None, None];
        let array = StringArray::from(original.clone());
        let compressor = LiquidByteArray::train_compressor(array.iter());
        let liquid_array = LiquidByteArray::from_string_array(&array, compressor);
        let result_array = liquid_array.filter(&BooleanArray::from(vec![true, false, true]));
        let result_array = result_array.to_arrow_array();

        assert_eq!(result_array.len(), 2);
        assert_eq!(result_array.null_count(), 2);
    }

    #[test]
    fn test_get_string_needle() {
        // Test Utf8 scalar value
        let utf8_value = ScalarValue::Utf8(Some("test_string".to_string()));
        assert_eq!(get_string_needle(&utf8_value), Some("test_string"));

        // Test Utf8View scalar value
        let utf8_view_value = ScalarValue::Utf8View(Some("test_view".to_string()));
        assert_eq!(get_string_needle(&utf8_view_value), Some("test_view"));

        // Test LargeUtf8 scalar value
        let large_utf8_value = ScalarValue::LargeUtf8(Some("test_large".to_string()));
        assert_eq!(get_string_needle(&large_utf8_value), Some("test_large"));

        // Test Dictionary scalar value
        let dict_inner = ScalarValue::Utf8(Some("test_dict".to_string()));
        let dict_value = ScalarValue::Dictionary(
            Box::new(arrow_schema::DataType::Int32),
            Box::new(dict_inner),
        );
        assert_eq!(get_string_needle(&dict_value), Some("test_dict"));

        // Test None values
        let utf8_none = ScalarValue::Utf8(None);
        assert_eq!(get_string_needle(&utf8_none), None);

        // Test non-string scalar value
        let int_value = ScalarValue::Int32(Some(42));
        assert_eq!(get_string_needle(&int_value), None);
    }

    #[test]
    fn test_try_eval_predicate_inner_string_equality() {
        use datafusion::logical_expr::Operator;
        use datafusion::physical_expr::expressions::BinaryExpr;
        use datafusion::physical_plan::expressions::{Column, Literal};
        use datafusion::scalar::ScalarValue;

        // Test the optimized string equality path
        let string_data = vec!["apple", "banana", "cherry", "apple", "grape"];
        let arrow_array = StringViewArray::from(string_data);
        let (_compressor, liquid_array) = LiquidByteArray::train_from_arrow_view(&arrow_array);

        // Create predicate: column = "apple"
        let expr: Arc<dyn PhysicalExpr> = Arc::new(BinaryExpr::new(
            Arc::new(Column::new("test_col", 0)),
            Operator::Eq,
            Arc::new(Literal::new(ScalarValue::Utf8(Some("apple".to_string())))),
        ));

        let result = try_eval_predicate_inner(&expr, &liquid_array).unwrap();
        assert!(result.is_some());

        let boolean_array = result.unwrap();
        assert_eq!(boolean_array.len(), 5);

        // Should match indices 0 and 3 (both "apple")
        assert!(boolean_array.value(0)); // "apple"
        assert!(!boolean_array.value(1)); // "banana"
        assert!(!boolean_array.value(2)); // "cherry"
        assert!(boolean_array.value(3)); // "apple"
        assert!(!boolean_array.value(4)); // "grape"
    }

    #[test]
    fn test_try_eval_predicate_inner_numeric_not_supported() {
        // Test that numeric comparisons are not supported and return None
        let numeric_data = vec![10, 20, 30, 15, 25];
        let arrow_array = Int32Array::from(numeric_data);
        let liquid_array = LiquidPrimitiveArray::<Int32Type>::from_arrow_array(arrow_array);
        let liquid_ref: LiquidArrayRef = Arc::new(liquid_array);

        // Create predicate: column > 20
        let expr: Arc<dyn PhysicalExpr> = Arc::new(BinaryExpr::new(
            Arc::new(Column::new("test_col", 0)),
            Operator::Gt,
            Arc::new(Literal::new(ScalarValue::Int32(Some(20)))),
        ));

        let filter = BooleanArray::from(vec![true; liquid_ref.len()]);
        let result = liquid_ref.try_eval_predicate(&expr, &filter).unwrap();
        // Numeric comparisons are not supported, should return None
        assert!(result.is_none());

        // Test other numeric operators that are also not supported
        let eq_expr: Arc<dyn PhysicalExpr> = Arc::new(BinaryExpr::new(
            Arc::new(Column::new("test_col", 0)),
            Operator::Eq,
            Arc::new(Literal::new(ScalarValue::Int32(Some(20)))),
        ));

        let result = liquid_ref.try_eval_predicate(&eq_expr, &filter).unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn test_try_eval_predicate_inner_unsupported_expression() {
        // Test unsupported expression types that should return None
        let numeric_data = vec![10, 20, 30, 15, 25];
        let arrow_array = Int32Array::from(numeric_data);
        let liquid_array = LiquidPrimitiveArray::<Int32Type>::from_arrow_array(arrow_array);
        let liquid_ref: LiquidArrayRef = Arc::new(liquid_array);

        // Create unsupported predicate: column + 5 (not column op literal)
        let add_expr: Arc<dyn PhysicalExpr> = Arc::new(BinaryExpr::new(
            Arc::new(Column::new("test_col", 0)),
            Operator::Plus,
            Arc::new(Literal::new(ScalarValue::Int32(Some(5)))),
        ));

        let filter = BooleanArray::from(vec![true; liquid_ref.len()]);
        let result = liquid_ref.try_eval_predicate(&add_expr, &filter).unwrap();
        assert!(result.is_none());

        // Test another unsupported case: literal op column (wrong order)
        let wrong_order_expr: Arc<dyn PhysicalExpr> = Arc::new(BinaryExpr::new(
            Arc::new(Literal::new(ScalarValue::Int32(Some(20)))),
            Operator::Eq,
            Arc::new(Column::new("test_col", 0)),
        ));

        let result = liquid_ref
            .try_eval_predicate(&wrong_order_expr, &filter)
            .unwrap();
        assert!(result.is_none());

        // Test column-column comparison (not column-literal)
        let col_col_expr: Arc<dyn PhysicalExpr> = Arc::new(BinaryExpr::new(
            Arc::new(Column::new("col1", 0)),
            Operator::Eq,
            Arc::new(Column::new("col2", 1)),
        ));

        let result = liquid_ref
            .try_eval_predicate(&col_col_expr, &filter)
            .unwrap();
        assert!(result.is_none());
    }
}
