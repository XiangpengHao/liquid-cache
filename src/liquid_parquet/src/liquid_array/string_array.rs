use std::any::Any;
use std::num::NonZero;
use std::sync::Arc;

use arrow::array::builder::StringDictionaryBuilder;
use arrow::array::{
    Array, ArrayRef, BinaryArray, BooleanArray, DictionaryArray, PrimitiveArray, RecordBatch,
    StringArray, cast::AsArray, types::UInt16Type,
};
use arrow::array::{
    ArrayAccessor, ArrayIter, GenericByteArray, GenericByteDictionaryBuilder, StringViewArray,
};
use arrow::compute::{cast, kernels};

use arrow::buffer::BooleanBuffer;
use arrow::datatypes::{BinaryType, ByteArrayType, Utf8Type};
use arrow_schema::{DataType, Field, Schema};
use fsst::Compressor;

use crate::liquid_array::{FsstArray, get_bit_width};

use super::{BitPackedArray, LiquidArray, LiquidArrayRef};

impl LiquidArray for LiquidStringArray {
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
        let dict = self.to_dict_string();
        Arc::new(dict)
    }

    fn filter(&self, selection: &BooleanArray) -> LiquidArrayRef {
        let values = self.values.clone();
        let keys = self.keys.clone();
        let primitive_keys = keys.to_primitive();
        let filtered_keys = arrow::compute::filter(&primitive_keys, selection)
            .unwrap()
            .as_primitive::<UInt16Type>()
            .clone();
        let bit_packed_array = BitPackedArray::from_primitive(filtered_keys, keys.bit_width);
        Arc::new(LiquidStringArray {
            keys: bit_packed_array,
            values,
            arrow_type: self.arrow_type,
        })
    }
}

/// Metadata for the EtcStringArray.
pub struct LiquidStringMetadata {
    compressor: Arc<Compressor>,
    uncompressed_len: u32,
    keys_original_len: u32,
    keys_bit_width: NonZero<u8>,
}

impl std::fmt::Debug for LiquidStringMetadata {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "LiquidStringMetadata")
    }
}

#[derive(Debug, Clone, Copy)]
enum ArrowStringType {
    Utf8,
    Utf8View,
    Dict16Binary, // DictionaryArray<UInt16Type>
    Dict16Utf8,   // DictionaryArray<UInt16Type>
    Binary,
    BinaryView,
}

impl ArrowStringType {
    pub fn to_arrow_type(self) -> DataType {
        match self {
            ArrowStringType::Utf8 => DataType::Utf8,
            ArrowStringType::Utf8View => DataType::Utf8View,
            ArrowStringType::Dict16Binary => {
                DataType::Dictionary(Box::new(DataType::UInt16), Box::new(DataType::Binary))
            }
            ArrowStringType::Dict16Utf8 => {
                DataType::Dictionary(Box::new(DataType::UInt16), Box::new(DataType::Utf8))
            }
            ArrowStringType::Binary => DataType::Binary,
            ArrowStringType::BinaryView => DataType::BinaryView,
        }
    }

    pub fn from_arrow_type(ty: &DataType) -> Self {
        match ty {
            DataType::Utf8 => ArrowStringType::Utf8,
            DataType::Utf8View => ArrowStringType::Utf8View,
            DataType::Binary => ArrowStringType::Binary,
            DataType::BinaryView => ArrowStringType::BinaryView,
            DataType::Dictionary(_, _) => {
                if ty
                    == &DataType::Dictionary(Box::new(DataType::UInt16), Box::new(DataType::Binary))
                {
                    ArrowStringType::Dict16Binary
                } else if ty
                    == &DataType::Dictionary(Box::new(DataType::UInt16), Box::new(DataType::Utf8))
                {
                    ArrowStringType::Dict16Utf8
                } else {
                    panic!("Unsupported arrow type: {:?}", ty)
                }
            }
            _ => panic!("Unsupported arrow type: {:?}", ty),
        }
    }
}

/// An array that stores strings in a dictionary format, with a bit-packed array for the keys and a FSST array for the values.
#[derive(Debug)]
pub struct LiquidStringArray {
    keys: BitPackedArray<UInt16Type>,
    /// TODO: we need to specify that the values in the FsstArray must be unique, this enables us some optimizations.
    values: FsstArray,
    arrow_type: ArrowStringType,
}

impl LiquidStringArray {
    pub fn from_string_view_array(array: &StringViewArray, compressor: Arc<Compressor>) -> Self {
        let dict = byte_array_to_dict_string::<Utf8Type, _>(array.iter());
        Self::from_dict_array_inner(dict, compressor, ArrowStringType::Utf8View)
    }

    fn train_compressor_bytes<'a, T: ArrayAccessor<Item = &'a [u8]>>(
        array: ArrayIter<T>,
    ) -> Arc<Compressor> {
        let strings = array.filter_map(|s| s.as_ref().map(|s| *s));
        Arc::new(FsstArray::train_compressor(strings))
    }

    fn train_compressor<'a, T: ArrayAccessor<Item = &'a str>>(
        array: ArrayIter<T>,
    ) -> Arc<Compressor> {
        let strings = array.filter_map(|s| s.as_ref().map(|s| s.as_bytes()));
        Arc::new(FsstArray::train_compressor(strings))
    }

    pub fn from_string_array(array: &StringArray, compressor: Arc<Compressor>) -> Self {
        Self::from_byte_array(array, compressor)
    }

    pub fn from_byte_array<T: ByteArrayType>(
        array: &GenericByteArray<T>,
        compressor: Arc<Compressor>,
    ) -> Self {
        let dict = byte_array_to_dict_string::<T, _>(array.iter());
        Self::from_dict_array_inner(
            dict,
            compressor,
            ArrowStringType::from_arrow_type(&T::DATA_TYPE),
        )
    }

    pub fn train_from_arrow_view(array: &StringViewArray) -> (Arc<Compressor>, Self) {
        let dict = byte_array_to_dict_string::<Utf8Type, _>(array.iter());
        let compressor = Self::train_compressor(dict.values().as_string::<i32>().iter());
        (
            compressor.clone(),
            Self::from_dict_array_inner(dict, compressor, ArrowStringType::Dict16Utf8),
        )
    }

    pub fn train_from_arrow<T: ByteArrayType>(
        array: &GenericByteArray<T>,
    ) -> (Arc<Compressor>, Self) {
        let dict = byte_array_to_dict_string::<T, _>(array.iter());
        let values = dict.values().as_string::<i32>();
        let compressor = Self::train_compressor(values.iter());
        (
            compressor.clone(),
            Self::from_dict_array_inner(
                dict,
                compressor,
                ArrowStringType::from_arrow_type(&T::DATA_TYPE),
            ),
        )
    }

    pub fn train_from_arrow_dict(array: &DictionaryArray<UInt16Type>) -> (Arc<Compressor>, Self) {
        if array.values().data_type() == &DataType::Utf8 {
            let values = array.values().as_string::<i32>();
            let compressor = Self::train_compressor(values.iter());
            (
                compressor.clone(),
                Self::from_dict_array_inner(array.clone(), compressor, ArrowStringType::Dict16Utf8),
            )
        } else if array.values().data_type() == &DataType::Binary {
            let values = array.values().as_binary::<i32>();
            let compressor = Self::train_compressor_bytes(values.iter());
            (
                compressor.clone(),
                Self::from_dict_array_inner(
                    array.clone(),
                    compressor,
                    ArrowStringType::Dict16Binary,
                ),
            )
        } else {
            panic!("Unsupported dictionary type: {:?}", array.data_type())
        }
    }

    fn from_dict_array_inner(
        array: DictionaryArray<UInt16Type>,
        compressor: Arc<Compressor>,
        arrow_type: ArrowStringType,
    ) -> Self {
        let (keys, values) = array.into_parts();

        let (_, keys, nulls) = keys.into_parts();
        let keys = PrimitiveArray::<UInt16Type>::try_new(keys, nulls).unwrap();

        let distinct_count = values.len();
        let max_bit_width = get_bit_width(distinct_count as u64);
        debug_assert!(2u64.pow(max_bit_width.get() as u32) >= distinct_count as u64);

        let bit_packed_array = BitPackedArray::from_primitive(keys, max_bit_width);

        let fsst_values = if let Some(values) = values.as_string_opt::<i32>() {
            FsstArray::from_byte_array_with_compressor(values, compressor)
        } else if let Some(values) = values.as_binary_opt::<i32>() {
            FsstArray::from_byte_array_with_compressor(values, compressor)
        } else {
            panic!("Unsupported dictionary type")
        };
        LiquidStringArray {
            keys: bit_packed_array,
            values: fsst_values,
            arrow_type,
        }
    }

    pub fn from_dict_array(
        array: &DictionaryArray<UInt16Type>,
        compressor: Arc<Compressor>,
    ) -> Self {
        if array.downcast_dict::<StringArray>().is_some() {
            let dict = array.downcast_dict::<StringArray>().unwrap();

            // We require the values to be unique, but the incoming dictionary may not be unique, has to deduplicate again here.
            let mut deduplicated = StringDictionaryBuilder::<UInt16Type>::new();
            for v in dict.into_iter() {
                deduplicated.append_option(v);
            }
            let dict = deduplicated.finish();
            Self::from_dict_array_inner(dict, compressor, ArrowStringType::Dict16Utf8)
        } else if array.downcast_dict::<BinaryArray>().is_some() {
            let dict = array.downcast_dict::<BinaryArray>().unwrap();

            // We require the values to be unique, but the incoming dictionary may not be unique, has to deduplicate again here.
            let mut deduplicated = GenericByteDictionaryBuilder::<UInt16Type, BinaryType>::new();
            for v in dict.into_iter() {
                deduplicated.append_option(v);
            }
            let dict = deduplicated.finish();
            Self::from_dict_array_inner(dict, compressor, ArrowStringType::Dict16Binary)
        } else {
            panic!("Unsupported dictionary type: {:?}", array.data_type())
        }
    }

    /// Get the compressor of the LiquidStringArray.
    pub fn compressor(&self) -> Arc<Compressor> {
        self.values.compressor.clone()
    }

    /// Convert the LiquidStringArray to a DictionaryArray.
    pub fn to_dict_string(&self) -> DictionaryArray<UInt16Type> {
        let primitive_key = self.keys.to_primitive().clone();
        let values: StringArray = StringArray::from(&self.values);
        unsafe { DictionaryArray::<UInt16Type>::new_unchecked(primitive_key, Arc::new(values)) }
    }

    /// Convert the LiquidStringArray to a DictionaryArray with a selection.
    pub fn to_dict_string_with_selection(
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
        let dict = self.to_dict_string();
        cast(&dict, &self.arrow_type.to_arrow_type()).unwrap()
    }

    /// Repackage the data into Arrow-compatible format, so that it can be written to disk, transferred over flight.
    pub fn to_record_batch(&self) -> (RecordBatch, LiquidStringMetadata) {
        let schema = Schema::new(vec![
            Field::new("keys", DataType::UInt32, false),
            Field::new("values", DataType::Binary, false),
        ]);
        let batch = RecordBatch::try_new(Arc::new(schema), vec![
            Arc::new(self.keys.values.clone()),
            Arc::new(self.values.compressed.clone()),
        ])
        .unwrap();
        (batch, self.metadata())
    }

    /// Reconstruct the LiquidStringArray from a RecordBatch.
    pub fn from_record_batch(batch: RecordBatch, metadata: &LiquidStringMetadata) -> Self {
        let key_column = batch.column(0).as_primitive::<UInt16Type>();
        let values_column = batch.column(1).as_binary();

        let keys = BitPackedArray::from_parts(
            key_column.clone(),
            metadata.keys_bit_width,
            metadata.keys_original_len as usize,
        );
        let values = FsstArray::from_parts(
            values_column.clone(),
            metadata.compressor.clone(),
            metadata.uncompressed_len as usize,
        );
        LiquidStringArray {
            keys,
            values,
            arrow_type: ArrowStringType::Dict16Binary,
        }
    }

    /// Get the metadata of the LiquidStringArray.
    pub fn metadata(&self) -> LiquidStringMetadata {
        LiquidStringMetadata {
            compressor: self.values.compressor.clone(),
            uncompressed_len: self.values.uncompressed_len as u32,
            keys_original_len: self.keys.values.len() as u32,
            keys_bit_width: self.keys.bit_width,
        }
    }

    /// Compare the values of the LiquidStringArray with a given string and return a BooleanArray of the result.
    pub fn compare_not_equals(&self, needle: &str) -> BooleanArray {
        let result = self.compare_equals(needle);
        let (values, nulls) = result.into_parts();
        let values = !&values;
        BooleanArray::new(values, nulls)
    }

    /// Compare the values of the LiquidStringArray with a given string.
    pub fn compare_equals(&self, needle: &str) -> BooleanArray {
        let compressor = &self.values.compressor;
        let compressed = compressor.compress(needle.as_bytes());
        let compressed = BinaryArray::new_scalar(compressed);

        let values = &self.values.compressed;

        let result = kernels::cmp::eq(values, &compressed).unwrap();
        let (values, nulls) = result.into_parts();
        assert!(
            nulls.is_none(),
            "The dictionary values should not have nulls"
        );

        let keys = self.keys.to_primitive();
        let mut return_mask = BooleanBuffer::new_unset(keys.len());
        for idx in values.set_indices() {
            let result =
                kernels::cmp::eq(&keys, &PrimitiveArray::<UInt16Type>::new_scalar(idx as u16))
                    .unwrap();
            let (values, _nulls) = result.into_parts();
            return_mask = &return_mask | &values;
        }
        BooleanArray::new(return_mask, keys.nulls().cloned())
    }
}

fn byte_array_to_dict_string<'a, T: ByteArrayType, I: ArrayAccessor<Item = &'a T::Native>>(
    input: ArrayIter<I>,
) -> DictionaryArray<UInt16Type> {
    let mut builder = GenericByteDictionaryBuilder::<UInt16Type, T>::new();
    for s in input {
        builder.append_option(s);
    }
    builder.finish()
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::Array;

    #[test]
    fn test_simple_roundtrip() {
        let input = StringArray::from(vec!["hello", "world", "hello", "rust"]);
        let compressor = LiquidStringArray::train_compressor(input.iter());
        let etc = LiquidStringArray::from_string_array(&input, compressor);
        let output = etc.to_arrow_array();
        let string_array = output.as_string::<i32>();
        assert_eq!(input.len(), string_array.len());
        for i in 0..input.len() {
            assert_eq!(input.value(i), string_array.value(i));
        }
    }

    #[test]
    fn test_to_arrow_array_preserve_arrow_type() {
        let input = StringArray::from(vec!["hello", "world", "hello", "rust"]);
        let compressor = LiquidStringArray::train_compressor(input.iter());
        let etc = LiquidStringArray::from_string_array(&input, compressor);
        let output = etc.to_arrow_array();
        assert_eq!(&input, output.as_string::<i32>());

        let input = cast(&input, &DataType::Utf8View)
            .unwrap()
            .as_string_view()
            .clone();
        let compressor = LiquidStringArray::train_compressor(input.iter());
        let etc = LiquidStringArray::from_string_view_array(&input, compressor);
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
            LiquidStringArray::train_compressor(input.values().as_string::<i32>().iter());
        let etc = LiquidStringArray::from_dict_array(&input, compressor);
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
        let compressor = LiquidStringArray::train_compressor(input.iter());
        let etc = LiquidStringArray::from_string_array(&input, compressor);
        let output = etc.to_arrow_array();
        let string_array = output.as_string::<i32>();

        assert_eq!(input.len(), string_array.len());
        for i in 0..input.len() {
            if input.is_null(i) {
                assert!(string_array.is_null(i));
            } else {
                assert_eq!(input.value(i), string_array.value(i));
            }
        }
    }

    #[test]
    fn test_roundtrip_with_many_duplicates() {
        let values = vec!["a", "b", "c"];
        let input: Vec<&str> = (0..1000).map(|i| values[i % values.len()]).collect();
        let input = StringArray::from(input);

        let compressor = LiquidStringArray::train_compressor(input.iter());
        let etc = LiquidStringArray::from_string_array(&input, compressor);
        let output = etc.to_arrow_array();
        let string_array = output.as_string::<i32>();

        assert_eq!(input.len(), string_array.len());
        for i in 0..input.len() {
            assert_eq!(input.value(i), string_array.value(i));
        }
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

        let compressor = LiquidStringArray::train_compressor(input.iter());
        let etc = LiquidStringArray::from_string_array(&input, compressor);
        let output = etc.to_arrow_array();
        let string_array = output.as_string::<i32>();
        assert_eq!(input.len(), string_array.len());
        for i in 0..input.len() {
            assert_eq!(input.value(i), string_array.value(i));
        }
    }

    #[test]
    fn test_empty_strings() {
        let input = StringArray::from(vec!["", "", "non-empty", ""]);
        let compressor = LiquidStringArray::train_compressor(input.iter());
        let etc = LiquidStringArray::from_string_array(&input, compressor);
        let output = etc.to_arrow_array();
        let string_array = output.as_string::<i32>();
        assert_eq!(input.len(), string_array.len());
        for i in 0..input.len() {
            assert_eq!(input.value(i), string_array.value(i));
        }
    }

    #[test]
    fn test_dictionary_roundtrip() {
        let input = StringArray::from(vec!["hello", "world", "hello", "rust"]);
        let compressor = LiquidStringArray::train_compressor(input.iter());
        let etc = LiquidStringArray::from_string_array(&input, compressor);
        let dict = etc.to_dict_string();

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
        ];

        for case in test_cases {
            let input_array: StringArray = StringArray::from(case.input.clone());

            let compressor = LiquidStringArray::train_compressor(input_array.iter());
            let etc = LiquidStringArray::from_string_array(&input_array, compressor);

            let result: BooleanArray = etc.compare_equals(case.needle);

            let expected_array: BooleanArray = BooleanArray::from(case.expected.clone());

            assert_eq!(result, expected_array,);
        }
    }
}
