use arrow::array::{
    Array, ArrayAccessor, ArrayIter, BinaryArray, BinaryViewArray, DictionaryArray,
    GenericByteArray, StringArray, StringViewArray, UInt16Array, cast::AsArray, types::UInt16Type,
};
use arrow::datatypes::ByteArrayType;
use arrow_schema::DataType;
use fsst::Compressor;
use std::sync::Arc;

use super::LiquidByteViewArray;
use crate::liquid_array::byte_array::ArrowByteType;
use crate::liquid_array::raw::fsst_buffer::{
    FsstArray, FsstBacking, PrefixKey, RawFsstBuffer, train_compressor,
};
use crate::utils::CheckedDictionaryArray;

impl<B: FsstBacking> LiquidByteViewArray<B> {
    /// Create a LiquidByteViewArray from an Arrow StringViewArray
    pub fn from_string_view_array(
        array: &StringViewArray,
        compressor: Arc<Compressor>,
    ) -> LiquidByteViewArray<FsstArray> {
        Self::from_view_array_inner(array, compressor, ArrowByteType::Utf8View)
    }

    /// Create a LiquidByteViewArray from an Arrow BinaryViewArray
    pub fn from_binary_view_array(
        array: &BinaryViewArray,
        compressor: Arc<Compressor>,
    ) -> LiquidByteViewArray<FsstArray> {
        Self::from_view_array_inner(array, compressor, ArrowByteType::BinaryView)
    }

    /// Create a LiquidByteViewArray from an Arrow StringArray
    pub fn from_string_array(
        array: &StringArray,
        compressor: Arc<Compressor>,
    ) -> LiquidByteViewArray<FsstArray> {
        Self::from_byte_array_inner(array, compressor, ArrowByteType::Utf8)
    }

    /// Create a LiquidByteViewArray from an Arrow BinaryArray
    pub fn from_binary_array(
        array: &BinaryArray,
        compressor: Arc<Compressor>,
    ) -> LiquidByteViewArray<FsstArray> {
        Self::from_byte_array_inner(array, compressor, ArrowByteType::Binary)
    }

    /// Train a compressor from an Arrow StringViewArray
    pub fn train_from_string_view(
        array: &StringViewArray,
    ) -> (Arc<Compressor>, LiquidByteViewArray<FsstArray>) {
        let compressor = Self::train_compressor(array.iter());
        (
            compressor.clone(),
            Self::from_view_array_inner(array, compressor, ArrowByteType::Utf8View),
        )
    }

    /// Train a compressor from an Arrow BinaryViewArray
    pub fn train_from_binary_view(
        array: &BinaryViewArray,
    ) -> (Arc<Compressor>, LiquidByteViewArray<FsstArray>) {
        let compressor = Self::train_compressor_bytes(array.iter());
        (
            compressor.clone(),
            Self::from_view_array_inner(array, compressor, ArrowByteType::BinaryView),
        )
    }

    /// Train a compressor from an Arrow ByteArray.
    pub fn train_from_arrow<T: ByteArrayType>(
        array: &GenericByteArray<T>,
    ) -> (Arc<Compressor>, LiquidByteViewArray<FsstArray>) {
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

    /// Only used when the dictionary is read from a trusted parquet reader,
    /// which reads a trusted parquet file, written by a trusted writer.
    ///
    /// # Safety
    /// The caller must ensure that the values in the dictionary are unique.
    pub unsafe fn from_unique_dict_array(
        array: &DictionaryArray<UInt16Type>,
        compressor: Arc<Compressor>,
    ) -> LiquidByteViewArray<FsstArray> {
        let arrow_type = ArrowByteType::from_arrow_type(array.values().data_type());
        Self::from_dict_array_inner(
            unsafe { CheckedDictionaryArray::new_unchecked_i_know_what_i_am_doing(array) },
            compressor,
            arrow_type,
        )
    }

    /// Train a compressor from an Arrow DictionaryArray.
    pub fn train_from_arrow_dict(
        array: &DictionaryArray<UInt16Type>,
    ) -> (Arc<Compressor>, LiquidByteViewArray<FsstArray>) {
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
    ) -> LiquidByteViewArray<FsstArray>
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
    ) -> LiquidByteViewArray<FsstArray> {
        let dict = CheckedDictionaryArray::from_byte_array::<T>(array);
        Self::from_dict_array_inner(dict, compressor, arrow_type)
    }

    /// Core implementation that converts a CheckedDictionaryArray to LiquidByteViewArray
    fn from_dict_array_inner(
        dict: CheckedDictionaryArray,
        compressor: Arc<Compressor>,
        arrow_type: ArrowByteType,
    ) -> LiquidByteViewArray<FsstArray> {
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

        // Prefix keys - one per unique value in dictionary.
        let mut prefix_keys = Vec::with_capacity(values.len());

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

        for i in 0..values.len() {
            let value_bytes = if let Some(string_values) = values.as_string_opt::<i32>() {
                string_values.value(i).as_bytes()
            } else if let Some(binary_values) = values.as_binary_opt::<i32>() {
                binary_values.value(i)
            } else {
                panic!("Unsupported dictionary value type")
            };

            let remaining_bytes = if shared_prefix_len < value_bytes.len() {
                &value_bytes[shared_prefix_len..]
            } else {
                &[]
            };

            prefix_keys.push(PrefixKey::new(remaining_bytes));
        }

        assert_eq!(values.len(), byte_offsets.len() - 1);

        let prefix_keys: Arc<[PrefixKey]> = prefix_keys.into();

        LiquidByteViewArray::from_parts(
            keys,
            prefix_keys,
            FsstArray::from_byte_offsets(Arc::new(raw_fsst_buffer), &byte_offsets, compressor),
            arrow_type,
            shared_prefix,
        )
    }

    /// Create LiquidByteViewArray from parts
    pub(super) fn from_parts(
        dictionary_keys: UInt16Array,
        prefix_keys: Arc<[PrefixKey]>,
        fsst_buffer: B,
        original_arrow_type: ArrowByteType,
        shared_prefix: Vec<u8>,
    ) -> Self {
        Self {
            dictionary_keys,
            prefix_keys,
            fsst_buffer,
            original_arrow_type,
            shared_prefix,
            string_fingerprints: None,
        }
    }
}
