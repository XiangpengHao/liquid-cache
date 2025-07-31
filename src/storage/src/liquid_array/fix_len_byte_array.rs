use std::{any::Any, sync::Arc};

use ahash::HashMap;
use arrow::{
    array::{
        Array, ArrayRef, AsArray, BooleanArray, BooleanBufferBuilder, DictionaryArray,
        PrimitiveArray, UInt16Array,
    },
    buffer::Buffer,
    compute::kernels::cast,
    datatypes::{Decimal128Type, Decimal256Type, DecimalType, UInt16Type},
};
use arrow_schema::DataType;
use fsst::Compressor;
use std::mem::MaybeUninit;

use crate::utils::CheckedDictionaryArray;

use super::{
    LiquidArray, LiquidArrayRef, LiquidDataType,
    raw::{BitPackedArray, FsstArray},
};

/// A fixed length byte array.
#[derive(Debug)]
pub struct LiquidFixedLenByteArray {
    arrow_type: ArrowFixedLenByteArrayType,
    keys: BitPackedArray<UInt16Type>,
    values: FsstArray,
}

#[derive(Debug, Clone)]
pub enum ArrowFixedLenByteArrayType {
    Decimal128(u8, i8),
    Decimal256(u8, i8),
}

impl From<&DataType> for ArrowFixedLenByteArrayType {
    fn from(value: &DataType) -> Self {
        match value {
            DataType::Decimal128(precision, scale) => {
                ArrowFixedLenByteArrayType::Decimal128(*precision, *scale)
            }
            DataType::Decimal256(precision, scale) => {
                ArrowFixedLenByteArrayType::Decimal256(*precision, *scale)
            }
            _ => panic!("Unsupported arrow type: {value:?}"),
        }
    }
}

impl From<&ArrowFixedLenByteArrayType> for DataType {
    fn from(value: &ArrowFixedLenByteArrayType) -> Self {
        match value {
            ArrowFixedLenByteArrayType::Decimal128(precision, scale) => {
                DataType::Decimal128(*precision, *scale)
            }
            ArrowFixedLenByteArrayType::Decimal256(precision, scale) => {
                DataType::Decimal256(*precision, *scale)
            }
        }
    }
}

impl ArrowFixedLenByteArrayType {
    pub fn value_width(&self) -> usize {
        match self {
            ArrowFixedLenByteArrayType::Decimal128(_, _) => Decimal128Type::BYTE_LENGTH,
            ArrowFixedLenByteArrayType::Decimal256(_, _) => Decimal256Type::BYTE_LENGTH,
        }
    }
}

impl LiquidArray for LiquidFixedLenByteArray {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn get_array_memory_size(&self) -> usize {
        self.keys.get_array_memory_size() + self.values.get_array_memory_size()
    }

    fn len(&self) -> usize {
        self.keys.len()
    }

    fn to_arrow_array(&self) -> ArrayRef {
        if self.keys.len() < 2048 || self.keys.len() < self.values.len() {
            // Use keyed decompression for smaller arrays
            self.to_arrow_array_decompress_keyed()
        } else {
            // Use full decompression for larger arrays
            self.to_arrow_array_decompress_all()
        }
    }

    fn to_best_arrow_array(&self) -> ArrayRef {
        self.to_arrow_array()
    }

    fn filter(&self, selection: &BooleanArray) -> LiquidArrayRef {
        let values = self.values.clone();
        let keys = self.keys.clone();
        let primitive_keys = keys.to_primitive();
        let filtered_keys = arrow::compute::filter(&primitive_keys, selection)
            .unwrap()
            .as_primitive::<UInt16Type>()
            .clone();
        let Some(bit_width) = keys.bit_width() else {
            return Arc::new(LiquidFixedLenByteArray {
                arrow_type: self.arrow_type.clone(),
                keys: BitPackedArray::new_null_array(filtered_keys.len()),
                values,
            });
        };
        let bit_packed_array = BitPackedArray::from_primitive(filtered_keys, bit_width);
        Arc::new(LiquidFixedLenByteArray {
            arrow_type: self.arrow_type.clone(),
            keys: bit_packed_array,
            values,
        })
    }

    fn to_bytes(&self) -> Vec<u8> {
        self.to_bytes_inner()
    }

    fn data_type(&self) -> LiquidDataType {
        LiquidDataType::FixedLenByteArray
    }
}

impl LiquidFixedLenByteArray {
    /// Create a new fixed length byte array from a decimal array.
    pub fn from_decimal_array<T: DecimalType>(
        array: &PrimitiveArray<T>,
        compressor: Arc<Compressor>,
    ) -> Self {
        let dict = CheckedDictionaryArray::from_decimal_array(array);
        Self::from_dict_array_inner(
            dict,
            compressor,
            ArrowFixedLenByteArrayType::from(array.data_type()),
        )
    }

    /// Train a new fixed length byte array from a decimal array.
    pub fn train_from_decimal_array<T: DecimalType>(
        array: &PrimitiveArray<T>,
    ) -> (Arc<Compressor>, Self) {
        let value_width = array.data_type().primitive_width().unwrap();
        let value_buffer = array.values().inner().chunks(value_width);
        let compressor = FsstArray::train_compressor(value_buffer);
        let compressor = Arc::new(compressor);
        let liquid_array = Self::from_decimal_array(array, compressor.clone());
        (compressor, liquid_array)
    }

    fn from_dict_array_inner(
        array: CheckedDictionaryArray,
        compressor: Arc<Compressor>,
        arrow_type: ArrowFixedLenByteArrayType,
    ) -> Self {
        let bit_width_for_key = array.bit_width_for_key();
        let (keys, values) = array.into_inner().into_parts();
        let bit_packed_array = BitPackedArray::from_primitive(keys, bit_width_for_key);

        let fsst_values = match arrow_type {
            ArrowFixedLenByteArrayType::Decimal128(_, _) => {
                let values = values.as_primitive::<Decimal128Type>();
                FsstArray::from_decimal128_array_with_compressor(values, compressor)
            }
            ArrowFixedLenByteArrayType::Decimal256(_, _) => {
                let values = values.as_primitive::<Decimal256Type>();
                FsstArray::from_decimal256_array_with_compressor(values, compressor)
            }
        };
        Self {
            arrow_type,
            keys: bit_packed_array,
            values: fsst_values,
        }
    }

    /// Convert to arrow array by decompressing all values
    fn to_arrow_array_decompress_all(&self) -> ArrayRef {
        match self.arrow_type {
            ArrowFixedLenByteArrayType::Decimal128(precision, scale) => {
                let array = self.values.to_decimal128_array(&self.arrow_type);
                let keys = self.keys.to_primitive();
                let dict =
                    unsafe { DictionaryArray::<UInt16Type>::new_unchecked(keys, Arc::new(array)) };
                cast(&dict, &DataType::Decimal128(precision, scale)).unwrap()
            }
            ArrowFixedLenByteArrayType::Decimal256(precision, scale) => {
                let array = self.values.to_decimal256_array(&self.arrow_type);
                let keys = self.keys.to_primitive();
                let dict =
                    unsafe { DictionaryArray::<UInt16Type>::new_unchecked(keys, Arc::new(array)) };
                cast(&dict, &DataType::Decimal256(precision, scale)).unwrap()
            }
        }
    }

    /// Convert to arrow array by only decompressing values referenced by keys
    fn to_arrow_array_decompress_keyed(&self) -> ArrayRef {
        let primitive_key = self.keys.to_primitive();
        let mut hit_mask = BooleanBufferBuilder::new(self.values.len());
        hit_mask.advance(self.values.len());
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

        let decompressed_values = self.decompress_keyed_values(&hit_mask);
        let dict =
            unsafe { DictionaryArray::<UInt16Type>::new_unchecked(new_keys, decompressed_values) };

        match self.arrow_type {
            ArrowFixedLenByteArrayType::Decimal128(precision, scale) => {
                cast(&dict, &DataType::Decimal128(precision, scale)).unwrap()
            }
            ArrowFixedLenByteArrayType::Decimal256(precision, scale) => {
                cast(&dict, &DataType::Decimal256(precision, scale)).unwrap()
            }
        }
    }

    /// Decompress only the values that are selected by the hit mask
    fn decompress_keyed_values(&self, hit_mask: &arrow::buffer::BooleanBuffer) -> ArrayRef {
        let value_width = self.arrow_type.value_width();
        let selected_cnt = hit_mask.count_set_bits();
        let decompressor = self.values.compressor().decompressor();

        let mut value_buffer: Vec<u8> = Vec::with_capacity(selected_cnt * value_width + 8);
        let mut dst = value_buffer.as_mut_ptr();

        assert_eq!(hit_mask.len(), self.values.len());
        for i in 0..hit_mask.len() {
            if unsafe { hit_mask.value_unchecked(i) } {
                let v = unsafe { self.values.compressed().value_unchecked(i) };
                let slice = unsafe {
                    std::slice::from_raw_parts_mut(dst as *mut MaybeUninit<u8>, value_width + 8)
                };
                let len = decompressor.decompress_into(v, slice);
                debug_assert!(len == value_width);
                unsafe {
                    dst = dst.add(value_width);
                }
            }
        }

        unsafe {
            value_buffer.set_len(dst as usize - value_buffer.as_ptr() as usize);
        }
        value_buffer.shrink_to_fit();
        let value_buffer = Buffer::from(value_buffer);

        match self.arrow_type {
            ArrowFixedLenByteArrayType::Decimal128(precision, scale) => {
                let array_data =
                    arrow::array::ArrayDataBuilder::new(DataType::Decimal128(precision, scale))
                        .len(selected_cnt)
                        .add_buffer(value_buffer)
                        .build()
                        .unwrap();
                Arc::new(arrow::array::Decimal128Array::from(array_data))
            }
            ArrowFixedLenByteArrayType::Decimal256(precision, scale) => {
                let array_data =
                    arrow::array::ArrayDataBuilder::new(DataType::Decimal256(precision, scale))
                        .len(selected_cnt)
                        .add_buffer(value_buffer)
                        .build()
                        .unwrap();
                Arc::new(arrow::array::Decimal256Array::from(array_data))
            }
        }
    }

    pub(crate) fn from_parts(
        arrow_type: ArrowFixedLenByteArrayType,
        keys: BitPackedArray<UInt16Type>,
        values: FsstArray,
    ) -> Self {
        Self {
            arrow_type,
            keys,
            values,
        }
    }

    pub(super) fn values(&self) -> &FsstArray {
        &self.values
    }

    pub(super) fn keys(&self) -> &BitPackedArray<UInt16Type> {
        &self.keys
    }

    pub(super) fn arrow_type(&self) -> &ArrowFixedLenByteArrayType {
        &self.arrow_type
    }
}

#[cfg(test)]
mod tests {
    use crate::liquid_array::utils::gen_test_decimal_array;

    use super::*;
    use arrow_schema::DataType;

    fn test_decimal_roundtrip<T: DecimalType>(data_type: DataType) {
        let original_array = gen_test_decimal_array::<T>(data_type);
        let (_compressor, liquid_array) =
            LiquidFixedLenByteArray::train_from_decimal_array(&original_array);

        let arrow_array = liquid_array.to_arrow_array();
        let roundtrip_array = arrow_array.as_primitive::<T>();

        assert_eq!(original_array.len(), roundtrip_array.len());

        for i in 0..original_array.len() {
            assert_eq!(original_array.is_null(i), roundtrip_array.is_null(i));
            if !original_array.is_null(i) {
                assert_eq!(original_array.value(i), roundtrip_array.value(i));
            }
        }
    }

    #[test]
    fn test_decimal128_roundtrip() {
        test_decimal_roundtrip::<Decimal128Type>(DataType::Decimal128(15, 3));
    }

    #[test]
    fn test_decimal256_roundtrip() {
        test_decimal_roundtrip::<Decimal256Type>(DataType::Decimal256(38, 6));
    }

    fn test_decimal_filter_operation<T: DecimalType>(data_type: DataType) {
        let original_array = gen_test_decimal_array::<T>(data_type);
        let (_compressor, liquid_array) =
            LiquidFixedLenByteArray::train_from_decimal_array(&original_array);

        let mut filter_builder = arrow::array::BooleanBuilder::new();
        for i in 0..liquid_array.len() {
            filter_builder.append_value(i.is_multiple_of(2));
        }
        let filter = filter_builder.finish();
        let filtered_array = liquid_array.filter(&filter);
        let arrow_filtered = filtered_array.to_arrow_array();
        let arrow_typed = arrow_filtered.as_primitive::<T>();

        assert_eq!(filtered_array.len(), original_array.len() / 2);

        for (i, val) in arrow_typed.iter().enumerate() {
            if original_array.is_null(i * 2) {
                assert!(arrow_typed.is_null(i));
            } else {
                assert_eq!(val.unwrap(), original_array.value(i * 2));
            }
        }
    }

    #[test]
    fn test_decimal128_filter_operation() {
        test_decimal_filter_operation::<Decimal128Type>(DataType::Decimal128(12, 2));
    }

    #[test]
    fn test_decimal256_filter_operation() {
        test_decimal_filter_operation::<Decimal256Type>(DataType::Decimal256(38, 4));
    }

    #[test]
    fn test_keyed_decompression_optimization() {
        // Create a larger decimal array to test the optimization logic
        let mut builder = arrow::array::Decimal128Builder::new();

        // Create 10 distinct values
        for i in 0..10 {
            builder.append_value(i as i128 * 1000);
        }
        let distinct_values = builder.finish().with_precision_and_scale(15, 3).unwrap();

        let (_compressor, mut liquid_array) =
            LiquidFixedLenByteArray::train_from_decimal_array(&distinct_values);

        // Create a small keys array that only references a few values
        // This should trigger the keyed decompression path (keys.len() < 2048)
        let small_keys = UInt16Array::from(vec![0, 2, 4, 2, 0]); // Only references indices 0, 2, 4
        liquid_array.keys =
            BitPackedArray::from_primitive(small_keys, std::num::NonZero::new(3).unwrap());

        // Test both decompress_all and decompress_keyed should give the same result
        let result_all = liquid_array.to_arrow_array_decompress_all();
        let result_keyed = liquid_array.to_arrow_array_decompress_keyed();

        // Both should be equal
        assert_eq!(
            result_all.as_primitive::<Decimal128Type>().values(),
            result_keyed.as_primitive::<Decimal128Type>().values()
        );

        // Verify the actual values are correct
        let expected_values = vec![0, 2000, 4000, 2000, 0]; // i * 1000 for i in [0, 2, 4, 2, 0]
        let actual_values: Vec<i128> = result_keyed
            .as_primitive::<Decimal128Type>()
            .values()
            .iter()
            .copied()
            .collect();
        assert_eq!(expected_values, actual_values);
    }

    #[test]
    fn test_large_array_uses_full_decompression() {
        // Test that large arrays (>= 2048) use full decompression
        let distinct_values = gen_test_decimal_array::<Decimal128Type>(DataType::Decimal128(15, 3));
        let (_compressor, mut liquid_array) =
            LiquidFixedLenByteArray::train_from_decimal_array(&distinct_values);

        // Create a large keys array
        let large_keys: Vec<u16> = (0..3000)
            .map(|i| (i % distinct_values.len()) as u16)
            .collect();
        let large_keys = UInt16Array::from(large_keys);
        liquid_array.keys = BitPackedArray::from_primitive(
            large_keys,
            std::num::NonZero::new(4).unwrap(), // Adjust bit width as needed
        );

        // This should use the full decompression path since keys.len() >= 2048
        let result = liquid_array.to_arrow_array();
        assert_eq!(result.len(), 3000);

        // Verify the result is valid by checking it matches decompress_all
        let result_all = liquid_array.to_arrow_array_decompress_all();
        assert_eq!(
            result.as_primitive::<Decimal128Type>().values(),
            result_all.as_primitive::<Decimal128Type>().values()
        );
    }
}
