use std::{any::Any, sync::Arc};

use arrow::{
    array::{Array, ArrayRef, AsArray, BooleanArray, DictionaryArray, PrimitiveArray},
    compute::kernels::cast,
    datatypes::{Decimal128Type, Decimal256Type, DecimalType, UInt16Type},
};
use arrow_schema::DataType;
use fsst::Compressor;

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
        let bit_packed_array = BitPackedArray::from_primitive(filtered_keys, keys.bit_width());
        Arc::new(LiquidFixedLenByteArray {
            arrow_type: self.arrow_type.clone(),
            keys: bit_packed_array,
            values,
        })
    }

    fn to_bytes(&self) -> Vec<u8> {
        unimplemented!()
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

    /// Train a new fixed length byte array from a [Decimal128Array].
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
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::Int64Builder;
    use arrow_schema::DataType;

    fn gen_test_array<T: DecimalType>(data_type: DataType) -> PrimitiveArray<T> {
        let mut builder = Int64Builder::new();
        for i in 0..4096i64 {
            if i % 97 == 0 {
                builder.append_null();
            } else {
                let value = if i % 5 == 0 {
                    i * 1000 + 123
                } else if i % 3 == 0 {
                    42
                } else if i % 7 == 0 {
                    i * 1_000_000 + 456789
                } else {
                    i * 100 + 42
                };
                builder.append_value(value as i64);
            }
        }
        let array = builder.finish();
        cast(&array, &data_type)
            .unwrap()
            .as_primitive::<T>()
            .clone()
    }

    fn test_decimal_roundtrip<T: DecimalType>(data_type: DataType) {
        let original_array = gen_test_array::<T>(data_type);
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
        let original_array = gen_test_array::<T>(data_type);
        let (_compressor, liquid_array) =
            LiquidFixedLenByteArray::train_from_decimal_array(&original_array);

        let mut filter_builder = arrow::array::BooleanBuilder::new();
        for i in 0..liquid_array.len() {
            filter_builder.append_value(i % 2 == 0);
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
}
