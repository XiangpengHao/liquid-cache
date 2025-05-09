use std::sync::Arc;

use arrow::array::types::*;
use arrow::array::{ArrayRef, AsArray};
use arrow_schema::DataType;

use crate::liquid_array::{
    LiquidArrayRef, LiquidByteArray, LiquidFixedLenByteArray, LiquidFloatArray,
    LiquidPrimitiveArray,
};

use super::LiquidCompressorStates;

/// This method is used to transcode an arrow array into a liquid array.
///
/// Returns the transcoded liquid array if successful, otherwise returns the original arrow array.
pub(super) fn transcode_liquid_inner<'a>(
    array: &'a ArrayRef,
    state: &LiquidCompressorStates,
) -> Result<LiquidArrayRef, &'a ArrayRef> {
    let data_type = array.data_type();
    if data_type.is_primitive() {
        // For primitive types, perform the transcoding.
        let liquid_array: LiquidArrayRef = match data_type {
            DataType::Int8 => Arc::new(LiquidPrimitiveArray::<Int8Type>::from_arrow_array(
                array.as_primitive::<Int8Type>().clone(),
            )),
            DataType::Int16 => Arc::new(LiquidPrimitiveArray::<Int16Type>::from_arrow_array(
                array.as_primitive::<Int16Type>().clone(),
            )),
            DataType::Int32 => Arc::new(LiquidPrimitiveArray::<Int32Type>::from_arrow_array(
                array.as_primitive::<Int32Type>().clone(),
            )),
            DataType::Int64 => Arc::new(LiquidPrimitiveArray::<Int64Type>::from_arrow_array(
                array.as_primitive::<Int64Type>().clone(),
            )),
            DataType::UInt8 => Arc::new(LiquidPrimitiveArray::<UInt8Type>::from_arrow_array(
                array.as_primitive::<UInt8Type>().clone(),
            )),
            DataType::UInt16 => Arc::new(LiquidPrimitiveArray::<UInt16Type>::from_arrow_array(
                array.as_primitive::<UInt16Type>().clone(),
            )),
            DataType::UInt32 => Arc::new(LiquidPrimitiveArray::<UInt32Type>::from_arrow_array(
                array.as_primitive::<UInt32Type>().clone(),
            )),
            DataType::UInt64 => Arc::new(LiquidPrimitiveArray::<UInt64Type>::from_arrow_array(
                array.as_primitive::<UInt64Type>().clone(),
            )),
            DataType::Date32 => Arc::new(LiquidPrimitiveArray::<Date32Type>::from_arrow_array(
                array.as_primitive::<Date32Type>().clone(),
            )),
            DataType::Date64 => Arc::new(LiquidPrimitiveArray::<Date64Type>::from_arrow_array(
                array.as_primitive::<Date64Type>().clone(),
            )),
            DataType::Float32 => Arc::new(LiquidFloatArray::<Float32Type>::from_arrow_array(
                array.as_primitive::<Float32Type>().clone(),
            )),
            DataType::Float64 => Arc::new(LiquidFloatArray::<Float64Type>::from_arrow_array(
                array.as_primitive::<Float64Type>().clone(),
            )),
            DataType::Decimal128(_, _) => {
                let compressor = state.fsst_compressor.read().unwrap();
                if let Some(compressor) = compressor.as_ref() {
                    let compressed = LiquidFixedLenByteArray::from_decimal_array(
                        array.as_primitive::<Decimal128Type>(),
                        compressor.clone(),
                    );
                    return Ok(Arc::new(compressed));
                }
                drop(compressor);
                let mut compressors = state.fsst_compressor.write().unwrap();
                let (compressor, liquid_array) = LiquidFixedLenByteArray::train_from_decimal_array(
                    array.as_primitive::<Decimal128Type>(),
                );
                *compressors = Some(compressor);
                return Ok(Arc::new(liquid_array));
            }
            DataType::Decimal256(_, _) => {
                let compressor = state.fsst_compressor.read().unwrap();
                if let Some(compressor) = compressor.as_ref() {
                    let compressed = LiquidFixedLenByteArray::from_decimal_array(
                        array.as_primitive::<Decimal256Type>(),
                        compressor.clone(),
                    );
                    return Ok(Arc::new(compressed));
                }
                drop(compressor);
                let mut compressors = state.fsst_compressor.write().unwrap();
                let (compressor, liquid_array) = LiquidFixedLenByteArray::train_from_decimal_array(
                    array.as_primitive::<Decimal256Type>(),
                );
                *compressors = Some(compressor);
                return Ok(Arc::new(liquid_array));
            }
            _ => {
                // For unsupported primitive types, leave the value unchanged.
                log::warn!("unsupported primitive type {data_type:?}");
                return Err(array);
            }
        };
        return Ok(liquid_array);
    }

    // Handle string/dictionary types.
    match array.data_type() {
        DataType::Utf8View => {
            let compressor = state.fsst_compressor.read().unwrap();
            if let Some(compressor) = compressor.as_ref() {
                let compressed = LiquidByteArray::from_string_view_array(
                    array.as_string_view(),
                    compressor.clone(),
                );
                return Ok(Arc::new(compressed));
            }
            drop(compressor);
            let mut compressors = state.fsst_compressor.write().unwrap();
            let (compressor, compressed) =
                LiquidByteArray::train_from_arrow_view(array.as_string_view());
            *compressors = Some(compressor);
            Ok(Arc::new(compressed))
        }
        DataType::Utf8 => {
            let compressor = state.fsst_compressor.read().unwrap();
            if let Some(compressor) = compressor.as_ref() {
                let compressed = LiquidByteArray::from_string_array(
                    array.as_string::<i32>(),
                    compressor.clone(),
                );
                return Ok(Arc::new(compressed));
            }
            drop(compressor);
            let mut compressors = state.fsst_compressor.write().unwrap();
            let (compressor, compressed) =
                LiquidByteArray::train_from_arrow(array.as_string::<i32>());
            *compressors = Some(compressor);
            Ok(Arc::new(compressed))
        }
        DataType::Dictionary(_, _) => {
            if let Some(dict_array) = array.as_dictionary_opt::<UInt16Type>() {
                let compressor = state.fsst_compressor.read().unwrap();
                if let Some(compressor) = compressor.as_ref() {
                    let liquid_array = unsafe {
                        LiquidByteArray::from_unique_dict_array(dict_array, compressor.clone())
                    };
                    return Ok(Arc::new(liquid_array));
                }
                drop(compressor);
                let mut compressors = state.fsst_compressor.write().unwrap();
                let (compressor, liquid_array) = LiquidByteArray::train_from_arrow_dict(dict_array);
                *compressors = Some(compressor);
                return Ok(Arc::new(liquid_array));
            }
            log::warn!("unsupported data type {:?}", array.data_type());
            Err(array)
        }
        _ => {
            log::warn!("unsupported data type {:?}", array.data_type());
            Err(array)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sync::RwLock;
    use arrow::array::{
        ArrayRef, BooleanArray, DictionaryArray, Float32Array, Float64Array, Int32Array,
        Int64Array, StringArray, UInt16Array,
    };
    use arrow::datatypes::UInt16Type;

    const TEST_ARRAY_SIZE: usize = 8192;

    fn create_compressor_states() -> LiquidCompressorStates {
        LiquidCompressorStates {
            fsst_compressor: RwLock::new(None),
        }
    }

    fn assert_transcode(original: &ArrayRef, transcoded: &LiquidArrayRef) {
        assert!(
            transcoded.get_array_memory_size() < original.get_array_memory_size(),
            "transcoded size: {}, original size: {}",
            transcoded.get_array_memory_size(),
            original.get_array_memory_size()
        );
        let back_to_arrow = transcoded.to_arrow_array();
        assert_eq!(original, &back_to_arrow);
    }

    #[test]
    fn test_transcode_int32() {
        let array: ArrayRef = Arc::new(Int32Array::from_iter_values(0..TEST_ARRAY_SIZE as i32));
        let state = create_compressor_states();
        let transcoded = transcode_liquid_inner(&array, &state).unwrap();
        assert_transcode(&array, &transcoded);
    }

    #[test]
    fn test_transcode_int64() {
        let array: ArrayRef = Arc::new(Int64Array::from_iter_values(0..TEST_ARRAY_SIZE as i64));
        let state = create_compressor_states();
        let transcoded = transcode_liquid_inner(&array, &state).unwrap();
        assert_transcode(&array, &transcoded);
    }

    #[test]
    fn test_transcode_float32() {
        let array: ArrayRef = Arc::new(Float32Array::from_iter_values(
            (0..TEST_ARRAY_SIZE).map(|i| i as f32),
        ));
        let state = create_compressor_states();
        let transcoded = transcode_liquid_inner(&array, &state).unwrap();
        assert_transcode(&array, &transcoded);
    }

    #[test]
    fn test_transcode_float64() {
        let array: ArrayRef = Arc::new(Float64Array::from_iter_values(
            (0..TEST_ARRAY_SIZE).map(|i| i as f64),
        ));
        let state = create_compressor_states();

        let transcoded = transcode_liquid_inner(&array, &state).unwrap();
        assert_transcode(&array, &transcoded);
    }

    #[test]
    fn test_transcode_string() {
        let array: ArrayRef = Arc::new(StringArray::from_iter_values(
            (0..TEST_ARRAY_SIZE).map(|i| format!("test_string_{}", i)),
        ));
        let state = create_compressor_states();

        let transcoded = transcode_liquid_inner(&array, &state).unwrap();
        assert_transcode(&array, &transcoded);
    }

    #[test]
    fn test_transcode_dictionary() {
        // Create a dictionary with many repeated values
        let values = StringArray::from_iter_values((0..100).map(|i| format!("value_{}", i)));
        let keys: Vec<u16> = (0..TEST_ARRAY_SIZE).map(|i| (i % 100) as u16).collect();

        let dict_array =
            DictionaryArray::<UInt16Type>::try_new(UInt16Array::from(keys), Arc::new(values))
                .unwrap();

        let array: ArrayRef = Arc::new(dict_array);
        let state = create_compressor_states();

        let transcoded = transcode_liquid_inner(&array, &state).unwrap();
        assert_transcode(&array, &transcoded);
    }

    #[test]
    fn test_transcode_unsupported_type() {
        // Create a boolean array which is not supported by the transcoder
        let values: Vec<bool> = (0..TEST_ARRAY_SIZE).map(|i| i % 2 == 0).collect();
        let array: ArrayRef = Arc::new(BooleanArray::from(values));
        let state = create_compressor_states();

        // Try to transcode and expect an error
        let result = transcode_liquid_inner(&array, &state);

        // Verify it returns Err with the original array
        assert!(result.is_err());
        if let Err(original) = result {
            assert_eq!(&array, original);
        }
    }
}
