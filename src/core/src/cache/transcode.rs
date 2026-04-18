use std::sync::Arc;

use arrow::array::types::*;
use arrow::array::{ArrayRef, AsArray};
use arrow_schema::{DataType, TimeUnit};

use crate::liquid_array::byte_view_array::ByteViewBuildOptions;
use crate::liquid_array::raw::FsstArray;
use crate::liquid_array::{
    LiquidArrayRef, LiquidByteViewArray, LiquidDecimalArray, LiquidFixedLenByteArray,
    LiquidFloatArray, LiquidPrimitiveArray,
};

use super::{CacheExpression, utils::LiquidCompressorStates};

fn with_fsst_compressor_or_train(
    state: &LiquidCompressorStates,
    use_compressor: impl FnOnce(Arc<fsst::Compressor>) -> LiquidArrayRef,
    train: impl FnOnce() -> (Arc<fsst::Compressor>, LiquidArrayRef),
) -> LiquidArrayRef {
    if let Some(compressor) = state.fsst_compressor() {
        return use_compressor(compressor);
    }

    let mut compressors = state.fsst_compressor_raw().write().unwrap();
    if let Some(compressor) = compressors.as_ref() {
        return use_compressor(compressor.clone());
    }

    let (compressor, liquid_array) = train();
    *compressors = Some(compressor);
    liquid_array
}

/// This method is used to transcode an arrow array into a liquid array.
///
/// Returns the transcoded liquid array if successful, otherwise returns the original arrow array.
pub fn transcode_liquid_inner<'a>(
    array: &'a ArrayRef,
    state: &LiquidCompressorStates,
) -> Result<LiquidArrayRef, &'a ArrayRef> {
    transcode_liquid_inner_with_hint(array, state, None)
}

/// Transcode with an optional hint to precompute metadata (e.g., substring fingerprints).
pub fn transcode_liquid_inner_with_hint<'a>(
    array: &'a ArrayRef,
    state: &LiquidCompressorStates,
    squeeze_hint: Option<&CacheExpression>,
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
            DataType::Timestamp(TimeUnit::Second, None) => Arc::new(LiquidPrimitiveArray::<
                TimestampSecondType,
            >::from_arrow_array(
                array.as_primitive::<TimestampSecondType>().clone(),
            )),
            DataType::Timestamp(TimeUnit::Millisecond, None) => Arc::new(LiquidPrimitiveArray::<
                TimestampMillisecondType,
            >::from_arrow_array(
                array.as_primitive::<TimestampMillisecondType>().clone(),
            )),
            DataType::Timestamp(TimeUnit::Microsecond, None) => Arc::new(LiquidPrimitiveArray::<
                TimestampMicrosecondType,
            >::from_arrow_array(
                array.as_primitive::<TimestampMicrosecondType>().clone(),
            )),
            DataType::Timestamp(TimeUnit::Nanosecond, None) => Arc::new(LiquidPrimitiveArray::<
                TimestampNanosecondType,
            >::from_arrow_array(
                array.as_primitive::<TimestampNanosecondType>().clone(),
            )),
            DataType::Timestamp(_, Some(_)) => {
                log::warn!("unsupported timestamp type with timezone {data_type:?}");
                return Err(array);
            }
            DataType::Float32 => Arc::new(LiquidFloatArray::<Float32Type>::from_arrow_array(
                array.as_primitive::<Float32Type>().clone(),
            )),
            DataType::Float64 => Arc::new(LiquidFloatArray::<Float64Type>::from_arrow_array(
                array.as_primitive::<Float64Type>().clone(),
            )),
            DataType::Decimal128(_, _) => {
                let decimals = array.as_primitive::<Decimal128Type>();
                if LiquidDecimalArray::fits_u64(decimals) {
                    return Ok(Arc::new(LiquidDecimalArray::from_decimal_array(decimals)));
                }
                let liquid_array = with_fsst_compressor_or_train(
                    state,
                    |compressor| {
                        Arc::new(LiquidFixedLenByteArray::from_decimal_array(
                            decimals, compressor,
                        ))
                    },
                    || {
                        let (compressor, liquid_array) =
                            LiquidFixedLenByteArray::train_from_decimal_array(decimals);
                        (compressor, Arc::new(liquid_array))
                    },
                );
                return Ok(liquid_array);
            }
            DataType::Decimal256(_, _) => {
                let decimals = array.as_primitive::<Decimal256Type>();
                if LiquidDecimalArray::fits_u64(decimals) {
                    return Ok(Arc::new(LiquidDecimalArray::from_decimal_array(decimals)));
                }
                let liquid_array = with_fsst_compressor_or_train(
                    state,
                    |compressor| {
                        Arc::new(LiquidFixedLenByteArray::from_decimal_array(
                            decimals, compressor,
                        ))
                    },
                    || {
                        let (compressor, liquid_array) =
                            LiquidFixedLenByteArray::train_from_decimal_array(decimals);
                        (compressor, Arc::new(liquid_array))
                    },
                );
                return Ok(liquid_array);
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
    let build_fingerprints = matches!(squeeze_hint, Some(CacheExpression::SubstringSearch));
    match array.data_type() {
        DataType::Utf8View => {
            let options =
                ByteViewBuildOptions::for_data_type(array.data_type(), build_fingerprints);
            let liquid_array = with_fsst_compressor_or_train(
                state,
                |compressor| {
                    Arc::new(LiquidByteViewArray::<FsstArray>::from_view_array_inner(
                        array.as_string_view(),
                        compressor,
                        options,
                    ))
                },
                || {
                    let (compressor, compressed) =
                        LiquidByteViewArray::<FsstArray>::train_from_string_view_inner(
                            array.as_string_view(),
                            options,
                        );
                    (compressor, Arc::new(compressed))
                },
            );
            Ok(liquid_array)
        }
        DataType::BinaryView => {
            let options =
                ByteViewBuildOptions::for_data_type(array.data_type(), build_fingerprints);
            let liquid_array = with_fsst_compressor_or_train(
                state,
                |compressor| {
                    Arc::new(LiquidByteViewArray::<FsstArray>::from_view_array_inner(
                        array.as_binary_view(),
                        compressor,
                        options,
                    ))
                },
                || {
                    let (compressor, compressed) =
                        LiquidByteViewArray::<FsstArray>::train_from_binary_view_inner(
                            array.as_binary_view(),
                            options,
                        );
                    (compressor, Arc::new(compressed))
                },
            );
            Ok(liquid_array)
        }
        DataType::Utf8 => {
            let options =
                ByteViewBuildOptions::for_data_type(array.data_type(), build_fingerprints);
            let liquid_array = with_fsst_compressor_or_train(
                state,
                |compressor| {
                    Arc::new(LiquidByteViewArray::<FsstArray>::from_byte_array_inner(
                        array.as_string::<i32>(),
                        compressor,
                        options,
                    ))
                },
                || {
                    let (compressor, compressed) =
                        LiquidByteViewArray::<FsstArray>::train_from_arrow_inner(
                            array.as_string::<i32>(),
                            options,
                        );
                    (compressor, Arc::new(compressed))
                },
            );
            Ok(liquid_array)
        }
        DataType::Binary => {
            let options =
                ByteViewBuildOptions::for_data_type(array.data_type(), build_fingerprints);
            let liquid_array = with_fsst_compressor_or_train(
                state,
                |compressor| {
                    Arc::new(LiquidByteViewArray::<FsstArray>::from_byte_array_inner(
                        array.as_binary::<i32>(),
                        compressor,
                        options,
                    ))
                },
                || {
                    let (compressor, compressed) =
                        LiquidByteViewArray::<FsstArray>::train_from_arrow_inner(
                            array.as_binary::<i32>(),
                            options,
                        );
                    (compressor, Arc::new(compressed))
                },
            );
            Ok(liquid_array)
        }
        DataType::Dictionary(_, _) => {
            if let Some(dict_array) = array.as_dictionary_opt::<UInt16Type>() {
                let options =
                    ByteViewBuildOptions::for_data_type(array.data_type(), build_fingerprints);
                let liquid_array = with_fsst_compressor_or_train(
                    state,
                    |compressor| unsafe {
                        Arc::new(
                            LiquidByteViewArray::<FsstArray>::from_unique_dict_array_with_options(
                                dict_array, compressor, options,
                            ),
                        )
                    },
                    || {
                        let (compressor, liquid_array) =
                            LiquidByteViewArray::<FsstArray>::train_from_arrow_dict_inner(
                                dict_array, options,
                            );
                        (compressor, Arc::new(liquid_array))
                    },
                );
                return Ok(liquid_array);
            }
            log::warn!("unsupported data type {:?}", array.data_type());
            Err(array)
        }
        _ => {
            log::debug!("unsupported data type {:?}", array.data_type());
            Err(array)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::{
        ArrayRef, BinaryArray, BinaryViewArray, BooleanArray, DictionaryArray, Float32Array,
        Float64Array, Int32Array, Int64Array, StringArray, TimestampMicrosecondArray, UInt16Array,
    };
    use arrow::datatypes::UInt16Type;

    const TEST_ARRAY_SIZE: usize = 8192;

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
        let state = LiquidCompressorStates::new();
        let transcoded = transcode_liquid_inner(&array, &state).unwrap();
        assert_transcode(&array, &transcoded);
    }

    #[test]
    fn test_transcode_int64() {
        let array: ArrayRef = Arc::new(Int64Array::from_iter_values(0..TEST_ARRAY_SIZE as i64));
        let state = LiquidCompressorStates::new();
        let transcoded = transcode_liquid_inner(&array, &state).unwrap();
        assert_transcode(&array, &transcoded);
    }

    #[test]
    fn test_transcode_float32() {
        let array: ArrayRef = Arc::new(Float32Array::from_iter_values(
            (0..TEST_ARRAY_SIZE).map(|i| i as f32),
        ));
        let state = LiquidCompressorStates::new();
        let transcoded = transcode_liquid_inner(&array, &state).unwrap();
        assert_transcode(&array, &transcoded);
    }

    #[test]
    fn test_transcode_float64() {
        let array: ArrayRef = Arc::new(Float64Array::from_iter_values(
            (0..TEST_ARRAY_SIZE).map(|i| i as f64),
        ));
        let state = LiquidCompressorStates::new();

        let transcoded = transcode_liquid_inner(&array, &state).unwrap();
        assert_transcode(&array, &transcoded);
    }

    #[test]
    fn test_transcode_timestamp_microsecond() {
        let array: ArrayRef = Arc::new(TimestampMicrosecondArray::from_iter_values(
            (0..TEST_ARRAY_SIZE).map(|i| (i as i64) * 1_000),
        ));
        let state = LiquidCompressorStates::new();

        let transcoded = transcode_liquid_inner(&array, &state).unwrap();
        assert_transcode(&array, &transcoded);
    }

    #[test]
    fn test_transcode_string() {
        let array: ArrayRef = Arc::new(StringArray::from_iter_values(
            (0..TEST_ARRAY_SIZE).map(|i| format!("test_string_{i}")),
        ));
        let state = LiquidCompressorStates::new();

        let transcoded = transcode_liquid_inner(&array, &state).unwrap();
        assert_transcode(&array, &transcoded);
    }

    #[test]
    fn test_transcode_binary_view() {
        let array: ArrayRef = Arc::new(BinaryViewArray::from_iter_values(
            (0..TEST_ARRAY_SIZE).map(|i| format!("test_binary_{i}").into_bytes()),
        ));
        let state = LiquidCompressorStates::new();

        let transcoded = transcode_liquid_inner(&array, &state).unwrap();
        assert_transcode(&array, &transcoded);
    }

    #[test]
    fn test_transcode_dictionary_uft8() {
        // Create a dictionary with many repeated values
        let values =
            StringArray::from_iter_values((0..100).map(|i| format!("value__longer_values_{i}")));
        let keys: Vec<u16> = (0..TEST_ARRAY_SIZE).map(|i| (i % 100) as u16).collect();

        let dict_array =
            DictionaryArray::<UInt16Type>::try_new(UInt16Array::from(keys), Arc::new(values))
                .unwrap();

        let array: ArrayRef = Arc::new(dict_array);
        let state = LiquidCompressorStates::new();

        let transcoded = transcode_liquid_inner(&array, &state).unwrap();
        assert_transcode(&array, &transcoded);
    }

    #[test]
    fn test_transcode_dictionary_binary() {
        // Create a dictionary with binary values and many repeated values
        let values = BinaryArray::from_iter_values(
            (0..100).map(|i| format!("binary_value_{i}").into_bytes()),
        );
        let keys: Vec<u16> = (0..TEST_ARRAY_SIZE).map(|i| (i % 100) as u16).collect();

        let dict_array =
            DictionaryArray::<UInt16Type>::try_new(UInt16Array::from(keys), Arc::new(values))
                .unwrap();

        let array: ArrayRef = Arc::new(dict_array);
        let state = LiquidCompressorStates::new();

        let transcoded = transcode_liquid_inner(&array, &state).unwrap();
        assert_transcode(&array, &transcoded);
    }

    #[test]
    fn test_transcode_unsupported_type() {
        // Create a boolean array which is not supported by the transcoder
        let values: Vec<bool> = (0..TEST_ARRAY_SIZE).map(|i| i.is_multiple_of(2)).collect();
        let array: ArrayRef = Arc::new(BooleanArray::from(values));
        let state = LiquidCompressorStates::new();

        // Try to transcode and expect an error
        let result = transcode_liquid_inner(&array, &state);

        // Verify it returns Err with the original array
        assert!(result.is_err());
        if let Err(original) = result {
            assert_eq!(&array, original);
        }
    }
}
