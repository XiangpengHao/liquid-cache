use std::sync::Arc;

use arrow::array::types::{
    Float32Type as ArrowFloat32Type, Float64Type as ArrowFloat64Type, Int8Type as ArrowInt8Type,
    Int16Type as ArrowInt16Type, Int32Type as ArrowInt32Type, Int64Type as ArrowInt64Type,
    UInt8Type as ArrowUInt8Type, UInt16Type as ArrowUInt16Type, UInt32Type as ArrowUInt32Type,
    UInt64Type as ArrowUInt64Type,
};
use arrow::array::{ArrayRef, AsArray};
use arrow_schema::DataType;

use crate::liquid_array::{
    LiquidArrayRef, LiquidByteArray, LiquidFloatArray, LiquidPrimitiveArray,
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
            DataType::Int8 => Arc::new(LiquidPrimitiveArray::<ArrowInt8Type>::from_arrow_array(
                array.as_primitive::<ArrowInt8Type>().clone(),
            )),
            DataType::Int16 => Arc::new(LiquidPrimitiveArray::<ArrowInt16Type>::from_arrow_array(
                array.as_primitive::<ArrowInt16Type>().clone(),
            )),
            DataType::Int32 => Arc::new(LiquidPrimitiveArray::<ArrowInt32Type>::from_arrow_array(
                array.as_primitive::<ArrowInt32Type>().clone(),
            )),
            DataType::Int64 => Arc::new(LiquidPrimitiveArray::<ArrowInt64Type>::from_arrow_array(
                array.as_primitive::<ArrowInt64Type>().clone(),
            )),
            DataType::UInt8 => Arc::new(LiquidPrimitiveArray::<ArrowUInt8Type>::from_arrow_array(
                array.as_primitive::<ArrowUInt8Type>().clone(),
            )),
            DataType::UInt16 => {
                Arc::new(LiquidPrimitiveArray::<ArrowUInt16Type>::from_arrow_array(
                    array.as_primitive::<ArrowUInt16Type>().clone(),
                ))
            }
            DataType::UInt32 => {
                Arc::new(LiquidPrimitiveArray::<ArrowUInt32Type>::from_arrow_array(
                    array.as_primitive::<ArrowUInt32Type>().clone(),
                ))
            }
            DataType::UInt64 => {
                Arc::new(LiquidPrimitiveArray::<ArrowUInt64Type>::from_arrow_array(
                    array.as_primitive::<ArrowUInt64Type>().clone(),
                ))
            }
            DataType::Float32 => Arc::new(LiquidFloatArray::<ArrowFloat32Type>::from_arrow_array(
                array.as_primitive::<ArrowFloat32Type>().clone(),
            )),
            DataType::Float64 => Arc::new(LiquidFloatArray::<ArrowFloat64Type>::from_arrow_array(
                array.as_primitive::<ArrowFloat64Type>().clone(),
            )),
            _ => {
                // For unsupported primitive types, leave the value unchanged.
                log::warn!("unsupported primitive type {:?}", data_type);
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
            if let Some(dict_array) = array.as_dictionary_opt::<ArrowUInt16Type>() {
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
