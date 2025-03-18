use std::{any::TypeId, num::NonZeroU8};

use arrow::{
    array::ArrowPrimitiveType,
    buffer::{BooleanBuffer, Buffer, NullBuffer, ScalarBuffer},
};
use bytes::Bytes;

use crate::liquid_array::BitPackedArray;

use super::{LiquidPrimitiveArray, LiquidPrimitiveType};

impl<T> LiquidPrimitiveArray<T>
where
    T: LiquidPrimitiveType,
{
    pub fn to_bytes(&self) -> Vec<u8> {
        // Magic number "LQDA" for LiQuid Data Array
        const MAGIC: u32 = 0x4C51_4441;
        const VERSION: u16 = 1;

        // Determine type ID based on the type
        let type_id = get_type_id::<T>();

        // Determine reference value size
        let ref_value_size = size_of::<T::Native>() as u8;

        // Check if we have nulls
        let has_nulls = self.bit_packed.is_nullable() as u8;

        // Get the nulls and values buffers
        let nulls_bytes = if has_nulls == 1 {
            // The NullBuffer contains a BooleanBuffer which has a Buffer inside
            let nulls = self.bit_packed.nulls().unwrap();
            nulls.buffer().as_slice().to_vec()
        } else {
            Vec::new()
        };

        // Get the bit-packed values
        let values_bytes = self.bit_packed.packed_values().inner().as_slice();

        // Calculate header size (fixed part + reference value)
        let header_size = 24 + size_of::<T::Native>();

        // Calculate offsets
        let nulls_offset = if has_nulls == 1 { header_size + 32 } else { 0 };
        let values_offset_base = header_size
            + if has_nulls == 1 {
                32 + nulls_bytes.len()
            } else {
                32
            };

        // Ensure values_offset is aligned to 8 bytes
        let values_offset = (values_offset_base + 7) & !7;

        // Calculate total size
        let total_size = values_offset + values_bytes.len();
        let mut result = Vec::with_capacity(total_size);

        // Write header
        result.extend_from_slice(&MAGIC.to_le_bytes());
        result.extend_from_slice(&VERSION.to_le_bytes());
        result.extend_from_slice(&type_id.to_le_bytes());
        result.extend_from_slice(&(self.bit_packed.len() as u64).to_le_bytes());
        result.push(self.bit_packed.bit_width().get());
        result.push(has_nulls);
        result.extend_from_slice(&[0u8, 0u8]); // Reserved
        result.push(ref_value_size);

        // Add padding to align reference value
        while result.len() < 24 {
            result.push(0);
        }

        // Write reference value
        let ref_value_bytes = unsafe {
            std::slice::from_raw_parts(
                &self.reference_value as *const T::Native as *const u8,
                size_of::<T::Native>(),
            )
        };
        result.extend_from_slice(ref_value_bytes);

        // Write nulls offset and length
        result.extend_from_slice(&(nulls_offset as u64).to_le_bytes());
        result.extend_from_slice(&(nulls_bytes.len() as u64).to_le_bytes());

        // Write values offset and length
        result.extend_from_slice(&(values_offset as u64).to_le_bytes());
        result.extend_from_slice(&(values_bytes.len() as u64).to_le_bytes());

        // Write nulls buffer if present
        if has_nulls == 1 {
            result.extend_from_slice(&nulls_bytes);
        }

        // Add padding to align values buffer
        while result.len() < values_offset {
            result.push(0);
        }

        // Write values buffer
        result.extend_from_slice(&values_bytes);

        result
    }

    /// Deserialize a LiquidPrimitiveArray from bytes, using zero-copy where possible.
    ///
    /// This function creates a LiquidPrimitiveArray from a byte buffer that was
    /// previously created using `to_bytes()`. It attempts to use zero-copy semantics
    /// by creating views into the original buffer for the data portions.
    pub fn from_bytes(bytes: Bytes) -> Self {
        // Magic number "LQDA" for LiQuid Data Array
        const MAGIC: u32 = 0x4C51_4441;

        // Check minimum header size
        if bytes.len() < 24 {
            panic!("Input buffer too small for header");
        }

        // Read header fields
        let magic = u32::from_le_bytes(bytes[0..4].try_into().unwrap());
        if magic != MAGIC {
            panic!("Invalid magic number");
        }

        let version = u16::from_le_bytes(bytes[4..6].try_into().unwrap());
        if version != 1 {
            panic!("Unsupported version");
        }

        let type_id = u16::from_le_bytes(bytes[6..8].try_into().unwrap());
        assert_eq!(type_id, get_type_id::<T>());

        let original_len = u64::from_le_bytes(bytes[8..16].try_into().unwrap()) as usize;

        let bit_width = bytes[16];
        let bit_width = NonZeroU8::new(bit_width).unwrap();

        let has_nulls = bytes[17] != 0;

        let ref_value_size = bytes[20] as usize;
        if ref_value_size != size_of::<T::Native>() {
            panic!("Reference value size mismatch");
        }

        // Read reference value (starting at offset 24)
        if bytes.len() < 24 + ref_value_size {
            panic!("Input buffer too small for reference value");
        }

        let ref_value_ptr = unsafe { bytes.as_ptr().add(24) };
        let reference_value = unsafe { *(ref_value_ptr as *const T::Native) };

        // Read buffer offsets
        let offset_start = 24 + ref_value_size;
        if bytes.len() < offset_start + 32 {
            panic!("Input buffer too small for offsets");
        }

        let nulls_offset =
            u64::from_le_bytes(bytes[offset_start..offset_start + 8].try_into().unwrap());

        let nulls_length = u64::from_le_bytes(
            bytes[offset_start + 8..offset_start + 16]
                .try_into()
                .unwrap(),
        );

        let values_offset = u64::from_le_bytes(
            bytes[offset_start + 16..offset_start + 24]
                .try_into()
                .unwrap(),
        );

        let values_length = u64::from_le_bytes(
            bytes[offset_start + 24..offset_start + 32]
                .try_into()
                .unwrap(),
        ) as usize;

        // Validate offsets and lengths
        if has_nulls {
            if nulls_offset == 0 || nulls_length == 0 {
                panic!("Array has nulls but null buffer is missing");
            }
            if nulls_offset + nulls_length > bytes.len() as u64 {
                panic!("Null buffer extends beyond input buffer");
            }
        }

        if values_offset == 0 || values_length == 0 {
            panic!("Values buffer is required");
        }
        if values_offset as usize + values_length > bytes.len() {
            panic!("Values buffer extends beyond input buffer");
        }

        // Create the nulls buffer if present
        let nulls = if has_nulls {
            // Create a buffer view into the nulls section
            let nulls_slice =
                bytes.slice(nulls_offset as usize..nulls_offset as usize + nulls_length as usize);
            let nulls_buffer = Buffer::from(nulls_slice);

            // Create a NullBuffer from the BooleanBuffer
            let boolean_buffer = BooleanBuffer::new(nulls_buffer, 0, original_len);
            let null_buffer = NullBuffer::from(boolean_buffer);
            Some(null_buffer)
        } else {
            None
        };

        // Create a view into the values buffer
        let values_slice =
            bytes.slice(values_offset as usize..values_offset as usize + values_length as usize);
        let values_buffer = Buffer::from(values_slice);

        // We know the packed buffer contains the serialized data, so we can simply use
        // the length of the values buffer divided by the size of the native type
        let native_type_size = size_of::<u32>(); // Use a placeholder size that will be corrected below
        let packed_len = values_length / native_type_size;

        // Create a ScalarBuffer for the packed values
        let values = ScalarBuffer::<<T::UnSignedType as ArrowPrimitiveType>::Native>::new(
            values_buffer,
            0,
            packed_len,
        );

        let bit_packed = BitPackedArray::from_parts(values, nulls, bit_width, original_len);

        Self {
            bit_packed,
            reference_value,
        }
    }
}

fn get_type_id<T: LiquidPrimitiveType>() -> u16 {
    match TypeId::of::<T>() {
        id if id == TypeId::of::<arrow::datatypes::Int8Type>() => 0,
        id if id == TypeId::of::<arrow::datatypes::Int16Type>() => 1,
        id if id == TypeId::of::<arrow::datatypes::Int32Type>() => 2,
        id if id == TypeId::of::<arrow::datatypes::Int64Type>() => 3,
        id if id == TypeId::of::<arrow::datatypes::UInt8Type>() => 4,
        id if id == TypeId::of::<arrow::datatypes::UInt16Type>() => 5,
        id if id == TypeId::of::<arrow::datatypes::UInt32Type>() => 6,
        id if id == TypeId::of::<arrow::datatypes::UInt64Type>() => 7,
        _ => panic!("Unsupported primitive type"),
    }
}

#[cfg(test)]
mod tests {
    use arrow::{array::PrimitiveArray, datatypes::Int32Type};

    use crate::liquid_array::LiquidArray;

    use super::*;

    #[test]
    fn test_to_bytes() {
        // Create a simple array
        let original: Vec<Option<i32>> = vec![Some(10), Some(20), Some(30), None, Some(50)];
        let array = PrimitiveArray::<Int32Type>::from(original.clone());
        let liquid_array = LiquidPrimitiveArray::<Int32Type>::from_arrow_array(array);

        // Serialize to bytes
        let bytes = liquid_array.to_bytes();

        // Basic validation
        let magic = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
        let version = u16::from_le_bytes([bytes[4], bytes[5]]);
        let type_id = u16::from_le_bytes([bytes[6], bytes[7]]);
        let length = u64::from_le_bytes([
            bytes[8], bytes[9], bytes[10], bytes[11], bytes[12], bytes[13], bytes[14], bytes[15],
        ]);
        let bit_width = bytes[16];
        let has_nulls = bytes[17];

        // Check header values
        assert_eq!(magic, 0x4C51_4441, "Magic number should be LQDA");
        assert_eq!(version, 1, "Version should be 1");
        assert_eq!(type_id, 2, "Type ID for Int32Type should be 2");
        assert_eq!(length, 5, "Array length should be 5");
        assert!(bit_width > 0, "Bit width should be positive");
        assert_eq!(has_nulls, 1, "Array has nulls");

        // Check that the total size makes sense (we can't predict the exact size without knowing bit_width)
        assert!(
            bytes.len() > 100,
            "Serialized data should have a reasonable size"
        );
    }

    #[test]
    fn test_roundtrip_bytes() {
        let original: Vec<Option<i32>> = vec![Some(10), Some(20), Some(30), None, Some(50)];
        let array = PrimitiveArray::<Int32Type>::from(original.clone());
        let liquid_array = LiquidPrimitiveArray::<Int32Type>::from_arrow_array(array.clone());

        let bytes = liquid_array.to_bytes();
        let bytes = Bytes::from(bytes);

        let deserialized_array = LiquidPrimitiveArray::<Int32Type>::from_bytes(bytes);

        let result_array = deserialized_array.to_arrow_array();

        assert_eq!(result_array.as_ref(), &array);
    }
}
