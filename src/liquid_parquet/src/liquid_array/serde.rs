use std::{any::TypeId, mem::size_of};

use bytes::Bytes;

use crate::liquid_array::BitPackedArray;
use crate::liquid_array::LiquidPrimitiveArray;

use super::LiquidPrimitiveType;

impl<T> LiquidPrimitiveArray<T>
where
    T: LiquidPrimitiveType,
{
    fn header_size() -> usize {
        let header_size = 8 + size_of::<T::Native>();
        (header_size + 7) & !7
    }

    /*
    Serialized LiquidPrimitiveArray Memory Layout:

    +--------------------------------------------------+
    | LiquidPrimitiveArray Header (16 bytes)           |
    +--------------------------------------------------+
    | MAGIC (4 bytes)                                  |  // Offset  0..3: "LQDA" magic number (0x4C51_4441)
    +--------------------------------------------------+
    | VERSION (2 bytes)                                |  // Offset  4..5: Version (currently 1)
    +--------------------------------------------------+
    | type_id (2 bytes)                                |  // Offset  6..7: Type identifier for T
    +--------------------------------------------------+
    | reference_value (size_of::<T::Native> bytes)     |  // Offset  8..(8 + size_of::<T::Native> - 1):
    |                                                  |  // the reference value (e.g. minimum value)
    +--------------------------------------------------+
    | Padding (to 16 bytes)                            |  // Padding to ensure 16 byte
    +--------------------------------------------------+

    +--------------------------------------------------+
    | BitPackedArray Data                              |
    +--------------------------------------------------+
    | [BitPackedArray Header & Bit-Packed Values]      |  // Written by self.bit_packed.to_bytes()
    +--------------------------------------------------+
    */
    pub fn to_bytes(&self) -> Vec<u8> {
        // Magic number "LQDA" for LiQuid Data Array
        const MAGIC: u32 = 0x4C51_4441;
        const VERSION: u16 = 1;

        // Determine type ID based on the type
        let type_id = get_type_id::<T>();

        // Create a buffer for the header (MAGIC + VERSION + type_id + reference value)
        let header_size = Self::header_size();
        let mut result = Vec::with_capacity(header_size + 256); // Pre-allocate a reasonable size

        // Write header
        result.extend_from_slice(&MAGIC.to_le_bytes());
        result.extend_from_slice(&VERSION.to_le_bytes());
        result.extend_from_slice(&type_id.to_le_bytes());

        // Write reference value
        let ref_value_bytes = unsafe {
            std::slice::from_raw_parts(
                &self.reference_value as *const T::Native as *const u8,
                size_of::<T::Native>(),
            )
        };
        result.extend_from_slice(ref_value_bytes);
        while result.len() < header_size {
            result.push(0);
        }

        // Let BitPackedArray write the rest of the data
        self.bit_packed.to_bytes(&mut result);

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
        if bytes.len() < 10 {
            // 4 (magic) + 2 (version) + 2 (type_id) + 2 (minimum reference value)
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

        // Get the reference value
        let ref_value_ptr = &bytes[8];
        let reference_value = unsafe { *(ref_value_ptr as *const u8 as *const T::Native) };

        // Skip ahead to the BitPackedArray data
        let bit_packed_data = bytes.slice(Self::header_size()..);
        let bit_packed = BitPackedArray::<T::UnSignedType>::from_bytes(bit_packed_data);

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
        let magic = u32::from_le_bytes(bytes[0..4].try_into().unwrap());
        let version = u16::from_le_bytes(bytes[4..6].try_into().unwrap());
        let type_id = u16::from_le_bytes(bytes[6..8].try_into().unwrap());

        // Check header values
        assert_eq!(magic, 0x4C51_4441, "Magic number should be LQDA");
        assert_eq!(version, 1, "Version should be 1");
        assert_eq!(type_id, 2, "Type ID for Int32Type should be 2");

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

    #[test]
    fn test_roundtrip_edge_cases() {
        // Test various edge cases

        // 1. All nulls array
        let all_nulls: Vec<Option<i32>> = vec![None; 1000];
        let array = PrimitiveArray::<Int32Type>::from(all_nulls);
        let liquid_array = LiquidPrimitiveArray::<Int32Type>::from_arrow_array(array.clone());
        let bytes = liquid_array.to_bytes();
        let bytes = Bytes::from(bytes);
        let deserialized = LiquidPrimitiveArray::<Int32Type>::from_bytes(bytes);
        let result = deserialized.to_arrow_array();
        assert_eq!(result.as_ref(), &array);

        // 2. No nulls array
        let no_nulls: Vec<Option<i32>> = (0..1000).map(|i| Some(i)).collect();
        let array = PrimitiveArray::<Int32Type>::from(no_nulls);
        let liquid_array = LiquidPrimitiveArray::<Int32Type>::from_arrow_array(array.clone());
        let bytes = liquid_array.to_bytes();
        let bytes = Bytes::from(bytes);
        let deserialized = LiquidPrimitiveArray::<Int32Type>::from_bytes(bytes);
        let result = deserialized.to_arrow_array();
        assert_eq!(result.as_ref(), &array);

        // 3. Single value array
        let single_value: Vec<Option<i32>> = vec![Some(42)];
        let array = PrimitiveArray::<Int32Type>::from(single_value);
        let liquid_array = LiquidPrimitiveArray::<Int32Type>::from_arrow_array(array.clone());
        let bytes = liquid_array.to_bytes();
        let bytes = Bytes::from(bytes);
        let deserialized = LiquidPrimitiveArray::<Int32Type>::from_bytes(bytes);
        let result = deserialized.to_arrow_array();
        assert_eq!(result.as_ref(), &array);

        // 4. Empty array
        let empty: Vec<Option<i32>> = vec![];
        let array = PrimitiveArray::<Int32Type>::from(empty);
        let liquid_array = LiquidPrimitiveArray::<Int32Type>::from_arrow_array(array.clone());
        let bytes = liquid_array.to_bytes();
        let bytes = Bytes::from(bytes);
        let deserialized = LiquidPrimitiveArray::<Int32Type>::from_bytes(bytes);
        let result = deserialized.to_arrow_array();
        assert_eq!(result.as_ref(), &array);

        // 5. Large array with very sparse nulls
        let sparse_nulls: Vec<Option<i32>> = (0..10_000)
            .map(|i| {
                if i == 1000 || i == 5000 || i == 9000 {
                    None
                } else {
                    Some(i)
                }
            })
            .collect();
        let array = PrimitiveArray::<Int32Type>::from(sparse_nulls);
        let liquid_array = LiquidPrimitiveArray::<Int32Type>::from_arrow_array(array.clone());
        let bytes = liquid_array.to_bytes();
        let bytes = Bytes::from(bytes);
        let deserialized = LiquidPrimitiveArray::<Int32Type>::from_bytes(bytes);
        let result = deserialized.to_arrow_array();
        assert_eq!(result.as_ref(), &array);
    }

    #[test]
    fn test_roundtrip_multiple_data_types() {
        use arrow::datatypes::{Int16Type, UInt32Type, UInt64Type};

        // Test with Int16Type
        let i16_values: Vec<Option<i16>> = (0..2000)
            .map(|i| {
                if i % 11 == 0 {
                    None
                } else {
                    Some((i % 300 - 150) as i16)
                }
            })
            .collect();
        let array = PrimitiveArray::<Int16Type>::from(i16_values);
        let liquid_array = LiquidPrimitiveArray::<Int16Type>::from_arrow_array(array.clone());
        let bytes = liquid_array.to_bytes();
        let bytes = Bytes::from(bytes);
        let deserialized = LiquidPrimitiveArray::<Int16Type>::from_bytes(bytes);
        let result = deserialized.to_arrow_array();
        assert_eq!(result.as_ref(), &array);

        // Test with UInt32Type
        let u32_values: Vec<Option<u32>> = (0..2000)
            .map(|i| {
                if i % 13 == 0 {
                    None
                } else {
                    Some(i as u32 * 10000)
                }
            })
            .collect();
        let array = PrimitiveArray::<UInt32Type>::from(u32_values);
        let liquid_array = LiquidPrimitiveArray::<UInt32Type>::from_arrow_array(array.clone());
        let bytes = liquid_array.to_bytes();
        let bytes = Bytes::from(bytes);
        let deserialized = LiquidPrimitiveArray::<UInt32Type>::from_bytes(bytes);
        let result = deserialized.to_arrow_array();
        assert_eq!(result.as_ref(), &array);

        // Test with UInt64Type
        let u64_values: Vec<Option<u64>> = (0..2000)
            .map(|i| {
                if i % 17 == 0 {
                    None
                } else {
                    Some(u64::MAX - (i as u64 * 1000000))
                }
            })
            .collect();
        let array = PrimitiveArray::<UInt64Type>::from(u64_values);
        let liquid_array = LiquidPrimitiveArray::<UInt64Type>::from_arrow_array(array.clone());
        let bytes = liquid_array.to_bytes();
        let bytes = Bytes::from(bytes);
        let deserialized = LiquidPrimitiveArray::<UInt64Type>::from_bytes(bytes);
        let result = deserialized.to_arrow_array();
        assert_eq!(result.as_ref(), &array);
    }
}
