use std::sync::Arc;
use std::{any::TypeId, mem::size_of};

use arrow::array::types::UInt16Type;
use arrow::datatypes::{
    Int8Type, Int16Type, Int32Type, Int64Type, UInt8Type, UInt32Type, UInt64Type,
};
use bytes::Bytes;
use fsst::Compressor;

use crate::liquid_array::LiquidPrimitiveArray;
use crate::liquid_array::LiquidPrimitiveType;
use crate::liquid_array::byte_array::ArrowStringType;
use crate::liquid_array::raw::{BitPackedArray, FsstArray};

use super::{LiquidArrayRef, LiquidByteArray, LiquidDataType};

const MAGIC: u32 = 0x4C51_4441; // "LQDA" for LiQuid Data Array
const VERSION: u16 = 1;

/*
    +--------------------------------------------------+
    | LiquidIPCHeader (16 bytes)                       |
    +--------------------------------------------------+
    | MAGIC (4 bytes)                                  |  // Offset  0..3: "LQDA" magic number (0x4C51_4441)
    +--------------------------------------------------+
    | VERSION (2 bytes)                                |  // Offset  4..5: Version (currently 1)
    +--------------------------------------------------+
    | logical_type_id (2 bytes)                        |  // Offset  6..7: Logical type identifier (e.g. Integer)
    +--------------------------------------------------+
    | physical_type_id (2 bytes)                       |  // Offset  8..9: Physical type identifier for T
    +--------------------------------------------------+
    | __padding (6 bytes)                              |  // Offset 10..15: Padding to ensure 16 byte header
    +--------------------------------------------------+
*/
#[repr(C)]
struct LiquidIPCHeader {
    magic: [u8; 4],
    version: u16,
    logical_type_id: u16,
    physical_type_id: u16,
    __padding: [u8; 6],
}

const _: () = assert!(size_of::<LiquidIPCHeader>() == LiquidIPCHeader::size());

impl LiquidIPCHeader {
    const fn size() -> usize {
        16
    }

    fn to_bytes(&self) -> [u8; Self::size()] {
        let mut bytes = [0; Self::size()];
        bytes[0..4].copy_from_slice(&self.magic);
        bytes[4..6].copy_from_slice(&self.version.to_le_bytes());
        bytes[6..8].copy_from_slice(&self.logical_type_id.to_le_bytes());
        bytes[8..10].copy_from_slice(&self.physical_type_id.to_le_bytes());
        bytes
    }

    fn from_bytes(bytes: &[u8]) -> Self {
        if bytes.len() < Self::size() {
            panic!(
                "value too small for LiquidIPCHeader, expected at least {} bytes, got {}",
                Self::size(),
                bytes.len()
            );
        }
        let magic = bytes[0..4].try_into().unwrap();
        let version = u16::from_le_bytes(bytes[4..6].try_into().unwrap());
        let logical_type_id = u16::from_le_bytes(bytes[6..8].try_into().unwrap());
        let physical_type_id = u16::from_le_bytes(bytes[8..10].try_into().unwrap());
        Self {
            magic,
            version,
            logical_type_id,
            physical_type_id,
            __padding: [0; 6],
        }
    }
}

pub struct LiquidIPCContext {
    compressor: Option<Arc<Compressor>>,
}

impl LiquidIPCContext {
    pub fn new(compressor: Option<Arc<Compressor>>) -> Self {
        Self { compressor }
    }
}

pub fn read_from_bytes(bytes: Bytes, context: &LiquidIPCContext) -> LiquidArrayRef {
    let header = LiquidIPCHeader::from_bytes(&bytes);
    let logical_type = LiquidDataType::from(header.logical_type_id);
    match logical_type {
        LiquidDataType::Integer => match header.physical_type_id {
            0 => Arc::new(LiquidPrimitiveArray::<Int8Type>::from_bytes(bytes)),
            1 => Arc::new(LiquidPrimitiveArray::<Int16Type>::from_bytes(bytes)),
            2 => Arc::new(LiquidPrimitiveArray::<Int32Type>::from_bytes(bytes)),
            3 => Arc::new(LiquidPrimitiveArray::<Int64Type>::from_bytes(bytes)),
            4 => Arc::new(LiquidPrimitiveArray::<UInt8Type>::from_bytes(bytes)),
            5 => Arc::new(LiquidPrimitiveArray::<UInt16Type>::from_bytes(bytes)),
            6 => Arc::new(LiquidPrimitiveArray::<UInt32Type>::from_bytes(bytes)),
            7 => Arc::new(LiquidPrimitiveArray::<UInt64Type>::from_bytes(bytes)),
            _ => panic!("Unsupported physical type"),
        },
        LiquidDataType::ByteArray => {
            let compressor = context.compressor.as_ref().expect("Expected a compressor");
            Arc::new(LiquidByteArray::from_bytes(bytes, compressor.clone()))
        }
        LiquidDataType::Float => {
            panic!("Conversion of float array to IPC format is not yet supported");
        }
    }
}

impl<T> LiquidPrimitiveArray<T>
where
    T: LiquidPrimitiveType,
{
    fn bit_pack_starting_loc() -> usize {
        let header_size = LiquidIPCHeader::size() + size_of::<T::Native>();
        (header_size + 7) & !7
    }

    /*
    Serialized LiquidPrimitiveArray Memory Layout:
    +--------------------------------------------------+
    | LiquidIPCHeader (16 bytes)                       |
    +--------------------------------------------------+

    +--------------------------------------------------+
    | reference_value (size_of::<T::Native> bytes)     |  // The reference value (e.g. minimum value)
    +--------------------------------------------------+
    | Padding (to 8-byte alignment)                    |  // Padding to ensure 8-byte alignment
    +--------------------------------------------------+

    +--------------------------------------------------+
    | BitPackedArray Data                              |
    +--------------------------------------------------+
    | [BitPackedArray Header & Bit-Packed Values]      |  // Written by self.bit_packed.to_bytes()
    +--------------------------------------------------+
    */
    pub(crate) fn to_bytes_inner(&self) -> Vec<u8> {
        // Determine type ID based on the type
        let physical_type_id = get_physical_type_id::<T>();
        let logical_type_id = LiquidDataType::Integer as u16;
        let header = LiquidIPCHeader {
            magic: MAGIC.to_le_bytes(),
            version: VERSION,
            logical_type_id,
            physical_type_id,
            __padding: [0; 6],
        };

        let bit_pack_starting_loc = Self::bit_pack_starting_loc();
        let mut result = Vec::with_capacity(bit_pack_starting_loc + 256); // Pre-allocate a reasonable size

        // Write header
        result.extend_from_slice(&header.to_bytes());

        // Write reference value
        let ref_value_bytes = unsafe {
            std::slice::from_raw_parts(
                &self.reference_value as *const T::Native as *const u8,
                size_of::<T::Native>(),
            )
        };
        result.extend_from_slice(ref_value_bytes);
        while result.len() < bit_pack_starting_loc {
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
        let header = LiquidIPCHeader::from_bytes(&bytes);

        if header.magic != MAGIC.to_le_bytes() {
            panic!("Invalid magic number");
        }

        if header.version != VERSION {
            panic!("Unsupported version");
        }

        let physical_id = header.physical_type_id;
        assert_eq!(physical_id, get_physical_type_id::<T>());
        let logical_id = header.logical_type_id;
        assert_eq!(logical_id, LiquidDataType::Integer as u16);

        // Get the reference value
        let ref_value_ptr = &bytes[LiquidIPCHeader::size()];
        let reference_value =
            unsafe { (ref_value_ptr as *const u8 as *const T::Native).read_unaligned() };

        // Skip ahead to the BitPackedArray data
        let bit_packed_data = bytes.slice(Self::bit_pack_starting_loc()..);
        let bit_packed = BitPackedArray::<T::UnSignedType>::from_bytes(bit_packed_data);

        Self {
            bit_packed,
            reference_value,
        }
    }
}

#[repr(C)]
struct ByteArrayHeader {
    key_size: u32,
    value_size: u32,
}

impl ByteArrayHeader {
    const fn size() -> usize {
        8
    }

    fn to_bytes(&self) -> [u8; Self::size()] {
        let mut bytes = [0; Self::size()];
        bytes[0..4].copy_from_slice(&self.key_size.to_le_bytes());
        bytes[4..8].copy_from_slice(&self.value_size.to_le_bytes());
        bytes
    }

    fn from_bytes(bytes: &[u8]) -> Self {
        if bytes.len() < Self::size() {
            panic!(
                "value too small for ByteArrayHeader, expected at least {} bytes, got {}",
                Self::size(),
                bytes.len()
            );
        }
        let key_size = u32::from_le_bytes(bytes[0..4].try_into().unwrap());
        let value_size = u32::from_le_bytes(bytes[4..8].try_into().unwrap());
        Self {
            key_size,
            value_size,
        }
    }
}
const _: () = assert!(size_of::<ByteArrayHeader>() == ByteArrayHeader::size());

impl LiquidByteArray {
    /*
    Serialized LiquidByteArray Memory Layout:

    +--------------------------------------------------+
    | LiquidIPCHeader (16 bytes)                       |
    +--------------------------------------------------+
    | ByteArrayHeader (8 bytes)                        |  // Contains key_size and value_size
    +--------------------------------------------------+
    | BitPackedArray Data (keys)                       |
    +--------------------------------------------------+
    | [BitPackedArray Header & Bit-Packed Key Values]  |  // Written by self.keys.to_bytes()
    +--------------------------------------------------+
    | Padding (to 8-byte alignment)                    |  // Padding to ensure 8-byte alignment
    +--------------------------------------------------+
    | FsstArray Data (values)                          |
    +--------------------------------------------------+
    | [FsstArray Data]                                 |  // Written by self.values.to_bytes()
    +--------------------------------------------------+
    */
    pub(crate) fn to_bytes_inner(&self) -> Vec<u8> {
        // Create a buffer for the final output data, starting with the header
        let header_size = LiquidIPCHeader::size() + ByteArrayHeader::size();
        let mut result = Vec::with_capacity(header_size + 1024); // Pre-allocate a reasonable size

        result.resize(header_size, 0);

        // Serialize the BitPackedArray (keys)
        let keys_start = result.len();
        self.keys.to_bytes(&mut result);
        let keys_size = result.len() - keys_start;

        // Add padding to ensure FsstArray starts at an 8-byte aligned position
        while result.len() % 8 != 0 {
            result.push(0);
        }

        // Serialize the FsstArray (values)
        let values_start = result.len();
        self.values.to_bytes(&mut result);
        let values_size = result.len() - values_start;

        // Go back and fill in the header
        let ipc_header = LiquidIPCHeader {
            magic: MAGIC.to_le_bytes(),
            version: VERSION,
            logical_type_id: LiquidDataType::ByteArray as u16,
            physical_type_id: self.original_arrow_type as u16,
            __padding: [0; 6],
        };
        let header = &mut result[0..header_size];
        header[0..LiquidIPCHeader::size()].copy_from_slice(&ipc_header.to_bytes());

        let byte_array_header = ByteArrayHeader {
            key_size: keys_size as u32,
            value_size: values_size as u32,
        };
        header[LiquidIPCHeader::size()..header_size].copy_from_slice(&byte_array_header.to_bytes());

        result
    }

    /// Deserialize a LiquidByteArray from bytes, using zero-copy where possible.
    pub fn from_bytes(bytes: Bytes, compressor: Arc<Compressor>) -> Self {
        let header_size = LiquidIPCHeader::size() + ByteArrayHeader::size();
        let header = LiquidIPCHeader::from_bytes(&bytes);

        if header.magic != MAGIC.to_le_bytes() {
            panic!("Invalid magic number");
        }

        if header.version != VERSION {
            panic!("Unsupported version");
        }

        let byte_array_header =
            ByteArrayHeader::from_bytes(&bytes[LiquidIPCHeader::size()..header_size]);

        let original_arrow_type = ArrowStringType::from(header.physical_type_id);

        let keys_size = byte_array_header.key_size as usize;
        let values_size = byte_array_header.value_size as usize;

        // Calculate offsets
        let keys_start = header_size;
        let keys_end = keys_start + keys_size;

        if keys_end > bytes.len() {
            panic!("Keys data extends beyond input buffer");
        }

        // Ensure values data starts at 8-byte aligned position
        let values_start = (keys_end + 7) & !7; // Round up to next 8-byte boundary
        let values_end = values_start + values_size;

        if values_end > bytes.len() {
            panic!("Values data extends beyond input buffer");
        }

        // Extract and deserialize components
        let keys_data = bytes.slice(keys_start..keys_end);
        let keys = BitPackedArray::<UInt16Type>::from_bytes(keys_data);

        let values_data = bytes.slice(values_start..values_end);
        let values = FsstArray::from_bytes(values_data, compressor);

        Self {
            keys,
            values,
            original_arrow_type,
        }
    }
}

fn get_physical_type_id<T: LiquidPrimitiveType>() -> u16 {
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
    use arrow::{
        array::{PrimitiveArray, StringArray},
        datatypes::Int32Type,
    };

    use crate::liquid_array::LiquidArray;

    use super::*;

    #[test]
    fn test_to_bytes() {
        // Create a simple array
        let original: Vec<Option<i32>> = vec![Some(10), Some(20), Some(30), None, Some(50)];
        let array = PrimitiveArray::<Int32Type>::from(original.clone());
        let liquid_array = LiquidPrimitiveArray::<Int32Type>::from_arrow_array(array);

        // Serialize to bytes
        let bytes = liquid_array.to_bytes_inner();

        // Basic validation
        let header = LiquidIPCHeader::from_bytes(&bytes);
        assert_eq!(
            header.magic,
            MAGIC.to_le_bytes(),
            "Magic number should be LQDA"
        );
        assert_eq!(header.version, VERSION, "Version should be 1");
        assert_eq!(
            header.physical_type_id, 2,
            "Type ID for Int32Type should be 2"
        );
        assert_eq!(
            header.logical_type_id,
            LiquidDataType::Integer as u16,
            "Logical type ID should be 1"
        );

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

        let bytes = liquid_array.to_bytes_inner();
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
        let bytes = liquid_array.to_bytes_inner();
        let bytes = Bytes::from(bytes);
        let deserialized = LiquidPrimitiveArray::<Int32Type>::from_bytes(bytes);
        let result = deserialized.to_arrow_array();
        assert_eq!(result.as_ref(), &array);

        // 2. No nulls array
        let no_nulls: Vec<Option<i32>> = (0..1000).map(|i| Some(i)).collect();
        let array = PrimitiveArray::<Int32Type>::from(no_nulls);
        let liquid_array = LiquidPrimitiveArray::<Int32Type>::from_arrow_array(array.clone());
        let bytes = liquid_array.to_bytes_inner();
        let bytes = Bytes::from(bytes);
        let deserialized = LiquidPrimitiveArray::<Int32Type>::from_bytes(bytes);
        let result = deserialized.to_arrow_array();
        assert_eq!(result.as_ref(), &array);

        // 3. Single value array
        let single_value: Vec<Option<i32>> = vec![Some(42)];
        let array = PrimitiveArray::<Int32Type>::from(single_value);
        let liquid_array = LiquidPrimitiveArray::<Int32Type>::from_arrow_array(array.clone());
        let bytes = liquid_array.to_bytes_inner();
        let bytes = Bytes::from(bytes);
        let deserialized = LiquidPrimitiveArray::<Int32Type>::from_bytes(bytes);
        let result = deserialized.to_arrow_array();
        assert_eq!(result.as_ref(), &array);

        // 4. Empty array
        let empty: Vec<Option<i32>> = vec![];
        let array = PrimitiveArray::<Int32Type>::from(empty);
        let liquid_array = LiquidPrimitiveArray::<Int32Type>::from_arrow_array(array.clone());
        let bytes = liquid_array.to_bytes_inner();
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
        let bytes = liquid_array.to_bytes_inner();
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
        let bytes = liquid_array.to_bytes_inner();
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
        let bytes = liquid_array.to_bytes_inner();
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
        let bytes = liquid_array.to_bytes_inner();
        let bytes = Bytes::from(bytes);
        let deserialized = LiquidPrimitiveArray::<UInt64Type>::from_bytes(bytes);
        let result = deserialized.to_arrow_array();
        assert_eq!(result.as_ref(), &array);
    }

    #[test]
    fn test_byte_array_roundtrip() {
        let string_array = StringArray::from(vec![
            Some("hello"),
            Some("world"),
            None,
            Some("liquid"),
            Some("byte"),
            Some("array"),
        ]);

        // Create a compressor and LiquidByteArray
        let compressor =
            FsstArray::train_compressor(string_array.iter().flat_map(|s| s.map(|s| s.as_bytes())));
        let compressor_arc = Arc::new(compressor);

        let original = LiquidByteArray::from_string_array(&string_array, compressor_arc.clone());

        let bytes = original.to_bytes_inner();
        let bytes = Bytes::from(bytes);
        let deserialized = LiquidByteArray::from_bytes(bytes, compressor_arc);

        let original_arrow = original.to_arrow_array();
        let deserialized_arrow = deserialized.to_arrow_array();

        assert_eq!(original_arrow.as_ref(), deserialized_arrow.as_ref());

        // Verify the original arrow type is preserved
        assert_eq!(
            original.original_arrow_type as u8,
            deserialized.original_arrow_type as u8
        );
    }
}
