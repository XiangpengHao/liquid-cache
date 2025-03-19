use arrow::{
    array::{
        Array, ArrayDataBuilder, BinaryArray, BufferBuilder, GenericByteArray, StringArray,
        builder::BinaryBuilder,
    },
    buffer::{BooleanBuffer, Buffer, NullBuffer},
    datatypes::{ArrowNativeType, ByteArrayType, Utf8Type},
};
use bytes;
use fsst::{Compressor, Decompressor};
use std::mem::MaybeUninit;
use std::sync::Arc;

#[derive(Clone)]
pub struct FsstArray {
    compressor: Arc<Compressor>,
    // TODO: should we do a values + offset array here? So that the offset array is a bit-packed array?
    pub(crate) compressed: BinaryArray,
    pub(crate) uncompressed_len: usize,
}

impl std::fmt::Debug for FsstArray {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "FsstArray")
    }
}

impl FsstArray {
    pub fn from_parts(
        compressed: BinaryArray,
        compressor: Arc<Compressor>,
        uncompressed_len: usize,
    ) -> Self {
        Self {
            compressor,
            compressed,
            uncompressed_len,
        }
    }

    pub fn train_compressor<'a>(input: impl Iterator<Item = &'a [u8]>) -> Compressor {
        let strings = input.collect::<Vec<_>>();
        fsst::Compressor::train(&strings)
    }

    pub fn from_byte_array_with_compressor<T: ByteArrayType>(
        input: &GenericByteArray<T>,
        compressor: Arc<Compressor>,
    ) -> Self {
        let default_offset = T::Offset::default();
        let data_capacity = input.offsets().last().unwrap_or(&default_offset);
        let item_capacity = input.len();

        let mut compress_buffer = Vec::with_capacity(2 * 1024 * 1024);
        let mut builder = BinaryBuilder::with_capacity(item_capacity, data_capacity.as_usize());
        let mut total_len = 0;

        for s in input.iter() {
            match s {
                Some(s) => {
                    let bytes: &[u8] = s.as_ref();
                    total_len += bytes.len();
                    unsafe {
                        compressor.compress_into(bytes, &mut compress_buffer);
                    }
                    builder.append_value(&compress_buffer);
                }
                None => {
                    builder.append_null();
                }
            }
        }

        let compressed = builder.finish();
        let compressed = {
            let (offsets, values, nulls) = compressed.into_parts();
            let mut shrunk_values = Vec::with_capacity(values.len());
            shrunk_values.extend_from_slice(&values);
            unsafe { GenericByteArray::new_unchecked(offsets, shrunk_values.into(), nulls) }
        };

        FsstArray {
            compressor,
            compressed,
            uncompressed_len: total_len,
        }
    }

    pub fn get_array_memory_size(&self) -> usize {
        self.compressed.get_array_memory_size() + std::mem::size_of::<FsstArray>()
    }

    pub fn to_arrow_byte_array<T: ByteArrayType>(&self) -> GenericByteArray<T> {
        // we can directly use the null buffer in the compressed array.
        let null_buffer = self.compressed.nulls().cloned();
        let mut value_buffer: Vec<u8> = Vec::with_capacity(self.uncompressed_len + 8);
        let mut offsets_builder = BufferBuilder::<i32>::new(self.compressed.len() + 1);
        offsets_builder.append(0);

        let decompressor = self.compressor.decompressor();

        for v in self.compressed.iter() {
            match v {
                Some(v) => {
                    let slice = unsafe {
                        std::slice::from_raw_parts_mut(
                            value_buffer.as_mut_ptr().add(value_buffer.len())
                                as *mut MaybeUninit<u8>,
                            value_buffer.capacity(), // we don't care about the capacity here
                        )
                    };
                    let len = decompressor.decompress_into(v, slice);
                    let new_len = value_buffer.len() + len;
                    debug_assert!(new_len <= value_buffer.capacity());
                    unsafe {
                        value_buffer.set_len(new_len);
                    }
                    offsets_builder.append(value_buffer.len() as i32);
                }
                None => {
                    offsets_builder.append(value_buffer.len() as i32);
                }
            }
        }
        assert_eq!(value_buffer.len(), self.uncompressed_len);
        let value_buffer = Buffer::from(value_buffer);
        let offsets_buffer = offsets_builder.finish();
        let array_builder = ArrayDataBuilder::new(T::DATA_TYPE)
            .len(self.compressed.len())
            .add_buffer(offsets_buffer)
            .add_buffer(value_buffer)
            .nulls(null_buffer);
        let array_data = unsafe { array_builder.build_unchecked() };
        GenericByteArray::from(array_data)
    }

    pub fn decompressor(&self) -> Decompressor {
        self.compressor.decompressor()
    }

    pub fn compressor(&self) -> &Compressor {
        &self.compressor
    }

    /*
    Memory Layout (serialized):

    +-------------------------------+  // Header (17 bytes total)
    | uncompressed_len (8 bytes)    |  // Offset  0 -  7: Original uncompressed length (u64)
    +-------------------------------+  //
    | has_nulls (1 byte)            |  // Offset      8: Null flag (1 if nulls present)
    +-------------------------------+  //
    | offsets_len (4 bytes)         |  // Offset  9 - 12: Length of offsets buffer (u32)
    +-------------------------------+  //
    | values_len (4 bytes)          |  // Offset 13 - 16: Length of values buffer (u32)
    +-------------------------------+

    [If has_nulls == 1]
    +-------------------------------+  // Nulls Buffer
    | nulls data (nulls_len bytes)  |  // Offset 16 - (16 + nulls_len - 1)
    +-------------------------------+

    +-------------------------------+
    | Padding for 8-byte alignment  |  // Ensure offsets buffer is 8-byte aligned
    +-------------------------------+

    +-------------------------------+  // Offsets Buffer
    | offsets data (offsets_len)    |  // Starts at the 8-byte aligned offset
    +-------------------------------+

    +-------------------------------+
    | Padding for 8-byte alignment  |  // Ensure values buffer is 8-byte aligned
    +-------------------------------+

    +-------------------------------+  // Values Buffer (compressed data)
    | values data (values_len)      |  // Starts at the 8-byte aligned offset
    +-------------------------------+
    */
    pub fn to_bytes(&self, buffer: &mut Vec<u8>) {
        let (offsets, values, nulls) = self.compressed.clone().into_parts();
        let has_nulls = nulls.is_some() as u8;

        let nulls_bytes = if has_nulls == 1 {
            let nulls = nulls.as_ref().unwrap();
            nulls.buffer().as_slice()
        } else {
            &[]
        };

        let offsets_bytes = offsets.as_ref();
        let values_bytes = values.as_slice();

        let header_size = 17;
        // Ensure nulls start at an 8-byte aligned boundary
        let nulls_offset_base = header_size;
        let nulls_offset = (nulls_offset_base + 7) & !7; // 8-byte aligned
        let nulls_size = nulls_bytes.len();

        // Calculate offsets with proper alignment
        let offsets_offset_base = nulls_offset + nulls_size;
        let offsets_offset = (offsets_offset_base + 7) & !7; // 8-byte aligned

        let values_offset_base = offsets_offset + std::mem::size_of_val(offsets_bytes);
        let values_offset = (values_offset_base + 7) & !7; // 8-byte aligned

        let total_size = values_offset + values_bytes.len();
        buffer.reserve(total_size);

        let start_offset = buffer.len();

        // Write header
        buffer.extend_from_slice(&(self.uncompressed_len as u64).to_le_bytes());
        buffer.push(has_nulls);
        buffer.extend_from_slice(&(std::mem::size_of_val(offsets_bytes) as u32).to_le_bytes());
        buffer.extend_from_slice(&(values_bytes.len() as u32).to_le_bytes());

        // Add padding to align nulls buffer
        while (buffer.len() - start_offset) < nulls_offset {
            buffer.push(0);
        }

        // Write nulls if present
        if has_nulls == 1 {
            buffer.extend_from_slice(nulls_bytes);
        }

        // Add padding for offsets alignment
        while (buffer.len() - start_offset) < offsets_offset {
            buffer.push(0);
        }

        // Write offsets
        for offset in offsets_bytes {
            buffer.extend_from_slice(&offset.to_le_bytes());
        }

        // Add padding for values alignment
        while (buffer.len() - start_offset) < values_offset {
            buffer.push(0);
        }

        // Write values
        buffer.extend_from_slice(values_bytes);
    }

    pub fn from_bytes(bytes: bytes::Bytes, compressor: Arc<Compressor>) -> Self {
        if bytes.len() < 17 {
            panic!("Input buffer too small for header");
        }

        // Read header fields
        let uncompressed_len = u64::from_le_bytes(bytes[0..8].try_into().unwrap()) as usize;
        let has_nulls = bytes[8] != 0;
        let offsets_len = u32::from_le_bytes(bytes[9..13].try_into().unwrap()) as usize;
        let values_len = u32::from_le_bytes(bytes[13..17].try_into().unwrap()) as usize;

        // Calculate offsets
        let header_size = 17;
        // Ensure nulls start at an 8-byte aligned boundary
        let nulls_offset_base = header_size;
        let nulls_offset = (nulls_offset_base + 7) & !7; // 8-byte aligned
        let nulls_size = if has_nulls {
            // Calculate length of the nulls buffer based on array length
            let array_len = offsets_len / std::mem::size_of::<i32>() - 1;
            array_len.div_ceil(8)
        } else {
            0
        };

        // Calculate aligned offsets
        let offsets_offset_base = nulls_offset + nulls_size;
        let offsets_offset = (offsets_offset_base + 7) & !7; // 8-byte aligned

        let values_offset_base = offsets_offset + offsets_len;
        let values_offset = (values_offset_base + 7) & !7; // 8-byte aligned

        // Validate offsets and lengths
        if has_nulls {
            if nulls_offset == 0 || nulls_size == 0 {
                panic!("Array has nulls but null buffer is missing");
            }
            if nulls_offset + nulls_size > bytes.len() {
                panic!("Null buffer extends beyond input buffer");
            }
        }

        if offsets_offset == 0 || offsets_len == 0 {
            panic!("Offsets buffer is required");
        }
        if offsets_offset + offsets_len > bytes.len() {
            panic!("Offsets buffer extends beyond input buffer");
        }

        if values_offset == 0 || values_len == 0 {
            panic!("Values buffer is required");
        }
        if values_offset + values_len > bytes.len() {
            panic!("Values buffer extends beyond input buffer");
        }

        // Create the nulls buffer if present
        let nulls = if has_nulls {
            // Create a buffer view into the nulls section
            let nulls_slice = bytes.slice(nulls_offset..nulls_offset + nulls_size);
            let nulls_buffer = Buffer::from(nulls_slice);
            let array_len = offsets_len / std::mem::size_of::<i32>() - 1;
            let boolean_buffer = BooleanBuffer::new(nulls_buffer, 0, array_len);
            Some(NullBuffer::from(boolean_buffer))
        } else {
            None
        };

        // Create the offsets buffer
        let offsets_slice = bytes.slice(offsets_offset..offsets_offset + offsets_len);
        let offsets_buffer = Buffer::from(offsets_slice);

        // Convert raw bytes to i32 offsets
        let array_len = offsets_len / std::mem::size_of::<i32>();
        let offsets_typed = arrow::buffer::ScalarBuffer::<i32>::new(offsets_buffer, 0, array_len);
        let offsets = arrow::buffer::OffsetBuffer::new(offsets_typed);

        // Create the values buffer
        let values_slice = bytes.slice(values_offset..values_offset + values_len);
        let values_buffer = Buffer::from(values_slice);

        // Create the BinaryArray directly from components
        let compressed = unsafe { GenericByteArray::new_unchecked(offsets, values_buffer, nulls) };

        Self {
            compressor,
            compressed,
            uncompressed_len,
        }
    }
}

impl From<&FsstArray> for StringArray {
    fn from(value: &FsstArray) -> Self {
        value.to_arrow_byte_array::<Utf8Type>()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::StringBuilder;

    #[test]
    fn test_liquid_string_roundtrip() {
        // Create test data with mix of strings and nulls
        let mut builder = StringBuilder::new();
        for i in 0..5000 {
            if i % 100 == 0 {
                builder.append_null();
            } else {
                match i % 5 {
                    0 => builder.append_value(&format!("hello world {}", i)),
                    1 => builder.append_value(&format!("🦀 rust is awesome {}", i)),
                    2 => builder.append_value(&format!(
                        "testing string compression with a longer string to test efficiency {}",
                        i
                    )),
                    3 => builder.append_value(&format!(
                        "The quick brown fox jumps over the lazy dog {}",
                        i
                    )),
                    _ => builder.append_value(&format!("Lorem ipsum dolor sit amet {}", i)),
                }
            }
        }
        let original = builder.finish();

        println!("original len: {}", original.get_array_memory_size());
        // Convert to EtcString and back
        let compressor =
            FsstArray::train_compressor(original.iter().flat_map(|s| s.map(|s| s.as_bytes())));
        let etc = FsstArray::from_byte_array_with_compressor(&original, Arc::new(compressor));
        println!("etc len: {}", etc.compressed.get_array_memory_size());

        let roundtrip = StringArray::from(&etc);

        // Verify length is preserved
        assert_eq!(original.len(), roundtrip.len());

        // Verify each value matches
        for (orig, round) in original.iter().zip(roundtrip.iter()) {
            assert_eq!(orig, round);
        }
    }

    #[test]
    fn test_bytes_roundtrip() {
        // Create test data
        let mut builder = StringBuilder::new();
        for i in 0..1000 {
            if i % 100 == 0 {
                builder.append_null();
            } else {
                builder.append_value(&format!("test string value {}", i));
            }
        }
        let original = builder.finish();

        // Create FsstArray
        let compressor =
            FsstArray::train_compressor(original.iter().flat_map(|s| s.map(|s| s.as_bytes())));
        let compressor_arc = Arc::new(compressor);
        let original_fsst =
            FsstArray::from_byte_array_with_compressor(&original, compressor_arc.clone());

        // Serialize to bytes
        let mut buffer = Vec::new();
        original_fsst.to_bytes(&mut buffer);

        // Deserialize from bytes
        let bytes = bytes::Bytes::from(buffer);
        let deserialized = FsstArray::from_bytes(bytes, compressor_arc);

        // Convert both to StringArray for comparison
        let original_strings = StringArray::from(&original_fsst);
        let deserialized_strings = StringArray::from(&deserialized);

        // Verify length is preserved
        assert_eq!(original_strings.len(), deserialized_strings.len());

        // Verify each value matches
        for (orig, deser) in original_strings.iter().zip(deserialized_strings.iter()) {
            assert_eq!(orig, deser);
        }
    }
}
