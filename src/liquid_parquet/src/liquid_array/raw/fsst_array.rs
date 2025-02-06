use arrow::{
    array::{
        Array, ArrayDataBuilder, BinaryArray, BufferBuilder, GenericByteArray, StringArray,
        builder::BinaryBuilder,
    },
    buffer::Buffer,
    datatypes::{ArrowNativeType, ByteArrayType, Utf8Type},
};
use fsst::Compressor;
use std::mem::MaybeUninit;
use std::sync::Arc;

#[derive(Clone)]
pub(crate) struct FsstArray {
    pub(crate) compressor: Arc<Compressor>,
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

    pub(crate) fn get_array_memory_size(&self) -> usize {
        self.compressed.get_array_memory_size() + std::mem::size_of::<FsstArray>()
    }

    pub(crate) fn to_arrow_byte_array<T: ByteArrayType>(&self) -> GenericByteArray<T> {
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
                    let len = unsafe { decompressor.decompress_into(v, slice) };
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
}
