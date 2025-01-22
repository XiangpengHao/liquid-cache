use arrow::array::{
    builder::{BinaryBuilder, StringBuilder},
    Array, BinaryArray, GenericByteArray, StringArray,
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

    pub fn from_string_array(input: &StringArray) -> Self {
        let strings = input
            .iter()
            .filter_map(|s| s.as_ref().map(|s| s.as_bytes()))
            .collect::<Vec<_>>();

        let compressor = Arc::new(fsst::Compressor::train(&strings));

        Self::from_string_array_with_compressor(input, compressor)
    }

    pub fn from_string_array_with_compressor(
        input: &StringArray,
        compressor: Arc<Compressor>,
    ) -> Self {
        let data_capacity = input.offsets().last().unwrap_or(&0);
        let item_capacity = input.offsets().len();

        let mut compress_buffer = Vec::with_capacity(2 * 1024 * 1024);
        let mut builder = BinaryBuilder::with_capacity(item_capacity, *data_capacity as usize);
        let mut total_len = 0;

        for s in input.iter() {
            match s {
                Some(s) => {
                    let bytes = s.as_bytes();
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
}

impl From<&FsstArray> for StringArray {
    fn from(value: &FsstArray) -> Self {
        let total_size = value.uncompressed_len;
        let mut builder = StringBuilder::with_capacity(value.compressed.len(), total_size);

        let decompressor = value.compressor.decompressor();
        let mut decompress_buffer: Vec<u8> = Vec::with_capacity(1024);
        for v in value.compressed.iter() {
            match v {
                Some(v) => {
                    let cap = decompressor.max_decompression_capacity(v);
                    let decompressed = unsafe {
                        std::slice::from_raw_parts_mut(
                            decompress_buffer.as_mut_ptr() as *mut MaybeUninit<u8>,
                            cap,
                        )
                    };
                    let len = decompressor.decompress_into(v, decompressed);
                    unsafe {
                        decompress_buffer.set_len(len);
                    }
                    let s = unsafe { std::str::from_utf8_unchecked(&decompress_buffer) };
                    builder.append_value(s);
                }
                None => {
                    builder.append_null();
                }
            }
        }
        builder.finish()
    }
}

impl From<&StringArray> for FsstArray {
    fn from(input: &StringArray) -> Self {
        FsstArray::from_string_array(input)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_etc_string_roundtrip() {
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
        let etc = FsstArray::from(&original);
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
