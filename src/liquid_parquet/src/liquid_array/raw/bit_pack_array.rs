use std::num::NonZero;

use arrow::array::{ArrowPrimitiveType, PrimitiveArray};
use arrow::buffer::{Buffer, NullBuffer, ScalarBuffer};
use arrow::datatypes::ArrowNativeType;
use fastlanes::BitPacking;

#[derive(Debug)]
pub struct BitPackedArray<T: ArrowPrimitiveType>
where
    T::Native: BitPacking,
{
    packed_values: ScalarBuffer<T::Native>,
    nulls: Option<NullBuffer>,
    bit_width: NonZero<u8>,
    original_len: usize,
}

/// Implement Clone for any T that implements ArrowPrimitiveType and BitPacking
/// This allows us to clone it without requiring T to implement Clone
impl<T: ArrowPrimitiveType> Clone for BitPackedArray<T>
where
    T::Native: BitPacking,
{
    fn clone(&self) -> Self {
        Self {
            packed_values: self.packed_values.clone(),
            nulls: self.nulls.clone(),
            bit_width: self.bit_width,
            original_len: self.original_len,
        }
    }
}

impl<T: ArrowPrimitiveType> BitPackedArray<T>
where
    T::Native: BitPacking,
{
    pub fn from_parts(
        packed_values: ScalarBuffer<T::Native>,
        nulls: Option<NullBuffer>,
        bit_width: NonZero<u8>,
        original_len: usize,
    ) -> Self {
        Self {
            packed_values,
            nulls,
            bit_width,
            original_len,
        }
    }

    pub fn new_null_array(len: usize) -> Self {
        Self {
            packed_values: vec![T::Native::usize_as(0); len].into(),
            nulls: Some(NullBuffer::new_null(len)),
            bit_width: NonZero::new(u8::MAX).unwrap(),
            original_len: len,
        }
    }

    pub(crate) fn len(&self) -> usize {
        self.original_len
    }

    pub(crate) fn nulls(&self) -> Option<&NullBuffer> {
        self.nulls.as_ref()
    }

    pub(crate) fn bit_width(&self) -> NonZero<u8> {
        self.bit_width
    }

    pub fn is_nullable(&self) -> bool {
        self.nulls.is_some()
    }

    pub fn from_primitive(array: PrimitiveArray<T>, bit_width: NonZero<u8>) -> Self {
        let original_len = array.len();
        let (_data_type, values, nulls) = array.into_parts();

        let bit_width_usize = bit_width.get() as usize;
        let num_chunks = original_len.div_ceil(1024);
        let num_full_chunks = original_len / 1024;
        let packed_len = (1024 * bit_width_usize).div_ceil(size_of::<T::Native>() * 8);

        let mut output = Vec::<T::Native>::with_capacity(num_chunks * packed_len);

        (0..num_full_chunks).for_each(|i| {
            let start_elem = i * 1024;

            output.reserve(packed_len);
            let output_len = output.len();
            unsafe {
                output.set_len(output_len + packed_len);
                BitPacking::unchecked_pack(
                    bit_width_usize,
                    &values[start_elem..][..1024],
                    &mut output[output_len..][..packed_len],
                );
            }
        });

        if num_chunks != num_full_chunks {
            let last_chunk_size = values.len() % 1024;
            let mut last_chunk = vec![T::Native::default(); 1024];
            last_chunk[..last_chunk_size]
                .copy_from_slice(&values[values.len() - last_chunk_size..]);

            output.reserve(packed_len);
            let output_len = output.len();
            unsafe {
                output.set_len(output_len + packed_len);
                BitPacking::unchecked_pack(
                    bit_width_usize,
                    &last_chunk,
                    &mut output[output_len..][..packed_len],
                );
            }
        }

        let buffer = Buffer::from(output);
        let scalar_buffer = ScalarBuffer::new(buffer, 0, num_chunks * packed_len);

        Self {
            packed_values: scalar_buffer,
            nulls,
            bit_width,
            original_len,
        }
    }

    pub fn to_primitive(&self) -> PrimitiveArray<T> {
        // Special case for all nulls, don't unpack
        if let Some(nulls) = self.nulls() {
            if nulls.null_count() == self.original_len {
                return PrimitiveArray::<T>::new_null(self.original_len);
            }
        }
        let bit_width = self.bit_width.get() as usize;
        let packed = self.packed_values.as_ref();
        let length = self.original_len;
        let offset = 0;

        let num_chunks = (offset + length).div_ceil(1024);
        let elements_per_chunk = (1024 * bit_width).div_ceil(size_of::<T::Native>() * 8);

        let mut output = Vec::<T::Native>::with_capacity(num_chunks * 1024 - offset);

        let first_full_chunk = if offset != 0 {
            let chunk: &[T::Native] = &packed[0..elements_per_chunk];
            let mut decoded = vec![T::Native::default(); 1024];
            unsafe { BitPacking::unchecked_unpack(bit_width, chunk, &mut decoded) };
            output.extend_from_slice(&decoded[offset..]);
            1
        } else {
            0
        };

        (first_full_chunk..num_chunks).for_each(|i| {
            let chunk: &[T::Native] = &packed[i * elements_per_chunk..][0..elements_per_chunk];
            unsafe {
                let output_len = output.len();
                output.set_len(output_len + 1024);
                BitPacking::unchecked_unpack(bit_width, chunk, &mut output[output_len..][..1024]);
            }
        });

        output.truncate(length);
        if output.len() < 1024 {
            output.shrink_to_fit();
        }

        let nulls = self.nulls.clone();
        PrimitiveArray::<T>::new(ScalarBuffer::from(output), nulls)
    }

    pub fn get_array_memory_size(&self) -> usize {
        self.packed_values.inner().capacity()
    }
}

#[allow(dead_code)]
fn best_arrow_primitive_width(bit_width: NonZero<u8>) -> usize {
    match bit_width.get() {
        0..=8 => 8,
        9..=16 => 16,
        17..=32 => 32,
        33..=64 => 64,
        _ => panic!("Unsupported bit width: {}", bit_width.get()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::{array::Array, datatypes::UInt32Type};

    #[test]
    fn test_bit_pack_roundtrip() {
        // Test with a full chunk (1024 elements)
        let values: Vec<u32> = (0..1024).collect();

        let array = PrimitiveArray::<UInt32Type>::from(values);
        let before_size = array.get_array_memory_size();
        let bit_packed = BitPackedArray::from_primitive(array, NonZero::new(10).unwrap());
        let after_size = bit_packed.get_array_memory_size();
        println!("before: {}, after: {}", before_size, after_size);
        let unpacked = bit_packed.to_primitive();

        assert_eq!(unpacked.len(), 1024);
        for i in 0..1024 {
            assert_eq!(unpacked.value(i), i as u32);
        }
    }

    #[test]
    fn test_bit_pack_partial_chunk() {
        // Test with a partial chunk (500 elements)
        let values: Vec<u32> = (0..500).collect();
        let array = PrimitiveArray::<UInt32Type>::from(values);
        let bit_packed = BitPackedArray::from_primitive(array, NonZero::new(10).unwrap());
        let unpacked = bit_packed.to_primitive();

        assert_eq!(unpacked.len(), 500);
        for i in 0..500 {
            assert_eq!(unpacked.value(i), i as u32);
        }
    }

    #[test]
    fn test_bit_pack_multiple_chunks() {
        // Test with multiple chunks (2048 elements = 2 full chunks)
        let values: Vec<u32> = (0..2048).collect();
        let array = PrimitiveArray::<UInt32Type>::from(values);
        let bit_packed = BitPackedArray::from_primitive(array, NonZero::new(11).unwrap());
        let unpacked = bit_packed.to_primitive();

        assert_eq!(unpacked.len(), 2048);
        for i in 0..2048 {
            assert_eq!(unpacked.value(i), i as u32);
        }
    }

    #[test]
    fn test_bit_pack_with_nulls() {
        let values: Vec<Option<u32>> = (0..1000)
            .map(|i| if i % 2 == 0 { Some(i as u32) } else { None })
            .collect();
        let array = PrimitiveArray::<UInt32Type>::from(values);
        let bit_packed = BitPackedArray::from_primitive(array, NonZero::new(10).unwrap());
        let unpacked = bit_packed.to_primitive();

        assert_eq!(unpacked.len(), 1000);
        for i in 0..1000 {
            if i % 2 == 0 {
                assert_eq!(unpacked.value(i), i as u32);
            } else {
                assert!(unpacked.is_null(i));
            }
        }
    }

    #[test]
    fn test_different_bit_widths() {
        // Test with different bit widths
        let values: Vec<u32> = (0..100).map(|i| i * 2).collect();
        let array = PrimitiveArray::<UInt32Type>::from(values);

        for bit_width in [8, 16, 24, 32] {
            let bit_packed =
                BitPackedArray::from_primitive(array.clone(), NonZero::new(bit_width).unwrap());
            let unpacked = bit_packed.to_primitive();

            assert_eq!(unpacked.len(), 100);
            for i in 0..100 {
                assert_eq!(unpacked.value(i), i as u32 * 2);
            }
        }
    }
}
