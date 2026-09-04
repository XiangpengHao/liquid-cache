use bytes::Bytes;
use std::{any::Any, mem::size_of, sync::Arc};

use arrow::array::{Array, ArrayRef, PrimitiveArray};
use arrow::buffer::ScalarBuffer;
use arrow::datatypes::{Decimal128Type, Decimal256Type, DecimalType, UInt64Type, i256};
use arrow_schema::DataType;
use num_traits::ToPrimitive;

use super::{LiquidArray, LiquidDataType};
use crate::liquid_array::ipc::{LiquidIPCHeader, get_physical_type_id};
use crate::liquid_array::raw::BitPackedArray;
use crate::utils::get_bit_width;

#[derive(Debug, Clone, Copy)]
struct DecimalMeta {
    precision: u8,
    scale: i8,
    is_256: bool,
}

impl DecimalMeta {
    fn from_data_type(data_type: &DataType) -> Self {
        match data_type {
            DataType::Decimal128(precision, scale) => Self {
                precision: *precision,
                scale: *scale,
                is_256: false,
            },
            DataType::Decimal256(precision, scale) => Self {
                precision: *precision,
                scale: *scale,
                is_256: true,
            },
            _ => panic!("unsupported decimal data type: {data_type:?}"),
        }
    }

    fn data_type(&self) -> DataType {
        if self.is_256 {
            DataType::Decimal256(self.precision, self.scale)
        } else {
            DataType::Decimal128(self.precision, self.scale)
        }
    }

    fn arrow_code(&self) -> u8 {
        if self.is_256 { 1 } else { 0 }
    }
}

#[repr(C)]
struct DecimalArrayHeader {
    arrow_type: u8, // 0 for Decimal128, 1 for Decimal256
    precision: u8,
    scale: i8,
    __padding: u8,
    __reserved: u32,
}

impl DecimalArrayHeader {
    const fn size() -> usize {
        8
    }

    fn from_meta(meta: DecimalMeta) -> Self {
        Self {
            arrow_type: meta.arrow_code(),
            precision: meta.precision,
            scale: meta.scale,
            __padding: 0,
            __reserved: 0,
        }
    }

    fn to_bytes(&self) -> [u8; Self::size()] {
        let mut bytes = [0; Self::size()];
        bytes[0] = self.arrow_type;
        bytes[1] = self.precision;
        bytes[2] = self.scale as u8;
        bytes
    }

    fn from_bytes(bytes: &[u8]) -> Self {
        if bytes.len() < Self::size() {
            panic!(
                "value too small for DecimalArrayHeader, expected at least {} bytes, got {}",
                Self::size(),
                bytes.len()
            );
        }
        Self {
            arrow_type: bytes[0],
            precision: bytes[1],
            scale: bytes[2] as i8,
            __padding: 0,
            __reserved: 0,
        }
    }
}

/// Liquid decimal array stored as a compressed u64 primitive.
#[derive(Debug)]
pub struct LiquidDecimalArray {
    meta: DecimalMeta,
    bit_packed: BitPackedArray<UInt64Type>,
    reference_value: u64,
}

impl LiquidDecimalArray {
    pub(crate) fn fits_u64<T: DecimalType>(array: &PrimitiveArray<T>) -> bool
    where
        T::Native: ToPrimitive,
    {
        array.iter().flatten().all(|v| v.to_u64().is_some())
    }

    pub(crate) fn from_decimal_array<T: DecimalType>(array: &PrimitiveArray<T>) -> Self
    where
        T::Native: ToPrimitive,
    {
        debug_assert!(Self::fits_u64(array));
        let meta = DecimalMeta::from_data_type(array.data_type());
        if array.null_count() == array.len() {
            return Self {
                meta,
                bit_packed: BitPackedArray::new_null_array(array.len()),
                reference_value: 0,
            };
        }

        let nulls = array.nulls().cloned();
        let mut min = u64::MAX;
        let mut max = 0u64;
        let values: Vec<u64> = array
            .iter()
            .map(|v| match v {
                Some(v) => {
                    let value = v.to_u64().expect("decimal fits u64");
                    if value < min {
                        min = value;
                    }
                    if value > max {
                        max = value;
                    }
                    value
                }
                None => 0,
            })
            .collect();

        let bit_width = get_bit_width(max - min);
        let offsets = ScalarBuffer::from_iter(values.iter().map(|v| v.saturating_sub(min)));
        let unsigned_array = PrimitiveArray::<UInt64Type>::new(offsets, nulls);
        let bit_packed = BitPackedArray::from_primitive(unsigned_array, bit_width);

        Self {
            meta,
            bit_packed,
            reference_value: min,
        }
    }

    fn bit_pack_starting_loc() -> usize {
        let header_size = LiquidIPCHeader::size() + DecimalArrayHeader::size();
        (header_size + size_of::<u64>() + 7) & !7
    }

    fn to_u64_array(&self) -> PrimitiveArray<UInt64Type> {
        let unsigned_array = self.bit_packed.to_primitive();
        let (_data_type, values, _nulls) = unsigned_array.into_parts();
        let nulls = self.bit_packed.nulls();
        let values = if self.reference_value != 0 {
            let reference_value = self.reference_value;
            ScalarBuffer::from_iter(values.iter().map(|v| v.wrapping_add(reference_value)))
        } else {
            values
        };
        PrimitiveArray::<UInt64Type>::new(values, nulls.cloned())
    }

    pub(crate) fn to_bytes_inner(&self) -> Vec<u8> {
        let header_size = LiquidIPCHeader::size() + DecimalArrayHeader::size();
        let mut result = Vec::with_capacity(Self::bit_pack_starting_loc() + 256);
        result.resize(header_size, 0);

        let logical_type_id = LiquidDataType::Decimal as u16;
        let physical_type_id = get_physical_type_id::<UInt64Type>();
        let ipc_header = LiquidIPCHeader::new(logical_type_id, physical_type_id);
        result[0..LiquidIPCHeader::size()].copy_from_slice(&ipc_header.to_bytes());

        let decimal_header = DecimalArrayHeader::from_meta(self.meta);
        result[LiquidIPCHeader::size()..header_size].copy_from_slice(&decimal_header.to_bytes());

        result.extend_from_slice(&self.reference_value.to_le_bytes());
        while result.len() < Self::bit_pack_starting_loc() {
            result.push(0);
        }
        self.bit_packed.to_bytes(&mut result);
        result
    }

    pub(crate) fn from_bytes(bytes: Bytes) -> Self {
        let header_size = LiquidIPCHeader::size() + DecimalArrayHeader::size();
        let header = LiquidIPCHeader::from_bytes(&bytes);

        assert_eq!(header.logical_type_id, LiquidDataType::Decimal as u16);
        assert_eq!(
            header.physical_type_id,
            get_physical_type_id::<UInt64Type>()
        );

        let decimal_header =
            DecimalArrayHeader::from_bytes(&bytes[LiquidIPCHeader::size()..header_size]);
        let meta = DecimalMeta {
            precision: decimal_header.precision,
            scale: decimal_header.scale,
            is_256: match decimal_header.arrow_type {
                0 => false,
                1 => true,
                _ => panic!(
                    "unsupported decimal type code: {}",
                    decimal_header.arrow_type
                ),
            },
        };

        let ref_start = header_size;
        let ref_end = ref_start + size_of::<u64>();
        let reference_value = u64::from_le_bytes(bytes[ref_start..ref_end].try_into().unwrap());

        let bit_packed_data = bytes.slice(Self::bit_pack_starting_loc()..);
        let bit_packed = BitPackedArray::<UInt64Type>::from_bytes(bit_packed_data);

        Self {
            meta,
            bit_packed,
            reference_value,
        }
    }
}

impl LiquidArray for LiquidDecimalArray {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn get_array_memory_size(&self) -> usize {
        self.bit_packed.get_array_memory_size() + size_of::<u64>() + size_of::<DecimalMeta>()
    }

    fn len(&self) -> usize {
        self.bit_packed.len()
    }

    fn to_arrow_array(&self) -> ArrayRef {
        let u64_array = self.to_u64_array();
        let (_data_type, values, nulls) = u64_array.into_parts();
        let data_type = self.meta.data_type();
        if self.meta.is_256 {
            let values_i256 =
                ScalarBuffer::from_iter(values.iter().map(|v| i256::from_i128(*v as i128)));
            let array = PrimitiveArray::<Decimal256Type>::new(values_i256, nulls);
            Arc::new(array.with_data_type(data_type))
        } else {
            let values_i128 = ScalarBuffer::from_iter(values.iter().map(|v| *v as i128));
            let array = PrimitiveArray::<Decimal128Type>::new(values_i128, nulls);
            Arc::new(array.with_data_type(data_type))
        }
    }

    fn original_arrow_data_type(&self) -> DataType {
        self.meta.data_type()
    }

    fn to_bytes(&self) -> Vec<u8> {
        self.to_bytes_inner()
    }

    fn data_type(&self) -> LiquidDataType {
        LiquidDataType::Decimal
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::Decimal128Builder;

    #[test]
    fn decimal_u64_roundtrip() {
        let mut builder = Decimal128Builder::new();
        builder.append_value(100_i128);
        builder.append_null();
        builder.append_value(250_i128);
        let original = builder.finish().with_precision_and_scale(10, 2).unwrap();

        let liquid = LiquidDecimalArray::from_decimal_array(&original);
        let arrow = liquid.to_arrow_array();
        assert_eq!(arrow.as_ref(), &original);
    }

    #[test]
    fn decimal_u64_ipc_roundtrip() {
        let mut builder = Decimal128Builder::new();
        builder.append_value(12345_i128);
        builder.append_value(67890_i128);
        let original = builder.finish().with_precision_and_scale(12, 3).unwrap();

        let liquid = LiquidDecimalArray::from_decimal_array(&original);
        let bytes = liquid.to_bytes();
        let decoded = LiquidDecimalArray::from_bytes(bytes.into());
        let arrow = decoded.to_arrow_array();
        assert_eq!(arrow.as_ref(), &original);
    }
}
