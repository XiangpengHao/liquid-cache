//! IPC for liquid array.

use std::mem::size_of;
use std::sync::Arc;

use arrow::array::ArrowPrimitiveType;
use arrow::datatypes::{
    Date32Type, Date64Type, Float32Type, Float64Type, Int8Type, Int16Type, Int32Type, Int64Type,
    TimestampMicrosecondType, TimestampMillisecondType, TimestampNanosecondType,
    TimestampSecondType, UInt8Type, UInt16Type, UInt32Type, UInt64Type,
};
use bytes::Bytes;
use fsst::Compressor;

use crate::liquid_array::LiquidByteViewArray;
use crate::liquid_array::LiquidPrimitiveArray;
use crate::liquid_array::raw::FsstArray;

use super::linear_integer_array::LiquidLinearArray;
use super::{
    LiquidArrayRef, LiquidByteArray, LiquidDataType, LiquidFixedLenByteArray, LiquidFloatArray,
};

const MAGIC: u32 = 0x4C51_4441; // "LQDA" for LiQuid Data Array
const VERSION: u16 = 1;

macro_rules! primitive_physical_type_entries {
    ($macro:ident) => {
        $macro!([
            (Int8, Int8Type, 0, Integer),
            (Int16, Int16Type, 1, Integer),
            (Int32, Int32Type, 2, Integer),
            (Int64, Int64Type, 3, Integer),
            (UInt8, UInt8Type, 4, Integer),
            (UInt16, UInt16Type, 5, Integer),
            (UInt32, UInt32Type, 6, Integer),
            (UInt64, UInt64Type, 7, Integer),
            (Float32, Float32Type, 8, Float),
            (Float64, Float64Type, 9, Float),
            (Date32, Date32Type, 10, Integer),
            (Date64, Date64Type, 11, Integer),
            (TimestampSecond, TimestampSecondType, 12, Integer),
            (TimestampMillisecond, TimestampMillisecondType, 13, Integer),
            (TimestampMicrosecond, TimestampMicrosecondType, 14, Integer),
            (TimestampNanosecond, TimestampNanosecondType, 15, Integer)
        ]);
    };
}

macro_rules! physical_type_integer_body {
    (Integer, $arrow_ty:ty, $bytes:expr, $self:expr) => {
        Arc::new(LiquidPrimitiveArray::<$arrow_ty>::from_bytes($bytes)) as LiquidArrayRef
    };
    (Float, $arrow_ty:ty, $bytes:expr, $self:expr) => {
        panic!(
            "Physical type {:?} cannot be decoded as an integer array",
            $self
        )
    };
}

macro_rules! physical_type_linear_body {
    (Integer, $arrow_ty:ty, $bytes:expr, $self:expr) => {
        Arc::new(LiquidLinearArray::<$arrow_ty>::from_bytes($bytes)) as LiquidArrayRef
    };
    (Float, $arrow_ty:ty, $bytes:expr, $self:expr) => {
        panic!(
            "Physical type {:?} cannot be decoded as a linear integer array",
            $self
        )
    };
}

macro_rules! physical_type_float_body {
    (Float, $arrow_ty:ty, $bytes:expr, $self:expr) => {
        Arc::new(LiquidFloatArray::<$arrow_ty>::from_bytes($bytes)) as LiquidArrayRef
    };
    (Integer, $arrow_ty:ty, $bytes:expr, $self:expr) => {
        panic!(
            "Physical type {:?} cannot be decoded as a float array",
            $self
        )
    };
}

macro_rules! define_physical_types {
    ( [ $(($variant:ident, $arrow_ty:ty, $id:expr, $category:ident)),+ $(,)? ] ) => {
        /// Physical primitive types supported by Liquid IPC.
        #[derive(Debug, Clone, Copy, PartialEq, Eq)]
        #[allow(missing_docs)]
        #[repr(u16)]
        pub enum PrimitivePhysicalType {
            $( $variant = $id, )+
        }

        /// Marker trait implemented for Arrow primitive types that have a Liquid physical ID.
        pub trait PhysicalTypeMarker: ArrowPrimitiveType {
            /// The physical type associated with the Arrow primitive.
            const PHYSICAL_TYPE: PrimitivePhysicalType;
        }

        $(impl PhysicalTypeMarker for $arrow_ty {
            const PHYSICAL_TYPE: PrimitivePhysicalType = PrimitivePhysicalType::$variant;
        })+

        impl PrimitivePhysicalType {
            fn from_arrow_type<T>() -> PrimitivePhysicalType
            where
                T: ArrowPrimitiveType + PhysicalTypeMarker,
            {
                T::PHYSICAL_TYPE
            }

            fn deserialize_integer(self, bytes: Bytes) -> LiquidArrayRef {
                match self {
                    $( PrimitivePhysicalType::$variant => {
                        physical_type_integer_body!($category, $arrow_ty, bytes, self)
                    }, )+
                }
            }

            fn deserialize_linear_integer(self, bytes: Bytes) -> LiquidArrayRef {
                match self {
                    $( PrimitivePhysicalType::$variant => {
                        physical_type_linear_body!($category, $arrow_ty, bytes, self)
                    }, )+
                }
            }

            fn deserialize_float(self, bytes: Bytes) -> LiquidArrayRef {
                match self {
                    $( PrimitivePhysicalType::$variant => {
                        physical_type_float_body!($category, $arrow_ty, bytes, self)
                    }, )+
                }
            }
        }

        impl TryFrom<u16> for PrimitivePhysicalType {
            type Error = u16;

            fn try_from(value: u16) -> Result<Self, Self::Error> {
                match value {
                    $( $id => Ok(PrimitivePhysicalType::$variant), )+
                    _ => Err(value),
                }
            }
        }
    };
}

primitive_physical_type_entries!(define_physical_types);

fn expect_physical_type(id: u16, label: &str) -> PrimitivePhysicalType {
    PrimitivePhysicalType::try_from(id)
        .unwrap_or_else(|value| panic!("Unsupported {label} physical type: {value}"))
}

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
pub(super) struct LiquidIPCHeader {
    pub(super) magic: [u8; 4],
    pub(super) version: u16,
    pub(super) logical_type_id: u16,
    pub(super) physical_type_id: u16,
    pub(super) __padding: [u8; 6],
}

const _: () = assert!(size_of::<LiquidIPCHeader>() == LiquidIPCHeader::size());

impl LiquidIPCHeader {
    pub(super) const fn size() -> usize {
        16
    }

    pub(super) fn new(logical_type_id: u16, physical_type_id: u16) -> Self {
        Self {
            magic: MAGIC.to_le_bytes(),
            version: VERSION,
            logical_type_id,
            physical_type_id,
            __padding: [0; 6],
        }
    }

    pub(super) fn to_bytes(&self) -> [u8; Self::size()] {
        let mut bytes = [0; Self::size()];
        bytes[0..4].copy_from_slice(&self.magic);
        bytes[4..6].copy_from_slice(&self.version.to_le_bytes());
        bytes[6..8].copy_from_slice(&self.logical_type_id.to_le_bytes());
        bytes[8..10].copy_from_slice(&self.physical_type_id.to_le_bytes());
        bytes
    }

    pub(super) fn from_bytes(bytes: &[u8]) -> Self {
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

        if magic != MAGIC.to_le_bytes() {
            panic!("Invalid magic number");
        }
        if version != VERSION {
            panic!("Unsupported version");
        }

        Self {
            magic,
            version,
            logical_type_id,
            physical_type_id,
            __padding: [0; 6],
        }
    }
}

/// Context for liquid IPC.
pub struct LiquidIPCContext {
    compressor: Option<Arc<Compressor>>,
}

impl LiquidIPCContext {
    /// Create a new instance of LiquidIPCContext.
    pub fn new(compressor: Option<Arc<Compressor>>) -> Self {
        Self { compressor }
    }
}

/// Read a liquid array from bytes.
pub fn read_from_bytes(bytes: Bytes, context: &LiquidIPCContext) -> LiquidArrayRef {
    let header = LiquidIPCHeader::from_bytes(&bytes);
    let logical_type = LiquidDataType::from(header.logical_type_id);
    match logical_type {
        LiquidDataType::Integer => {
            let physical_type = expect_physical_type(header.physical_type_id, "integer");
            physical_type.deserialize_integer(bytes)
        }
        LiquidDataType::ByteArray => {
            let compressor = context.compressor.as_ref().expect("Expected a compressor");
            Arc::new(LiquidByteArray::from_bytes(bytes, compressor.clone()))
        }
        LiquidDataType::ByteViewArray => {
            let compressor = context.compressor.as_ref().expect("Expected a compressor");
            Arc::new(LiquidByteViewArray::<FsstArray>::from_bytes(
                bytes,
                compressor.clone(),
            ))
        }
        LiquidDataType::Float => {
            let physical_type = expect_physical_type(header.physical_type_id, "float");
            physical_type.deserialize_float(bytes)
        }
        LiquidDataType::FixedLenByteArray => {
            let compressor = context.compressor.as_ref().expect("Expected a compressor");
            Arc::new(LiquidFixedLenByteArray::from_bytes(
                bytes,
                compressor.clone(),
            ))
        }
        LiquidDataType::LinearInteger => {
            let physical_type = expect_physical_type(header.physical_type_id, "linear-integer");
            physical_type.deserialize_linear_integer(bytes)
        }
    }
}

pub(super) fn get_physical_type_id<T>() -> u16
where
    T: ArrowPrimitiveType + PhysicalTypeMarker,
{
    PrimitivePhysicalType::from_arrow_type::<T>() as u16
}

#[cfg(test)]
mod tests {
    use arrow::{
        array::{AsArray, BinaryViewArray, PrimitiveArray, StringArray},
        datatypes::{
            Decimal128Type, Decimal256Type, DecimalType, Int32Type, TimestampMicrosecondType,
            TimestampMillisecondType, TimestampNanosecondType, TimestampSecondType, i256,
        },
    };
    use arrow_schema::DataType;

    use crate::liquid_array::raw::FsstArray;
    use crate::liquid_array::{LiquidArray, utils::gen_test_decimal_array};

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
        let no_nulls: Vec<Option<i32>> = (0..1000).map(Some).collect();
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
    fn test_date_types_ipc_roundtrip() {
        // Test Date32Type
        let date32_array = PrimitiveArray::<Date32Type>::from(vec![Some(18628), None, Some(0)]);
        let liquid_array =
            LiquidPrimitiveArray::<Date32Type>::from_arrow_array(date32_array.clone());
        let bytes = Bytes::from(liquid_array.to_bytes());
        let context = LiquidIPCContext::new(None);
        let deserialized = read_from_bytes(bytes, &context);
        assert_eq!(deserialized.to_arrow_array().as_ref(), &date32_array);

        // Test Date64Type
        let date64_array =
            PrimitiveArray::<Date64Type>::from(vec![Some(1609459200000), None, Some(0)]);
        let liquid_array =
            LiquidPrimitiveArray::<Date64Type>::from_arrow_array(date64_array.clone());
        let bytes = Bytes::from(liquid_array.to_bytes());
        let context = LiquidIPCContext::new(None);
        let deserialized = read_from_bytes(bytes, &context);
        assert_eq!(deserialized.to_arrow_array().as_ref(), &date64_array);
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

        // Verify the original arrow type is preserved via Arrow array types
        assert_eq!(
            original.to_arrow_array().data_type(),
            deserialized.to_arrow_array().data_type()
        );
    }

    #[test]
    fn test_ipc_roundtrip_utf8_for_both_byte_and_view() {
        let input = StringArray::from(vec![
            Some("hello"),
            Some("world"),
            None,
            Some("liquid"),
            Some("byte"),
            Some("array"),
            Some("hello"),
        ]);

        // LiquidByteArray
        let compressor_ba = LiquidByteArray::train_compressor(input.iter());
        let original_ba = LiquidByteArray::from_string_array(&input, compressor_ba.clone());
        let bytes_ba = Bytes::from(original_ba.to_bytes());
        let deserialized_ba = LiquidByteArray::from_bytes(bytes_ba, compressor_ba);
        let output_ba = deserialized_ba.to_arrow_array();
        assert_eq!(output_ba.as_string::<i32>(), &input);

        // LiquidByteViewArray
        let compressor_bv = LiquidByteViewArray::<FsstArray>::train_compressor(input.iter());
        let original_bv =
            LiquidByteViewArray::<FsstArray>::from_string_array(&input, compressor_bv.clone());
        let bytes_bv = Bytes::from(original_bv.to_bytes());
        let deserialized_bv = LiquidByteViewArray::<FsstArray>::from_bytes(bytes_bv, compressor_bv);
        let output_bv = deserialized_bv.to_arrow_array().unwrap();
        assert_eq!(output_bv.as_string::<i32>(), &input);
    }

    #[test]
    fn test_ipc_roundtrip_binaryview_for_both_byte_and_view() {
        let input = BinaryViewArray::from(vec![
            Some(b"hello".as_slice()),
            Some(b"world".as_slice()),
            Some(b"hello".as_slice()),
            Some(b"rust\x00".as_slice()),
            None,
            Some(b"This is a very long string that should be compressed well"),
            Some(b""),
            Some(b"This is a very long string that should be compressed well"),
        ]);

        // LiquidByteArray via BinaryView
        let (compressor_ba, original_ba) = LiquidByteArray::train_from_binary_view(&input);
        let bytes_ba = Bytes::from(original_ba.to_bytes());
        let deserialized_ba = LiquidByteArray::from_bytes(bytes_ba, compressor_ba);
        let output_ba = deserialized_ba.to_arrow_array();
        assert_eq!(output_ba.as_binary_view(), &input);

        // LiquidByteViewArray via BinaryView
        let (compressor_bv, original_bv) =
            LiquidByteViewArray::<FsstArray>::train_from_binary_view(&input);
        let bytes_bv = Bytes::from(original_bv.to_bytes());
        let deserialized_bv = LiquidByteViewArray::<FsstArray>::from_bytes(bytes_bv, compressor_bv);
        let output_bv = deserialized_bv.to_arrow_array().unwrap();
        assert_eq!(output_bv.as_binary_view(), &input);
    }

    #[test]
    fn test_float32_array_roundtrip() {
        let arr = PrimitiveArray::<Float32Type>::from(vec![
            Some(-1.3e7),
            Some(1.9),
            Some(6.6e4),
            None,
            Some(9.1e-5),
        ]);
        let original = LiquidFloatArray::<Float32Type>::from_arrow_array(arr.clone());
        let serialized = Bytes::from(original.to_bytes_inner());
        let deserialized = LiquidFloatArray::<Float32Type>::from_bytes(serialized).to_arrow_array();
        assert_eq!(deserialized.as_ref(), &arr);
    }

    #[test]
    fn test_float64_array_roundtrip() {
        let arr = PrimitiveArray::<Float64Type>::from(vec![
            Some(-1.3e7),
            Some(1.9),
            Some(6.6e4),
            None,
            Some(9.1e-5),
        ]);
        let original = LiquidFloatArray::<Float64Type>::from_arrow_array(arr.clone());
        let serialized = Bytes::from(original.to_bytes_inner());
        let deserialized = LiquidFloatArray::<Float64Type>::from_bytes(serialized).to_arrow_array();
        assert_eq!(deserialized.as_ref(), &arr);
    }

    fn test_decimal_roundtrip<T: DecimalType>(data_type: DataType) {
        let original_array = gen_test_decimal_array::<T>(data_type);
        let (compressor, liquid_array) =
            LiquidFixedLenByteArray::train_from_decimal_array(&original_array);

        let bytes = liquid_array.to_bytes_inner();
        let bytes = Bytes::from(bytes);
        let deserialized = LiquidFixedLenByteArray::from_bytes(bytes, compressor);
        let deserialized_arrow = deserialized.to_arrow_array();
        assert_eq!(deserialized_arrow.as_ref(), &original_array);
    }

    #[test]
    fn test_decimal128_array_roundtrip() {
        test_decimal_roundtrip::<Decimal128Type>(DataType::Decimal128(10, 2));
    }

    #[test]
    fn test_decimal256_array_roundtrip() {
        test_decimal_roundtrip::<Decimal256Type>(DataType::Decimal256(38, 6));
    }

    #[test]
    fn test_fixed_len_byte_array_ipc_roundtrip() {
        // Test both Decimal128 and Decimal256 through the full IPC pipeline

        let decimal128_array =
            gen_test_decimal_array::<Decimal128Type>(DataType::Decimal128(15, 3));
        let (compressor, liquid_array) =
            LiquidFixedLenByteArray::train_from_decimal_array(&decimal128_array);

        let bytes = liquid_array.to_bytes();
        let bytes = Bytes::from(bytes);

        let context = LiquidIPCContext::new(Some(compressor.clone()));
        let deserialized_ref = read_from_bytes(bytes, &context);
        assert!(matches!(
            deserialized_ref.data_type(),
            LiquidDataType::FixedLenByteArray
        ));
        let result_arrow = deserialized_ref.to_arrow_array();
        assert_eq!(result_arrow.as_ref(), &decimal128_array);

        // Test Decimal256
        let decimal256_array =
            gen_test_decimal_array::<Decimal256Type>(DataType::Decimal256(38, 6));
        let (compressor, liquid_array) =
            LiquidFixedLenByteArray::train_from_decimal_array(&decimal256_array);

        let bytes = liquid_array.to_bytes();
        let bytes = Bytes::from(bytes);

        let context = LiquidIPCContext::new(Some(compressor.clone()));
        let deserialized_ref = read_from_bytes(bytes, &context);

        assert!(matches!(
            deserialized_ref.data_type(),
            LiquidDataType::FixedLenByteArray
        ));

        let result_arrow = deserialized_ref.to_arrow_array();
        assert_eq!(result_arrow.as_ref(), &decimal256_array);
    }

    #[test]
    fn test_fixed_len_byte_array_ipc_edge_cases() {
        // Test edge cases with FixedLenByteArray IPC

        let mut builder = arrow::array::Decimal128Builder::new();
        builder.append_value(123456789_i128);
        builder.append_null();
        builder.append_value(-987654321_i128);
        builder.append_null();
        builder.append_value(0_i128);
        let array_with_nulls = builder.finish().with_precision_and_scale(15, 3).unwrap();

        let (compressor, liquid_array) =
            LiquidFixedLenByteArray::train_from_decimal_array(&array_with_nulls);

        let bytes = liquid_array.to_bytes();
        let bytes = Bytes::from(bytes);

        let context = LiquidIPCContext::new(Some(compressor));
        let deserialized_ref = read_from_bytes(bytes, &context);
        let result_arrow = deserialized_ref.to_arrow_array();

        assert_eq!(result_arrow.as_ref(), &array_with_nulls);

        // Test with single value
        let mut builder = arrow::array::Decimal256Builder::new();
        builder.append_value(i256::from_i128(42_i128));
        let single_value_array = builder.finish().with_precision_and_scale(38, 6).unwrap();

        let (compressor, liquid_array) =
            LiquidFixedLenByteArray::train_from_decimal_array(&single_value_array);

        let bytes = liquid_array.to_bytes();
        let bytes = Bytes::from(bytes);

        let context = LiquidIPCContext::new(Some(compressor));
        let deserialized_ref = read_from_bytes(bytes, &context);
        let result_arrow = deserialized_ref.to_arrow_array();

        assert_eq!(result_arrow.as_ref(), &single_value_array);
    }

    #[test]
    fn test_timestamp_physical_type_ids() {
        assert_eq!(get_physical_type_id::<TimestampSecondType>(), 12);
        assert_eq!(get_physical_type_id::<TimestampMillisecondType>(), 13);
        assert_eq!(get_physical_type_id::<TimestampMicrosecondType>(), 14);
        assert_eq!(get_physical_type_id::<TimestampNanosecondType>(), 15);
    }
}
