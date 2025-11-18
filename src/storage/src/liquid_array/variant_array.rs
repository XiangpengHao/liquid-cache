use std::sync::Arc;

use arrow::array::{Array, ArrayRef, BinaryViewArray, StructArray};
use arrow::buffer::NullBuffer;
use arrow::error::ArrowError;
use arrow::ipc::{reader::StreamReader, writer::StreamWriter};
use arrow::record_batch::RecordBatch;
use arrow_schema::{DataType, Field, Fields, Schema};
use bytes::Bytes;
use std::io::Cursor;

use crate::liquid_array::{
    HybridBacking, IoRange, LiquidArray, LiquidArrayRef, LiquidDataType, LiquidHybridArray,
};

/// Hybrid representation for variant arrays that contain multiple typed fields.
#[derive(Debug)]
pub struct VariantStructHybridArray {
    metadata: Arc<BinaryViewArray>,
    typed_struct: Arc<StructArray>,
    nulls: Option<NullBuffer>,
    original_arrow_type: DataType,
}

impl VariantStructHybridArray {
    /// Create a hybrid representation that keeps the metadata and typed variant columns resident
    /// while dropping the untyped `value` column.
    pub fn new(
        metadata: Arc<BinaryViewArray>,
        typed_struct: Arc<StructArray>,
        nulls: Option<NullBuffer>,
        original_arrow_type: DataType,
    ) -> Self {
        Self {
            metadata,
            typed_struct,
            nulls,
            original_arrow_type,
        }
    }

    fn build_root_struct(&self) -> StructArray {
        let len = self.typed_struct.len();
        let metadata_field = Arc::new(Field::new("metadata", DataType::BinaryView, false));
        let value_field = Arc::new(Field::new("value", DataType::BinaryView, true));
        let typed_field = Arc::new(Field::new(
            "typed_value",
            self.typed_struct.data_type().clone(),
            true,
        ));
        let placeholder = Arc::new(BinaryViewArray::from(vec![None::<&[u8]>; len])) as ArrayRef;

        StructArray::new(
            Fields::from(vec![metadata_field, value_field, typed_field]),
            vec![
                self.metadata.clone() as ArrayRef,
                placeholder,
                self.typed_struct.clone() as ArrayRef,
            ],
            self.nulls.clone(),
        )
    }
}

impl LiquidHybridArray for VariantStructHybridArray {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn get_array_memory_size(&self) -> usize {
        self.metadata.get_array_memory_size() + self.typed_struct.get_array_memory_size()
    }

    fn len(&self) -> usize {
        self.typed_struct.len()
    }

    fn to_arrow_array(&self) -> Result<ArrayRef, IoRange> {
        Ok(Arc::new(self.build_root_struct()) as ArrayRef)
    }

    fn data_type(&self) -> LiquidDataType {
        LiquidDataType::ByteArray
    }

    fn original_arrow_data_type(&self) -> DataType {
        self.original_arrow_type.clone()
    }

    fn to_bytes(&self) -> Result<Vec<u8>, IoRange> {
        serialize_variant_array(&(Arc::new(self.build_root_struct()) as ArrayRef))
            .map(|bytes| bytes.to_vec())
            .map_err(|_| IoRange { range: 0..0 })
    }

    fn soak(&self, data: Bytes) -> LiquidArrayRef {
        let mut reader =
            StreamReader::try_new(Cursor::new(data), None).expect("invalid variant IPC stream");
        let batch = reader
            .next()
            .expect("variant IPC batch")
            .expect("read variant batch");
        let column = batch.column(0).clone();
        Arc::new(VariantStructLiquidArray::new(
            column,
            self.original_arrow_type.clone(),
        )) as LiquidArrayRef
    }

    fn to_liquid(&self) -> IoRange {
        IoRange { range: 0..0 }
    }

    fn disk_backing(&self) -> HybridBacking {
        HybridBacking::Arrow
    }
}

#[derive(Debug)]
struct VariantStructLiquidArray {
    array: ArrayRef,
    original_arrow_type: DataType,
}

impl VariantStructLiquidArray {
    fn new(array: ArrayRef, original_arrow_type: DataType) -> Self {
        Self {
            array,
            original_arrow_type,
        }
    }
}

impl LiquidArray for VariantStructLiquidArray {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn get_array_memory_size(&self) -> usize {
        self.array.get_array_memory_size()
    }

    fn len(&self) -> usize {
        self.array.len()
    }

    fn to_arrow_array(&self) -> ArrayRef {
        self.array.clone()
    }

    fn data_type(&self) -> LiquidDataType {
        LiquidDataType::ByteArray
    }

    fn original_arrow_data_type(&self) -> DataType {
        self.original_arrow_type.clone()
    }

    fn to_bytes(&self) -> Vec<u8> {
        serialize_variant_array(&self.array)
            .expect("failed to serialize variant struct")
            .to_vec()
    }
}

fn serialize_variant_array(array: &ArrayRef) -> Result<Bytes, ArrowError> {
    let field = Arc::new(Field::new(
        "column",
        array.data_type().clone(),
        array.null_count() > 0,
    ));
    let schema = Arc::new(Schema::new(vec![field]));
    let batch = RecordBatch::try_new(schema.clone(), vec![array.clone()])?;
    let mut buffer = Vec::new();
    {
        let mut writer = StreamWriter::try_new(&mut buffer, &schema)?;
        writer.write(&batch)?;
        writer.finish()?;
    }
    Ok(Bytes::from(buffer))
}
