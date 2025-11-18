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

/// Hybrid representation for a single top-level variant field.
#[derive(Debug)]
pub struct VariantExtractedArray {
    field_name: Arc<str>,
    values: LiquidArrayRef,
    metadata: Arc<BinaryViewArray>,
    nulls: Option<NullBuffer>,
    original_arrow_type: DataType,
}

impl VariantExtractedArray {
    /// Build a hybrid array for the provided top-level variant field.
    pub fn new(
        field_name: impl Into<Arc<str>>,
        values: LiquidArrayRef,
        metadata: Arc<BinaryViewArray>,
        nulls: Option<NullBuffer>,
        original_arrow_type: DataType,
    ) -> Self {
        Self {
            field_name: field_name.into(),
            values,
            metadata,
            nulls,
            original_arrow_type,
        }
    }

    /// Field name carried by this hybrid array.
    pub fn field(&self) -> &str {
        self.field_name.as_ref()
    }
}

impl LiquidHybridArray for VariantExtractedArray {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn get_array_memory_size(&self) -> usize {
        self.metadata.get_array_memory_size() + self.values.get_array_memory_size()
    }

    fn len(&self) -> usize {
        self.metadata.len()
    }

    fn to_arrow_array(&self) -> Result<ArrayRef, IoRange> {
        let arrow_values = self.values.to_arrow_array();
        let leaf_field = Arc::new(Field::new(
            "typed_value",
            arrow_values.data_type().clone(),
            arrow_values.null_count() > 0,
        ));
        let leaf_struct = Arc::new(StructArray::new(
            Fields::from(vec![leaf_field]),
            vec![arrow_values.clone()],
            arrow_values.nulls().cloned(),
        ));

        let named_field = Arc::new(Field::new(
            self.field_name.as_ref(),
            leaf_struct.data_type().clone(),
            true,
        ));
        let typed_struct = Arc::new(StructArray::new(
            Fields::from(vec![named_field]),
            vec![leaf_struct as ArrayRef],
            self.nulls.clone(),
        ));

        let value_placeholder =
            Arc::new(BinaryViewArray::from(vec![None::<&[u8]>; self.len()])) as ArrayRef;

        let root_fields = Fields::from(vec![
            Arc::new(Field::new("metadata", DataType::BinaryView, false)),
            Arc::new(Field::new("value", DataType::BinaryView, true)),
            Arc::new(Field::new(
                "typed_value",
                typed_struct.data_type().clone(),
                true,
            )),
        ]);

        Ok(Arc::new(StructArray::new(
            root_fields,
            vec![
                self.metadata.clone() as ArrayRef,
                value_placeholder,
                typed_struct,
            ],
            self.nulls.clone(),
        )))
    }

    fn data_type(&self) -> LiquidDataType {
        LiquidDataType::ByteArray
    }

    fn original_arrow_data_type(&self) -> DataType {
        self.original_arrow_type.clone()
    }

    fn to_bytes(&self) -> Result<Vec<u8>, IoRange> {
        Err(IoRange { range: 0..0 })
    }

    fn filter(&self, selection: &arrow::buffer::BooleanBuffer) -> Result<ArrayRef, IoRange> {
        let array = self.to_arrow_array()?;
        let selection = arrow::array::BooleanArray::new(selection.clone(), None);
        Ok(arrow::compute::filter(&array, &selection).unwrap())
    }

    fn try_eval_predicate(
        &self,
        _predicate: &Arc<dyn datafusion::physical_plan::PhysicalExpr>,
        _filter: &arrow::buffer::BooleanBuffer,
    ) -> Result<Option<arrow::array::BooleanArray>, IoRange> {
        Ok(None)
    }

    fn soak(&self, _data: bytes::Bytes) -> LiquidArrayRef {
        Arc::clone(&self.values)
    }

    fn to_liquid(&self) -> IoRange {
        IoRange { range: 0..0 }
    }

    fn disk_backing(&self) -> HybridBacking {
        HybridBacking::Arrow
    }
}

/// Hybrid representation for variant arrays that contain multiple typed fields.
#[derive(Debug)]
pub struct VariantStructHybridArray {
    metadata: Arc<BinaryViewArray>,
    value: ArrayRef,
    typed_struct: Arc<StructArray>,
    nulls: Option<NullBuffer>,
    original_arrow_type: DataType,
}

impl VariantStructHybridArray {
    /// Create a hybrid representation that keeps all typed variant columns resident.
    pub fn new(
        metadata: Arc<BinaryViewArray>,
        value: ArrayRef,
        typed_struct: Arc<StructArray>,
        nulls: Option<NullBuffer>,
        original_arrow_type: DataType,
    ) -> Self {
        Self {
            metadata,
            value,
            typed_struct,
            nulls,
            original_arrow_type,
        }
    }

    fn build_root_struct(&self) -> StructArray {
        let metadata_field = Arc::new(Field::new("metadata", DataType::BinaryView, false));
        let value_field = Arc::new(Field::new("value", self.value.data_type().clone(), true));
        let typed_field = Arc::new(Field::new(
            "typed_value",
            self.typed_struct.data_type().clone(),
            true,
        ));

        StructArray::new(
            Fields::from(vec![metadata_field, value_field, typed_field]),
            vec![
                self.metadata.clone() as ArrayRef,
                self.value.clone(),
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
        self.metadata.get_array_memory_size()
            + self.value.get_array_memory_size()
            + self.typed_struct.get_array_memory_size()
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
