use std::sync::Arc;

use arrow::array::{Array, ArrayRef, BinaryViewArray, StructArray};
use arrow::buffer::NullBuffer;
use arrow::error::ArrowError;
use arrow::ipc::writer::StreamWriter;
use arrow::record_batch::RecordBatch;
use arrow_schema::{DataType, Field, Fields, Schema};
use bytes::Bytes;

use crate::liquid_array::{
    HybridBacking, LiquidArrayRef, LiquidDataType, LiquidHybridArray, NeedsBacking,
};
use ahash::AHashMap;

/// Hybrid representation for variant arrays that contain multiple typed fields.
#[derive(Debug)]
pub struct VariantStructHybridArray {
    values: AHashMap<Arc<str>, LiquidArrayRef>,
    len: usize,
    nulls: Option<NullBuffer>,
    original_arrow_type: DataType,
}

impl VariantStructHybridArray {
    /// Create a hybrid representation that keeps only the typed variant columns resident.
    pub fn new(
        values: Vec<(Arc<str>, LiquidArrayRef)>,
        nulls: Option<NullBuffer>,
        original_arrow_type: DataType,
    ) -> Self {
        let len = values.first().map(|(_, array)| array.len()).unwrap_or(0);
        let mut map = AHashMap::with_capacity(values.len());
        for (path, array) in values {
            debug_assert_eq!(array.len(), len, "variant paths must share length");
            map.insert(path, array);
        }
        Self {
            values: map,
            len,
            nulls,
            original_arrow_type,
        }
    }

    fn build_root_struct(&self) -> StructArray {
        let metadata = Arc::new(BinaryViewArray::from_iter_values(std::iter::repeat_n(
            b"" as &[u8],
            self.len,
        ))) as ArrayRef;
        let value_placeholder =
            Arc::new(BinaryViewArray::from(vec![None::<&[u8]>; self.len])) as ArrayRef;
        let typed_struct = self.build_typed_struct();

        let metadata_field = Arc::new(Field::new("metadata", DataType::BinaryView, false));
        let value_field = Arc::new(Field::new("value", DataType::BinaryView, true));
        let typed_field = Arc::new(Field::new(
            "typed_value",
            typed_struct.data_type().clone(),
            true,
        ));

        StructArray::new(
            Fields::from(vec![metadata_field, value_field, typed_field]),
            vec![metadata, value_placeholder, typed_struct as ArrayRef],
            self.nulls.clone(),
        )
    }

    fn build_typed_struct(&self) -> Arc<StructArray> {
        let mut root = VariantTreeNode::new(self.len);
        for (path, array) in &self.values {
            let segments: Vec<&str> = path
                .split('.')
                .filter(|segment| !segment.is_empty())
                .collect();
            if segments.is_empty() {
                continue;
            }
            root.insert(&segments, array.to_arrow_array());
        }
        root.into_struct_array()
    }

    /// Returns true if the hybrid contains the provided variant path.
    pub fn contains_path(&self, path: &str) -> bool {
        self.values.contains_key(path)
    }
}

impl LiquidHybridArray for VariantStructHybridArray {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn get_array_memory_size(&self) -> usize {
        self.values
            .values()
            .map(|array| array.get_array_memory_size())
            .sum()
    }

    fn len(&self) -> usize {
        self.len
    }

    fn to_arrow_array(&self) -> Result<ArrayRef, NeedsBacking> {
        Ok(Arc::new(self.build_root_struct()) as ArrayRef)
    }

    fn data_type(&self) -> LiquidDataType {
        LiquidDataType::ByteArray
    }

    fn original_arrow_data_type(&self) -> DataType {
        self.original_arrow_type.clone()
    }

    fn to_bytes(&self) -> Result<Vec<u8>, NeedsBacking> {
        serialize_variant_array(&(Arc::new(self.build_root_struct()) as ArrayRef))
            .map(|bytes| bytes.to_vec())
            .map_err(|_| NeedsBacking)
    }

    fn disk_backing(&self) -> HybridBacking {
        HybridBacking::Arrow
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

#[derive(Default)]
struct VariantTreeNode {
    len: usize,
    leaf: Option<ArrayRef>,
    children: AHashMap<String, VariantTreeNode>,
}

impl VariantTreeNode {
    fn new(len: usize) -> Self {
        Self {
            len,
            leaf: None,
            children: AHashMap::new(),
        }
    }

    fn insert(&mut self, segments: &[&str], values: ArrayRef) {
        if segments.is_empty() {
            self.leaf = Some(values);
            return;
        }
        let (head, tail) = segments.split_first().unwrap();
        self.children
            .entry(head.to_string())
            .or_insert_with(|| VariantTreeNode::new(self.len))
            .insert(tail, values);
    }

    fn into_struct_array(self) -> Arc<StructArray> {
        let mut fields = Vec::with_capacity(self.children.len());
        let mut arrays = Vec::with_capacity(self.children.len());
        let mut entries: Vec<_> = self.children.into_iter().collect();
        entries.sort_by(|a, b| a.0.cmp(&b.0));
        for (name, child) in entries {
            let field_array = child.into_field_array();
            fields.push(Arc::new(Field::new(
                name.as_str(),
                field_array.data_type().clone(),
                false,
            )));
            arrays.push(field_array);
        }
        Arc::new(StructArray::new(Fields::from(fields), arrays, None))
    }

    fn into_field_array(self) -> ArrayRef {
        let len = self.len;
        if self.children.is_empty() {
            let values = self.leaf.expect("variant leaf value present");
            wrap_typed_value(len, values)
        } else {
            let typed_struct = self.into_struct_array() as ArrayRef;
            wrap_typed_value(len, typed_struct)
        }
    }
}

fn wrap_typed_value(len: usize, values: ArrayRef) -> ArrayRef {
    let placeholder = Arc::new(BinaryViewArray::from(vec![None::<&[u8]>; len])) as ArrayRef;
    Arc::new(StructArray::new(
        Fields::from(vec![
            Arc::new(Field::new("value", DataType::BinaryView, true)),
            Arc::new(Field::new("typed_value", values.data_type().clone(), true)),
        ]),
        vec![placeholder, values],
        None,
    )) as ArrayRef
}
