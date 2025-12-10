use std::sync::Arc;

use arrow::array::{Array, ArrayRef, BinaryViewArray, StructArray};
use arrow::buffer::NullBuffer;
use arrow::error::ArrowError;
use arrow::ipc::writer::StreamWriter;
use arrow::record_batch::RecordBatch;
use arrow_schema::{DataType, Field, Fields, Schema};
use bytes::Bytes;

use crate::liquid_array::{
    LiquidArrayRef, LiquidDataType, LiquidSqueezedArray, NeedsBacking, SqueezedBacking,
};
use ahash::AHashMap;

/// Squeezed representation for variant arrays that contain multiple typed fields.
#[derive(Debug)]
pub struct VariantStructSqueezedArray {
    values: AHashMap<Arc<str>, LiquidArrayRef>,
    len: usize,
    nulls: Option<NullBuffer>,
    original_arrow_type: DataType,
}

impl VariantStructSqueezedArray {
    /// Create a squeezed representation that keeps only the typed variant columns resident.
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

    /// Returns true if the squeezed contains the provided variant path.
    pub fn contains_path(&self, path: &str) -> bool {
        self.values.contains_key(path)
    }

    /// Build an Arrow array that includes only the provided variant paths.
    /// If `paths` is empty or none match, it falls back to the full array.
    pub fn to_arrow_array_with_paths<'a>(
        &self,
        paths: impl IntoIterator<Item = &'a str>,
    ) -> Result<ArrayRef, NeedsBacking> {
        let mut filtered: Vec<(Arc<str>, LiquidArrayRef)> = Vec::new();
        for path in paths.into_iter() {
            if let Some(array) = self.values.get(path) {
                filtered.push((Arc::from(path.to_string()), array.clone()));
            }
        }

        if filtered.is_empty() {
            return self.to_arrow_array();
        }

        let filtered = VariantStructSqueezedArray::new(
            filtered,
            self.nulls.clone(),
            self.original_arrow_type.clone(),
        );
        filtered.to_arrow_array()
    }

    /// Clone the stored typed values keyed by variant path.
    pub fn typed_values(&self) -> Vec<(Arc<str>, LiquidArrayRef)> {
        self.values
            .iter()
            .map(|(path, array)| (path.clone(), array.clone()))
            .collect()
    }

    /// Null buffer shared by all stored paths, if present.
    pub fn nulls(&self) -> Option<NullBuffer> {
        self.nulls.clone()
    }
}

impl LiquidSqueezedArray for VariantStructSqueezedArray {
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

    fn disk_backing(&self) -> SqueezedBacking {
        SqueezedBacking::Arrow
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

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::{Int64Array, StringArray};
    use arrow_schema::DataType;

    use crate::liquid_array::{LiquidByteArray, LiquidPrimitiveArray};

    #[test]
    fn to_arrow_array_with_paths_prunes_extra_fields() {
        // Build squeezed variant with two typed paths: did (utf8) and time_us (int64).
        let did_arrow = StringArray::from(vec![Some("d")]);
        let (_comp, did_liquid) = LiquidByteArray::train_from_arrow(&did_arrow);
        let did_liquid: LiquidArrayRef = Arc::new(did_liquid);

        let time_arrow = Int64Array::from(vec![1_i64]);
        let time_liquid =
            LiquidPrimitiveArray::<arrow::datatypes::Int64Type>::from_arrow_array(time_arrow);
        let time_liquid: LiquidArrayRef = Arc::new(time_liquid);

        let squeezed = VariantStructSqueezedArray::new(
            vec![
                (Arc::from("did"), did_liquid),
                (Arc::from("time_us"), time_liquid),
            ],
            None,
            DataType::Struct(Fields::from(Vec::<Arc<Field>>::new())),
        );

        // Request only time_us; did should be pruned from typed_value.
        let array = squeezed
            .to_arrow_array_with_paths(["time_us"])
            .expect("arrow array");
        let root = array
            .as_any()
            .downcast_ref::<StructArray>()
            .expect("struct root");
        let typed_value = root
            .column_by_name("typed_value")
            .unwrap()
            .as_any()
            .downcast_ref::<StructArray>()
            .unwrap();
        let field_names: Vec<_> = typed_value
            .fields()
            .iter()
            .map(|f| f.name().clone())
            .collect();
        assert_eq!(field_names, vec!["time_us"]);
    }
}
