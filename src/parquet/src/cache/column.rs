use arrow::{
    array::{Array, ArrayRef, BooleanArray, StructArray},
    buffer::BooleanBuffer,
    compute::prep_null_mask_filter,
    record_batch::RecordBatch,
};
use arrow_schema::{ArrowError, DataType, Field, Fields, Schema};
use liquid_cache_storage::cache::{CacheExpression, CacheStorage, ColumnID};
use parquet::arrow::arrow_reader::ArrowPredicate;
use parquet_variant_compute::{VariantArray, VariantType, shred_variant, unshred_variant};

use crate::{
    LiquidPredicate,
    cache::{BatchID, ColumnAccessPath, ParquetArrayID},
    optimizers::{DATE_MAPPING_METADATA_KEY, variant_mappings_from_field},
};
use std::sync::Arc;

/// A column in the cache.
#[derive(Debug)]
pub struct LiquidCachedColumn {
    cache_store: Arc<CacheStorage>,
    field: Arc<Field>,
    column_path: ColumnAccessPath,
    expression: Option<Arc<CacheExpression>>,
}

/// A reference to a cached column.
pub type LiquidCachedColumnRef = Arc<LiquidCachedColumn>;

fn infer_expression(field: &Field) -> Option<CacheExpression> {
    if let Some(mapping) = field.metadata().get(DATE_MAPPING_METADATA_KEY)
        && field.data_type() == &DataType::Date32
        && let Some(expr) = CacheExpression::try_from_date_part_str(mapping)
    {
        return Some(expr);
    }
    if field.try_extension_type::<VariantType>().is_ok()
        && let Some(mappings) = variant_mappings_from_field(field)
        && let Some((path, data_type)) = mappings.iter().find_map(|mapping| {
            mapping
                .data_type
                .as_ref()
                .map(|data_type| (mapping.path.clone(), data_type.clone()))
        })
    {
        return Some(CacheExpression::variant_get(path, data_type));
    }
    None
}

/// Error type for inserting an arrow array into the cache.
#[derive(Debug)]
pub enum InsertArrowArrayError {
    /// The array is already cached.
    AlreadyCached,
}

impl LiquidCachedColumn {
    pub(crate) fn new(
        field: Arc<Field>,
        cache_store: Arc<CacheStorage>,
        column_id: u64,
        row_group_id: u64,
        file_id: u64,
    ) -> Self {
        let column_path = ColumnAccessPath::new(file_id, row_group_id, column_id);
        column_path.initialize_dir(cache_store.config().cache_root_dir());
        let registry = cache_store.expression_registry();
        let expression = infer_expression(field.as_ref()).map(|expr| {
            let col_id = ColumnID::new(file_id, row_group_id, column_id);
            registry.register(expr, Some(col_id))
        });
        Self {
            field,
            cache_store,
            column_path,
            expression,
        }
    }

    /// row_id must be on a batch boundary.
    pub(crate) fn entry_id(&self, batch_id: BatchID) -> ParquetArrayID {
        self.column_path.entry_id(batch_id)
    }

    pub(crate) fn is_cached(&self, batch_id: BatchID) -> bool {
        self.cache_store.is_cached(&self.entry_id(batch_id).into())
    }

    /// Returns the Arrow field metadata for this cached column.
    pub fn field(&self) -> Arc<Field> {
        self.field.clone()
    }

    /// Returns the expression metadata associated with this column, if any.
    pub fn expression(&self) -> Option<Arc<CacheExpression>> {
        self.expression.clone()
    }

    fn array_to_record_batch(&self, array: ArrayRef) -> Result<RecordBatch, ArrowError> {
        let schema = Arc::new(Schema::new(vec![self.field.clone()]));
        RecordBatch::try_new(schema, vec![array])
    }

    /// Evaluates a predicate on a cached column.
    pub async fn eval_predicate_with_filter(
        &self,
        batch_id: BatchID,
        filter: &BooleanBuffer,
        predicate: &mut LiquidPredicate,
    ) -> Option<Result<BooleanArray, ArrowError>> {
        let entry_id = self.entry_id(batch_id).into();
        let result = self
            .cache_store
            .eval_predicate(&entry_id, predicate.physical_expr_physical_column_index())
            .with_selection(filter)
            .await?;
        match result {
            Ok(boolean_array) => {
                let predicate_filter = match boolean_array.null_count() {
                    0 => boolean_array,
                    _ => prep_null_mask_filter(&boolean_array),
                };
                Some(Ok(predicate_filter))
            }
            Err(array) => {
                let record_batch = match self.array_to_record_batch(array) {
                    Ok(batch) => batch,
                    Err(err) => return Some(Err(err)),
                };
                let boolean_array = match predicate.evaluate(record_batch) {
                    Ok(arr) => arr,
                    Err(err) => return Some(Err(err)),
                };
                let predicate_filter = match boolean_array.null_count() {
                    0 => boolean_array,
                    _ => prep_null_mask_filter(&boolean_array),
                };
                Some(Ok(predicate_filter))
            }
        }
    }

    /// Get an arrow array with a filter applied.
    pub async fn get_arrow_array_with_filter(
        &self,
        batch_id: BatchID,
        filter: &BooleanBuffer,
    ) -> Option<ArrayRef> {
        let entry_id = self.entry_id(batch_id).into();
        let array = self
            .cache_store
            .get(&entry_id)
            .with_selection(filter)
            .with_optional_expression_hint(self.expression())
            .read()
            .await?;
        Some(array)
    }

    #[cfg(test)]
    pub(crate) async fn get_arrow_array_test_only(&self, batch_id: BatchID) -> Option<ArrayRef> {
        let entry_id = self.entry_id(batch_id).into();
        self.cache_store.get(&entry_id).await
    }

    /// Insert an array into the cache.
    pub async fn insert(
        self: &Arc<Self>,
        batch_id: BatchID,
        array: ArrayRef,
    ) -> Result<(), InsertArrowArrayError> {
        if self.is_cached(batch_id) {
            return Err(InsertArrowArrayError::AlreadyCached);
        }
        let mut array = array;
        if self.expression.is_some()
            && let Some(transformed) = maybe_shred_variant_array(&array, self.field.as_ref())
        {
            array = transformed;
        }
        self.cache_store
            .insert(self.entry_id(batch_id).into(), array)
            .await;
        Ok(())
    }
}

fn maybe_shred_variant_array(array: &ArrayRef, field: &Field) -> Option<ArrayRef> {
    let mappings = variant_mappings_from_field(field)?;
    let typed_specs: Vec<(String, DataType)> = mappings
        .into_iter()
        .filter_map(|mapping| mapping.data_type.map(|data_type| (mapping.path, data_type)))
        .collect();
    if typed_specs.is_empty() {
        return None;
    }
    shred_variant_array(array, field, &typed_specs)
}

fn shred_variant_array(
    array: &ArrayRef,
    field: &Field,
    specs: &[(String, DataType)],
) -> Option<ArrayRef> {
    if specs.is_empty() {
        return None;
    }

    let variant_array = VariantArray::try_new(array.as_ref()).ok()?;
    let missing_specs: Vec<_> = specs
        .iter()
        .filter(|(path, _)| !variant_contains_typed_field(&variant_array, path))
        .collect();
    if missing_specs.is_empty() {
        return None;
    }

    let target_fields = match field.data_type() {
        DataType::Struct(fields) => fields.clone(),
        _ => return None,
    };
    let typed_schema = target_fields
        .iter()
        .find(|child| child.name() == "typed_value")
        .cloned()?;
    let shredding_schema = build_shredding_schema(typed_schema.as_ref())?;
    let unshredded = unshred_variant(&variant_array).ok()?;
    let shredded = shred_variant(&unshredded, &shredding_schema).ok()?;
    let inner = shredded.inner();
    let metadata_array = inner.column_by_name("metadata")?.clone();
    let value_array = inner.column_by_name("value")?.clone();
    let typed_value_column = inner.column_by_name("typed_value")?.clone();
    let typed_struct = typed_value_column
        .as_any()
        .downcast_ref::<StructArray>()?;
    let rebuilt_typed_value = rebuild_typed_value_array(typed_schema.as_ref(), typed_struct)?;

    let mut columns = Vec::with_capacity(target_fields.len());
    for target_field in target_fields.iter() {
        let column = match target_field.name().as_str() {
            "metadata" => metadata_array.clone(),
            "value" => value_array.clone(),
            "typed_value" => rebuilt_typed_value.clone(),
            other => inner.column_by_name(other)?.clone(),
        };
        columns.push(column);
    }

    let root_struct = StructArray::new(target_fields, columns, inner.nulls().cloned());
    Some(Arc::new(root_struct))
}

fn split_variant_path(path: &str) -> Vec<String> {
    path
        .split('.')
        .filter(|segment| !segment.is_empty())
        .map(|segment| segment.to_string())
        .collect()
}

fn build_shredding_schema(typed_schema: &Field) -> Option<DataType> {
    match typed_schema.data_type() {
        DataType::Struct(children) => build_struct_from_children(children),
        other => Some(other.clone()),
    }
}

fn build_struct_from_children(children: &Fields) -> Option<DataType> {
    let mut new_children = Vec::with_capacity(children.len());
    for child in children.iter() {
        if let Some(field) = typed_schema_field(child.as_ref()) {
            new_children.push(field);
        }
    }
    if new_children.is_empty() {
        None
    } else {
        Some(DataType::Struct(Fields::from(new_children)))
    }
}

fn typed_schema_field(field: &Field) -> Option<Arc<Field>> {
    let Some(typed_child) = typed_value_child(field) else {
        return None;
    };
    match typed_child.data_type() {
        DataType::Struct(grand_children) => {
            let Some(child_type) = build_struct_from_children(grand_children) else {
                return None;
            };
            Some(Arc::new(Field::new(field.name(), child_type, true)))
        }
        other => Some(Arc::new(Field::new(field.name(), other.clone(), true))),
    }
}

fn typed_value_child(field: &Field) -> Option<&Field> {
    match field.data_type() {
        DataType::Struct(children) => children
            .iter()
            .find(|child| child.name() == "typed_value")
            .map(|child| child.as_ref()),
        _ => None,
    }
}

fn rebuild_typed_value_array(
    typed_schema: &Field,
    shredded_struct: &StructArray,
) -> Option<ArrayRef> {
    let DataType::Struct(children) = typed_schema.data_type() else {
        return None;
    };
    rebuild_struct_children(children, shredded_struct)
}

fn rebuild_struct_children(children: &Fields, shredded_struct: &StructArray) -> Option<ArrayRef> {
    let mut columns = Vec::with_capacity(children.len());
    for child in children.iter() {
        let shredded_child = shredded_struct.column_by_name(child.name())?;
        let shredded_field = shredded_child
            .as_any()
            .downcast_ref::<StructArray>()?;
        let converted = convert_shredded_field(child.as_ref(), shredded_field)?;
        columns.push(converted);
    }
    Some(Arc::new(StructArray::new(
        children.clone(),
        columns,
        shredded_struct.nulls().cloned(),
    )) as ArrayRef)
}

fn convert_shredded_field(schema_field: &Field, shredded_field: &StructArray) -> Option<ArrayRef> {
    let Some(typed_child) = typed_value_child(schema_field) else {
        return None;
    };
    let typed_value_array = shredded_field.column_by_name("typed_value")?.clone();
    match typed_child.data_type() {
        DataType::Struct(grand_children) => {
            let typed_struct = typed_value_array
                .as_any()
                .downcast_ref::<StructArray>()?;
            let rebuilt = rebuild_struct_children(grand_children, typed_struct)?;
            let struct_array = StructArray::new(
                Fields::from(vec![Arc::new(typed_child.clone())]),
                vec![rebuilt],
                typed_struct.nulls().cloned(),
            );
            Some(Arc::new(struct_array) as ArrayRef)
        }
        _ => {
            let struct_array = StructArray::new(
                Fields::from(vec![Arc::new(typed_child.clone())]),
                vec![typed_value_array],
                shredded_field.nulls().cloned(),
            );
            Some(Arc::new(struct_array) as ArrayRef)
        }
    }
}

fn variant_contains_typed_field(array: &VariantArray, path: &str) -> bool {
    let Some(typed_root_array) = array.typed_value_field() else {
        return false;
    };
    let Some(typed_root) = typed_root_array
        .as_any()
        .downcast_ref::<StructArray>()
    else {
        return false;
    };

    if typed_root.column_by_name(path).is_some() {
        return true;
    }

    let segments = split_variant_path(path);
    if segments.is_empty() {
        return false;
    }

    typed_struct_contains_path(typed_root, &segments)
}

fn typed_struct_contains_path(current: &StructArray, segments: &[String]) -> bool {
    let mut cursor = current;
    for (idx, segment) in segments.iter().enumerate() {
        let Some(field) = cursor.column_by_name(segment) else {
            return false;
        };
        let Some(struct_field) = field.as_any().downcast_ref::<StructArray>() else {
            return false;
        };
        if idx == segments.len() - 1 {
            return struct_field.column_by_name("typed_value").is_some();
        }
        let Some(next) = struct_field
            .column_by_name("typed_value")
            .and_then(|col| col.as_any().downcast_ref::<StructArray>())
        else {
            return false;
        };
        cursor = next;
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::optimizers::{
        VARIANT_MAPPING_METADATA_KEY, VariantField, enrich_variant_field_type,
    };
    use arrow::array::{ArrayRef, StringArray, StructArray};
    use parquet::variant::{VariantType, json_to_variant};
    use serde_json::json;
    use std::collections::HashMap;

    #[test]
    fn shredding_adds_all_variant_paths() {
        let values = StringArray::from(vec![
            Some(r#"{"name":"Alice","age":30}"#),
            Some(r#"{"name":"Bob","age":27}"#),
        ]);
        let variant = json_to_variant(&(Arc::new(values) as ArrayRef)).expect("variant");

        let mut metadata = HashMap::new();
        metadata.insert(
            VARIANT_MAPPING_METADATA_KEY.to_string(),
            serde_json::to_string(&vec![
                json!({"path": "name", "type": "Utf8"}),
                json!({"path": "age", "type": "Int64"}),
            ])
            .unwrap(),
        );

        let variant_fields = vec![
            VariantField {
                path: "name".to_string(),
                data_type: Some(DataType::Utf8),
            },
            VariantField {
                path: "age".to_string(),
                data_type: Some(DataType::Int64),
            },
        ];

        let base_field = Field::new("variant", variant.inner().data_type().clone(), true)
            .with_extension_type(VariantType)
            .with_metadata(metadata);
        let enriched = enrich_variant_field_type(base_field.as_ref(), &variant_fields)
            .with_metadata(base_field.metadata().clone());
        let array: ArrayRef = ArrayRef::from(variant);

        let shredded = maybe_shred_variant_array(&array, enriched.as_ref())
            .expect("variant should be shredded");
        let shredded_struct = shredded
            .as_any()
            .downcast_ref::<StructArray>()
            .expect("struct array");
        let typed_value = shredded_struct
            .column_by_name("typed_value")
            .expect("typed_value column");
        let typed_struct = typed_value
            .as_any()
            .downcast_ref::<StructArray>()
            .expect("typed struct");

        let name_struct = typed_struct
            .column_by_name("name")
            .expect("name path")
            .as_any()
            .downcast_ref::<StructArray>()
            .expect("name struct");
        let name_values = name_struct
            .column_by_name("typed_value")
            .expect("name typed value");
        assert_eq!(name_values.data_type(), &DataType::Utf8);

        let age_struct = typed_struct
            .column_by_name("age")
            .expect("age path")
            .as_any()
            .downcast_ref::<StructArray>()
            .expect("age struct");
        let age_values = age_struct
            .column_by_name("typed_value")
            .expect("age typed value");
        assert_eq!(age_values.data_type(), &DataType::Int64);
    }
}
