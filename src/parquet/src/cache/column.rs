use arrow::{
    array::{Array, ArrayRef, AsArray, BooleanArray},
    buffer::BooleanBuffer,
    compute::prep_null_mask_filter,
    record_batch::RecordBatch,
};
use arrow_schema::{ArrowError, DataType, Field, Schema};
use liquid_cache_storage::cache::{CacheExpression, LiquidCache};
use liquid_cache_storage::utils::VariantSchema;
use liquid_cache_storage::utils::typed_struct_contains_path;
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
pub struct CachedColumn {
    cache_store: Arc<LiquidCache>,
    field: Arc<Field>,
    column_path: ColumnAccessPath,
    expression: Option<Arc<CacheExpression>>,
}

/// A reference to a cached column.
pub type CachedColumnRef = Arc<CachedColumn>;

fn infer_expression(field: &Field) -> Option<CacheExpression> {
    if let Some(mapping) = field.metadata().get(DATE_MAPPING_METADATA_KEY)
        && field.data_type() == &DataType::Date32
        && let Some(expr) = CacheExpression::try_from_date_part_str(mapping)
    {
        return Some(expr);
    }
    if field.try_extension_type::<VariantType>().is_ok()
        && let Some(mappings) = variant_mappings_from_field(field)
    {
        let typed_specs: Vec<_> = mappings
            .into_iter()
            .filter_map(|mapping| mapping.data_type.map(|data_type| (mapping.path, data_type)))
            .collect();
        if !typed_specs.is_empty() {
            return Some(CacheExpression::variant_get_many(typed_specs));
        }
    }
    None
}

/// Error type for inserting an arrow array into the cache.
#[derive(Debug)]
pub enum InsertArrowArrayError {
    /// The array is already cached.
    AlreadyCached,
}

impl CachedColumn {
    pub(crate) fn new(
        field: Arc<Field>,
        cache_store: Arc<LiquidCache>,
        column_access_path: ColumnAccessPath,
        is_predicate_column: bool,
    ) -> Self {
        column_access_path.initialize_dir(cache_store.config().cache_root_dir());

        let expression = infer_expression(field.as_ref()).map(|expr| {
            let expr = Arc::new(expr);
            let hint_entry_id = column_access_path.entry_id(BatchID::from_raw(0)).into();
            cache_store.add_squeeze_hint(&hint_entry_id, expr.clone());
            expr
        });
        if is_predicate_column {
            let expression = Arc::new(CacheExpression::PredicateColumn);
            let hint_entry_id = column_access_path.entry_id(BatchID::from_raw(0)).into();
            cache_store.add_squeeze_hint(&hint_entry_id, expression);
        }
        Self {
            field,
            cache_store,
            column_path: column_access_path,
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

    fn array_to_record_batch(&self, array: ArrayRef) -> RecordBatch {
        let schema = Arc::new(Schema::new(vec![self.field.clone()]));
        RecordBatch::try_new(schema, vec![array]).unwrap()
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
                let mut array = array;
                if let Some(transformed) = maybe_shred_variant_array(&array, self.field.as_ref()) {
                    array = transformed;
                }
                let record_batch = self.array_to_record_batch(array);
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
        let mut array = self
            .cache_store
            .get(&entry_id)
            .with_selection(filter)
            .with_optional_expression_hint(self.expression())
            .read()
            .await?;
        if let Some(transformed) = maybe_shred_variant_array(&array, self.field.as_ref()) {
            array = transformed;
        }
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
    let mut schema = VariantSchema::new(Some(typed_schema.as_ref()));
    for (path, data_type) in missing_specs {
        schema.insert_path(path, data_type);
    }
    let shredding_schema = schema.shredding_type()?;
    let unshredded = unshred_variant(&variant_array).ok()?;
    let shredded = shred_variant(&unshredded, &shredding_schema).ok()?;
    Some(Arc::new(shredded.into_inner()))
}

fn variant_contains_typed_field(array: &VariantArray, path: &str) -> bool {
    let Some(typed_field) = array.typed_value_field() else {
        return false;
    };
    let Some(typed_root) = typed_field.as_struct_opt() else {
        return false;
    };
    typed_struct_contains_path(typed_root, path)
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
