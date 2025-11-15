use arrow::{
    array::{Array, ArrayRef, BinaryViewArray, BooleanArray, StructArray},
    buffer::{BooleanBuffer, NullBuffer},
    compute::prep_null_mask_filter,
    record_batch::RecordBatch,
};
use arrow_schema::{ArrowError, DataType, Field, Fields, Schema};
use liquid_cache_storage::cache::{CacheExpression, CacheStorage, ColumnID};
use parquet::arrow::arrow_reader::ArrowPredicate;
use parquet::variant::{GetOptions, VariantArray, VariantPath, VariantType, variant_get};

use crate::{
    LiquidPredicate,
    cache::{BatchID, ColumnAccessPath, ParquetArrayID},
    optimizers::{
        DATE_MAPPING_METADATA_KEY, VARIANT_MAPPING_METADATA_KEY, VARIANT_MAPPING_TYPE_METADATA_KEY,
    },
};
use std::{str::FromStr, sync::Arc};

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
    if let Some(path) = field.metadata().get(VARIANT_MAPPING_METADATA_KEY)
        && let Some(data_type) = field
            .metadata()
            .get(VARIANT_MAPPING_TYPE_METADATA_KEY)
            .and_then(|ty| DataType::from_str(ty).ok())
        && field.try_extension_type::<VariantType>().is_ok()
    {
        return Some(CacheExpression::variant_get(path.to_string(), data_type));
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
        if let Some(expr) = &self.expression {
            if let Some(transformed) =
                maybe_shred_variant_array(&array, expr.as_ref(), self.field.as_ref())
            {
                array = transformed;
            }
        }
        self.cache_store
            .insert(self.entry_id(batch_id).into(), array)
            .await;
        Ok(())
    }
}

fn maybe_shred_variant_array(
    array: &ArrayRef,
    expression: &CacheExpression,
    field: &Field,
) -> Option<ArrayRef> {
    match expression {
        CacheExpression::VariantGet { path, data_type } => {
            shred_variant_array(array, field, path.as_ref(), data_type.as_ref())
        }
        _ => None,
    }
}

fn shred_variant_array(
    array: &ArrayRef,
    field: &Field,
    path: &str,
    data_type: &DataType,
) -> Option<ArrayRef> {
    let variant_array = VariantArray::try_new(array.as_ref()).ok()?;
    if variant_contains_typed_field(&variant_array, path) {
        return None;
    }

    let typed_field = Arc::new(Field::new("typed_value", data_type.clone(), true));
    let options =
        GetOptions::new_with_path(VariantPath::from(path)).with_as_type(Some(typed_field));
    let typed_values = variant_get(array, options).ok()?;

    let typed_struct =
        build_typed_value_struct(path, typed_values, variant_array.inner().nulls().cloned());

    let inner = variant_array.inner();
    let target_fields = match field.data_type() {
        DataType::Struct(fields) => fields.clone(),
        _ => return None,
    };

    let metadata_array = inner
        .column_by_name("metadata")
        .cloned()
        .unwrap_or_else(|| Arc::new(variant_array.metadata_field().clone()) as ArrayRef);
    let value_array = inner.column_by_name("value").cloned().unwrap_or_else(|| {
        Arc::new(BinaryViewArray::from(vec![None::<&[u8]>; inner.len()])) as ArrayRef
    });

    let mut columns = Vec::with_capacity(target_fields.len());
    for target_field in target_fields.iter() {
        let column = match target_field.name().as_str() {
            "metadata" => metadata_array.clone(),
            "value" => value_array.clone(),
            "typed_value" => typed_struct.clone(),
            other => inner.column_by_name(other)?.clone(),
        };
        columns.push(column);
    }

    let root_struct = StructArray::new(target_fields, columns, inner.nulls().cloned());

    Some(Arc::new(root_struct))
}

fn build_typed_value_struct(
    path: &str,
    typed_values: ArrayRef,
    nulls: Option<NullBuffer>,
) -> ArrayRef {
    let leaf_field = Arc::new(Field::new(
        "typed_value",
        typed_values.data_type().clone(),
        typed_values.null_count() > 0,
    ));
    let leaf_struct = Arc::new(StructArray::new(
        Fields::from(vec![leaf_field]),
        vec![typed_values.clone()],
        typed_values.nulls().cloned(),
    ));

    let named_field = Arc::new(Field::new(path, leaf_struct.data_type().clone(), true));
    Arc::new(StructArray::new(
        Fields::from(vec![named_field]),
        vec![leaf_struct as ArrayRef],
        nulls,
    ))
}

fn variant_contains_typed_field(array: &VariantArray, path: &str) -> bool {
    array
        .typed_value_field()
        .and_then(|typed| typed.as_any().downcast_ref::<StructArray>())
        .map(|typed_struct| typed_struct.column_by_name(path).is_some())
        .unwrap_or(false)
}

#[cfg(test)]
mod tests {}
