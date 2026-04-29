use arrow::{
    array::{Array, ArrayRef, BooleanArray},
    buffer::BooleanBuffer,
    compute::prep_null_mask_filter,
    record_batch::RecordBatch,
};
use arrow_schema::{ArrowError, DataType, Field, Schema};
use liquid_cache::cache::{CacheExpression, LiquidCache, LiquidExpr};
use parquet::arrow::arrow_reader::ArrowPredicate;

use crate::{
    LiquidPredicate,
    cache::{BatchID, ColumnAccessPath, ParquetArrayID},
    optimizers::{DATE_MAPPING_METADATA_KEY, STRING_FINGERPRINT_METADATA_KEY},
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
        && matches!(
            field.data_type(),
            DataType::Date32 | DataType::Timestamp(_, _)
        )
        && let Some(expr) = CacheExpression::try_from_date_part_str(mapping)
    {
        return Some(expr);
    }
    if field
        .metadata()
        .contains_key(STRING_FINGERPRINT_METADATA_KEY)
        && is_string_type(field.data_type())
    {
        return Some(CacheExpression::substring_search());
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
        let expression = infer_expression(field.as_ref()).map(Arc::new);
        if let Some(expr) = expression.as_ref() {
            let hint_entry_id = column_access_path.entry_id(BatchID::from_raw(0)).into();
            cache_store.add_squeeze_hint(&hint_entry_id, expr.clone());
        } else if is_predicate_column {
            let hint_entry_id = column_access_path.entry_id(BatchID::from_raw(0)).into();
            cache_store
                .add_squeeze_hint(&hint_entry_id, Arc::new(CacheExpression::PredicateColumn));
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
        let liquid_expr = LiquidExpr::try_new(
            Arc::clone(predicate.physical_expr()),
            self.field.data_type(),
            self.expression.as_deref(),
        );

        if let Some(liquid_expr) = liquid_expr
            && let Some(boolean_array) = self
                .cache_store
                .eval_predicate(&entry_id, &liquid_expr)
                .with_selection(filter)
                .await
        {
            let predicate_filter = match boolean_array.null_count() {
                0 => boolean_array,
                _ => prep_null_mask_filter(&boolean_array),
            };
            return Some(Ok(predicate_filter));
        }

        let array = self.get_arrow_array_with_filter(batch_id, filter).await?;
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

    fn liquid_expr_for(
        &self,
        expr: Arc<dyn datafusion::physical_plan::PhysicalExpr>,
    ) -> Option<LiquidExpr> {
        LiquidExpr::try_new(expr, self.field.data_type(), self.expression.as_deref())
    }

    pub(crate) fn liquid_expr_for_predicate(
        &self,
        expr: Arc<dyn datafusion::physical_plan::PhysicalExpr>,
    ) -> Option<LiquidExpr> {
        self.liquid_expr_for(expr)
    }

    /// Get an arrow array with a filter applied.
    pub async fn get_arrow_array_with_filter(
        &self,
        batch_id: BatchID,
        filter: &BooleanBuffer,
    ) -> Option<ArrayRef> {
        let entry_id = self.entry_id(batch_id).into();
        self.cache_store
            .get(&entry_id)
            .with_selection(filter)
            .with_optional_expression_hint(self.expression())
            .read()
            .await
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

fn is_string_type(data_type: &DataType) -> bool {
    match data_type {
        DataType::Utf8 | DataType::Utf8View | DataType::LargeUtf8 => true,
        DataType::Dictionary(_, value_type) => is_string_type(value_type.as_ref()),
        _ => false,
    }
}
