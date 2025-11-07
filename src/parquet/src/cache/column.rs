use std::sync::Arc;

use arrow::{
    array::{Array, ArrayRef, BooleanArray},
    buffer::BooleanBuffer,
    compute::prep_null_mask_filter,
    record_batch::RecordBatch,
};
use arrow_schema::{ArrowError, DataType, Field, Schema};
use liquid_cache_storage::cache::{CacheExpression, CacheStorage};
use parquet::arrow::arrow_reader::ArrowPredicate;

use crate::{
    LiquidPredicate,
    cache::{BatchID, ColumnAccessPath, ParquetArrayID},
    optimizers::DATE_MAPPING_METADATA_KEY,
};

/// A column in the cache.
#[derive(Debug)]
pub struct LiquidCachedColumn {
    cache_store: Arc<CacheStorage>,
    field: Arc<Field>,
    column_path: ColumnAccessPath,
    expression: Option<CacheExpression>,
}

/// A reference to a cached column.
pub type LiquidCachedColumnRef = Arc<LiquidCachedColumn>;

fn infer_expression(field: &Field) -> Option<CacheExpression> {
    let mapping = field.metadata().get(DATE_MAPPING_METADATA_KEY)?;
    if field.data_type() != &DataType::Date32 {
        return None;
    }
    CacheExpression::try_from_date_part_str(mapping)
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
        let expression = infer_expression(field.as_ref());
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
    pub fn expression(&self) -> Option<&CacheExpression> {
        self.expression.as_ref()
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
            .with_optional_expression_hint(self.expression())
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
        self.cache_store
            .insert(self.entry_id(batch_id).into(), array)
            .with_optional_expression_hint(self.expression())
            .await;
        Ok(())
    }
}

#[cfg(test)]
mod tests {}
