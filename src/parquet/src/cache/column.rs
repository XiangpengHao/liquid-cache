use std::sync::Arc;

use arrow::{
    array::{Array, ArrayRef, BooleanArray, RecordBatch},
    buffer::BooleanBuffer,
    compute::prep_null_mask_filter,
};
use arrow_schema::{ArrowError, DataType, Field, Schema};
use liquid_cache_storage::cache::{CacheExpression, CacheStorage, GetWithPredicateResult};
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

    pub(crate) fn arrow_array_to_record_batch(
        &self,
        array: ArrayRef,
        field: &Arc<Field>,
    ) -> RecordBatch {
        let schema = Arc::new(Schema::new(vec![field.clone()]));
        RecordBatch::try_new(schema, vec![array]).unwrap()
    }

    /// Returns the Arrow field metadata for this cached column.
    pub fn field(&self) -> Arc<Field> {
        self.field.clone()
    }

    /// Returns the expression metadata associated with this column, if any.
    pub fn expression(&self) -> Option<&CacheExpression> {
        self.expression.as_ref()
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
            .get_with_predicate(
                &entry_id,
                filter,
                predicate.physical_expr_physical_column_index(),
            )
            .await?;

        match result {
            GetWithPredicateResult::Evaluated(buffer) => Some(Ok(buffer)),
            GetWithPredicateResult::Filtered(array) => {
                let record_batch = self.arrow_array_to_record_batch(array, &self.field);
                let boolean_array = predicate.evaluate(record_batch).unwrap();
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
        if let Some(expression) = self.expression()
            && filter.count_set_bits() == filter.len()
        {
            // test only path
            return self
                .get_arrow_array_with_expression(batch_id, Some(expression))
                .await;
        }
        let result = self
            .cache_store
            .get_with_selection(&entry_id, filter)
            .await?;
        result.ok()
    }

    /// Retrieve an arrow array optionally tailored to a cache expression.
    pub async fn get_arrow_array_with_expression(
        &self,
        batch_id: BatchID,
        expression: Option<&CacheExpression>,
    ) -> Option<ArrayRef> {
        let entry_id = self.entry_id(batch_id).into();
        self.cache_store
            .get_arrow_array_with_expression(&entry_id, expression)
            .await
    }

    #[cfg(test)]
    pub(crate) async fn get_arrow_array_test_only(&self, batch_id: BatchID) -> Option<ArrayRef> {
        let entry_id = self.entry_id(batch_id).into();
        self.cache_store.get_arrow_array(&entry_id).await
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

#[cfg(test)]
mod tests {}
