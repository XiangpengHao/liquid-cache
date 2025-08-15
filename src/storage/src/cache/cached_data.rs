//! Cached data in the cache.

use std::{fmt::Display, sync::Arc};

use arrow::{
    array::{ArrayRef, BooleanArray},
    buffer::BooleanBuffer,
};
use arrow_schema::ArrowError;
use datafusion::physical_plan::PhysicalExpr;

use crate::{cache::EntryID, liquid_array::LiquidArrayRef};

/// A wrapper around the actual data in the cache.
#[derive(Debug)]
pub struct CachedData<'a> {
    data: CachedBatch,
    id: EntryID,
    io_worker: &'a dyn super::core::IoWorker,
}

/// The result of predicate pushdown.
#[derive(Debug, PartialEq)]
pub enum PredicatePushdownResult {
    /// The predicate is evaluated on the filtered data and the result is a boolean buffer.
    Evaluated(BooleanBuffer),

    /// The predicate is not evaluated but data is filtered.
    Filtered(ArrayRef),
}

impl<'a> CachedData<'a> {
    pub(crate) fn new(
        data: CachedBatch,
        id: EntryID,
        io_worker: &'a dyn super::core::IoWorker,
    ) -> Self {
        Self {
            data,
            id,
            io_worker,
        }
    }

    #[cfg(test)]
    pub(crate) fn raw_data(&self) -> &CachedBatch {
        &self.data
    }

    /// Get the arrow array from the cached data.
    pub fn get_arrow_array(&self) -> ArrayRef {
        match &self.data {
            CachedBatch::MemoryArrow(array) => array.clone(),
            CachedBatch::MemoryLiquid(array) => array.to_best_arrow_array(),
            CachedBatch::DiskLiquid => self
                .io_worker
                .read_liquid_from_disk(&self.id)
                .unwrap()
                .to_best_arrow_array(),
            CachedBatch::DiskArrow => self.io_worker.read_arrow_from_disk(&self.id).unwrap(),
        }
    }

    /// Try to read the liquid array from the cached data.
    /// Return None if the cached data is not a liquid array.
    pub fn try_read_liquid(&self) -> Option<LiquidArrayRef> {
        match &self.data {
            CachedBatch::MemoryLiquid(array) => Some(array.clone()),
            CachedBatch::DiskLiquid => {
                Some(self.io_worker.read_liquid_from_disk(&self.id).unwrap())
            }
            _ => None,
        }
    }

    /// Get the arrow array with selection pushdown.
    pub fn get_with_selection(&self, selection: &BooleanArray) -> Result<ArrayRef, ArrowError> {
        match &self.data {
            CachedBatch::MemoryArrow(array) => {
                let filtered = arrow::compute::filter(array, selection)?;
                Ok(filtered)
            }
            CachedBatch::MemoryLiquid(array) => {
                let filtered = array.filter_to_arrow(selection);
                Ok(filtered)
            }
            CachedBatch::DiskLiquid => {
                let array = self.io_worker.read_liquid_from_disk(&self.id)?;
                let filtered = array.filter_to_arrow(selection);
                Ok(filtered)
            }
            CachedBatch::DiskArrow => {
                let array = self.io_worker.read_arrow_from_disk(&self.id)?;
                let filtered = arrow::compute::filter(&array, selection)?;
                Ok(filtered)
            }
        }
    }

    /// Get the arrow array with predicate pushdown.
    ///
    /// The `selection` is applied **before** predicate evaluation.
    ///
    /// Returns:
    /// - `PredicatePushdownResult::Evaluated(buffer)`: the predicate is evaluated on the filtered data and the result is a boolean buffer.
    /// - `PredicatePushdownResult::Filtered(array)`: the predicate is not evaluated (e.g., predicate is not supported or error happens) but data is filtered.
    ///
    /// ```rust
    /// use liquid_cache_storage::cache::{CacheStorageBuilder, EntryID, cached_data::PredicatePushdownResult};
    /// use liquid_cache_storage::common::LiquidCacheMode;
    /// use arrow::array::{StringArray, BooleanArray};
    /// use arrow::buffer::BooleanBuffer;
    /// use datafusion::logical_expr::Operator;
    /// use datafusion::physical_plan::expressions::{BinaryExpr, Column, Literal};
    /// use datafusion::physical_plan::PhysicalExpr;
    /// use datafusion::scalar::ScalarValue;
    /// use std::sync::Arc;
    ///
    /// let storage = CacheStorageBuilder::new()
    ///     .with_cache_mode(LiquidCacheMode::LiquidBlocking)
    ///     .build();
    ///
    /// let entry_id = EntryID::from(9);
    /// let data = Arc::new(StringArray::from(vec![
    ///     Some("apple"), Some("banana"), None, Some("apple"), Some("cherry"),
    /// ]));
    /// storage.insert(entry_id, data.clone());
    ///
    /// let selection = BooleanArray::from(vec![true, true, false, true, true]);
    ///
    /// let expr: Arc<dyn PhysicalExpr> = Arc::new(BinaryExpr::new(
    ///     Arc::new(Column::new("col", 0)),
    ///     Operator::Eq,
    ///     Arc::new(Literal::new(ScalarValue::Utf8(Some("apple".to_string())))),
    /// ));
    ///
    /// let cached = storage.get(&entry_id).unwrap();
    /// let result = cached
    ///     .get_with_predicate(&selection, &expr)
    ///     .unwrap();
    /// let expected = BooleanBuffer::from(vec![true, false, true, false]);
    /// assert_eq!(result, PredicatePushdownResult::Evaluated(expected));
    /// ```
    pub fn get_with_predicate(
        &self,
        selection: &BooleanArray,
        predicate: &Arc<dyn PhysicalExpr>,
    ) -> Result<PredicatePushdownResult, ArrowError> {
        match &self.data {
            CachedBatch::MemoryArrow(array) => {
                let selected = arrow::compute::filter(array, selection)?;
                Ok(PredicatePushdownResult::Filtered(selected))
            }
            CachedBatch::DiskArrow => {
                let array = self.io_worker.read_arrow_from_disk(&self.id)?;
                let selected = arrow::compute::filter(&array, selection)?;
                Ok(PredicatePushdownResult::Filtered(selected))
            }
            CachedBatch::MemoryLiquid(array) => {
                self.eval_predicate_with_filter_inner(predicate, array, selection)
            }
            CachedBatch::DiskLiquid => {
                let array = self.io_worker.read_liquid_from_disk(&self.id).unwrap();
                self.eval_predicate_with_filter_inner(predicate, &array, selection)
            }
        }
    }

    fn eval_predicate_with_filter_inner(
        &self,
        predicate: &Arc<dyn PhysicalExpr>,
        array: &LiquidArrayRef,
        selection: &BooleanArray,
    ) -> Result<PredicatePushdownResult, ArrowError> {
        match array.try_eval_predicate(predicate, selection)? {
            Some(new_filter) => {
                let (buffer, _) = new_filter.into_parts();
                Ok(PredicatePushdownResult::Evaluated(buffer))
            }
            None => {
                let filtered = array.filter_to_arrow(selection);
                Ok(PredicatePushdownResult::Filtered(filtered))
            }
        }
    }
}

/// Cached batch.
#[derive(Debug, Clone)]
pub enum CachedBatch {
    /// Cached batch in memory as Arrow array.
    MemoryArrow(ArrayRef),
    /// Cached batch in memory as liquid array.
    MemoryLiquid(LiquidArrayRef),
    /// Cached batch on disk as liquid array.
    DiskLiquid,
    /// Cached batch on disk as Arrow array.
    DiskArrow,
}

impl CachedBatch {
    /// Get the memory usage of the cached batch.
    pub fn memory_usage_bytes(&self) -> usize {
        match self {
            Self::MemoryArrow(array) => array.get_array_memory_size(),
            Self::MemoryLiquid(array) => array.get_array_memory_size(),
            Self::DiskLiquid => 0,
            Self::DiskArrow => 0,
        }
    }

    /// Get the reference count of the cached batch.
    pub fn reference_count(&self) -> usize {
        match self {
            Self::MemoryArrow(array) => Arc::strong_count(array),
            Self::MemoryLiquid(array) => Arc::strong_count(array),
            Self::DiskLiquid => 0,
            Self::DiskArrow => 0,
        }
    }
}

impl Display for CachedBatch {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::MemoryArrow(_) => write!(f, "MemoryArrow"),
            Self::MemoryLiquid(_) => write!(f, "MemoryLiquid"),
            Self::DiskLiquid => write!(f, "DiskLiquid"),
            Self::DiskArrow => write!(f, "DiskArrow"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::{Array, Int64Array, StringArray};
    use arrow::compute as compute_kernels;
    use datafusion::logical_expr::Operator;
    use datafusion::physical_plan::expressions::{BinaryExpr, Column, Literal};
    use datafusion::scalar::ScalarValue;

    fn io_worker() -> (tempfile::TempDir, super::super::core::DefaultIoWorker) {
        let tmp = tempfile::tempdir().unwrap();
        let io = super::super::core::DefaultIoWorker::new(tmp.path().to_path_buf());
        (tmp, io)
    }

    #[test]
    fn test_get_arrow_array_memory_arrow() {
        let array: ArrayRef = Arc::new(Int64Array::from_iter_values(0..8));
        let id = EntryID::from(1usize);
        let (_tmp, io) = io_worker();
        let cached = CachedData::new(CachedBatch::MemoryArrow(array.clone()), id, &io);

        let out = cached.get_arrow_array();
        assert_eq!(out.as_ref(), array.as_ref());
    }

    #[test]
    fn test_try_read_liquid_memory_liquid() {
        // Build a small liquid string array
        let input = StringArray::from(vec!["a", "b", "a", "c"]);
        let (_compressor, etc) = crate::liquid_array::LiquidByteArray::train_from_arrow(&input);
        let liquid_ref: LiquidArrayRef = Arc::new(etc);

        let id = EntryID::from(2usize);
        let (_tmp, io) = io_worker();
        let cached = CachedData::new(CachedBatch::MemoryLiquid(liquid_ref.clone()), id, &io);

        let got = cached.try_read_liquid().expect("should be liquid");
        assert_eq!(got.to_best_arrow_array().len(), input.len());
    }

    #[test]
    fn test_get_with_selection_memory_arrow() {
        let array: ArrayRef = Arc::new(Int64Array::from_iter_values(0..10));
        let selection = BooleanArray::from((0..10).map(|i| i % 2 == 0).collect::<Vec<_>>());

        let id = EntryID::from(3usize);
        let (_tmp, io) = io_worker();
        let cached = CachedData::new(CachedBatch::MemoryArrow(array.clone()), id, &io);

        let filtered = cached.get_with_selection(&selection).unwrap();
        let expected = compute_kernels::filter(&array, &selection).unwrap();
        assert_eq!(filtered.as_ref(), expected.as_ref());
    }

    #[test]
    fn test_get_with_predicate_evaluated_for_strings() {
        // Mirror the doc example: evaluate predicate on filtered liquid strings
        let data = Arc::new(StringArray::from(vec![
            Some("apple"),
            Some("banana"),
            None,
            Some("apple"),
            Some("cherry"),
        ]));

        // Build a liquid array in-memory so we hit the Evaluated branch
        let (_compressor, liquid) = crate::liquid_array::LiquidByteArray::train_from_arrow(&data);
        let liquid_ref: LiquidArrayRef = Arc::new(liquid);

        let selection = BooleanArray::from(vec![true, true, false, true, true]);

        let expr: Arc<dyn PhysicalExpr> = Arc::new(BinaryExpr::new(
            Arc::new(Column::new("col", 0)),
            Operator::Eq,
            Arc::new(Literal::new(ScalarValue::Utf8(Some("apple".to_string())))),
        ));

        let id = EntryID::from(9usize);
        let (_tmp, io) = io_worker();
        let cached = CachedData::new(CachedBatch::MemoryLiquid(liquid_ref), id, &io);

        let result = cached
            .get_with_predicate(&selection, &expr)
            .expect("predicate should succeed");

        match result {
            PredicatePushdownResult::Evaluated(buf) => {
                let expected = BooleanBuffer::from(vec![true, false, true, false]);
                assert_eq!(buf, expected);
            }
            other => panic!("expected Evaluated, got {other:?}"),
        }
    }

    #[test]
    fn test_cached_batch_memory_usage_and_refcount_arrow() {
        let array: ArrayRef = Arc::new(Int64Array::from_iter_values(0..16));
        let batch = CachedBatch::MemoryArrow(array.clone());

        assert_eq!(batch.memory_usage_bytes(), array.get_array_memory_size());
        assert_eq!(batch.reference_count(), Arc::strong_count(&array));
    }
}
