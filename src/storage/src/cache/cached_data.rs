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
pub enum PredicatePushdownResult {
    /// The predicate is evaluated on the filtered data and the result is a boolean buffer.
    PredicateEvaluated(BooleanBuffer),

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
    pub fn get_with_selection_pushdown(
        &self,
        selection: &BooleanArray,
    ) -> Result<ArrayRef, ArrowError> {
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
    pub fn get_with_predicate_pushdown(
        &self,
        selection: &BooleanArray,
        predicate: &Arc<dyn PhysicalExpr>,
    ) -> Result<PredicatePushdownResult, ArrowError> {
        match &self.data {
            CachedBatch::MemoryArrow(array) => {
                let selected = arrow::compute::filter(array, selection)?;
                return Ok(PredicatePushdownResult::Filtered(selected));
            }
            CachedBatch::DiskArrow => {
                let array = self.io_worker.read_arrow_from_disk(&self.id)?;
                let selected = arrow::compute::filter(&array, selection)?;
                return Ok(PredicatePushdownResult::Filtered(selected));
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
                Ok(PredicatePushdownResult::PredicateEvaluated(buffer))
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
