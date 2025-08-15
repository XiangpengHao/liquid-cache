use std::{fmt::Display, sync::Arc};

use arrow::{
    array::{Array, ArrayRef, BooleanArray, RecordBatch},
    buffer::BooleanBuffer,
    compute::prep_null_mask_filter,
};
use arrow_schema::{ArrowError, Field, Schema};
use parquet::arrow::arrow_reader::ArrowPredicate;

use crate::{LiquidPredicate, cache::EntryID, liquid_array::LiquidArrayRef};

#[derive(Debug)]
pub struct CachedData<'a> {
    data: CachedBatch,
    id: EntryID,
    io_worker: &'a dyn super::core::IoWorker,
}

impl<'a> CachedData<'a> {
    pub fn new(data: CachedBatch, id: EntryID, io_worker: &'a dyn super::core::IoWorker) -> Self {
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

    fn arrow_array_to_record_batch(&self, array: ArrayRef, field: &Arc<Field>) -> RecordBatch {
        let schema = Arc::new(Schema::new(vec![field.clone()]));
        RecordBatch::try_new(schema, vec![array]).unwrap()
    }

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

    pub fn try_read_liquid(&self) -> Option<LiquidArrayRef> {
        match &self.data {
            CachedBatch::MemoryLiquid(array) => Some(array.clone()),
            CachedBatch::DiskLiquid => {
                Some(self.io_worker.read_liquid_from_disk(&self.id).unwrap())
            }
            _ => None,
        }
    }

    pub fn get_arrow_with_filter(&self, filter: &BooleanArray) -> Result<ArrayRef, std::io::Error> {
        match &self.data {
            CachedBatch::MemoryArrow(array) => {
                let filtered = arrow::compute::filter(array, filter).unwrap();
                Ok(filtered)
            }
            CachedBatch::MemoryLiquid(array) => {
                let filtered = array.filter_to_arrow(filter);
                Ok(filtered)
            }
            CachedBatch::DiskLiquid => {
                let array = self.io_worker.read_liquid_from_disk(&self.id).unwrap();
                let filtered = array.filter_to_arrow(filter);
                Ok(filtered)
            }
            CachedBatch::DiskArrow => {
                let array = self.io_worker.read_arrow_from_disk(&self.id).unwrap();
                let filtered = arrow::compute::filter(&array, filter).unwrap();
                Ok(filtered)
            }
        }
    }

    fn eval_predicate_with_filter_inner(
        &self,
        predicate: &mut LiquidPredicate,
        array: &LiquidArrayRef,
        filter: &BooleanArray,
        field: &Arc<Field>,
    ) -> Result<BooleanBuffer, ArrowError> {
        match array.try_eval_predicate(predicate.physical_expr_physical_column_index(), filter)? {
            Some(new_filter) => {
                let (buffer, _) = new_filter.into_parts();
                Ok(buffer)
            }
            None => {
                let arrow_batch = array.filter_to_arrow(filter);
                let schema = Schema::new(vec![field.clone()]);
                let record_batch =
                    RecordBatch::try_new(Arc::new(schema), vec![arrow_batch]).unwrap();
                let boolean_array = predicate.evaluate(record_batch).unwrap();
                let (buffer, _) = boolean_array.into_parts();
                Ok(buffer)
            }
        }
    }

    pub fn eval_predicate_with_filter(
        &self,
        filter: &BooleanBuffer,
        predicate: &mut LiquidPredicate,
        field: &Arc<Field>,
    ) -> Result<BooleanBuffer, ArrowError> {
        match &self.data {
            CachedBatch::MemoryArrow(array) => {
                let boolean_array = BooleanArray::new(filter.clone(), None);
                let selected = arrow::compute::filter(array, &boolean_array).unwrap();
                let record_batch = self.arrow_array_to_record_batch(selected, field);
                let boolean_array = predicate.evaluate(record_batch).unwrap();
                let predicate_filter = match boolean_array.null_count() {
                    0 => boolean_array,
                    _ => prep_null_mask_filter(&boolean_array),
                };
                let (buffer, _) = predicate_filter.into_parts();
                Ok(buffer)
            }

            CachedBatch::DiskArrow => {
                let array = self.io_worker.read_arrow_from_disk(&self.id).unwrap();
                let boolean_array = BooleanArray::new(filter.clone(), None);
                let selected = arrow::compute::filter(&array, &boolean_array).unwrap();
                let record_batch = self.arrow_array_to_record_batch(selected, field);
                let boolean_array = predicate.evaluate(record_batch).unwrap();
                let predicate_filter = match boolean_array.null_count() {
                    0 => boolean_array,
                    _ => prep_null_mask_filter(&boolean_array),
                };
                let (buffer, _) = predicate_filter.into_parts();
                Ok(buffer)
            }
            CachedBatch::MemoryLiquid(array) => {
                let boolean_array = BooleanArray::new(filter.clone(), None);
                self.eval_predicate_with_filter_inner(predicate, array, &boolean_array, field)
            }
            CachedBatch::DiskLiquid => {
                let array = self.io_worker.read_liquid_from_disk(&self.id).unwrap();
                let boolean_array = BooleanArray::new(filter.clone(), None);
                self.eval_predicate_with_filter_inner(predicate, &array, &boolean_array, field)
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
