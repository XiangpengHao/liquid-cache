use crate::LiquidPredicate;
use crate::liquid_array::LiquidArrayRef;
use crate::sync::{Arc, RwLock};
use arrow::array::{Array, ArrayRef, BooleanArray, RecordBatch};
use arrow::buffer::BooleanBuffer;
use arrow::compute::prep_null_mask_filter;
use arrow_schema::{ArrowError, Field, Schema};
use liquid_cache_common::LiquidCacheMode;
use parquet::arrow::arrow_reader::ArrowPredicate;
use std::fmt::Display;
use std::path::PathBuf;

#[derive(Debug)]
pub struct CacheConfig {
    batch_size: usize,
    max_cache_bytes: usize,
    cache_root_dir: PathBuf,
    cache_mode: LiquidCacheMode,
}

impl CacheConfig {
    pub(super) fn new(
        batch_size: usize,
        max_cache_bytes: usize,
        cache_root_dir: PathBuf,
        cache_mode: LiquidCacheMode,
    ) -> Self {
        Self {
            batch_size,
            max_cache_bytes,
            cache_root_dir,
            cache_mode,
        }
    }

    pub fn batch_size(&self) -> usize {
        self.batch_size
    }

    pub fn max_cache_bytes(&self) -> usize {
        self.max_cache_bytes
    }

    pub fn cache_root_dir(&self) -> &PathBuf {
        &self.cache_root_dir
    }

    pub fn cache_mode(&self) -> &LiquidCacheMode {
        &self.cache_mode
    }
}

// Helper methods
#[cfg(test)]
pub(crate) fn create_test_array(size: usize) -> CachedBatch {
    use arrow::array::Int64Array;
    use std::sync::Arc;

    CachedBatch::MemoryArrow(Arc::new(Int64Array::from_iter_values(0..size as i64)))
}

// Helper methods
#[cfg(test)]
pub(crate) fn create_test_arrow_array(size: usize) -> ArrayRef {
    use arrow::array::Int64Array;
    Arc::new(Int64Array::from_iter_values(0..size as i64))
}

#[cfg(test)]
pub(crate) fn create_cache_store(
    max_cache_bytes: usize,
    policy: Box<dyn super::policies::CachePolicy>,
) -> Arc<super::core::CacheStorage> {
    use tempfile::tempdir;

    use crate::cache::core::{CacheStorage, DefaultIoWorker};

    let temp_dir = tempdir().unwrap();
    let base_dir = temp_dir.keep();
    let batch_size = 128;

    Arc::new(CacheStorage::new(
        batch_size,
        max_cache_bytes,
        base_dir.clone(),
        LiquidCacheMode::LiquidBlocking,
        policy,
        Arc::new(DefaultIoWorker::new(base_dir)),
    ))
}

/// Advice given by the cache policy.
#[derive(PartialEq, Eq, Debug)]
pub enum CacheAdvice {
    /// Evict the entry with the given ID.
    Evict(EntryID),
    /// Transcode the entry to disk.
    TranscodeToDisk(EntryID),
    /// Transcode the entry to liquid memory.
    Transcode(EntryID),
    /// Write the entry to disk as-is (preserve format).
    ToDisk(EntryID),
    /// Discard the incoming entry, do not cache.
    // Note that discarding a previously cached entry is disallowed,
    // as it creates race conditions when one thread reads that a entry is cached,
    // and later only to find it is not cached.
    Discard,
}

/// EntryID is a unique identifier for a batch of rows, i.e., the cache key.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Ord, PartialOrd)]
pub struct EntryID {
    val: usize,
}

impl From<usize> for EntryID {
    fn from(val: usize) -> Self {
        Self { val }
    }
}

impl From<EntryID> for usize {
    fn from(val: EntryID) -> Self {
        val.val
    }
}

/// States for liquid compressor.
pub struct LiquidCompressorStates {
    fsst_compressor: RwLock<Option<Arc<fsst::Compressor>>>,
}

impl std::fmt::Debug for LiquidCompressorStates {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "EtcCompressorStates")
    }
}

impl Default for LiquidCompressorStates {
    fn default() -> Self {
        Self::new()
    }
}

impl LiquidCompressorStates {
    /// Create a new instance of LiquidCompressorStates.
    pub fn new() -> Self {
        Self {
            fsst_compressor: RwLock::new(None),
        }
    }

    /// Create a new instance of LiquidCompressorStates with an fsst compressor.
    pub fn new_with_fsst_compressor(fsst_compressor: Arc<fsst::Compressor>) -> Self {
        Self {
            fsst_compressor: RwLock::new(Some(fsst_compressor)),
        }
    }

    /// Get the fsst compressor.
    pub fn fsst_compressor(&self) -> Option<Arc<fsst::Compressor>> {
        self.fsst_compressor.read().unwrap().clone()
    }

    /// Get the fsst compressor .
    pub fn fsst_compressor_raw(&self) -> &RwLock<Option<Arc<fsst::Compressor>>> {
        &self.fsst_compressor
    }
}

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
