#[cfg(test)]
use crate::cache::cached_data::CachedBatch;
use crate::sync::{Arc, RwLock};
use arrow::array::ArrayRef;
use arrow_schema::ArrowError;
use bytes::Bytes;
use liquid_cache_common::LiquidCacheMode;
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

    use crate::cache::{CacheStorageBuilder, core::DefaultIoContext};

    let temp_dir = tempdir().unwrap();
    let base_dir = temp_dir.keep();
    let batch_size = 128;

    let builder = CacheStorageBuilder::new()
        .with_batch_size(batch_size)
        .with_max_cache_bytes(max_cache_bytes)
        .with_cache_dir(base_dir.clone())
        .with_cache_mode(LiquidCacheMode::LiquidBlocking)
        .with_policy(policy)
        .with_io_worker(Arc::new(DefaultIoContext::new(base_dir)));
    builder.build()
}

/// Advice given by the cache policy.
#[derive(PartialEq, Eq, Debug)]
pub enum CacheAdvice {
    /// Evict the entry with the given ID.
    TranscodeToDisk(EntryID),
    /// Transcode the entry to liquid memory.
    Transcode(EntryID),
    /// Write the entry to disk as-is (preserve format).
    ToDisk(EntryID),
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

pub(crate) fn arrow_to_bytes(array: &ArrayRef) -> Result<Bytes, ArrowError> {
    use arrow::array::RecordBatch;
    use arrow::ipc::writer::StreamWriter;

    let mut bytes = Vec::new();

    // Create a record batch with the single array
    // We need to create a dummy field since we don't have the original field here
    let field =
        arrow_schema::Field::new("column", array.data_type().clone(), array.null_count() > 0);
    let schema = std::sync::Arc::new(arrow_schema::Schema::new(vec![field]));
    let batch = RecordBatch::try_new(schema.clone(), vec![array.clone()])?;

    let mut stream_writer = StreamWriter::try_new(&mut bytes, &schema)?;
    stream_writer.write(&batch)?;
    stream_writer.finish()?;

    Ok(Bytes::from(bytes))
}
