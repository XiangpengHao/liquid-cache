use std::{
    path::{Path, PathBuf},
    sync::{
        Arc, LazyLock,
        atomic::{AtomicUsize, Ordering},
    },
};

use arrow::array::ArrayRef;
use dashmap::DashMap;
use log::warn;
use tokio::runtime::Runtime;

use super::{CachedBatch, LiquidCompressorStates, transcode_liquid_inner};

/// A dedicated Tokio thread pool for background transcoding tasks.
/// This pool is built with 4 worker threads.
static TRANSCODE_THREAD_POOL: LazyLock<Runtime> = LazyLock::new(|| {
    tokio::runtime::Builder::new_multi_thread()
        .worker_threads(4)
        .thread_name("transcode-worker")
        .enable_all()
        .build()
        .unwrap()
});

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Ord, PartialOrd)]
pub(super) struct CacheEntryID {
    // This is a unique identifier for a row in a parquet file.
    // It is composed of 8 bytes:
    // - 2 bytes for the file id
    // - 2 bytes for the row group id
    // - 2 bytes for the column id
    // - 2 bytes for the row id
    // The numerical order of val is meaningful: sorted by each of the fields.
    val: u64,
}

impl CacheEntryID {
    pub(super) fn new(file_id: u64, row_group_id: u64, column_id: u64, row_id: u64) -> Self {
        debug_assert!(file_id <= u16::MAX as u64);
        debug_assert!(row_group_id <= u16::MAX as u64);
        debug_assert!(column_id <= u16::MAX as u64);
        debug_assert!(row_id <= u16::MAX as u64);
        Self {
            val: (file_id as u64) << 48
                | (row_group_id as u64) << 32
                | (column_id as u64) << 16
                | row_id as u64,
        }
    }

    fn row_id_inner(&self) -> u64 {
        self.val & 0x0000_0000_0000_FFFF
    }

    fn file_id_inner(&self) -> u64 {
        self.val >> 48
    }

    fn row_group_id_inner(&self) -> u64 {
        (self.val >> 32) & 0x0000_0000_FFFF
    }

    fn column_id_inner(&self) -> u64 {
        (self.val >> 16) & 0x0000_0000_FFFF
    }

    fn on_disk_path(&self, cache_root_dir: &Path) -> PathBuf {
        let row_id = self.row_id_inner();
        cache_root_dir
            .join(format!("file_{}", self.file_id_inner()))
            .join(format!("row_group_{}", self.row_group_id_inner()))
            .join(format!("column_{}", self.column_id_inner()))
            .join(format!("row_{}.bin", row_id))
    }

    fn column_id(&self) -> ColumnID {
        ColumnID::new(
            self.file_id_inner(),
            self.row_group_id_inner(),
            self.column_id_inner(),
        )
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Ord, PartialOrd)]
struct ColumnID {
    val: u64,
}

impl ColumnID {
    fn new(file_id: u64, row_group_id: u64, column_id: u64) -> Self {
        Self {
            val: (file_id as u64) << 48 | (row_group_id as u64) << 32 | (column_id as u64) << 16,
        }
    }
}

#[derive(Debug)]
pub(super) struct CacheConfig {
    batch_size: usize,
    max_cache_bytes: usize,
    cache_root_dir: PathBuf,
}

impl CacheConfig {
    pub fn new(batch_size: usize, max_cache_bytes: usize, cache_root_dir: PathBuf) -> Self {
        Self {
            batch_size,
            max_cache_bytes,
            cache_root_dir,
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
}

#[derive(Debug)]
pub(super) struct BudgetAccounting {
    max_memory_bytes: usize,
    used_memory_bytes: AtomicUsize,
    used_disk_bytes: AtomicUsize,
}

impl BudgetAccounting {
    fn new(max_memory_bytes: usize) -> Self {
        Self {
            max_memory_bytes,
            used_memory_bytes: AtomicUsize::new(0),
            used_disk_bytes: AtomicUsize::new(0),
        }
    }

    fn reset_usage(&self) {
        self.used_memory_bytes.store(0, Ordering::Relaxed);
        self.used_disk_bytes.store(0, Ordering::Relaxed);
    }

    /// Try to reserve space in the cache.
    /// Returns true if the space was reserved, false if the cache is full.
    pub fn try_reserve_memory(&self, request_bytes: usize) -> Result<(), ()> {
        let used = self.used_memory_bytes.load(Ordering::Relaxed);
        if used + request_bytes > self.max_memory_bytes {
            return Err(());
        }

        match self.used_memory_bytes.compare_exchange(
            used,
            used + request_bytes,
            Ordering::Relaxed,
            Ordering::Relaxed,
        ) {
            Ok(_) => Ok(()),
            Err(_) => self.try_reserve_memory(request_bytes),
        }
    }

    /// Adjust the cache size after transcoding.
    /// Returns true if the size was adjusted, false if the cache is full, when new_size is larger than old_size.
    pub fn try_update_memory_usage_after_transcoding(
        &self,
        old_size: usize,
        new_size: usize,
    ) -> Result<(), ()> {
        if old_size < new_size {
            let diff = new_size - old_size;
            if diff > 1024 * 1024 {
                warn!(
                    "Transcoding increased the size of the array by at least 1MB, previous size: {}, new size: {}, double check this is correct",
                    old_size, new_size
                );
            }

            self.try_reserve_memory(diff)?;
            Ok(())
        } else {
            self.used_memory_bytes
                .fetch_sub(old_size - new_size, Ordering::Relaxed);
            Ok(())
        }
    }

    pub fn memory_usage_bytes(&self) -> usize {
        self.used_memory_bytes.load(Ordering::Relaxed)
    }

    pub fn disk_usage_bytes(&self) -> usize {
        self.used_disk_bytes.load(Ordering::Relaxed)
    }

    pub fn add_used_disk_bytes(&self, bytes: usize) {
        self.used_disk_bytes.fetch_add(bytes, Ordering::Relaxed);
    }

    #[allow(unused)]
    pub fn update_usage_after_eviction(&self, evicted_size: usize) {
        self.used_memory_bytes
            .fetch_sub(evicted_size, Ordering::Relaxed);
        self.used_disk_bytes
            .fetch_add(evicted_size, Ordering::Relaxed);
    }
}

#[derive(Debug)]
pub(crate) struct CacheStore {
    cached_data: DashMap<CacheEntryID, CachedBatch>,
    compressor_states: DashMap<ColumnID, Arc<LiquidCompressorStates>>,
    config: CacheConfig,
    budget: BudgetAccounting,
    advisor: Arc<dyn CacheAdvisor>,
    background_transcode: bool,
}

#[derive(Debug)]
struct NoReplacementCacheAdvisor {
    budget: BudgetAccounting,
}

enum CacheAdvice {
    /// The cache store should insert the batch.
    InsertToMemory,
    /// The cache store should insert the batch to disk.
    InsertToDisk,
    /// The cache store should evict the batch and consult the advisor again.
    EvictAndRetry(CacheEntryID),
}

impl NoReplacementCacheAdvisor {
    fn new(max_memory_bytes: usize) -> Self {
        Self {
            budget: BudgetAccounting::new(max_memory_bytes),
        }
    }
}

impl CacheAdvisor for NoReplacementCacheAdvisor {
    fn advice_on(&self, _id: CacheEntryID, batch: &ArrayRef) -> CacheAdvice {
        let size = batch.get_array_memory_size();
        if self.budget.try_reserve_memory(size).is_ok() {
            CacheAdvice::InsertToMemory
        } else {
            CacheAdvice::InsertToDisk
        }
    }

    fn inform_read(&self, _id: &CacheEntryID) {}
}

trait CacheAdvisor: std::fmt::Debug + Sync + Send {
    /// Returns the advice for the cache store what to do with this batch.
    fn advice_on(&self, id: CacheEntryID, batch: &ArrayRef) -> CacheAdvice;

    /// Inform the advisor that a batch is being read.
    fn inform_read(&self, id: &CacheEntryID);
}

impl CacheStore {
    pub(super) fn new(config: CacheConfig, background_transcode: bool) -> Self {
        Self {
            cached_data: DashMap::new(),
            compressor_states: DashMap::new(),
            budget: BudgetAccounting::new(config.max_cache_bytes()),
            advisor: Arc::new(NoReplacementCacheAdvisor::new(config.max_cache_bytes())),
            config,
            background_transcode,
        }
    }

    pub(super) fn insert(self: &Arc<Self>, entry_id: CacheEntryID, cached_batch: ArrayRef) {
        let advice = self.advisor.advice_on(entry_id, &cached_batch);
        match advice {
            CacheAdvice::InsertToMemory => {
                if self.background_transcode {
                    self.cached_data
                        .insert(entry_id, CachedBatch::ArrowMemory(cached_batch));
                    let self_clone = self.clone();
                    TRANSCODE_THREAD_POOL.spawn(async move {
                        let Some(entry) = self_clone.cached_data.get(&entry_id) else {
                            return;
                        };
                        let states = self_clone
                            .compressor_states
                            .entry(entry_id.column_id())
                            .or_insert_with(|| Arc::new(LiquidCompressorStates::new()))
                            .clone();
                        let transcoded = match entry.transcode_to_liquid(&states) {
                            Ok(transcoded) => transcoded,
                            Err(_) => {
                                warn!("Failed to transcode batch, inserting as arrow array");
                                entry.clone()
                            }
                        };
                        let column_id = entry_id.column_id();
                        let compressor_states = self
                            .compressor_states
                            .entry(column_id)
                            .or_insert_with(|| Arc::new(LiquidCompressorStates::new()))
                            .clone();
                    });
                } else {
                    let column_id = entry_id.column_id();
                    let compressor_states = self
                        .compressor_states
                        .entry(column_id)
                        .or_insert_with(|| Arc::new(LiquidCompressorStates::new()))
                        .clone();
                    let transcoded = match transcode_liquid_inner(&cached_batch, &compressor_states)
                    {
                        Ok(transcoded) => CachedBatch::LiquidMemory(transcoded),
                        Err(_) => {
                            warn!("Failed to transcode batch, inserting as arrow array");
                            CachedBatch::ArrowMemory(cached_batch)
                        }
                    };
                    self.cached_data.insert(entry_id, transcoded);
                }
            }
            CacheAdvice::InsertToDisk => {
                let path = entry_id.on_disk_path(self.config.cache_root_dir());
                let transcoded = cached_batch.write_to_file(&path);
                self.cached_data.insert(entry_id, transcoded);
            }
            CacheAdvice::EvictAndRetry(to_evict) => {
                self.cached_data.remove(&to_evict);
                self.insert(entry_id, cached_batch);
            }
        }
    }

    pub(super) fn get(&self, entry_id: &CacheEntryID) -> Option<CachedBatch> {
        let v = self
            .cached_data
            .get(entry_id)
            .map(|entry| entry.value().clone());
        self.advisor.inform_read(entry_id);
        v
    }

    pub(super) fn reset(&self) {
        self.cached_data.clear();
        self.budget.reset_usage();
    }

    pub(super) fn is_cached(&self, entry_id: &CacheEntryID) -> bool {
        self.cached_data.contains_key(entry_id)
    }

    pub(super) fn config(&self) -> &CacheConfig {
        &self.config
    }

    pub(super) fn budget(&self) -> &BudgetAccounting {
        &self.budget
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_reservation_and_accounting() {
        let config = BudgetAccounting::new(1000);

        // Check initial state
        assert_eq!(config.memory_usage_bytes(), 0);

        // Basic reservation
        assert!(config.try_reserve_memory(500).is_ok());
        assert_eq!(config.memory_usage_bytes(), 500);

        // Additional reservation within limits
        assert!(config.try_reserve_memory(300).is_ok());
        assert_eq!(config.memory_usage_bytes(), 800);

        // Reservation exceeding limit
        assert!(config.try_reserve_memory(300).is_err());
        assert_eq!(config.memory_usage_bytes(), 800);

        // Test eviction accounting
        config.update_usage_after_eviction(200);
        assert_eq!(config.memory_usage_bytes(), 600);

        // Reset usage
        config.reset_usage();
        assert_eq!(config.memory_usage_bytes(), 0);
    }

    #[test]
    fn test_memory_transcoding_accounting() {
        let config = BudgetAccounting::new(1000);

        // Initial reservation
        assert!(config.try_reserve_memory(400).is_ok());
        assert_eq!(config.memory_usage_bytes(), 400);

        // Update after transcoding - size decrease
        assert!(
            config
                .try_update_memory_usage_after_transcoding(300, 200)
                .is_ok()
        );
        assert_eq!(config.memory_usage_bytes(), 300); // 400 - (300 - 200)

        // Update after transcoding - size increase within limits
        assert!(
            config
                .try_update_memory_usage_after_transcoding(200, 500)
                .is_ok()
        );
        assert_eq!(config.memory_usage_bytes(), 600); // 300 + (500 - 200)

        // Update after transcoding - size increase exceeding limits
        assert!(
            config
                .try_update_memory_usage_after_transcoding(100, 600)
                .is_err()
        );
        assert_eq!(config.memory_usage_bytes(), 600); // Unchanged because update failed
    }
}
