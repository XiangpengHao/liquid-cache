use std::{
    collections::VecDeque,
    ops::Deref,
    path::{Path, PathBuf},
    sync::{
        Mutex,
        atomic::{AtomicUsize, Ordering},
    },
};

use dashmap::{DashMap, Entry};
use log::warn;

use super::CachedBatch;

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

/// BatchID is a unique identifier for a batch of rows,
/// it is row id divided by the batch size.
///
// It's very easy to misinterpret this as row id, so we use new type idiom to avoid confusion:
// https://doc.rust-lang.org/rust-by-example/generics/new_types.html
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Ord, PartialOrd)]
pub struct BatchID {
    v: u16,
}

impl BatchID {
    /// Creates a new BatchID from a row id and a batch size.
    /// row id must be on the batch boundary.
    pub(crate) fn from_row_id(row_id: usize, batch_size: usize) -> Self {
        debug_assert!(row_id % batch_size == 0);
        Self {
            v: (row_id / batch_size) as u16,
        }
    }

    pub(crate) fn from_raw(v: u16) -> Self {
        Self { v }
    }

    pub(crate) fn inc(&mut self) {
        debug_assert!(self.v < u16::MAX);
        self.v += 1;
    }
}

impl Deref for BatchID {
    type Target = u16;

    fn deref(&self) -> &Self::Target {
        &self.v
    }
}

impl CacheEntryID {
    pub(super) fn new(file_id: u64, row_group_id: u64, column_id: u64, batch_id: BatchID) -> Self {
        debug_assert!(file_id <= u16::MAX as u64);
        debug_assert!(row_group_id <= u16::MAX as u64);
        debug_assert!(column_id <= u16::MAX as u64);
        Self {
            val: (file_id) << 48 | (row_group_id) << 32 | (column_id) << 16 | batch_id.v as u64,
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

    #[allow(unused)]
    fn on_disk_path(&self, cache_root_dir: &Path) -> PathBuf {
        let row_id = self.row_id_inner();
        cache_root_dir
            .join(format!("file_{}", self.file_id_inner()))
            .join(format!("row_group_{}", self.row_group_id_inner()))
            .join(format!("column_{}", self.column_id_inner()))
            .join(format!("row_{}.bin", row_id))
    }
}

#[derive(Debug)]
pub(super) struct CacheConfig {
    batch_size: usize,
    max_cache_bytes: usize,
    cache_root_dir: PathBuf,
}

impl CacheConfig {
    fn new(batch_size: usize, max_cache_bytes: usize, cache_root_dir: PathBuf) -> Self {
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
    fn try_reserve_memory(&self, request_bytes: usize) -> Result<(), ()> {
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
    fn try_update_memory_usage(&self, old_size: usize, new_size: usize) -> Result<(), ()> {
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

    pub fn sub_memory_usage(&self, evicted_size: usize) {
        self.used_memory_bytes
            .fetch_sub(evicted_size, Ordering::Relaxed);
    }
}

#[derive(Debug)]
pub(crate) struct CacheStore {
    cached_data: DashMap<CacheEntryID, CachedBatch>,
    config: CacheConfig,
    budget: BudgetAccounting,
    fifo_queue: Mutex<VecDeque<CacheEntryID>>,
}

#[derive(Debug)]
pub(super) enum CacheAdvice {
    EvictAndRetry(CacheEntryID),
    InsertToDisk,
}

impl CacheStore {
    pub(super) fn new(batch_size: usize, max_cache_bytes: usize, cache_root_dir: PathBuf) -> Self {
        let config = CacheConfig::new(batch_size, max_cache_bytes, cache_root_dir);
        Self {
            cached_data: DashMap::new(),
            budget: BudgetAccounting::new(config.max_cache_bytes()),
            config,
            fifo_queue: Mutex::new(VecDeque::new()),
        }
    }

    pub(super) fn insert(
        &self,
        entry_id: CacheEntryID,
        cached_batch: CachedBatch,
    ) -> Result<(), CacheAdvice> {
        let new_memory_size = cached_batch.memory_usage_bytes();

        match self.cached_data.entry(entry_id) {
            Entry::Occupied(mut entry) => {
                let old = entry.get();
                let old_memory_size = old.memory_usage_bytes();
                self.budget
                    .try_update_memory_usage(old_memory_size, new_memory_size)
                    .map_err(|_| {
                        let mut fifo_queue = self.fifo_queue.lock().unwrap();
                        let v = fifo_queue.pop_front();
                        if let Some(v) = v {
                            CacheAdvice::EvictAndRetry(v)
                        } else {
                            CacheAdvice::InsertToDisk
                        }
                    })?;
                entry.insert(cached_batch);
            }
            Entry::Vacant(entry) => {
                self.budget
                    .try_reserve_memory(new_memory_size)
                    .map_err(|_| CacheAdvice::InsertToDisk)?;
                entry.insert(cached_batch);
            }
        }
        self.fifo_queue.lock().unwrap().push_back(entry_id);
        Ok(())
    }

    pub(super) fn get(&self, entry_id: &CacheEntryID) -> Option<CachedBatch> {
        self.cached_data
            .get(entry_id)
            .map(|entry| entry.value().clone())
    }

    pub(super) fn remove(&self, entry_id: &CacheEntryID) -> Option<CachedBatch> {
        let v = self.cached_data.remove(entry_id).map(|(_, batch)| batch)?;
        let memory_usage = v.memory_usage_bytes();
        self.budget.sub_memory_usage(memory_usage);
        Some(v)
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
        config.sub_memory_usage(200);
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
        assert!(config.try_update_memory_usage(300, 200).is_ok());
        assert_eq!(config.memory_usage_bytes(), 300); // 400 - (300 - 200)

        // Update after transcoding - size increase within limits
        assert!(config.try_update_memory_usage(200, 500).is_ok());
        assert_eq!(config.memory_usage_bytes(), 600); // 300 + (500 - 200)

        // Update after transcoding - size increase exceeding limits
        assert!(config.try_update_memory_usage(100, 600).is_err());
        assert_eq!(config.memory_usage_bytes(), 600); // Unchanged because update failed
    }
}
