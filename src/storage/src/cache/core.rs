use arrow::array::ArrayRef;
use std::fs::File;
use std::io::Write;
use std::path::PathBuf;
use std::{fmt::Debug, path::Path};

use super::{
    CachedBatch, budget::BudgetAccounting, policies::CachePolicy, tracer::CacheTracer,
    transcode::transcode_liquid_inner, utils::CacheConfig,
};
use crate::cache::transcode::submit_background_transcoding_task;
use crate::cache::utils::{CacheAdvice, LiquidCompressorStates};
use crate::cache::{index::ArtIndex, utils::EntryID};
use crate::liquid_array::LiquidArrayRef;
use crate::sync::Arc;
use liquid_cache_common::LiquidCacheMode;

/// A trait for objects that can handle IO operations for the cache.
pub trait IoWorker: Debug + Send + Sync {
    /// Get the compressor for an entry.
    fn get_compressor_for_entry(&self, entry_id: &EntryID) -> Arc<LiquidCompressorStates>;
    /// Get the path to the arrow file for an entry.
    fn entry_arrow_path(&self, base_dir: &Path, entry_id: &EntryID) -> PathBuf;
    /// Get the path to the liquid file for an entry.
    fn entry_liquid_path(&self, base_dir: &Path, entry_id: &EntryID) -> PathBuf;
    /// Write an arrow array to disk.
    fn write_arrow_to_disk(
        &self,
        base_dir: &Path,
        entry_id: &EntryID,
        array: &ArrayRef,
    ) -> Result<usize, std::io::Error> {
        use arrow::array::RecordBatch;
        use arrow::ipc::writer::StreamWriter;
        use std::fs::File;
        use std::io::BufWriter;

        let file_path = self.entry_arrow_path(base_dir, entry_id);

        // Ensure parent directory exists
        if let Some(parent) = file_path.parent() {
            std::fs::create_dir_all(parent)?;
        }

        let file = File::create(file_path)?;
        let buf_writer = BufWriter::new(file);

        // Create a record batch with the single array
        // We need to create a dummy field since we don't have the original field here
        let field =
            arrow_schema::Field::new("column", array.data_type().clone(), array.null_count() > 0);
        let schema = std::sync::Arc::new(arrow_schema::Schema::new(vec![field]));
        let batch = RecordBatch::try_new(schema.clone(), vec![array.clone()]).unwrap();

        let mut stream_writer = StreamWriter::try_new(buf_writer, &schema).unwrap();
        stream_writer.write(&batch).unwrap();
        stream_writer.finish().unwrap();

        // Return approximate size for disk usage tracking
        Ok(array.get_array_memory_size())
    }

    /// Write a liquid array to disk.
    fn write_liquid_to_disk(
        &self,
        base_dir: &Path,
        entry_id: &EntryID,
        liquid_array: &LiquidArrayRef,
    ) -> Result<usize, std::io::Error> {
        let path = self.entry_liquid_path(base_dir, entry_id);
        let bytes = liquid_array.to_bytes();
        let mut file = File::create(&path)?;
        file.write_all(&bytes)?;
        Ok(bytes.len())
    }
}

/// A default implementation of IoWorker that uses the default compressor.
#[derive(Debug)]
pub struct DefaultIoWorker {
    compressor_state: Arc<LiquidCompressorStates>,
}

impl Default for DefaultIoWorker {
    fn default() -> Self {
        Self::new()
    }
}

impl DefaultIoWorker {
    /// Create a new instance of DefaultIoWorker.
    pub fn new() -> Self {
        Self {
            compressor_state: Arc::new(LiquidCompressorStates::new()),
        }
    }
}

impl IoWorker for DefaultIoWorker {
    fn get_compressor_for_entry(&self, _entry_id: &EntryID) -> Arc<LiquidCompressorStates> {
        self.compressor_state.clone()
    }

    fn entry_arrow_path(&self, base_dir: &Path, entry_id: &EntryID) -> PathBuf {
        base_dir.join(format!("{:016x}.arrow", usize::from(*entry_id)))
    }

    fn entry_liquid_path(&self, base_dir: &Path, entry_id: &EntryID) -> PathBuf {
        base_dir.join(format!("{:016x}.liquid", usize::from(*entry_id)))
    }
}

/// Cache storage for liquid cache.
#[derive(Debug)]
pub struct CacheStorage {
    index: ArtIndex,
    config: CacheConfig,
    budget: BudgetAccounting,
    policy: Box<dyn CachePolicy>,
    tracer: CacheTracer,
    io_worker: Arc<dyn IoWorker>,
}

impl CacheStorage {
    /// Create a new instance of CacheStorage.
    pub fn new(
        batch_size: usize,
        max_cache_bytes: usize,
        cache_dir: PathBuf,
        cache_mode: LiquidCacheMode,
        policy: Box<dyn CachePolicy>,
        io_worker: Arc<dyn IoWorker>,
    ) -> Self {
        let config = CacheConfig::new(batch_size, max_cache_bytes, cache_dir, cache_mode);
        Self {
            index: ArtIndex::new(),
            budget: BudgetAccounting::new(config.max_cache_bytes()),
            config,
            policy,
            tracer: CacheTracer::new(),
            io_worker,
        }
    }

    fn try_insert(
        &self,
        entry_id: EntryID,
        cached_batch: CachedBatch,
    ) -> Result<(), (CacheAdvice, CachedBatch)> {
        let new_memory_size = cached_batch.memory_usage_bytes();
        if let Some(entry) = self.index.get(&entry_id) {
            let old_memory_size = entry.memory_usage_bytes();
            if self
                .budget
                .try_update_memory_usage(old_memory_size, new_memory_size)
                .is_err()
            {
                let advice = self.policy.advise(&entry_id, self.config.cache_mode());
                return Err((advice, cached_batch));
            }
            self.index.insert(&entry_id, cached_batch);
        } else {
            if self.budget.try_reserve_memory(new_memory_size).is_err() {
                let advice = self.policy.advise(&entry_id, self.config.cache_mode());
                return Err((advice, cached_batch));
            }
            self.index.insert(&entry_id, cached_batch);
        }

        Ok(())
    }

    /// Returns Some(CachedBatch) if need to retry the insert.
    #[must_use]
    fn apply_advice(&self, advice: CacheAdvice, not_inserted: CachedBatch) -> Option<CachedBatch> {
        match advice {
            CacheAdvice::Transcode(to_transcode) => {
                let compressor_states = self.io_worker.get_compressor_for_entry(&to_transcode);
                let Some(to_transcode_batch) = self.index.get(&to_transcode) else {
                    // The batch is gone, no need to transcode
                    return Some(not_inserted);
                };

                if let CachedBatch::MemoryArrow(array) = to_transcode_batch {
                    let liquid_array = transcode_liquid_inner(&array, compressor_states.as_ref())
                        .expect("Failed to transcode to liquid array");
                    let liquid_array = CachedBatch::MemoryLiquid(liquid_array);
                    self.try_insert(to_transcode, liquid_array)
                        .expect("Failed to insert the transcoded batch");
                }

                Some(not_inserted)
            }
            CacheAdvice::Evict(to_evict) => {
                let compressor_states = self.io_worker.get_compressor_for_entry(&to_evict);
                let Some(to_evict_batch) = self.index.get(&to_evict) else {
                    return Some(not_inserted);
                };
                let liquid_array = match to_evict_batch {
                    CachedBatch::MemoryArrow(array) => {
                        transcode_liquid_inner(&array, compressor_states.as_ref())
                            .expect("Failed to transcode to liquid array")
                    }
                    CachedBatch::MemoryLiquid(liquid_array) => liquid_array,
                    CachedBatch::DiskLiquid => {
                        // do nothing, already on disk
                        return Some(not_inserted);
                    }
                    CachedBatch::DiskArrow => {
                        // do nothing, already on disk
                        return Some(not_inserted);
                    }
                };
                self.write_liquid_to_disk(&to_evict, &liquid_array);
                self.try_insert(to_evict, CachedBatch::DiskLiquid)
                    .expect("failed to insert on disk liquid");
                Some(not_inserted)
            }
            CacheAdvice::TranscodeToDisk(to_transcode) => {
                let compressor_states = self.io_worker.get_compressor_for_entry(&to_transcode);
                let liquid_array = match not_inserted {
                    CachedBatch::MemoryArrow(array) => {
                        transcode_liquid_inner(&array, compressor_states.as_ref())
                            .expect("Failed to transcode to liquid array")
                    }
                    CachedBatch::MemoryLiquid(liquid_array) => liquid_array,
                    CachedBatch::DiskLiquid => {
                        return None;
                    }
                    CachedBatch::DiskArrow => {
                        return None;
                    }
                };
                self.write_liquid_to_disk(&to_transcode, &liquid_array);
                self.try_insert(to_transcode, CachedBatch::DiskLiquid)
                    .expect("failed to insert on disk liquid");
                None
            }
            CacheAdvice::ToDisk(to_disk) => {
                match not_inserted {
                    CachedBatch::MemoryArrow(array) => {
                        self.write_arrow_to_disk(&to_disk, &array);
                        self.try_insert(to_disk, CachedBatch::DiskArrow)
                            .expect("failed to insert on disk arrow");
                    }
                    CachedBatch::MemoryLiquid(liquid_array) => {
                        self.write_liquid_to_disk(&to_disk, &liquid_array);
                        self.try_insert(to_disk, CachedBatch::DiskLiquid)
                            .expect("failed to insert on disk liquid");
                    }
                    CachedBatch::DiskLiquid | CachedBatch::DiskArrow => {
                        // Already on disk, nothing to do
                        return None;
                    }
                }
                None
            }
            CacheAdvice::Discard => None,
        }
    }

    fn write_liquid_to_disk(&self, entry_id: &EntryID, liquid_array: &LiquidArrayRef) {
        let disk_usage = self
            .io_worker
            .write_liquid_to_disk(self.config.cache_root_dir(), entry_id, liquid_array)
            .expect("failed to write liquid to disk");
        self.budget.add_used_disk_bytes(disk_usage);
    }

    fn write_arrow_to_disk(&self, entry_id: &EntryID, array: &arrow::array::ArrayRef) {
        let disk_usage = self
            .io_worker
            .write_arrow_to_disk(self.config.cache_root_dir(), entry_id, array)
            .expect("failed to write arrow to disk");
        self.budget.add_used_disk_bytes(disk_usage);
    }

    /// Insert a batch into the cache.
    pub fn insert(self: &Arc<Self>, entry_id: EntryID, batch_to_cache: ArrayRef) {
        let batch = {
            match self.config.cache_mode() {
                LiquidCacheMode::Arrow => CachedBatch::MemoryArrow(batch_to_cache),
                LiquidCacheMode::Liquid => {
                    let original_batch = batch_to_cache.clone();
                    submit_background_transcoding_task(batch_to_cache, self.clone(), entry_id);
                    CachedBatch::MemoryArrow(original_batch)
                }
                LiquidCacheMode::LiquidBlocking => {
                    let compressor_states = self.io_worker.get_compressor_for_entry(&entry_id);
                    let liquid_array =
                        transcode_liquid_inner(&batch_to_cache, compressor_states.as_ref())
                            .expect("Failed to transcode to liquid array");
                    CachedBatch::MemoryLiquid(liquid_array)
                }
            }
        };

        self.insert_inner(entry_id, batch);
    }

    /// Insert a batch into the cache, it will run cache replacement policy until the batch is inserted.
    pub(crate) fn insert_inner(&self, entry_id: EntryID, mut batch_to_cache: CachedBatch) {
        let mut loop_count = 0;
        loop {
            let Err((advice, not_inserted)) = self.try_insert(entry_id, batch_to_cache) else {
                self.policy.notify_insert(&entry_id);
                return;
            };

            let Some(not_inserted) = self.apply_advice(advice, not_inserted) else {
                return;
            };

            batch_to_cache = not_inserted;
            crate::utils::yield_now_if_shuttle();

            loop_count += 1;
            if loop_count > 20 {
                log::warn!("Cache store insert looped 20 times");
            }
        }
    }

    /// Get a batch from the cache.
    pub fn get(&self, entry_id: &EntryID) -> Option<CachedBatch> {
        let batch = self.index.get(entry_id);
        let batch_size = batch.as_ref().map(|b| b.memory_usage_bytes()).unwrap_or(0);
        self.tracer
            .trace_get(*entry_id, self.budget.memory_usage_bytes(), batch_size);
        // Notify the advisor that this entry was accessed
        self.policy.notify_access(entry_id);

        batch
    }

    /// Iterate over all entries in the cache.
    /// No guarantees are made about the order of the entries.
    /// Isolation level: read-committed
    pub fn for_each_entry(&self, mut f: impl FnMut(&EntryID, &CachedBatch)) {
        self.index.for_each(&mut f);
    }

    /// Reset the cache.
    pub fn reset(&self) {
        self.index.reset();
        self.budget.reset_usage();
    }

    /// Check if a batch is cached.
    pub fn is_cached(&self, entry_id: &EntryID) -> bool {
        self.index.is_cached(entry_id)
    }

    /// Get the config of the cache.
    pub fn config(&self) -> &CacheConfig {
        &self.config
    }

    /// Get the budget of the cache.
    pub fn budget(&self) -> &BudgetAccounting {
        &self.budget
    }

    /// Get the tracer of the cache.
    pub fn tracer(&self) -> &CacheTracer {
        &self.tracer
    }

    /// Get the compressor states of the cache.
    pub fn compressor_states(&self, entry_id: &EntryID) -> Arc<LiquidCompressorStates> {
        self.io_worker.get_compressor_for_entry(entry_id)
    }

    /// Flush all entries to disk.
    pub fn flush_all_to_disk(&self) {
        // Collect all entries that need to be flushed to disk
        let mut entries_to_flush = Vec::new();

        self.for_each_entry(|entry_id, batch| {
            match batch {
                CachedBatch::MemoryArrow(_) | CachedBatch::MemoryLiquid(_) => {
                    entries_to_flush.push((*entry_id, batch.clone()));
                }
                CachedBatch::DiskArrow | CachedBatch::DiskLiquid => {
                    // Already on disk, skip
                }
            }
        });

        // Now flush each entry to disk
        for (entry_id, batch) in entries_to_flush {
            match batch {
                CachedBatch::MemoryArrow(array) => {
                    self.write_arrow_to_disk(&entry_id, &array);
                    self.try_insert(entry_id, CachedBatch::DiskArrow)
                        .expect("failed to insert disk arrow entry");
                }
                CachedBatch::MemoryLiquid(liquid_array) => {
                    self.write_liquid_to_disk(&entry_id, &liquid_array);
                    self.try_insert(entry_id, CachedBatch::DiskLiquid)
                        .expect("failed to insert disk liquid entry");
                }
                CachedBatch::DiskArrow | CachedBatch::DiskLiquid => {
                    // Should not happen since we filtered these out above
                    unreachable!("Unexpected disk batch in flush operation");
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cache::{
        core::ArtIndex,
        policies::{CachePolicy, LruPolicy},
        utils::{create_cache_store, create_test_array, create_test_arrow_array},
    };
    use crate::sync::thread;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use arrow::array::Array;
    use liquid_cache_common::LiquidCacheMode;

    mod partitioned_hash_store_tests {
        use super::*;

        #[test]
        fn test_get_and_is_cached() {
            let store = ArtIndex::new();
            let entry_id1: EntryID = EntryID::from(1);
            let entry_id2: EntryID = EntryID::from(2);
            let array1 = create_test_array(100);

            // Initially, entries should not be cached
            assert!(!store.is_cached(&entry_id1));
            assert!(!store.is_cached(&entry_id2));
            assert!(store.get(&entry_id1).is_none());

            // Insert an entry and verify it's cached
            {
                store.insert(&entry_id1, array1.clone());
            }

            assert!(store.is_cached(&entry_id1));
            assert!(!store.is_cached(&entry_id2));

            // Get should return the cached value
            match store.get(&entry_id1) {
                Some(CachedBatch::MemoryArrow(arr)) => assert_eq!(arr.len(), 100),
                _ => panic!("Expected ArrowMemory batch"),
            }
        }

        #[test]
        fn test_reset() {
            let store = ArtIndex::new();
            let entry_id: EntryID = EntryID::from(1);
            let array = create_test_array(100);

            store.insert(&entry_id, array.clone());

            let entry_id: EntryID = EntryID::from(1);
            assert!(store.is_cached(&entry_id));

            store.reset();
            let entry_id: EntryID = EntryID::from(1);
            assert!(!store.is_cached(&entry_id));
        }
    }

    // Unified advice type for more concise testing
    #[derive(Debug)]
    struct TestPolicy {
        advice_type: AdviceType,
        target_id: Option<EntryID>,
        advice_count: AtomicUsize,
    }

    #[derive(Debug, Clone, Copy)]
    enum AdviceType {
        Evict,
        Transcode,
        TranscodeToDisk,
        ToDisk,
    }

    impl TestPolicy {
        fn new(advice_type: AdviceType, target_id: Option<EntryID>) -> Self {
            Self {
                advice_type,
                target_id,
                advice_count: AtomicUsize::new(0),
            }
        }
    }

    impl CachePolicy for TestPolicy {
        fn advise(&self, entry_id: &EntryID, _cache_mode: &LiquidCacheMode) -> CacheAdvice {
            self.advice_count.fetch_add(1, Ordering::SeqCst);
            match self.advice_type {
                AdviceType::Evict => {
                    let id_to_use = self.target_id.unwrap_or(*entry_id);
                    CacheAdvice::Evict(id_to_use)
                }
                AdviceType::Transcode => {
                    let id_to_use = self.target_id.unwrap_or(*entry_id);
                    CacheAdvice::Transcode(id_to_use)
                }
                AdviceType::TranscodeToDisk => CacheAdvice::TranscodeToDisk(*entry_id),
                AdviceType::ToDisk => CacheAdvice::ToDisk(*entry_id),
            }
        }
    }

    #[test]
    fn test_basic_cache_operations() {
        // Test basic insert, get, and size tracking in one test
        let budget_size = 10 * 1024;
        let store = create_cache_store(budget_size, Box::new(LruPolicy::new()));

        // 1. Initial budget should be empty
        assert_eq!(store.budget.memory_usage_bytes(), 0);

        // 2. Insert and verify first entry
        let entry_id1: EntryID = EntryID::from(1);
        let array1 = create_test_array(100);
        let size1 = array1.memory_usage_bytes();
        store.insert_inner(entry_id1, array1);

        // Verify budget usage and data correctness
        assert_eq!(store.budget.memory_usage_bytes(), size1);
        let retrieved1 = store.get(&entry_id1).unwrap();
        match retrieved1 {
            CachedBatch::MemoryArrow(arr) => assert_eq!(arr.len(), 100),
            _ => panic!("Expected ArrowMemory"),
        }

        let entry_id2: EntryID = EntryID::from(2);
        let array2 = create_test_array(200);
        let size2 = array2.memory_usage_bytes();
        store.insert_inner(entry_id2, array2);

        assert_eq!(store.budget.memory_usage_bytes(), size1 + size2);

        let array3 = create_test_array(150);
        let size3 = array3.memory_usage_bytes();
        store.insert_inner(entry_id1, array3);

        assert_eq!(store.budget.memory_usage_bytes(), size3 + size2);
        assert!(store.get(&EntryID::from(999)).is_none());
    }

    #[test]
    fn test_cache_advice_strategies() {
        // Comprehensive test of all three advice types

        // Create entry IDs we'll use throughout the test
        let entry_id1 = EntryID::from(1);
        let entry_id2 = EntryID::from(2);
        let entry_id3 = EntryID::from(3);

        // 1. Test EVICT advice
        {
            let advisor = TestPolicy::new(AdviceType::Evict, Some(entry_id1));
            let store = create_cache_store(8000, Box::new(advisor)); // Small budget to force advice

            store.insert_inner(entry_id1, create_test_array(800));
            match store.get(&entry_id1).unwrap() {
                CachedBatch::MemoryArrow(_) => {}
                other => panic!("Expected ArrowMemory, got {other:?}"),
            }

            store.insert_inner(entry_id2, create_test_array(800));
            match store.get(&entry_id1).unwrap() {
                CachedBatch::DiskLiquid => {}
                other => panic!("Expected OnDiskLiquid after eviction, got {other:?}"),
            }
        }

        // 2. Test TRANSCODE advice
        {
            let advisor = TestPolicy::new(AdviceType::Transcode, Some(entry_id1));
            let store = create_cache_store(8000, Box::new(advisor)); // Small budget

            store.insert_inner(entry_id1, create_test_array(800));
            match store.get(&entry_id1).unwrap() {
                CachedBatch::MemoryArrow(_) => {}
                other => panic!("Expected ArrowMemory, got {other:?}"),
            }

            store.insert_inner(entry_id2, create_test_array(800));
            match store.get(&entry_id1).unwrap() {
                CachedBatch::MemoryLiquid(_) => {}
                other => panic!("Expected LiquidMemory after transcoding, got {other:?}"),
            }
        }

        // 3. Test TRANSCODE_TO_DISK advice
        {
            let advisor = TestPolicy::new(AdviceType::TranscodeToDisk, None);
            let store = create_cache_store(8000, Box::new(advisor)); // Tiny budget to force disk storage

            store.insert_inner(entry_id1, create_test_array(800));
            store.insert_inner(entry_id3, create_test_array(800));
            match store.get(&entry_id3).unwrap() {
                CachedBatch::DiskLiquid => {}
                other => panic!("Expected OnDiskLiquid, got {other:?}"),
            }
        }

        // 4. Test TO_DISK advice - preserve format as-is
        {
            let advisor = TestPolicy::new(AdviceType::ToDisk, None);
            let store = create_cache_store(8000, Box::new(advisor)); // Small budget to force disk storage

            // Insert arrow data - should be written as DiskArrow (preserving format)
            store.insert_inner(entry_id1, create_test_array(800));
            store.insert_inner(entry_id2, create_test_array(800)); // This should trigger ToDisk advice

            match store.get(&entry_id2).unwrap() {
                CachedBatch::DiskArrow => {}
                other => panic!("Expected DiskArrow after ToDisk advice, got {other:?}"),
            }
        }
    }

    #[test]
    fn test_concurrent_cache_operations() {
        concurrent_cache_operations();
    }

    #[cfg(feature = "shuttle")]
    #[test]
    fn shuttle_cache_operations() {
        crate::utils::shuttle_test(concurrent_cache_operations);
    }

    fn concurrent_cache_operations() {
        let num_threads = 3;
        let ops_per_thread = 50;

        let budget_size = num_threads * ops_per_thread * 100 * 8 / 2;
        let store = Arc::new(create_cache_store(budget_size, Box::new(LruPolicy::new())));

        let mut handles = vec![];
        for thread_id in 0..num_threads {
            let store = store.clone();
            handles.push(thread::spawn(move || {
                for i in 0..ops_per_thread {
                    let unique_id = thread_id * ops_per_thread + i;
                    let entry_id: EntryID = EntryID::from(unique_id);
                    let array = create_test_arrow_array(100);
                    store.insert(entry_id, array);
                }
            }));
        }
        for handle in handles {
            handle.join().unwrap();
        }

        // Invariant 1: Every previously inserted entry can be retrieved
        for thread_id in 0..num_threads {
            for i in 0..ops_per_thread {
                let unique_id = thread_id * ops_per_thread + i;
                let entry_id: EntryID = EntryID::from(unique_id);
                assert!(store.get(&entry_id).is_some());
            }
        }

        // Invariant 2: Number of entries matches number of insertions
        assert_eq!(store.index.keys().len(), num_threads * ops_per_thread);
    }
}
