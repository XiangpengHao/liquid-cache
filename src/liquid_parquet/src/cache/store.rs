use congee::Congee;
use std::fmt::{Debug, Formatter};
use std::{fs::File, io::Write, path::PathBuf};

use super::{
    budget::BudgetAccounting,
    policies::CachePolicy,
    tracer::CacheTracer,
    transcode_liquid_inner,
    utils::{CacheConfig, ColumnAccessPath},
    CacheEntryID, CachedBatch, LiquidCompressorStates,
};
use crate::liquid_array::LiquidArrayRef;
use crate::sync::{Arc, RwLock};
use ahash::AHashMap;

#[derive(Debug)]
struct CompressorStates {
    states: RwLock<AHashMap<ColumnAccessPath, Arc<LiquidCompressorStates>>>,
}

impl CompressorStates {
    fn new() -> Self {
        Self {
            states: RwLock::new(AHashMap::new()),
        }
    }

    fn get_compressor(&self, entry_id: &CacheEntryID) -> Arc<LiquidCompressorStates> {
        let column_path = ColumnAccessPath::from(*entry_id);
        let mut states = self.states.write().unwrap();
        states
            .entry(column_path)
            .or_insert_with(|| Arc::new(LiquidCompressorStates::new()))
            .clone()
    }
}

struct ArtStore {
    art: Congee<CacheEntryID, CachedBatch>,
}

impl Debug for ArtStore {
    fn fmt(&self, _f: &mut Formatter<'_>) -> std::fmt::Result {
        Ok(())
    }
}

// TODO: Replace PartitionedHashStore to use congee instead

impl ArtStore {
    fn new() -> Self {
        let art: Congee<CacheEntryID, CachedBatch> = Congee::default();
        Self { art }
    }

    fn get(&self, entry_id: &CacheEntryID) -> Option<CachedBatch> {
        let guard = self.art.pin();
        self.art.get(entry_id, &guard)
    }

    fn is_cached(&self, entry_id: &CacheEntryID) -> bool {
        let guard = self.art.pin();
        self.art.get(entry_id, &guard).is_some()
    }

    fn insert(&self, entry_id: &CacheEntryID, batch: CachedBatch) {
        let guard = self.art.pin();
        self.art
            .insert(*entry_id, batch, &guard)
            .expect("Insertion failed");
    }

    fn reset(&mut self) {
        self.art = Congee::default();
    }

    fn for_each(&self, mut f: impl FnMut(&CacheEntryID, &CachedBatch)) {
        let guard = self.art.pin();
        for id in self.id_list.iter() {
            f(
                &CacheEntryID::from(id),
                &self
                    .art
                    .get(&CacheEntryID::from(id), &guard)
                    .expect("Failed to get value from ART"),
            );
        }
    }
}

#[derive(Debug)]
pub(crate) struct CacheStore {
    cached_data: ArtStore,
    config: CacheConfig,
    budget: BudgetAccounting,
    policy: Box<dyn CachePolicy>,
    tracer: CacheTracer,
    compressor_states: CompressorStates,
}

/// Advice given by the cache policy.
#[derive(PartialEq, Eq, Debug)]
pub enum CacheAdvice {
    /// Evict the entry with the given ID.
    Evict(CacheEntryID),
    /// Transcode the entry to disk.
    TranscodeToDisk(CacheEntryID),
    /// Transcode the entry to liquid memory.
    Transcode(CacheEntryID),
}

impl CacheStore {
    pub(super) fn new(
        batch_size: usize,
        max_cache_bytes: usize,
        cache_root_dir: PathBuf,
        policy: Box<dyn CachePolicy>,
    ) -> Self {
        let config = CacheConfig::new(batch_size, max_cache_bytes, cache_root_dir);
        Self {
            cached_data: ArtStore::new(),
            budget: BudgetAccounting::new(config.max_cache_bytes()),
            config,
            policy,
            tracer: CacheTracer::new(),
            compressor_states: CompressorStates::new(),
        }
    }

    fn insert_inner(
        &mut self,
        entry_id: CacheEntryID,
        cached_batch: CachedBatch,
    ) -> Result<(), (CacheAdvice, CachedBatch)> {
        let new_memory_size = cached_batch.memory_usage_bytes();
        if let Some(entry) = self.cached_data.get(&entry_id) {
            let old_memory_size = entry.memory_usage_bytes();
            if self
                .budget
                .try_update_memory_usage(old_memory_size, new_memory_size)
                .is_err()
            {
                let advice = self.policy.advise(&entry_id, &cached_batch);
                return Err((advice, cached_batch));
            }
            self.cached_data.insert(&entry_id, cached_batch);
        } else {
            if self.budget.try_reserve_memory(new_memory_size).is_err() {
                let advice = self.policy.advise(&entry_id, &cached_batch);
                return Err((advice, cached_batch));
            }
            self.cached_data.insert(&entry_id, cached_batch);
        }

        Ok(())
    }

    /// Returns Some(CachedBatch) if need to retry the insert.
    #[must_use]
    fn apply_advice(
        &mut self,
        advice: CacheAdvice,
        not_inserted: CachedBatch,
    ) -> Option<CachedBatch> {
        match advice {
            CacheAdvice::Transcode(to_transcode) => {
                let compressor_states = self.compressor_states.get_compressor(&to_transcode);
                let Some(to_transcode_batch) = self.cached_data.get(&to_transcode) else {
                    // The batch is gone, no need to transcode
                    return Some(not_inserted);
                };

                if let CachedBatch::ArrowMemory(array) = to_transcode_batch {
                    let liquid_array = transcode_liquid_inner(&array, compressor_states.as_ref())
                        .expect("Failed to transcode to liquid array");
                    let liquid_array = CachedBatch::LiquidMemory(liquid_array);
                    self.insert_inner(to_transcode, liquid_array)
                        .expect("Failed to insert the transcoded batch");
                }

                Some(not_inserted)
            }
            CacheAdvice::Evict(to_evict) => {
                let compressor_states = self.compressor_states.get_compressor(&to_evict);
                let Some(to_evict_batch) = self.cached_data.get(&to_evict) else {
                    return Some(not_inserted);
                };
                let liquid_array = match to_evict_batch {
                    CachedBatch::ArrowMemory(array) => {
                        transcode_liquid_inner(&array, compressor_states.as_ref())
                            .expect("Failed to transcode to liquid array")
                    }
                    CachedBatch::LiquidMemory(liquid_array) => liquid_array,
                    CachedBatch::OnDiskLiquid => {
                        // do nothing, already on disk
                        return Some(not_inserted);
                    }
                };
                self.write_to_disk(&to_evict, &liquid_array);
                self.insert_inner(to_evict, CachedBatch::OnDiskLiquid)
                    .expect("failed to insert on disk liquid");
                self.policy.notify_evict(&to_evict);
                Some(not_inserted)
            }
            CacheAdvice::TranscodeToDisk(to_transcode) => {
                let compressor_states = self.compressor_states.get_compressor(&to_transcode);
                let liquid_array = match not_inserted {
                    CachedBatch::ArrowMemory(array) => {
                        transcode_liquid_inner(&array, compressor_states.as_ref())
                            .expect("Failed to transcode to liquid array")
                    }
                    CachedBatch::LiquidMemory(liquid_array) => liquid_array,
                    CachedBatch::OnDiskLiquid => {
                        return None;
                    }
                };
                self.write_to_disk(&to_transcode, &liquid_array);
                self.insert_inner(to_transcode, CachedBatch::OnDiskLiquid)
                    .expect("failed to insert on disk liquid");
                None
            }
        }
    }

    fn write_to_disk(&self, entry_id: &CacheEntryID, liquid_array: &LiquidArrayRef) {
        let bytes = liquid_array.to_bytes();
        let file_path = entry_id.on_disk_path(self.config.cache_root_dir());
        let mut file = File::create(file_path).unwrap();
        file.write_all(&bytes).unwrap();
        self.budget.add_used_disk_bytes(bytes.len());
    }

    pub(super) fn insert(&mut self, entry_id: CacheEntryID, mut batch_to_cache: CachedBatch) {
        let mut loop_count = 0;
        loop {
            if batch_to_cache.memory_usage_bytes() > self.budget.max_memory_bytes() {
                let advice = CacheAdvice::TranscodeToDisk(entry_id);
                let not_inserted = self.apply_advice(advice, batch_to_cache);
                assert!(
                    not_inserted.is_none(),
                    "If batch is too big, it should be transcoded to disk"
                );
                return;
            }

            let Err((advice, not_inserted)) = self.insert_inner(entry_id, batch_to_cache) else {
                self.policy.notify_insert(&entry_id);
                return;
            };

            let Some(not_inserted) = self.apply_advice(advice, not_inserted) else {
                return;
            };

            batch_to_cache = not_inserted;
            crate::utils::yield_now_if_shuttle();

            loop_count += 1;
            if loop_count > 10 {
                log::warn!("Cache store insert looped 10 times");
            }
        }
    }

    pub(super) fn get(&self, entry_id: &CacheEntryID) -> Option<CachedBatch> {
        self.tracer
            .trace_get(*entry_id, self.budget.memory_usage_bytes());

        // Notify the advisor that this entry was accessed
        self.policy.notify_access(entry_id);

        self.cached_data.get(entry_id)
    }

    /// Iterate over all entries in the cache.
    /// No guarantees are made about the order of the entries.
    /// Isolation level: read-committed
    pub(super) fn for_each_entry(&self, mut f: impl FnMut(&CacheEntryID, &CachedBatch)) {
        self.cached_data.for_each(&mut f);
    }

    pub(super) fn reset(&mut self) {
        self.cached_data.reset();
        self.budget.reset_usage();
    }

    pub(super) fn is_cached(&self, entry_id: &CacheEntryID) -> bool {
        self.cached_data.is_cached(entry_id)
    }

    pub(super) fn config(&self) -> &CacheConfig {
        &self.config
    }

    pub(super) fn budget(&self) -> &BudgetAccounting {
        &self.budget
    }

    pub(super) fn tracer(&self) -> &CacheTracer {
        &self.tracer
    }

    pub(super) fn compressor_states(&self, entry_id: &CacheEntryID) -> Arc<LiquidCompressorStates> {
        self.compressor_states.get_compressor(entry_id)
    }
}

#[cfg(test)]
mod tests {
    use crate::cache::{
        policies::{CachePolicy, LruPolicy},
        utils::{create_cache_store, create_entry_id, create_test_array},
    };

    use super::*;
    use crate::sync::{
        atomic::{AtomicUsize, Ordering},
        thread,
    };
    use arrow::array::Array;

    mod partitioned_hash_store_tests {
        use super::*;

        #[test]
        fn test_get_and_is_cached() {
            let mut store = ArtStore::new();
            let entry_id1 = create_entry_id(1, 1, 1, 1);
            let entry_id2 = create_entry_id(2, 2, 2, 2);
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
                Some(CachedBatch::ArrowMemory(arr)) => assert_eq!(arr.len(), 100),
                _ => panic!("Expected ArrowMemory batch"),
            }
        }

        #[test]
        fn test_reset() {
            let mut store = ArtStore::new();
            let entry_id = create_entry_id(1, 1 as u64, 1, 1);
            let array = create_test_array(100);

            store.insert(&entry_id, array.clone());

            let entry_id = create_entry_id(1, 1 as u64, 1, 1);
            assert!(store.is_cached(&entry_id));

            store.reset();
            let entry_id = create_entry_id(1, 1 as u64, 1, 1);
            assert!(!store.is_cached(&entry_id));
        }
    }

    // Unified advice type for more concise testing
    #[derive(Debug)]
    struct TestPolicy {
        advice_type: AdviceType,
        target_id: Option<CacheEntryID>,
        advice_count: AtomicUsize,
    }

    #[derive(Debug, Clone, Copy)]
    enum AdviceType {
        Evict,
        Transcode,
        TranscodeToDisk,
    }

    impl TestPolicy {
        fn new(advice_type: AdviceType, target_id: Option<CacheEntryID>) -> Self {
            Self {
                advice_type,
                target_id,
                advice_count: AtomicUsize::new(0),
            }
        }
    }

    impl CachePolicy for TestPolicy {
        fn advise(&self, entry_id: &CacheEntryID, _to_insert: &CachedBatch) -> CacheAdvice {
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
            }
        }

        fn notify_evict(&self, _entry_id: &CacheEntryID) {}
    }

    #[test]
    fn test_basic_cache_operations() {
        // Test basic insert, get, and size tracking in one test
        let budget_size = 10 * 1024;
        let mut store = create_cache_store(budget_size, Box::new(LruPolicy::new()));

        // 1. Initial budget should be empty
        assert_eq!(store.budget.memory_usage_bytes(), 0);

        // 2. Insert and verify first entry
        let entry_id1 = create_entry_id(1, 1, 1, 1);
        let array1 = create_test_array(100);
        let size1 = array1.memory_usage_bytes();
        store.insert(entry_id1, array1);

        // Verify budget usage and data correctness
        assert_eq!(store.budget.memory_usage_bytes(), size1);
        let retrieved1 = store.get(&entry_id1).unwrap();
        match retrieved1 {
            CachedBatch::ArrowMemory(arr) => assert_eq!(arr.len(), 100),
            _ => panic!("Expected ArrowMemory"),
        }

        let entry_id2 = create_entry_id(2, 2, 2, 2);
        let array2 = create_test_array(200);
        let size2 = array2.memory_usage_bytes();
        store.insert(entry_id2, array2);

        assert_eq!(store.budget.memory_usage_bytes(), size1 + size2);

        let array3 = create_test_array(150);
        let size3 = array3.memory_usage_bytes();
        store.insert(entry_id1, array3);

        assert_eq!(store.budget.memory_usage_bytes(), size3 + size2);
        assert!(store.get(&create_entry_id(999, 999, 999, 999)).is_none());
    }

    #[test]
    fn test_cache_advice_strategies() {
        // Comprehensive test of all three advice types

        // Create entry IDs we'll use throughout the test
        let entry_id1 = create_entry_id(1, 1, 1, 1);
        let entry_id2 = create_entry_id(2, 2, 2, 2);
        let entry_id3 = create_entry_id(3, 3, 3, 3);

        // 1. Test EVICT advice
        {
            let advisor = TestPolicy::new(AdviceType::Evict, Some(entry_id1));
            let mut store = create_cache_store(8000, Box::new(advisor)); // Small budget to force advice

            let on_disk_path = entry_id1.on_disk_path(&store.config.cache_root_dir());
            std::fs::create_dir_all(on_disk_path.parent().unwrap()).unwrap();

            store.insert(entry_id1, create_test_array(800));
            match store.get(&entry_id1).unwrap() {
                CachedBatch::ArrowMemory(_) => {}
                other => panic!("Expected ArrowMemory, got {:?}", other),
            }

            store.insert(entry_id2, create_test_array(800));
            match store.get(&entry_id1).unwrap() {
                CachedBatch::OnDiskLiquid => {}
                other => panic!("Expected OnDiskLiquid after eviction, got {:?}", other),
            }
        }

        // 2. Test TRANSCODE advice
        {
            let advisor = TestPolicy::new(AdviceType::Transcode, Some(entry_id1));
            let mut store = create_cache_store(8000, Box::new(advisor)); // Small budget

            store.insert(entry_id1, create_test_array(800));
            match store.get(&entry_id1).unwrap() {
                CachedBatch::ArrowMemory(_) => {}
                other => panic!("Expected ArrowMemory, got {:?}", other),
            }

            store.insert(entry_id2, create_test_array(800));
            match store.get(&entry_id1).unwrap() {
                CachedBatch::LiquidMemory(_) => {}
                other => panic!("Expected LiquidMemory after transcoding, got {:?}", other),
            }
        }

        // 3. Test TRANSCODE_TO_DISK advice
        {
            let advisor = TestPolicy::new(AdviceType::TranscodeToDisk, None);
            let mut store = create_cache_store(8000, Box::new(advisor)); // Tiny budget to force disk storage

            let on_disk_path = entry_id3.on_disk_path(&store.config.cache_root_dir());
            std::fs::create_dir_all(on_disk_path.parent().unwrap()).unwrap();

            store.insert(entry_id1, create_test_array(800));
            store.insert(entry_id3, create_test_array(800));
            match store.get(&entry_id3).unwrap() {
                CachedBatch::OnDiskLiquid => {}
                other => panic!("Expected OnDiskLiquid, got {:?}", other),
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

    impl ArtStore {
        fn len(&self) -> usize {
            0
        }
    }

    fn concurrent_cache_operations() {
        let num_threads = 3;
        let ops_per_thread = 50;

        let budget_size = num_threads * ops_per_thread * 100 * 8 / 2;
        let mut store = Arc::new(create_cache_store(budget_size, Box::new(LruPolicy::new())));
        let entry_id = create_entry_id(1, 1, 1, 1);
        let on_disk_path = entry_id.on_disk_path(&store.config().cache_root_dir());
        std::fs::create_dir_all(on_disk_path.parent().unwrap()).unwrap();

        let mut handles = vec![];
        for thread_id in 0..num_threads {
            let store = store.clone();
            handles.push(thread::spawn(move || {
                for i in 0..ops_per_thread {
                    let unique_id = thread_id * ops_per_thread + i;
                    let entry_id = create_entry_id(1, 1, 1, unique_id as u16);
                    let array = create_test_array(100);
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
                let entry_id = create_entry_id(1, 1, 1, unique_id as u16);
                assert!(store.get(&entry_id).is_some());
            }
        }

        // Invariant 2: Number of entries matches number of insertions
        assert_eq!(store.cached_data.len(), num_threads * ops_per_thread);
    }
}
