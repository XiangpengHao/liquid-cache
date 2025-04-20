use std::{fs::File, io::Write, path::PathBuf, sync::Arc};

use dashmap::{DashMap, Entry, OccupiedEntry};

use super::{
    CacheEntryID, CachedBatch, LiquidCompressorStates,
    budget::BudgetAccounting,
    policies::CachePolicy,
    tracer::CacheTracer,
    transcode_liquid_inner,
    utils::{CacheConfig, ColumnAccessPath},
};
use crate::liquid_array::LiquidArrayRef;

#[derive(Debug)]
struct CompressorStates {
    states: DashMap<ColumnAccessPath, Arc<LiquidCompressorStates>>,
}

impl CompressorStates {
    fn new() -> Self {
        Self {
            states: DashMap::new(),
        }
    }

    fn get_compressor(&self, entry_id: &CacheEntryID) -> Arc<LiquidCompressorStates> {
        let column_path = ColumnAccessPath::from(*entry_id);
        self.states
            .entry(column_path)
            .or_insert_with(|| Arc::new(LiquidCompressorStates::new()))
            .clone()
    }
}

#[derive(Debug)]
pub(crate) struct CacheStore {
    cached_data: DashMap<CacheEntryID, CachedBatch>,
    config: CacheConfig,
    budget: BudgetAccounting,
    policy: Box<dyn CachePolicy>,
    tracer: CacheTracer,
    compressor_states: CompressorStates,
}

#[allow(unused)]
#[derive(Debug)]
pub enum CacheAdvice {
    Evict(CacheEntryID),
    TranscodeToDisk(CacheEntryID),
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
            cached_data: DashMap::new(),
            budget: BudgetAccounting::new(config.max_cache_bytes()),
            config,
            policy,
            tracer: CacheTracer::new(),
            compressor_states: CompressorStates::new(),
        }
    }

    fn insert_inner(
        &self,
        entry_id: CacheEntryID,
        cached_batch: CachedBatch,
    ) -> Result<OccupiedEntry<'_, CacheEntryID, CachedBatch>, (CacheAdvice, CachedBatch)> {
        let new_memory_size = cached_batch.memory_usage_bytes();
        let entry = self.cached_data.entry(entry_id);
        let entry = match entry {
            Entry::Occupied(mut entry) => {
                let old = entry.get();
                let old_memory_size = old.memory_usage_bytes();

                if self
                    .budget
                    .try_update_memory_usage(old_memory_size, new_memory_size)
                    .is_err()
                {
                    let advice = self
                        .policy
                        .advise(&entry_id, &cached_batch, &self.cached_data);
                    return Err((advice, cached_batch));
                }
                entry.insert(cached_batch);
                entry
            }
            Entry::Vacant(entry) => {
                if self.budget.try_reserve_memory(new_memory_size).is_err() {
                    let advice = self
                        .policy
                        .advise(&entry_id, &cached_batch, &self.cached_data);
                    return Err((advice, cached_batch));
                }
                entry.insert_entry(cached_batch)
            }
        };
        Ok(entry)
    }

    /// Returns Some(CachedBatch) if need to retry the insert.
    #[must_use]
    fn apply_advice(&self, advice: CacheAdvice, not_inserted: CachedBatch) -> Option<CachedBatch> {
        match advice {
            CacheAdvice::Transcode(to_transcode) => {
                let compressor_states = self.compressor_states.get_compressor(&to_transcode);
                let Some(to_transcode_batch) = self
                    .cached_data
                    .get(&to_transcode)
                    .map(|entry| entry.value().clone())
                else {
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
                let Some(to_evict_batch) = self
                    .cached_data
                    .get(&to_evict)
                    .map(|entry| entry.value().clone())
                else {
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
                None
            }
        }
    }

    fn write_to_disk(&self, entry_id: &CacheEntryID, liquid_array: &LiquidArrayRef) {
        let bytes = liquid_array.to_bytes();
        let file_path = entry_id.on_disk_path(self.config.cache_root_dir());
        let mut file = File::create(file_path).unwrap();
        file.write_all(&bytes).unwrap();
        self.insert_inner(*entry_id, CachedBatch::OnDiskLiquid)
            .expect("failed to insert on disk liquid");
        self.budget.add_used_disk_bytes(bytes.len());
    }

    pub(super) fn insert(&self, entry_id: CacheEntryID, batch_to_cache: CachedBatch) {
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

        // retry the insert
        self.insert(entry_id, not_inserted);
    }

    pub(super) fn get(&self, entry_id: &CacheEntryID) -> Option<CachedBatch> {
        self.tracer
            .trace_get(*entry_id, self.budget.memory_usage_bytes());

        // Notify the advisor that this entry was accessed
        self.policy.notify_access(entry_id);

        self.cached_data
            .get(entry_id)
            .map(|entry| entry.value().clone())
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

    pub(super) fn iter(&self) -> dashmap::iter::Iter<'_, CacheEntryID, CachedBatch> {
        self.cached_data.iter()
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
    use arrow::array::Array;
    use std::sync::atomic::{AtomicUsize, Ordering};

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
        fn advise(
            &self,
            entry_id: &CacheEntryID,
            _to_insert: &CachedBatch,
            _cached: &DashMap<CacheEntryID, CachedBatch>,
        ) -> CacheAdvice {
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
    }

    #[test]
    fn test_basic_cache_operations() {
        // Test basic insert, get, and size tracking in one test
        let budget_size = 10 * 1024;
        let store = create_cache_store(budget_size, Box::new(LruPolicy::new()));

        // 1. Initial budget should be empty
        assert_eq!(store.budget.memory_usage_bytes(), 0);

        // 2. Insert and verify first entry
        let entry_id1 = create_entry_id(1, 1, 1, 1);
        let array1 = create_test_array(100);
        let size1 = array1.get_array_memory_size();
        store.insert(entry_id1, CachedBatch::ArrowMemory(array1.clone()));

        // Verify budget usage and data correctness
        assert_eq!(store.budget.memory_usage_bytes(), size1);
        let retrieved1 = store.get(&entry_id1).unwrap();
        match retrieved1 {
            CachedBatch::ArrowMemory(arr) => assert_eq!(arr.len(), 100),
            _ => panic!("Expected ArrowMemory"),
        }

        let entry_id2 = create_entry_id(2, 2, 2, 2);
        let array2 = create_test_array(200);
        let size2 = array2.get_array_memory_size();
        store.insert(entry_id2, CachedBatch::ArrowMemory(array2));

        assert_eq!(store.budget.memory_usage_bytes(), size1 + size2);

        let array3 = create_test_array(150);
        let size3 = array3.get_array_memory_size();
        store.insert(entry_id1, CachedBatch::ArrowMemory(array3));

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
            let store = create_cache_store(8000, Box::new(advisor)); // Small budget to force advice

            let on_disk_path = entry_id1.on_disk_path(&store.config.cache_root_dir());
            std::fs::create_dir_all(on_disk_path.parent().unwrap()).unwrap();

            store.insert(entry_id1, CachedBatch::ArrowMemory(create_test_array(800)));
            match store.get(&entry_id1).unwrap() {
                CachedBatch::ArrowMemory(_) => {}
                other => panic!("Expected ArrowMemory, got {:?}", other),
            }

            store.insert(entry_id2, CachedBatch::ArrowMemory(create_test_array(800)));
            match store.get(&entry_id1).unwrap() {
                CachedBatch::OnDiskLiquid => {}
                other => panic!("Expected OnDiskLiquid after eviction, got {:?}", other),
            }
        }

        // 2. Test TRANSCODE advice
        {
            let advisor = TestPolicy::new(AdviceType::Transcode, Some(entry_id1));
            let store = create_cache_store(8000, Box::new(advisor)); // Small budget

            store.insert(entry_id1, CachedBatch::ArrowMemory(create_test_array(800)));
            match store.get(&entry_id1).unwrap() {
                CachedBatch::ArrowMemory(_) => {}
                other => panic!("Expected ArrowMemory, got {:?}", other),
            }

            store.insert(entry_id2, CachedBatch::ArrowMemory(create_test_array(800)));
            match store.get(&entry_id1).unwrap() {
                CachedBatch::LiquidMemory(_) => {}
                other => panic!("Expected LiquidMemory after transcoding, got {:?}", other),
            }
        }

        // 3. Test TRANSCODE_TO_DISK advice
        {
            let advisor = TestPolicy::new(AdviceType::TranscodeToDisk, None);
            let store = create_cache_store(8000, Box::new(advisor)); // Tiny budget to force disk storage

            let on_disk_path = entry_id3.on_disk_path(&store.config.cache_root_dir());
            std::fs::create_dir_all(on_disk_path.parent().unwrap()).unwrap();

            store.insert(entry_id1, CachedBatch::ArrowMemory(create_test_array(800)));
            store.insert(entry_id3, CachedBatch::ArrowMemory(create_test_array(800)));
            match store.get(&entry_id3).unwrap() {
                CachedBatch::OnDiskLiquid => {}
                other => panic!("Expected OnDiskLiquid, got {:?}", other),
            }
        }
    }
}
