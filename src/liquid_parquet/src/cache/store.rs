use std::{fs::File, io::Write, path::PathBuf, sync::Arc};

use dashmap::{DashMap, Entry, OccupiedEntry};

use crate::liquid_array::LiquidArrayRef;

use super::{
    CacheEntryID, CachedBatch, LiquidCompressorStates, budget::BudgetAccounting,
    tracer::CacheTracer, transcode_liquid_inner, utils::ColumnPath,
};
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
struct CompressorStates {
    states: DashMap<ColumnPath, Arc<LiquidCompressorStates>>,
}

impl CompressorStates {
    fn new() -> Self {
        Self {
            states: DashMap::new(),
        }
    }

    fn get_compressor(&self, entry_id: &CacheEntryID) -> Arc<LiquidCompressorStates> {
        let column_path = ColumnPath::from(*entry_id);
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
    advisor: Box<dyn CacheAdvisor>,
    tracer: CacheTracer,
    compressor_states: CompressorStates,
}

trait CacheAdvisor: std::fmt::Debug + Send + Sync {
    /// Give advice on what to do when cache is full.
    fn advise(
        &self,
        entry_id: &CacheEntryID,
        to_insert: &CachedBatch,
        cached: &DashMap<CacheEntryID, CachedBatch>,
    ) -> CacheAdvice;
}

#[derive(Debug)]
struct AlwaysWriteToDiskAdvisor;

impl CacheAdvisor for AlwaysWriteToDiskAdvisor {
    fn advise(
        &self,
        entry_id: &CacheEntryID,
        _to_insert: &CachedBatch,
        _cached: &DashMap<CacheEntryID, CachedBatch>,
    ) -> CacheAdvice {
        CacheAdvice::TranscodeToDisk(*entry_id)
    }
}

#[allow(dead_code)]
#[derive(Debug)]
struct EvictOddAdvisor;

impl CacheAdvisor for EvictOddAdvisor {
    fn advise(
        &self,
        entry_id: &CacheEntryID,
        _to_insert: &CachedBatch,
        _cached: &DashMap<CacheEntryID, CachedBatch>,
    ) -> CacheAdvice {
        let column_id = entry_id.column_id_inner();
        if column_id % 2 == 0 {
            CacheAdvice::Evict(*entry_id)
        } else {
            CacheAdvice::Transcode(*entry_id)
        }
    }
}

#[derive(Debug)]
pub(super) enum CacheAdvice {
    Evict(CacheEntryID),
    TranscodeToDisk(CacheEntryID),
    Transcode(CacheEntryID),
}

impl CacheStore {
    pub(super) fn new(batch_size: usize, max_cache_bytes: usize, cache_root_dir: PathBuf) -> Self {
        let config = CacheConfig::new(batch_size, max_cache_bytes, cache_root_dir);
        Self {
            cached_data: DashMap::new(),
            budget: BudgetAccounting::new(config.max_cache_bytes()),
            config,
            advisor: Box::new(AlwaysWriteToDiskAdvisor),
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
                        .advisor
                        .advise(&entry_id, &cached_batch, &self.cached_data);
                    return Err((advice, cached_batch));
                }
                entry.insert(cached_batch);
                entry
            }
            Entry::Vacant(entry) => {
                if self.budget.try_reserve_memory(new_memory_size).is_err() {
                    let advice = self
                        .advisor
                        .advise(&entry_id, &cached_batch, &self.cached_data);
                    return Err((advice, cached_batch));
                }
                entry.insert_entry(cached_batch)
            }
        };
        Ok(entry)
    }

    /// Returns Some(CachedBatch) if need to retry the insert.
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
                match to_transcode_batch {
                    CachedBatch::ArrowMemory(array) => {
                        let liquid_array =
                            transcode_liquid_inner(&array, compressor_states.as_ref())
                                .expect("Failed to transcode to liquid array");
                        let liquid_array = CachedBatch::LiquidMemory(liquid_array);
                        self.insert_inner(to_transcode, liquid_array)
                            .expect("Failed to insert the transcoded batch");
                    }
                    _ => {}
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
                Some(not_inserted)
            }
            CacheAdvice::TranscodeToDisk(to_transcode) => {
                let compressor_states = self.compressor_states.get_compressor(&to_transcode);
                let liquid_array = match not_inserted {
                    CachedBatch::ArrowMemory(array) => {
                        let liquid_array =
                            transcode_liquid_inner(&array, compressor_states.as_ref())
                                .expect("Failed to transcode to liquid array");
                        liquid_array
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
        let file_path = entry_id.on_disk_path(&self.config.cache_root_dir());
        let mut file = File::create(file_path).unwrap();
        file.write_all(&bytes).unwrap();
        self.insert_inner(*entry_id, CachedBatch::OnDiskLiquid)
            .expect("failed to insert on disk liquid");
        self.budget.add_used_disk_bytes(bytes.len());
    }

    pub(super) fn insert(&self, entry_id: CacheEntryID, cached_batch: CachedBatch) {
        let Err((advice, not_inserted)) = self.insert_inner(entry_id, cached_batch) else {
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
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use crate::cache::utils::BatchID;

    use super::*;
    use arrow::array::{ArrayRef, Int32Array};
    use tempfile::tempdir;
    #[test]
    fn test_cache_store_basic_operations() {
        let temp_dir = tempdir().unwrap();
        let cache_dir = temp_dir.path().to_path_buf();
        let batch_size = 128;
        let cache_max_bytes = 1000;

        let store = CacheStore::new(batch_size, cache_max_bytes, cache_dir);

        let entry_id = CacheEntryID::new(1, 2, 3, BatchID::from_raw(4));
        let cached_batch = create_test_cached_batch(200);

        assert!(!store.is_cached(&entry_id));
        assert_eq!(store.budget().memory_usage_bytes(), 0);

        store.insert(entry_id, cached_batch.clone());
        assert_eq!(
            store.budget().memory_usage_bytes(),
            cached_batch.memory_usage_bytes()
        );
        assert!(store.is_cached(&entry_id));

        let retrieved = store.get(&entry_id).unwrap();
        assert_eq!(
            retrieved.memory_usage_bytes(),
            cached_batch.memory_usage_bytes()
        );

        store.insert(entry_id, cached_batch.clone());
        assert!(store.is_cached(&entry_id));
        store.reset();
        assert!(!store.is_cached(&entry_id));
        assert_eq!(store.budget().memory_usage_bytes(), 0);
    }

    #[test]
    fn test_cache_store_memory_management() {
        let temp_dir = tempdir().unwrap();
        let cache_dir = temp_dir.path().to_path_buf();
        let batch_size = 128;
        let cache_max_bytes = 600;

        let store = CacheStore::new(batch_size, cache_max_bytes, cache_dir);

        let entry_id1 = CacheEntryID::new(1, 1, 1, BatchID::from_raw(1));
        let entry_id2 = CacheEntryID::new(2, 2, 2, BatchID::from_raw(2));
        let entry_id3 = CacheEntryID::new(3, 3, 3, BatchID::from_raw(3));

        let batch1 = create_test_cached_batch(50);
        let batch2 = create_test_cached_batch(50);
        let batch3 = create_test_cached_batch(50);

        store.insert(entry_id1, batch1.clone());
        store.insert(entry_id2, batch2.clone());
        assert_eq!(
            store.budget().memory_usage_bytes(),
            batch1.memory_usage_bytes() + batch2.memory_usage_bytes()
        );

        store.insert(entry_id3, batch3.clone());
        let is_id1_cached = store.is_cached(&entry_id1);
        let is_id2_cached = store.is_cached(&entry_id2);
        let is_id3_cached = store.is_cached(&entry_id3);

        assert!(
            is_id3_cached,
            "Entry 3 should be cached after successful insert"
        );
        assert!(
            !(is_id1_cached && is_id2_cached),
            "At least one of the previous entries should have been evicted"
        );

        if store.is_cached(&entry_id1) {
            let larger_batch = create_test_cached_batch(400);
            store.insert(entry_id1, larger_batch);
        }
    }

    fn create_test_cached_batch(element_count: usize) -> CachedBatch {
        let array = Arc::new(Int32Array::from(vec![0; element_count])) as ArrayRef;
        CachedBatch::ArrowMemory(array)
    }
}
