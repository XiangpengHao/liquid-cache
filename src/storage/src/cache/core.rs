use arrow::array::ArrayRef;
use std::path::PathBuf;
use std::{fmt::Debug, path::Path};

use super::{
    budget::BudgetAccounting, cache_policies::CachePolicy, cached_data::CachedBatch,
    cached_data::CachedData, tracer::CacheTracer, transcode::transcode_liquid_inner,
    utils::CacheConfig,
};
use crate::cache::squeeze_policies::{
    SqueezeNoHybridPolicy, SqueezePolicy, SqueezeToDiskPolicy, SqueezeToLiquidPolicy,
};
use crate::cache::transcode::submit_background_transcoding_task;
use crate::cache::utils::{LiquidCompressorStates, arrow_to_bytes};
use crate::cache::{index::ArtIndex, utils::EntryID};
use crate::cache_policies::FiloPolicy;
use crate::sync::Arc;
use liquid_cache_common::LiquidCacheMode;

/// A trait for objects that can handle IO operations for the cache.
pub trait IoContext: Debug + Send + Sync {
    /// Get the base directory for the cache eviction, i.e., evicted data will be written to this directory.
    fn base_dir(&self) -> &Path;

    /// Get the compressor for an entry.
    fn get_compressor_for_entry(&self, entry_id: &EntryID) -> Arc<LiquidCompressorStates>;

    /// Get the path to the arrow file for an entry.
    fn entry_arrow_path(&self, entry_id: &EntryID) -> PathBuf;

    /// Get the path to the liquid file for an entry.
    fn entry_liquid_path(&self, entry_id: &EntryID) -> PathBuf;
}

/// A default implementation of [IoContext] that uses the default compressor.
#[derive(Debug)]
pub struct DefaultIoContext {
    compressor_state: Arc<LiquidCompressorStates>,
    base_dir: PathBuf,
}

impl DefaultIoContext {
    /// Create a new instance of [DefaultIoContext].
    pub fn new(base_dir: PathBuf) -> Self {
        Self {
            compressor_state: Arc::new(LiquidCompressorStates::new()),
            base_dir,
        }
    }
}

impl IoContext for DefaultIoContext {
    fn base_dir(&self) -> &Path {
        &self.base_dir
    }

    fn get_compressor_for_entry(&self, _entry_id: &EntryID) -> Arc<LiquidCompressorStates> {
        self.compressor_state.clone()
    }

    fn entry_arrow_path(&self, entry_id: &EntryID) -> PathBuf {
        self.base_dir()
            .join(format!("{:016x}.arrow", usize::from(*entry_id)))
    }

    fn entry_liquid_path(&self, entry_id: &EntryID) -> PathBuf {
        self.base_dir()
            .join(format!("{:016x}.liquid", usize::from(*entry_id)))
    }
}

/// Snapshot of cache statistics.
#[derive(Debug, Clone)]
pub struct CacheStats {
    /// Total number of entries in the cache.
    pub total_entries: usize,
    /// Number of in-memory Arrow entries.
    pub memory_arrow_entries: usize,
    /// Number of in-memory Liquid entries.
    pub memory_liquid_entries: usize,
    /// Number of in-memory Hybrid-Liquid entries.
    pub memory_hybrid_liquid_entries: usize,
    /// Number of on-disk Liquid entries.
    pub disk_liquid_entries: usize,
    /// Number of on-disk Arrow entries.
    pub disk_arrow_entries: usize,
    /// Total memory usage of the cache.
    pub memory_usage_bytes: usize,
    /// Total disk usage of the cache.
    pub disk_usage_bytes: usize,
    /// Maximum cache size.
    pub max_cache_bytes: usize,
    /// Cache mode.
    pub cache_mode: LiquidCacheMode,
    /// Cache root directory.
    pub cache_root_dir: PathBuf,
}

/// Builder for [CacheStorage].
///
/// Example:
/// ```rust
/// use liquid_cache_storage::cache::CacheStorageBuilder;
/// use liquid_cache_storage::common::LiquidCacheMode;
///
///
/// let _storage = CacheStorageBuilder::new()
///     .with_batch_size(8192)
///     .with_max_cache_bytes(1024 * 1024 * 1024)
///     .with_cache_mode(LiquidCacheMode::Liquid)
///     .with_policy(Box::new(liquid_cache_storage::cache_policies::FiloPolicy::new()))
///     .build();
/// ```
pub struct CacheStorageBuilder {
    batch_size: usize,
    max_cache_bytes: usize,
    cache_dir: Option<PathBuf>,
    cache_mode: LiquidCacheMode,
    policy: Box<dyn CachePolicy>,
    io_worker: Option<Arc<dyn IoContext>>,
}

impl Default for CacheStorageBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl CacheStorageBuilder {
    /// Create a new instance of CacheStorageBuilder.
    pub fn new() -> Self {
        Self {
            batch_size: 8192,
            max_cache_bytes: 1024 * 1024 * 1024,
            cache_dir: None,
            cache_mode: LiquidCacheMode::Liquid,
            policy: Box::new(FiloPolicy::new()),
            io_worker: None,
        }
    }

    /// Set the cache directory for the cache.
    /// Default is a temporary directory.
    pub fn with_cache_dir(mut self, cache_dir: PathBuf) -> Self {
        self.cache_dir = Some(cache_dir);
        self
    }

    /// Set the batch size for the cache.
    /// Default is 8192.
    pub fn with_batch_size(mut self, batch_size: usize) -> Self {
        self.batch_size = batch_size;
        self
    }

    /// Set the max cache bytes for the cache.
    /// Default is 1GB.
    pub fn with_max_cache_bytes(mut self, max_cache_bytes: usize) -> Self {
        self.max_cache_bytes = max_cache_bytes;
        self
    }

    /// Set the cache mode for the cache.
    /// Default is LiquidCacheMode::Liquid.
    pub fn with_cache_mode(mut self, cache_mode: LiquidCacheMode) -> Self {
        self.cache_mode = cache_mode;
        self
    }

    /// Set the cache policy for the cache.
    /// Default is FiloPolicy.
    pub fn with_policy(mut self, policy: Box<dyn CachePolicy>) -> Self {
        self.policy = policy;
        self
    }

    /// Set the io worker for the cache.
    /// Default is [DefaultIoContext].
    pub fn with_io_worker(mut self, io_worker: Arc<dyn IoContext>) -> Self {
        self.io_worker = Some(io_worker);
        self
    }

    /// Build the cache storage.
    ///
    /// The cache storage is wrapped in an [Arc] to allow for concurrent access.
    pub fn build(self) -> Arc<CacheStorage> {
        let cache_dir = self
            .cache_dir
            .unwrap_or_else(|| tempfile::tempdir().unwrap().keep());
        let io_worker = self
            .io_worker
            .unwrap_or_else(|| Arc::new(DefaultIoContext::new(cache_dir.clone())));
        Arc::new(CacheStorage::new(
            self.batch_size,
            self.max_cache_bytes,
            cache_dir,
            self.cache_mode,
            self.policy,
            io_worker,
        ))
    }
}

/// Cache storage for liquid cache.
///
/// Example (sans-IO read):
/// ```rust
/// use liquid_cache_storage::cache::{CacheStorageBuilder, EntryID};
/// use liquid_cache_storage::cache::io_state::{SansIo, TryGet, IoRequest, IoStateMachine};
/// use arrow::array::UInt64Array;
/// use std::sync::Arc;
///
/// // Tiny IO helper for sans-IO reads
/// fn read_all(req: &IoRequest) -> bytes::Bytes {
///     bytes::Bytes::from(std::fs::read(req.path()).unwrap())
/// }
///
/// let storage = CacheStorageBuilder::new().build();
///
/// let entry_id = EntryID::from(0);
/// let arrow_array = Arc::new(UInt64Array::from_iter_values(0..32));
/// storage.insert(entry_id, arrow_array.clone());
///
/// // Move to disk so we can see the Pending path too
/// storage.flush_all_to_disk();
///
/// let batch = storage.get(&entry_id).unwrap();
/// match batch.get_arrow_array() {
///     SansIo::Ready(out) => assert_eq!(out.as_ref(), arrow_array.as_ref()),
///     SansIo::Pending((mut state, req)) => {
///         state.feed(read_all(&req));
///         let TryGet::Ready(out) = state.try_get() else { panic!("still pending") };
///         assert_eq!(out.as_ref(), arrow_array.as_ref());
///     }
/// }
/// ```
#[derive(Debug)]
pub struct CacheStorage {
    index: ArtIndex,
    config: CacheConfig,
    budget: BudgetAccounting,
    cache_policy: Box<dyn CachePolicy>,
    squeeze_policy: Box<dyn SqueezePolicy>,
    tracer: CacheTracer,
    io_context: Arc<dyn IoContext>,
}

impl CacheStorage {
    /// Return current cache statistics: counts and resource usage.
    pub fn stats(&self) -> CacheStats {
        // Count entries by residency/format
        let total_entries = self.index.entry_count();

        let mut memory_arrow_entries = 0usize;
        let mut memory_liquid_entries = 0usize;
        let mut memory_hybrid_liquid_entries = 0usize;
        let mut disk_liquid_entries = 0usize;
        let mut disk_arrow_entries = 0usize;

        self.index.for_each(|_, batch| match batch {
            CachedBatch::MemoryArrow(_) => memory_arrow_entries += 1,
            CachedBatch::MemoryLiquid(_) => memory_liquid_entries += 1,
            CachedBatch::MemoryHybridLiquid(_) => memory_hybrid_liquid_entries += 1,
            CachedBatch::DiskLiquid => disk_liquid_entries += 1,
            CachedBatch::DiskArrow => disk_arrow_entries += 1,
        });

        let memory_usage_bytes = self.budget.memory_usage_bytes();
        let disk_usage_bytes = self.budget.disk_usage_bytes();

        CacheStats {
            total_entries,
            memory_arrow_entries,
            memory_liquid_entries,
            memory_hybrid_liquid_entries,
            disk_liquid_entries,
            disk_arrow_entries,
            memory_usage_bytes,
            disk_usage_bytes,
            max_cache_bytes: self.config.max_cache_bytes(),
            cache_mode: *self.config.cache_mode(),
            cache_root_dir: self.config.cache_root_dir().clone(),
        }
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
                    let compressor_states = self.io_context.get_compressor_for_entry(&entry_id);
                    let liquid_array =
                        transcode_liquid_inner(&batch_to_cache, compressor_states.as_ref())
                            .expect("Failed to transcode to liquid array");
                    CachedBatch::MemoryLiquid(liquid_array)
                }
            }
        };

        self.insert_inner(entry_id, batch);
    }

    /// Get a batch from the cache.
    pub fn get(&self, entry_id: &EntryID) -> Option<CachedData<'_>> {
        let batch = self.index.get(entry_id);
        let batch_size = batch.as_ref().map(|b| b.memory_usage_bytes()).unwrap_or(0);
        self.tracer
            .trace_get(*entry_id, self.budget.memory_usage_bytes(), batch_size);
        // Notify the advisor that this entry was accessed
        self.cache_policy.notify_access(entry_id);

        batch.map(|b| CachedData::new(b, *entry_id, self.io_context.as_ref()))
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
        self.io_context.get_compressor_for_entry(entry_id)
    }

    /// Flush all entries to disk.
    pub fn flush_all_to_disk(&self) {
        // Collect all entries that need to be flushed to disk
        let mut entries_to_flush = Vec::new();

        self.for_each_entry(|entry_id, batch| {
            match batch {
                CachedBatch::MemoryArrow(_)
                | CachedBatch::MemoryLiquid(_)
                | CachedBatch::MemoryHybridLiquid(_) => {
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
                    let bytes = arrow_to_bytes(&array).expect("failed to convert arrow to bytes");
                    let path = self.io_context.entry_arrow_path(&entry_id);
                    self.write_to_disk_blocking(&path, &bytes);
                    self.try_insert(entry_id, CachedBatch::DiskArrow)
                        .expect("failed to insert disk arrow entry");
                }
                CachedBatch::MemoryLiquid(liquid_array) => {
                    let liquid_bytes = liquid_array.to_bytes();
                    let path = self.io_context.entry_liquid_path(&entry_id);
                    self.write_to_disk_blocking(&path, &liquid_bytes);
                    self.try_insert(entry_id, CachedBatch::DiskLiquid)
                        .expect("failed to insert disk liquid entry");
                }
                CachedBatch::MemoryHybridLiquid(_) => {
                    // We don't have to do anything, because it's already on disk
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

impl CacheStorage {
    /// returns the batch that was written to disk
    fn write_in_memory_batch_to_disk(&self, entry_id: EntryID, batch: CachedBatch) -> CachedBatch {
        match batch {
            CachedBatch::MemoryArrow(array) => {
                let bytes = arrow_to_bytes(&array).expect("failed to convert arrow to bytes");
                let path = self.io_context.entry_arrow_path(&entry_id);
                self.write_to_disk_blocking(&path, &bytes);
                CachedBatch::DiskArrow
            }
            CachedBatch::MemoryLiquid(liquid_array) => {
                let liquid_bytes = liquid_array.to_bytes();
                let path = self.io_context.entry_liquid_path(&entry_id);
                self.write_to_disk_blocking(&path, &liquid_bytes);
                CachedBatch::DiskLiquid
            }
            CachedBatch::DiskLiquid
            | CachedBatch::DiskArrow
            | CachedBatch::MemoryHybridLiquid(_) => {
                unreachable!("Unexpected batch in write_in_memory_batch_to_disk")
            }
        }
    }

    /// Insert a batch into the cache, it will run cache replacement policy until the batch is inserted.
    pub(crate) fn insert_inner(&self, entry_id: EntryID, mut batch_to_cache: CachedBatch) {
        let mut loop_count = 0;
        loop {
            let Err(not_inserted) = self.try_insert(entry_id, batch_to_cache) else {
                self.cache_policy.notify_insert(&entry_id);
                return;
            };

            let advice = self.cache_policy.advise(8);
            if advice.is_empty() {
                // no advice, because the cache is already empty
                // this can happen if the entry to be inserted is too large, in that case,
                // we write it to disk
                let on_disk_batch = self.write_in_memory_batch_to_disk(entry_id, not_inserted);
                batch_to_cache = on_disk_batch;
                continue;
            }
            self.apply_advice(advice);

            batch_to_cache = not_inserted;
            crate::utils::yield_now_if_shuttle();

            loop_count += 1;
            if loop_count > 20 {
                log::warn!("Cache store insert looped {loop_count} times");
            }
        }
    }

    /// Create a new instance of CacheStorage.
    fn new(
        batch_size: usize,
        max_cache_bytes: usize,
        cache_dir: PathBuf,
        cache_mode: LiquidCacheMode,
        policy: Box<dyn CachePolicy>,
        io_worker: Arc<dyn IoContext>,
    ) -> Self {
        let squeeze_policy: Box<dyn SqueezePolicy> = match cache_mode {
            LiquidCacheMode::Arrow => Box::new(SqueezeToDiskPolicy),
            LiquidCacheMode::Liquid => Box::new(SqueezeToLiquidPolicy),
            LiquidCacheMode::LiquidBlocking => Box::new(SqueezeNoHybridPolicy),
        };
        let config = CacheConfig::new(batch_size, max_cache_bytes, cache_dir, cache_mode);
        Self {
            index: ArtIndex::new(),
            budget: BudgetAccounting::new(config.max_cache_bytes()),
            config,
            cache_policy: policy,
            squeeze_policy,
            tracer: CacheTracer::new(),
            io_context: io_worker,
        }
    }

    fn try_insert(&self, entry_id: EntryID, cached_batch: CachedBatch) -> Result<(), CachedBatch> {
        let new_memory_size = cached_batch.memory_usage_bytes();
        if let Some(entry) = self.index.get(&entry_id) {
            let old_memory_size = entry.memory_usage_bytes();
            if self
                .budget
                .try_update_memory_usage(old_memory_size, new_memory_size)
                .is_err()
            {
                return Err(cached_batch);
            }
            self.index.insert(&entry_id, cached_batch);
        } else {
            if self.budget.try_reserve_memory(new_memory_size).is_err() {
                return Err(cached_batch);
            }
            self.index.insert(&entry_id, cached_batch);
        }

        Ok(())
    }

    fn apply_advice(&self, advice: Vec<EntryID>) {
        #[cfg(target_os = "linux")]
        {
            let mut executor = super::io::Executor::new();
            for adv in advice {
                executor.spawn(self.apply_advice_inner(adv));
            }
            executor.join();
        }
        #[cfg(not(target_os = "linux"))]
        {
            for adv in advice {
                self.apply_advice_inner_blocking(adv);
            }
        }
    }

    #[cfg(not(target_os = "linux"))]
    fn apply_advice_inner_blocking(&self, to_squeeze: EntryID) {
        let Some(to_squeeze_batch) = self.index.get(&to_squeeze) else {
            return;
        };
        let compressor = self.io_context.get_compressor_for_entry(&to_squeeze);
        let (new_batch, bytes_to_write) = self
            .squeeze_policy
            .squeeze(to_squeeze_batch, compressor.as_ref());

        if let Some(bytes_to_write) = bytes_to_write {
            match new_batch {
                CachedBatch::DiskArrow => {
                    let path = self.io_context.entry_arrow_path(&to_squeeze);
                    self.write_to_disk_blocking(&path, &bytes_to_write);
                }
                CachedBatch::DiskLiquid | CachedBatch::MemoryHybridLiquid(_) => {
                    let path = self.io_context.entry_liquid_path(&to_squeeze);
                    self.write_to_disk_blocking(&path, &bytes_to_write);
                }
                CachedBatch::MemoryArrow(_) | CachedBatch::MemoryLiquid(_) => {
                    unreachable!()
                }
            }
        }
        self.try_insert(to_squeeze, new_batch)
            .expect("failed to insert");
    }

    async fn apply_advice_inner(&self, to_squeeze: EntryID) {
        let Some(to_squeeze_batch) = self.index.get(&to_squeeze) else {
            return;
        };
        let compressor = self.io_context.get_compressor_for_entry(&to_squeeze);
        let (new_batch, bytes_to_write) = self
            .squeeze_policy
            .squeeze(to_squeeze_batch, compressor.as_ref());

        if let Some(bytes_to_write) = bytes_to_write {
            match new_batch {
                CachedBatch::DiskArrow => {
                    let path = self.io_context.entry_arrow_path(&to_squeeze);
                    self.write_to_disk(&path, &bytes_to_write).await;
                }
                CachedBatch::DiskLiquid | CachedBatch::MemoryHybridLiquid(_) => {
                    let path = self.io_context.entry_liquid_path(&to_squeeze);
                    self.write_to_disk(&path, &bytes_to_write).await;
                }
                CachedBatch::MemoryArrow(_) | CachedBatch::MemoryLiquid(_) => {
                    unreachable!()
                }
            }
        }
        self.try_insert(to_squeeze, new_batch)
            .expect("failed to insert");
    }

    fn write_to_disk_blocking(&self, path: impl AsRef<Path>, bytes: &[u8]) {
        use std::io::Write;
        let mut file = std::fs::File::create(&path).expect("failed to create file");
        file.write_all(bytes).expect("failed to write to file");
        let disk_usage = bytes.len();
        self.budget.add_used_disk_bytes(disk_usage);
    }

    async fn write_to_disk(&self, path: impl AsRef<Path>, bytes: &[u8]) {
        #[cfg(not(target_os = "linux"))]
        {
            use tokio::io::AsyncWriteExt as _;
            let mut file = tokio::fs::File::create(&path)
                .await
                .expect("failed to create file");
            file.write_all(bytes)
                .await
                .expect("failed to write to file");
        }
        #[cfg(target_os = "linux")]
        {
            let mut file = super::io::File::create(&path)
                .await
                .expect("failed to create file");
            file.write_all(bytes)
                .await
                .expect("failed to write to file");
        }
        let disk_usage = bytes.len();
        self.budget.add_used_disk_bytes(disk_usage);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cache::{
        cache_policies::{CachePolicy, LruPolicy},
        utils::{create_cache_store, create_test_array, create_test_arrow_array},
    };
    use crate::sync::thread;
    use arrow::array::{Array, Int32Array};
    use std::sync::atomic::{AtomicUsize, Ordering};

    // Unified advice type for more concise testing
    #[derive(Debug)]
    struct TestPolicy {
        target_id: Option<EntryID>,
        advice_count: AtomicUsize,
    }

    impl TestPolicy {
        fn new(target_id: Option<EntryID>) -> Self {
            Self {
                target_id,
                advice_count: AtomicUsize::new(0),
            }
        }
    }

    impl CachePolicy for TestPolicy {
        fn advise(&self, _cnt: usize) -> Vec<EntryID> {
            self.advice_count.fetch_add(1, Ordering::SeqCst);
            let id_to_use = self.target_id.unwrap();
            vec![id_to_use]
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
        match retrieved1.raw_data() {
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

        // 1. Test EVICT advice
        {
            let advisor = TestPolicy::new(Some(entry_id1));
            let store = create_cache_store(8000, Box::new(advisor)); // Small budget to force advice

            store.insert_inner(entry_id1, create_test_array(800));
            match store.get(&entry_id1).unwrap().raw_data() {
                CachedBatch::MemoryArrow(_) => {}
                other => panic!("Expected ArrowMemory, got {other:?}"),
            }

            store.insert_inner(entry_id2, create_test_array(800));
            match store.get(&entry_id1).unwrap().raw_data() {
                CachedBatch::MemoryLiquid(_) => {}
                other => panic!("Expected LiquidMemory after eviction, got {other:?}"),
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

    #[test]
    fn test_cache_stats_memory_and_disk_usage() {
        // Build a small cache in blocking liquid mode to avoid background tasks
        let storage = CacheStorageBuilder::new()
            .with_max_cache_bytes(10 * 1024 * 1024)
            .with_cache_mode(LiquidCacheMode::LiquidBlocking)
            .build();

        // Insert two small batches
        let arr1: ArrayRef = Arc::new(Int32Array::from_iter_values(0..64));
        let arr2: ArrayRef = Arc::new(Int32Array::from_iter_values(0..128));
        storage.insert(EntryID::from(1usize), arr1);
        storage.insert(EntryID::from(2usize), arr2);

        // Stats after insert: 2 entries, memory usage > 0, disk usage == 0
        let s = storage.stats();
        assert_eq!(s.total_entries, 2);
        assert!(s.memory_usage_bytes > 0);
        assert_eq!(s.disk_usage_bytes, 0);
        assert_eq!(s.max_cache_bytes, 10 * 1024 * 1024);

        // Flush to disk and verify memory usage drops and disk usage increases
        storage.flush_all_to_disk();
        let s2 = storage.stats();
        assert_eq!(s2.total_entries, 2);
        assert!(s2.disk_usage_bytes > 0);
        // In-memory usage should be reduced after moving to on-disk formats
        assert!(s2.memory_usage_bytes <= s.memory_usage_bytes);
    }
}
