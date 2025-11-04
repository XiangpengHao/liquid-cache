use arrow::array::{ArrayRef, BooleanArray};
use arrow::buffer::BooleanBuffer;
use datafusion::physical_plan::PhysicalExpr;
use std::path::PathBuf;
use std::{fmt::Debug, path::Path};

use super::{
    budget::BudgetAccounting,
    cache_policies::CachePolicy,
    cached_batch::{CachedBatch, CachedBatchType},
    tracer::CacheTracer,
    utils::CacheConfig,
};
use crate::cache::squeeze_policies::{SqueezePolicy, TranscodeSqueezeEvict};
use crate::cache::stats::{CacheStats, RuntimeStats};
use crate::cache::utils::{LiquidCompressorStates, arrow_to_bytes};
use crate::cache::{CacheExpression, index::ArtIndex, utils::EntryID};
use crate::cache_policies::LiquidPolicy;
use crate::liquid_array::SqueezedDate32Array;
use crate::sync::Arc;
use std::future::IntoFuture;
use std::pin::Pin;

use bytes::Bytes;
use std::ops::Range;

/// A trait for objects that can handle IO operations for the cache.
#[async_trait::async_trait]
pub trait IoContext: Debug + Send + Sync {
    /// Get the base directory for the cache eviction, i.e., evicted data will be written to this directory.
    fn base_dir(&self) -> &Path;

    /// Get the compressor for an entry.
    fn get_compressor(&self, entry_id: &EntryID) -> Arc<LiquidCompressorStates>;

    /// Get the path to the arrow file for an entry.
    fn arrow_path(&self, entry_id: &EntryID) -> PathBuf;

    /// Get the path to the liquid file for an entry.
    fn liquid_path(&self, entry_id: &EntryID) -> PathBuf;

    /// Read bytes from the file at the given path, optionally restricted to the provided range.
    async fn read(&self, path: PathBuf, range: Option<Range<u64>>)
    -> Result<Bytes, std::io::Error>;

    /// Write the entire buffer to a file at the given path.
    async fn write_file(&self, path: PathBuf, data: Bytes) -> Result<(), std::io::Error>;
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

#[async_trait::async_trait]
impl IoContext for DefaultIoContext {
    fn base_dir(&self) -> &Path {
        &self.base_dir
    }

    fn get_compressor(&self, _entry_id: &EntryID) -> Arc<LiquidCompressorStates> {
        self.compressor_state.clone()
    }

    fn arrow_path(&self, entry_id: &EntryID) -> PathBuf {
        self.base_dir()
            .join(format!("{:016x}.arrow", usize::from(*entry_id)))
    }

    fn liquid_path(&self, entry_id: &EntryID) -> PathBuf {
        self.base_dir()
            .join(format!("{:016x}.liquid", usize::from(*entry_id)))
    }

    async fn read(
        &self,
        path: PathBuf,
        range: Option<Range<u64>>,
    ) -> Result<Bytes, std::io::Error> {
        use tokio::io::AsyncReadExt;
        use tokio::io::AsyncSeekExt;
        let mut file = tokio::fs::File::open(path).await?;

        match range {
            Some(range) => {
                let len = (range.end - range.start) as usize;
                let mut bytes = vec![0u8; len];
                file.seek(tokio::io::SeekFrom::Start(range.start)).await?;
                file.read_exact(&mut bytes).await?;
                Ok(Bytes::from(bytes))
            }
            None => {
                let mut bytes = Vec::new();
                file.read_to_end(&mut bytes).await?;
                Ok(Bytes::from(bytes))
            }
        }
    }

    async fn write_file(&self, path: PathBuf, data: Bytes) -> Result<(), std::io::Error> {
        use tokio::io::AsyncWriteExt;
        let mut file = tokio::fs::File::create(path).await?;
        file.write_all(&data).await?;
        Ok(())
    }
}

/// A blocking implementation of [IoContext] that uses the default compressor.
/// This is used for testing purposes as all io operations are blocking.
#[derive(Debug)]
pub struct BlockingIoContext {
    compressor_state: Arc<LiquidCompressorStates>,
    base_dir: PathBuf,
}

impl BlockingIoContext {
    /// Create a new instance of [BlockingIoContext].
    pub fn new(base_dir: PathBuf) -> Self {
        Self {
            compressor_state: Arc::new(LiquidCompressorStates::new()),
            base_dir,
        }
    }
}

#[async_trait::async_trait]
impl IoContext for BlockingIoContext {
    fn base_dir(&self) -> &Path {
        &self.base_dir
    }

    fn get_compressor(&self, _entry_id: &EntryID) -> Arc<LiquidCompressorStates> {
        self.compressor_state.clone()
    }

    fn arrow_path(&self, entry_id: &EntryID) -> PathBuf {
        self.base_dir()
            .join(format!("{:016x}.arrow", usize::from(*entry_id)))
    }

    fn liquid_path(&self, entry_id: &EntryID) -> PathBuf {
        self.base_dir()
            .join(format!("{:016x}.liquid", usize::from(*entry_id)))
    }

    async fn read(
        &self,
        path: PathBuf,
        range: Option<Range<u64>>,
    ) -> Result<Bytes, std::io::Error> {
        let mut file = std::fs::File::open(path)?;
        match range {
            Some(range) => {
                let len = (range.end - range.start) as usize;
                let mut bytes = vec![0u8; len];
                std::io::Seek::seek(&mut file, std::io::SeekFrom::Start(range.start))?;
                std::io::Read::read_exact(&mut file, &mut bytes)?;
                Ok(Bytes::from(bytes))
            }
            None => {
                let mut bytes = Vec::new();
                std::io::Read::read_to_end(&mut file, &mut bytes)?;
                Ok(Bytes::from(bytes))
            }
        }
    }

    async fn write_file(&self, path: PathBuf, data: Bytes) -> Result<(), std::io::Error> {
        let mut file = std::fs::File::create(path)?;
        std::io::Write::write_all(&mut file, &data)?;
        Ok(())
    }
}

// CacheStats and RuntimeStats moved to stats.rs

/// Builder for [CacheStorage].
///
/// Example:
/// ```rust
/// use liquid_cache_storage::cache::CacheStorageBuilder;
/// use liquid_cache_storage::cache_policies::LiquidPolicy;
///
///
/// let _storage = CacheStorageBuilder::new()
///     .with_batch_size(8192)
///     .with_max_cache_bytes(1024 * 1024 * 1024)
///     .with_cache_policy(Box::new(LiquidPolicy::new()))
///     .build();
/// ```
pub struct CacheStorageBuilder {
    batch_size: usize,
    max_cache_bytes: usize,
    cache_dir: Option<PathBuf>,
    cache_policy: Box<dyn CachePolicy>,
    squeeze_policy: Box<dyn SqueezePolicy>,
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
            cache_policy: Box::new(LiquidPolicy::new()),
            squeeze_policy: Box::new(TranscodeSqueezeEvict),
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

    /// Set the cache policy for the cache.
    /// Default is [LiquidPolicy].
    pub fn with_cache_policy(mut self, policy: Box<dyn CachePolicy>) -> Self {
        self.cache_policy = policy;
        self
    }

    /// Set the squeeze policy for the cache.
    /// Default is [TranscodeSqueezeEvict].
    pub fn with_squeeze_policy(mut self, policy: Box<dyn SqueezePolicy>) -> Self {
        self.squeeze_policy = policy;
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
            self.squeeze_policy,
            self.cache_policy,
            io_worker,
        ))
    }
}

/// Cache storage for liquid cache.
///
/// Example (async read):
/// ```rust
/// use liquid_cache_storage::cache::{CacheStorageBuilder, EntryID};
/// use arrow::array::UInt64Array;
/// use std::sync::Arc;
///
/// tokio_test::block_on(async {
/// let storage = CacheStorageBuilder::new().build();
///
/// let entry_id = EntryID::from(0);
/// let arrow_array = Arc::new(UInt64Array::from_iter_values(0..32));
/// storage.insert(entry_id, arrow_array.clone()).await;
///
/// // Get the arrow array back asynchronously
/// let retrieved = storage.get(&entry_id).await.unwrap();
/// assert_eq!(retrieved.as_ref(), arrow_array.as_ref());
/// });
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
    runtime_stats: RuntimeStats,
}

/// Builder returned by [`CacheStorage::get`] for configuring cache reads.
#[derive(Debug)]
pub struct Get<'a> {
    storage: &'a CacheStorage,
    entry_id: &'a EntryID,
    selection: Option<&'a BooleanBuffer>,
    expression_hint: Option<&'a CacheExpression>,
}

/// Builder for predicate evaluation on cached data.
#[derive(Debug)]
pub struct EvaluatePredicate<'a> {
    storage: &'a CacheStorage,
    entry_id: &'a EntryID,
    predicate: &'a Arc<dyn PhysicalExpr>,
    selection: Option<&'a BooleanBuffer>,
    expression_hint: Option<&'a CacheExpression>,
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
            CachedBatch::DiskLiquid(_) => disk_liquid_entries += 1,
            CachedBatch::DiskArrow(_) => disk_arrow_entries += 1,
        });

        let memory_usage_bytes = self.budget.memory_usage_bytes();
        let disk_usage_bytes = self.budget.disk_usage_bytes();
        let runtime = self.runtime_stats.consume_snapshot();

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
            cache_root_dir: self.config.cache_root_dir().clone(),
            runtime,
        }
    }

    /// Insert a batch into the cache.
    pub async fn insert(self: &Arc<Self>, entry_id: EntryID, batch_to_cache: ArrayRef) {
        self.insert_inner(entry_id, CachedBatch::MemoryArrow(batch_to_cache))
            .await;
    }

    /// Create a [`Get`] builder for the provided entry.
    pub fn get<'a>(&'a self, entry_id: &'a EntryID) -> Get<'a> {
        Get::new(self, entry_id)
    }

    /// Create an [`EvaluatePredicate`] builder for evaluating predicates on cached data.
    pub fn eval_predicate<'a>(
        &'a self,
        entry_id: &'a EntryID,
        predicate: &'a Arc<dyn PhysicalExpr>,
    ) -> EvaluatePredicate<'a> {
        EvaluatePredicate::new(self, entry_id, predicate)
    }

    /// Try to read a liquid array from the cache.
    /// Returns None if the cached data is not in liquid format.
    pub async fn try_read_liquid(
        &self,
        entry_id: &EntryID,
    ) -> Option<crate::liquid_array::LiquidArrayRef> {
        self.runtime_stats.incr_try_read_liquid();
        let batch = self.index.get(entry_id)?;
        self.cache_policy
            .notify_access(entry_id, CachedBatchType::from(&batch));

        match batch {
            CachedBatch::MemoryLiquid(array) => Some(array),
            CachedBatch::DiskLiquid(_) => {
                let path = self.io_context.liquid_path(entry_id);
                let bytes = self.io_context.read(path, None).await.ok()?;
                let compressor_states = self.io_context.get_compressor(entry_id);
                let compressor = compressor_states.fsst_compressor();
                let liquid = crate::liquid_array::ipc::read_from_bytes(
                    bytes,
                    &crate::liquid_array::ipc::LiquidIPCContext::new(compressor),
                );
                Some(liquid)
            }
            CachedBatch::MemoryHybridLiquid(array) => {
                let io_range = array.to_liquid();
                let path = self.io_context.liquid_path(entry_id);
                let bytes = self
                    .io_context
                    .read(path, Some(io_range.range().clone()))
                    .await
                    .ok()?;
                Some(array.soak(bytes))
            }
            CachedBatch::DiskArrow(_) | CachedBatch::MemoryArrow(_) => None,
        }
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

    /// Get the index of the cache.
    #[cfg(test)]
    pub(crate) fn index(&self) -> &ArtIndex {
        &self.index
    }

    /// Get the compressor states of the cache.
    pub fn compressor_states(&self, entry_id: &EntryID) -> Arc<LiquidCompressorStates> {
        self.io_context.get_compressor(entry_id)
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
                CachedBatch::DiskArrow(_) | CachedBatch::DiskLiquid(_) => {
                    // Already on disk, skip
                }
            }
        });

        // Now flush each entry to disk
        for (entry_id, batch) in entries_to_flush {
            match batch {
                CachedBatch::MemoryArrow(array) => {
                    let bytes = arrow_to_bytes(&array).expect("failed to convert arrow to bytes");
                    let path = self.io_context.arrow_path(&entry_id);
                    self.write_to_disk_blocking(&path, &bytes);
                    self.try_insert(entry_id, CachedBatch::DiskArrow(array.data_type().clone()))
                        .expect("failed to insert disk arrow entry");
                }
                CachedBatch::MemoryLiquid(liquid_array) => {
                    let liquid_bytes = liquid_array.to_bytes();
                    let path = self.io_context.liquid_path(&entry_id);
                    self.write_to_disk_blocking(&path, &liquid_bytes);
                    self.try_insert(
                        entry_id,
                        CachedBatch::DiskLiquid(liquid_array.original_arrow_data_type()),
                    )
                    .expect("failed to insert disk liquid entry");
                }
                CachedBatch::MemoryHybridLiquid(array) => {
                    // We don't have to do anything, because it's already on disk
                    self.try_insert(
                        entry_id,
                        CachedBatch::DiskLiquid(array.original_arrow_data_type()),
                    )
                    .expect("failed to insert disk liquid entry");
                }
                CachedBatch::DiskArrow(_) | CachedBatch::DiskLiquid(_) => {
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
                let path = self.io_context.arrow_path(&entry_id);
                self.write_to_disk_blocking(&path, &bytes);
                CachedBatch::DiskArrow(array.data_type().clone())
            }
            CachedBatch::MemoryLiquid(liquid_array) => {
                let liquid_bytes = liquid_array.to_bytes();
                let path = self.io_context.liquid_path(&entry_id);
                self.write_to_disk_blocking(&path, &liquid_bytes);
                CachedBatch::DiskLiquid(liquid_array.original_arrow_data_type())
            }
            CachedBatch::DiskLiquid(_)
            | CachedBatch::DiskArrow(_)
            | CachedBatch::MemoryHybridLiquid(_) => {
                unreachable!("Unexpected batch in write_in_memory_batch_to_disk")
            }
        }
    }

    /// Insert a batch into the cache, it will run cache replacement policy until the batch is inserted.
    pub(crate) async fn insert_inner(&self, entry_id: EntryID, mut batch_to_cache: CachedBatch) {
        loop {
            let batch_type = CachedBatchType::from(&batch_to_cache);
            let Err(not_inserted) = self.try_insert(entry_id, batch_to_cache) else {
                self.cache_policy.notify_insert(&entry_id, batch_type);
                return;
            };

            let victims = self.cache_policy.find_victim(8);
            if victims.is_empty() {
                // no advice, because the cache is already empty
                // this can happen if the entry to be inserted is too large, in that case,
                // we write it to disk
                let on_disk_batch = self.write_in_memory_batch_to_disk(entry_id, not_inserted);
                batch_to_cache = on_disk_batch;
                continue;
            }
            self.squeeze_victims(victims).await;

            batch_to_cache = not_inserted;
            crate::utils::yield_now_if_shuttle();
        }
    }

    /// Create a new instance of CacheStorage.
    fn new(
        batch_size: usize,
        max_cache_bytes: usize,
        cache_dir: PathBuf,
        squeeze_policy: Box<dyn SqueezePolicy>,
        cache_policy: Box<dyn CachePolicy>,
        io_worker: Arc<dyn IoContext>,
    ) -> Self {
        let config = CacheConfig::new(batch_size, max_cache_bytes, cache_dir);
        Self {
            index: ArtIndex::new(),
            budget: BudgetAccounting::new(config.max_cache_bytes()),
            config,
            cache_policy,
            squeeze_policy,
            tracer: CacheTracer::new(),
            io_context: io_worker,
            runtime_stats: RuntimeStats::default(),
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

    #[fastrace::trace]
    async fn squeeze_victims(&self, victims: Vec<EntryID>) {
        // Run squeeze operations sequentially using async I/O
        for victim in victims {
            self.squeeze_victim_inner(victim).await;
        }
    }

    async fn squeeze_victim_inner(&self, to_squeeze: EntryID) {
        let Some(mut to_squeeze_batch) = self.index.get(&to_squeeze) else {
            return;
        };
        let compressor = self.io_context.get_compressor(&to_squeeze);

        loop {
            let (new_batch, bytes_to_write) = self
                .squeeze_policy
                .squeeze(to_squeeze_batch, compressor.as_ref());

            if let Some(bytes_to_write) = bytes_to_write {
                let path = match new_batch {
                    CachedBatch::DiskArrow(_) => self.io_context.arrow_path(&to_squeeze),
                    CachedBatch::DiskLiquid(_) | CachedBatch::MemoryHybridLiquid(_) => {
                        self.io_context.liquid_path(&to_squeeze)
                    }
                    CachedBatch::MemoryArrow(_) | CachedBatch::MemoryLiquid(_) => {
                        unreachable!()
                    }
                };
                // Use IoContext's write_file for async I/O
                if let Err(e) = self
                    .io_context
                    .write_file(path, bytes_to_write.clone())
                    .await
                {
                    eprintln!("Failed to write to disk: {}", e);
                    return;
                }
                self.budget.add_used_disk_bytes(bytes_to_write.len());
            }
            let batch_type = CachedBatchType::from(&new_batch);
            match self.try_insert(to_squeeze, new_batch) {
                Ok(()) => {
                    self.cache_policy.notify_insert(&to_squeeze, batch_type);
                    break;
                }
                Err(batch) => {
                    to_squeeze_batch = batch;
                }
            }
        }
    }

    fn write_to_disk_blocking(&self, path: impl AsRef<Path>, bytes: &[u8]) {
        use std::io::Write;
        let mut file = std::fs::File::create(&path).expect("failed to create file");
        file.write_all(bytes).expect("failed to write to file");
        let disk_usage = bytes.len();
        self.budget.add_used_disk_bytes(disk_usage);
    }

    async fn read_arrow_array(
        &self,
        entry_id: &EntryID,
        selection: Option<&BooleanBuffer>,
        expression: Option<&CacheExpression>,
    ) -> Option<ArrayRef> {
        use arrow::array::BooleanArray;

        let batch = self.index.get(entry_id)?;
        self.cache_policy
            .notify_access(entry_id, CachedBatchType::from(&batch));

        match batch {
            CachedBatch::MemoryArrow(array) => match selection {
                Some(selection) => {
                    let selection_array = BooleanArray::new(selection.clone(), None);
                    arrow::compute::filter(&array, &selection_array).ok()
                }
                None => Some(array.clone()),
            },
            CachedBatch::MemoryLiquid(array) => match selection {
                Some(selection) => Some(array.filter_to_arrow(selection)),
                None => Some(array.to_arrow_array()),
            },
            CachedBatch::DiskLiquid(data_type) => match selection {
                Some(selection) => {
                    if selection.count_set_bits() == 0 {
                        return Some(arrow::array::new_empty_array(&data_type));
                    }
                    let path = self.io_context.liquid_path(entry_id);
                    let bytes = self.io_context.read(path, None).await.ok()?;
                    let compressor_states = self.io_context.get_compressor(entry_id);
                    let compressor = compressor_states.fsst_compressor();
                    let liquid = crate::liquid_array::ipc::read_from_bytes(
                        bytes,
                        &crate::liquid_array::ipc::LiquidIPCContext::new(compressor),
                    );
                    Some(liquid.filter_to_arrow(selection))
                }
                None => {
                    let path = self.io_context.liquid_path(entry_id);
                    let bytes = self.io_context.read(path, None).await.ok()?;
                    let compressor_states = self.io_context.get_compressor(entry_id);
                    let compressor = compressor_states.fsst_compressor();
                    let liquid = crate::liquid_array::ipc::read_from_bytes(
                        bytes,
                        &crate::liquid_array::ipc::LiquidIPCContext::new(compressor),
                    );
                    Some(liquid.to_arrow_array())
                }
            },
            CachedBatch::MemoryHybridLiquid(array) => {
                if let Some(CacheExpression::ExtractDate32 { field }) = expression
                    && let Some(squeezed) = array.as_any().downcast_ref::<SqueezedDate32Array>()
                    && squeezed.field() == *field
                {
                    let component = Arc::new(squeezed.to_component_date32()) as ArrayRef;
                    self.runtime_stats.incr_hit_date32_expression();
                    if let Some(selection) = selection {
                        let selection_array = BooleanArray::new(selection.clone(), None);
                        let filtered = arrow::compute::filter(&component, &selection_array).ok()?;
                        return Some(filtered);
                    }
                    return Some(component);
                }
                if let Some(selection) = selection {
                    match array.filter_to_arrow(selection) {
                        Ok(arr) => Some(arr),
                        Err(io_range) => {
                            let path = self.io_context.liquid_path(entry_id);
                            let bytes = self
                                .io_context
                                .read(path, Some(io_range.range().clone()))
                                .await
                                .ok()?;
                            let new_array = array.soak(bytes);
                            Some(new_array.filter_to_arrow(selection))
                        }
                    }
                } else {
                    match array.to_arrow_array() {
                        Ok(arr) => Some(arr),
                        Err(io_range) => {
                            let path = self.io_context.liquid_path(entry_id);
                            let bytes = self
                                .io_context
                                .read(path, Some(io_range.range().clone()))
                                .await
                                .ok()?;
                            let new_array = array.soak(bytes);
                            Some(new_array.to_arrow_array())
                        }
                    }
                }
            }
            CachedBatch::DiskArrow(_) => {
                let path = self.io_context.arrow_path(entry_id);
                let bytes = self.io_context.read(path, None).await.ok()?;
                let cursor = std::io::Cursor::new(bytes.to_vec());
                let mut reader = arrow::ipc::reader::StreamReader::try_new(cursor, None).ok()?;
                let batch = reader.next()?.ok()?;
                let array = batch.column(0).clone();
                match selection {
                    Some(selection) => {
                        let selection_array = BooleanArray::new(selection.clone(), None);
                        arrow::compute::filter(&array, &selection_array).ok()
                    }
                    None => Some(array),
                }
            }
        }
    }

    async fn eval_predicate_internal(
        &self,
        entry_id: &EntryID,
        selection_opt: Option<&BooleanBuffer>,
        predicate: &Arc<dyn PhysicalExpr>,
        _expression_hint: Option<&CacheExpression>,
    ) -> Option<Result<BooleanArray, ArrayRef>> {
        use arrow::array::BooleanArray;

        self.runtime_stats.incr_get_with_predicate();
        let batch = self.index.get(entry_id)?;
        self.cache_policy
            .notify_access(entry_id, CachedBatchType::from(&batch));

        match batch {
            CachedBatch::MemoryArrow(array) => {
                let mut owned = None;
                let selection = selection_opt.unwrap_or_else(|| {
                    owned = Some(BooleanBuffer::new_set(array.len()));
                    owned.as_ref().unwrap()
                });
                let selection_array = BooleanArray::new(selection.clone(), None);
                let filtered = arrow::compute::filter(&array, &selection_array).ok()?;
                Some(Err(filtered))
            }
            CachedBatch::DiskArrow(_) => {
                let path = self.io_context.arrow_path(entry_id);
                let bytes = self.io_context.read(path, None).await.ok()?;
                let cursor = std::io::Cursor::new(bytes.to_vec());
                let mut reader = arrow::ipc::reader::StreamReader::try_new(cursor, None).ok()?;
                let batch = reader.next()?.ok()?;
                let array = batch.column(0).clone();
                let mut owned = None;
                let selection = selection_opt.unwrap_or_else(|| {
                    owned = Some(BooleanBuffer::new_set(array.len()));
                    owned.as_ref().unwrap()
                });
                let selection_array = BooleanArray::new(selection.clone(), None);
                let filtered = arrow::compute::filter(&array, &selection_array).ok()?;
                Some(Err(filtered))
            }
            CachedBatch::MemoryLiquid(array) => {
                let mut owned = None;
                let selection = selection_opt.unwrap_or_else(|| {
                    owned = Some(BooleanBuffer::new_set(array.len()));
                    owned.as_ref().unwrap()
                });
                match array.try_eval_predicate(predicate, selection) {
                    Some(buf) => Some(Ok(buf)),
                    None => {
                        let filtered = array.filter_to_arrow(selection);
                        Some(Err(filtered))
                    }
                }
            }
            CachedBatch::DiskLiquid(_) => {
                let path = self.io_context.liquid_path(entry_id);
                let bytes = self.io_context.read(path, None).await.ok()?;
                let compressor_states = self.io_context.get_compressor(entry_id);
                let compressor = compressor_states.fsst_compressor();
                let liquid = crate::liquid_array::ipc::read_from_bytes(
                    bytes,
                    &crate::liquid_array::ipc::LiquidIPCContext::new(compressor),
                );
                let mut owned = None;
                let selection = selection_opt.unwrap_or_else(|| {
                    owned = Some(BooleanBuffer::new_set(liquid.len()));
                    owned.as_ref().unwrap()
                });
                match liquid.try_eval_predicate(predicate, selection) {
                    Some(buf) => Some(Ok(buf)),
                    None => {
                        let filtered = liquid.filter_to_arrow(selection);
                        Some(Err(filtered))
                    }
                }
            }
            CachedBatch::MemoryHybridLiquid(array) => {
                let mut owned = None;
                let selection = selection_opt.unwrap_or_else(|| {
                    owned = Some(BooleanBuffer::new_set(array.len()));
                    owned.as_ref().unwrap()
                });
                match array.try_eval_predicate(predicate, selection) {
                    Ok(Some(buf)) => {
                        self.runtime_stats.incr_get_predicate_hybrid_success();
                        Some(Ok(buf))
                    }
                    Ok(None) => {
                        self.runtime_stats.incr_get_predicate_hybrid_unsupported();
                        match array.filter_to_arrow(selection) {
                            Ok(arr) => Some(Err(arr)),
                            Err(io_range) => {
                                self.runtime_stats.incr_get_predicate_hybrid_needs_io();
                                let path = self.io_context.liquid_path(entry_id);
                                let bytes = self
                                    .io_context
                                    .read(path, Some(io_range.range().clone()))
                                    .await
                                    .ok()?;
                                let new_array = array.soak(bytes);
                                match new_array.try_eval_predicate(predicate, selection) {
                                    Some(buf) => Some(Ok(buf)),
                                    None => {
                                        let filtered = new_array.filter_to_arrow(selection);
                                        Some(Err(filtered))
                                    }
                                }
                            }
                        }
                    }
                    Err(io_range) => {
                        self.runtime_stats.incr_get_predicate_hybrid_needs_io();
                        let path = self.io_context.liquid_path(entry_id);
                        let bytes = self
                            .io_context
                            .read(path, Some(io_range.range().clone()))
                            .await
                            .ok()?;
                        let new_array = array.soak(bytes);
                        match new_array.try_eval_predicate(predicate, selection) {
                            Some(buf) => Some(Ok(buf)),
                            None => {
                                let filtered = new_array.filter_to_arrow(selection);
                                Some(Err(filtered))
                            }
                        }
                    }
                }
            }
        }
    }
}

impl<'a> Get<'a> {
    fn new(storage: &'a CacheStorage, entry_id: &'a EntryID) -> Self {
        Self {
            storage,
            entry_id,
            selection: None,
            expression_hint: None,
        }
    }

    /// Attach a selection bitmap used to filter rows prior to materialization.
    pub fn with_selection(mut self, selection: &'a BooleanBuffer) -> Self {
        self.selection = Some(selection);
        self
    }

    /// Attach an expression hint that may help optimize cache materialization.
    pub fn with_expression_hint(mut self, expression: &'a CacheExpression) -> Self {
        self.expression_hint = Some(expression);
        self
    }

    /// Attach an optional expression hint.
    pub fn with_optional_expression_hint(
        mut self,
        expression: Option<&'a CacheExpression>,
    ) -> Self {
        if let Some(expr) = expression {
            self.expression_hint = Some(expr);
        }
        self
    }

    /// Materialize the cached array as [`ArrayRef`].
    pub async fn read(self) -> Option<ArrayRef> {
        if self.selection.is_some() {
            self.storage.runtime_stats.incr_get_with_selection();
        } else {
            self.storage.runtime_stats.incr_get_arrow_array();
        }
        self.storage
            .read_arrow_array(self.entry_id, self.selection, self.expression_hint)
            .await
    }
}

impl<'a> IntoFuture for Get<'a> {
    type Output = Option<ArrayRef>;
    type IntoFuture = Pin<Box<dyn std::future::Future<Output = Option<ArrayRef>> + Send + 'a>>;

    fn into_future(self) -> Self::IntoFuture {
        Box::pin(async move { self.read().await })
    }
}

impl<'a> EvaluatePredicate<'a> {
    fn new(
        storage: &'a CacheStorage,
        entry_id: &'a EntryID,
        predicate: &'a Arc<dyn PhysicalExpr>,
    ) -> Self {
        Self {
            storage,
            entry_id,
            predicate,
            selection: None,
            expression_hint: None,
        }
    }

    /// Attach a selection bitmap used to pre-filter rows before predicate evaluation.
    pub fn with_selection(mut self, selection: &'a BooleanBuffer) -> Self {
        self.selection = Some(selection);
        self
    }

    /// Attach an expression hint that may help optimize predicate evaluation.
    pub fn with_expression_hint(mut self, expression: &'a CacheExpression) -> Self {
        self.expression_hint = Some(expression);
        self
    }

    /// Attach an optional expression hint without manual matching at call sites.
    pub fn with_optional_expression_hint(
        mut self,
        expression: Option<&'a CacheExpression>,
    ) -> Self {
        if let Some(expr) = expression {
            self.expression_hint = Some(expr);
        }
        self
    }

    /// Evaluate the predicate against the cached data.
    pub async fn read(self) -> Option<Result<BooleanArray, ArrayRef>> {
        self.storage
            .eval_predicate_internal(
                self.entry_id,
                self.selection,
                self.predicate,
                self.expression_hint,
            )
            .await
    }
}

impl<'a> IntoFuture for EvaluatePredicate<'a> {
    type Output = Option<Result<BooleanArray, ArrayRef>>;
    type IntoFuture = Pin<
        Box<dyn std::future::Future<Output = Option<Result<BooleanArray, ArrayRef>>> + Send + 'a>,
    >;

    fn into_future(self) -> Self::IntoFuture {
        Box::pin(async move { self.read().await })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cache::{
        CacheExpression, CachedBatch,
        cache_policies::{CachePolicy, LruPolicy},
        utils::{create_cache_store, create_test_array, create_test_arrow_array},
    };
    use crate::liquid_array::{
        Date32Field, LiquidHybridArrayRef, LiquidPrimitiveArray, SqueezedDate32Array,
    };
    use crate::sync::thread;
    use arrow::array::{Array, Date32Array, Int32Array};
    use arrow::datatypes::Date32Type;
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
        fn find_victim(&self, _cnt: usize) -> Vec<EntryID> {
            self.advice_count.fetch_add(1, Ordering::SeqCst);
            let id_to_use = self.target_id.unwrap();
            vec![id_to_use]
        }
    }

    #[tokio::test]
    async fn test_basic_cache_operations() {
        // Test basic insert, get, and size tracking in one test
        let budget_size = 10 * 1024;
        let store = create_cache_store(budget_size, Box::new(LruPolicy::new()));

        // 1. Initial budget should be empty
        assert_eq!(store.budget.memory_usage_bytes(), 0);

        // 2. Insert and verify first entry
        let entry_id1: EntryID = EntryID::from(1);
        let array1 = create_test_array(100);
        let size1 = array1.memory_usage_bytes();
        store.insert_inner(entry_id1, array1).await;

        // Verify budget usage and data correctness
        assert_eq!(store.budget.memory_usage_bytes(), size1);
        let retrieved1 = store.index().get(&entry_id1).unwrap();
        match retrieved1 {
            CachedBatch::MemoryArrow(arr) => assert_eq!(arr.len(), 100),
            _ => panic!("Expected ArrowMemory"),
        }

        let entry_id2: EntryID = EntryID::from(2);
        let array2 = create_test_array(200);
        let size2 = array2.memory_usage_bytes();
        store.insert_inner(entry_id2, array2).await;

        assert_eq!(store.budget.memory_usage_bytes(), size1 + size2);

        let array3 = create_test_array(150);
        let size3 = array3.memory_usage_bytes();
        store.insert_inner(entry_id1, array3).await;

        assert_eq!(store.budget.memory_usage_bytes(), size3 + size2);
        assert!(store.index().get(&EntryID::from(999)).is_none());
    }

    #[tokio::test]
    async fn get_arrow_array_with_expression_extracts_year() {
        let store = create_cache_store(1 << 20, Box::new(LruPolicy::new()));
        let entry_id = EntryID::from(42);

        let date_values = Date32Array::from(vec![Some(0), Some(365), None, Some(730)]);
        let liquid = LiquidPrimitiveArray::<Date32Type>::from_arrow_array(date_values.clone());
        let squeezed = SqueezedDate32Array::from_liquid_date32(&liquid, Date32Field::Year);
        let hybrid: LiquidHybridArrayRef = Arc::new(squeezed);

        store
            .insert_inner(entry_id, CachedBatch::MemoryHybridLiquid(hybrid.clone()))
            .await;

        let expr = CacheExpression::extract_date32(Date32Field::Year);
        let result = store
            .get(&entry_id)
            .with_expression_hint(&expr)
            .read()
            .await
            .expect("array present");

        let result = result
            .as_any()
            .downcast_ref::<Date32Array>()
            .expect("date32 result");
        assert_eq!(result.len(), 4);
        assert_eq!(result.value(0), 1970);
        assert_eq!(result.value(1), 1971);
        assert!(result.is_null(2));
        assert_eq!(result.value(3), 1972);
    }

    #[tokio::test]
    async fn test_cache_advice_strategies() {
        // Comprehensive test of all three advice types

        // Create entry IDs we'll use throughout the test
        let entry_id1 = EntryID::from(1);
        let entry_id2 = EntryID::from(2);

        // 1. Test EVICT advice
        {
            let advisor = TestPolicy::new(Some(entry_id1));
            let store = create_cache_store(8000, Box::new(advisor)); // Small budget to force advice

            store.insert_inner(entry_id1, create_test_array(800)).await;
            match store.index().get(&entry_id1).unwrap() {
                CachedBatch::MemoryArrow(_) => {}
                other => panic!("Expected ArrowMemory, got {other:?}"),
            }

            store.insert_inner(entry_id2, create_test_array(800)).await;
            match store.index().get(&entry_id1).unwrap() {
                CachedBatch::MemoryLiquid(_) => {}
                other => panic!("Expected LiquidMemory after eviction, got {other:?}"),
            }
        }
    }

    #[tokio::test]
    async fn test_concurrent_cache_operations() {
        concurrent_cache_operations().await;
    }

    #[cfg(feature = "shuttle")]
    #[tokio::test]
    async fn shuttle_cache_operations() {
        crate::utils::shuttle_test(|| {
            tokio::runtime::Runtime::new()
                .unwrap()
                .block_on(concurrent_cache_operations());
        });
    }
    pub fn block_on<F: Future>(future: F) -> F::Output {
        #[cfg(feature = "shuttle")]
        {
            shuttle::future::block_on(future)
        }
        #[cfg(not(feature = "shuttle"))]
        {
            tokio_test::block_on(future)
        }
    }

    async fn concurrent_cache_operations() {
        let num_threads = 3;
        let ops_per_thread = 50;

        let budget_size = num_threads * ops_per_thread * 100 * 8 / 2;
        let store = Arc::new(create_cache_store(budget_size, Box::new(LruPolicy::new())));

        let mut handles = vec![];
        for thread_id in 0..num_threads {
            let store = store.clone();
            handles.push(thread::spawn(move || {
                block_on(async {
                    for i in 0..ops_per_thread {
                        let unique_id = thread_id * ops_per_thread + i;
                        let entry_id: EntryID = EntryID::from(unique_id);
                        let array = create_test_arrow_array(100);
                        store.insert(entry_id, array).await;
                    }
                });
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
                assert!(store.index().get(&entry_id).is_some());
            }
        }

        // Invariant 2: Number of entries matches number of insertions
        assert_eq!(store.index().keys().len(), num_threads * ops_per_thread);
    }

    #[tokio::test]
    async fn test_cache_stats_memory_and_disk_usage() {
        // Build a small cache in blocking liquid mode to avoid background tasks
        let storage = CacheStorageBuilder::new()
            .with_max_cache_bytes(10 * 1024 * 1024)
            .with_squeeze_policy(Box::new(TranscodeSqueezeEvict))
            .build();

        // Insert two small batches
        let arr1: ArrayRef = Arc::new(Int32Array::from_iter_values(0..64));
        let arr2: ArrayRef = Arc::new(Int32Array::from_iter_values(0..128));
        storage.insert(EntryID::from(1usize), arr1).await;
        storage.insert(EntryID::from(2usize), arr2).await;

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
