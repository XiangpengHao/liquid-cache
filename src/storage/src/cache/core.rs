use arrow::array::{ArrayRef, BooleanArray};
use arrow::buffer::BooleanBuffer;
use arrow_schema::DataType;
use bytes::Bytes;
use datafusion::physical_plan::PhysicalExpr;
use futures::StreamExt;
use std::path::PathBuf;

use super::{
    budget::BudgetAccounting,
    builders::{EvaluatePredicate, Get, Insert},
    cached_batch::{CacheEntry, CachedBatchType},
    io_context::IoContext,
    observer::{CacheTracer, InternalEvent, Observer},
    policies::{CachePolicy, HydrationPolicy, HydrationRequest, MaterializedEntry},
    utils::CacheConfig,
};
use crate::cache::policies::SqueezePolicy;
use crate::cache::utils::{LiquidCompressorStates, arrow_to_bytes};
use crate::cache::{CacheExpression, index::ArtIndex, utils::EntryID};
use crate::cache::{CacheStats, EventTrace};
use crate::cache::{DefaultSqueezeIo, RuntimeStats};
use crate::liquid_array::{
    LiquidSqueezedArrayRef, SqueezeIoHandler, SqueezedBacking, SqueezedDate32Array,
    VariantStructSqueezedArray,
};
use crate::sync::Arc;

// CacheStats and RuntimeStats moved to stats.rs

/// Cache storage for liquid cache.
///
/// Example (async read):
/// ```rust
/// use liquid_cache_storage::cache::{LiquidCacheBuilder, EntryID};
/// use arrow::array::UInt64Array;
/// use std::sync::Arc;
///
/// tokio_test::block_on(async {
/// let storage = LiquidCacheBuilder::new().build();
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
pub struct LiquidCache {
    index: ArtIndex,
    config: CacheConfig,
    budget: BudgetAccounting,
    cache_policy: Box<dyn CachePolicy>,
    hydration_policy: Box<dyn HydrationPolicy>,
    squeeze_policy: Box<dyn SqueezePolicy>,
    observer: Arc<Observer>,
    io_context: Arc<dyn IoContext>,
}

/// Builder returned by [`LiquidCache::insert`] for configuring cache writes.
impl LiquidCache {
    /// Return current cache statistics: counts and resource usage.
    pub fn stats(&self) -> CacheStats {
        // Count entries by residency/format
        let total_entries = self.index.entry_count();

        let mut memory_arrow_entries = 0usize;
        let mut memory_liquid_entries = 0usize;
        let mut memory_squeezed_liquid_entries = 0usize;
        let mut disk_liquid_entries = 0usize;
        let mut disk_arrow_entries = 0usize;

        self.index.for_each(|_, batch| match batch {
            CacheEntry::MemoryArrow(_) => memory_arrow_entries += 1,
            CacheEntry::MemoryLiquid(_) => memory_liquid_entries += 1,
            CacheEntry::MemorySqueezedLiquid(_) => memory_squeezed_liquid_entries += 1,
            CacheEntry::DiskLiquid(_) => disk_liquid_entries += 1,
            CacheEntry::DiskArrow(_) => disk_arrow_entries += 1,
        });

        let memory_usage_bytes = self.budget.memory_usage_bytes();
        let disk_usage_bytes = self.budget.disk_usage_bytes();
        let runtime = self.observer.runtime_snapshot();

        CacheStats {
            total_entries,
            memory_arrow_entries,
            memory_liquid_entries,
            memory_squeezed_liquid_entries,
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
    pub fn insert<'a>(
        self: &'a Arc<Self>,
        entry_id: EntryID,
        batch_to_cache: ArrayRef,
    ) -> Insert<'a> {
        Insert::new(self, entry_id, batch_to_cache)
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
        self.observer.on_try_read_liquid();
        self.trace(InternalEvent::TryReadLiquid { entry: *entry_id });
        let batch = self.index.get(entry_id)?;
        self.cache_policy
            .notify_access(entry_id, CachedBatchType::from(batch.as_ref()));

        match batch.as_ref() {
            CacheEntry::MemoryLiquid(array) => Some(array.clone()),
            entry @ CacheEntry::DiskLiquid(_) => {
                let liquid = self.read_disk_liquid_array(entry_id).await;
                self.maybe_hydrate(entry_id, entry, MaterializedEntry::Liquid(&liquid), None)
                    .await;
                Some(liquid)
            }
            CacheEntry::MemorySqueezedLiquid(array) => match array.disk_backing() {
                SqueezedBacking::Liquid => {
                    let liquid = self.read_disk_liquid_array(entry_id).await;
                    Some(liquid)
                }
                SqueezedBacking::Arrow => None,
            },
            CacheEntry::DiskArrow(_) | CacheEntry::MemoryArrow(_) => None,
        }
    }

    /// Iterate over all entries in the cache.
    /// No guarantees are made about the order of the entries.
    /// Isolation level: read-committed
    pub fn for_each_entry(&self, mut f: impl FnMut(&EntryID, &CacheEntry)) {
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
        self.observer.cache_tracer()
    }

    /// Access the cache observer (runtime stats, debug event trace, and optional cache tracing).
    pub fn observer(&self) -> &Observer {
        &self.observer
    }

    fn runtime_stats(&self) -> &RuntimeStats {
        self.observer.runtime_stats()
    }

    /// Get the compressor states of the cache.
    pub fn compressor_states(&self, entry_id: &EntryID) -> Arc<LiquidCompressorStates> {
        self.io_context.get_compressor(entry_id)
    }

    /// Add a squeeze hint for an entry.
    pub fn add_squeeze_hint(&self, entry_id: &EntryID, expression: Arc<CacheExpression>) {
        self.io_context.add_squeeze_hint(entry_id, expression);
    }

    /// Flush all entries to disk.
    pub async fn flush_all_to_disk(&self) {
        let mut entires = Vec::new();
        self.for_each_entry(|entry_id, batch| {
            entires.push((*entry_id, batch.clone()));
        });
        for (entry_id, batch) in entires {
            match &batch {
                CacheEntry::MemoryArrow(array) => {
                    let bytes = arrow_to_bytes(array).expect("failed to convert arrow to bytes");
                    self.write_batch_to_disk(entry_id, &batch, bytes).await;
                    self.try_insert(entry_id, CacheEntry::disk_arrow(array.data_type().clone()))
                        .expect("failed to insert disk arrow entry");
                }
                CacheEntry::MemoryLiquid(liquid_array) => {
                    let liquid_bytes = liquid_array.to_bytes();
                    self.write_batch_to_disk(entry_id, &batch, Bytes::from(liquid_bytes))
                        .await;
                    self.try_insert(
                        entry_id,
                        CacheEntry::disk_liquid(liquid_array.original_arrow_data_type()),
                    )
                    .expect("failed to insert disk liquid entry");
                }
                CacheEntry::MemorySqueezedLiquid(array) => {
                    // We don't have to do anything, because it's already on disk
                    let disk_entry = Self::disk_entry_from_squeezed(array);
                    self.try_insert(entry_id, disk_entry)
                        .expect("failed to insert disk entry");
                }
                CacheEntry::DiskArrow(_) | CacheEntry::DiskLiquid(_) => {
                    // Already on disk, skip
                }
            }
        }
    }
}

impl LiquidCache {
    /// returns the batch that was written to disk
    async fn write_in_memory_batch_to_disk(
        &self,
        entry_id: EntryID,
        batch: CacheEntry,
    ) -> CacheEntry {
        match &batch {
            batch @ CacheEntry::MemoryArrow(_) => {
                let squeeze_io: Arc<dyn SqueezeIoHandler> = Arc::new(DefaultSqueezeIo::new(
                    self.io_context.clone(),
                    entry_id,
                    self.observer.clone(),
                ));
                let (new_batch, bytes_to_write) = self.squeeze_policy.squeeze(
                    batch,
                    self.io_context.get_compressor(&entry_id).as_ref(),
                    None,
                    &squeeze_io,
                );
                if let Some(bytes_to_write) = bytes_to_write {
                    self.write_batch_to_disk(entry_id, &new_batch, bytes_to_write)
                        .await;
                }
                new_batch
            }
            CacheEntry::MemoryLiquid(liquid_array) => {
                let liquid_bytes = Bytes::from(liquid_array.to_bytes());
                self.write_batch_to_disk(entry_id, &batch, liquid_bytes)
                    .await;
                CacheEntry::disk_liquid(liquid_array.original_arrow_data_type())
            }
            CacheEntry::MemorySqueezedLiquid(squeezed_array) => {
                // The full data is already on disk, so we just need to mark ourself as disk entry
                let backing = squeezed_array.disk_backing();
                if backing == SqueezedBacking::Liquid {
                    CacheEntry::disk_liquid(squeezed_array.original_arrow_data_type())
                } else {
                    CacheEntry::disk_arrow(squeezed_array.original_arrow_data_type())
                }
            }
            CacheEntry::DiskLiquid(_) | CacheEntry::DiskArrow(_) => {
                unreachable!("Unexpected batch in write_in_memory_batch_to_disk")
            }
        }
    }

    /// Insert a batch into the cache, it will run cache replacement policy until the batch is inserted.
    pub(crate) async fn insert_inner(&self, entry_id: EntryID, mut batch_to_cache: CacheEntry) {
        loop {
            let Err(not_inserted) = self.try_insert(entry_id, batch_to_cache) else {
                return;
            };
            self.trace(InternalEvent::InsertFailed {
                entry: entry_id,
                kind: CachedBatchType::from(&not_inserted),
            });

            let victims = self.cache_policy.find_victim(8);
            if victims.is_empty() {
                // no advice, because the cache is already empty
                // this can happen if the entry to be inserted is too large, in that case,
                // we write it to disk
                let on_disk_batch = self
                    .write_in_memory_batch_to_disk(entry_id, not_inserted)
                    .await;
                batch_to_cache = on_disk_batch;
                continue;
            }
            self.squeeze_victims(victims).await;

            batch_to_cache = not_inserted;
            crate::utils::yield_now_if_shuttle();
        }
    }

    /// Create a new instance of CacheStorage.
    pub(crate) fn new(
        batch_size: usize,
        max_cache_bytes: usize,
        cache_dir: PathBuf,
        squeeze_policy: Box<dyn SqueezePolicy>,
        cache_policy: Box<dyn CachePolicy>,
        hydration_policy: Box<dyn HydrationPolicy>,
        io_worker: Arc<dyn IoContext>,
    ) -> Self {
        let config = CacheConfig::new(batch_size, max_cache_bytes, cache_dir);
        Self {
            index: ArtIndex::new(),
            budget: BudgetAccounting::new(config.max_cache_bytes()),
            config,
            cache_policy,
            hydration_policy,
            squeeze_policy,
            observer: Arc::new(Observer::new()),
            io_context: io_worker,
        }
    }

    fn try_insert(&self, entry_id: EntryID, to_insert: CacheEntry) -> Result<(), CacheEntry> {
        let new_memory_size = to_insert.memory_usage_bytes();
        let cached_batch_type = if let Some(entry) = self.index.get(&entry_id) {
            let old_memory_size = entry.memory_usage_bytes();
            if self
                .budget
                .try_update_memory_usage(old_memory_size, new_memory_size)
                .is_err()
            {
                return Err(to_insert);
            }
            let batch_type = CachedBatchType::from(&to_insert);
            self.index.insert(&entry_id, to_insert);
            batch_type
        } else {
            if self.budget.try_reserve_memory(new_memory_size).is_err() {
                return Err(to_insert);
            }
            let batch_type = CachedBatchType::from(&to_insert);
            self.index.insert(&entry_id, to_insert);
            batch_type
        };

        self.trace(InternalEvent::InsertSuccess {
            entry: entry_id,
            kind: cached_batch_type,
        });
        self.cache_policy
            .notify_insert(&entry_id, cached_batch_type);

        Ok(())
    }

    /// Consume the trace of the cache, for testing only.
    pub fn consume_event_trace(&self) -> EventTrace {
        self.observer.consume_event_trace()
    }

    pub(crate) fn trace(&self, event: InternalEvent) {
        self.observer.record_internal(event);
    }

    /// Get the index of the cache.
    #[cfg(test)]
    pub(crate) fn index(&self) -> &ArtIndex {
        &self.index
    }

    #[fastrace::trace]
    async fn squeeze_victims(&self, victims: Vec<EntryID>) {
        // Run squeeze operations sequentially using async I/O
        self.trace(InternalEvent::SqueezeBegin {
            victims: victims.clone(),
        });
        futures::stream::iter(victims)
            .for_each_concurrent(None, |victim| async move {
                self.squeeze_victim_inner(victim).await;
            })
            .await;
    }

    async fn squeeze_victim_inner(&self, to_squeeze: EntryID) {
        let Some(mut to_squeeze_batch) = self.index.get(&to_squeeze) else {
            return;
        };
        self.trace(InternalEvent::SqueezeVictim { entry: to_squeeze });
        let compressor = self.io_context.get_compressor(&to_squeeze);
        let squeeze_hint_arc = self.io_context.squeeze_hint(&to_squeeze);
        let squeeze_hint = squeeze_hint_arc.as_deref();
        let squeeze_io: Arc<dyn SqueezeIoHandler> = Arc::new(DefaultSqueezeIo::new(
            self.io_context.clone(),
            to_squeeze,
            self.observer.clone(),
        ));

        loop {
            let (new_batch, bytes_to_write) = self.squeeze_policy.squeeze(
                to_squeeze_batch.as_ref(),
                compressor.as_ref(),
                squeeze_hint,
                &squeeze_io,
            );

            if let Some(bytes_to_write) = bytes_to_write {
                self.write_batch_to_disk(to_squeeze, &new_batch, bytes_to_write)
                    .await;
            }
            match self.try_insert(to_squeeze, new_batch) {
                Ok(()) => {
                    break;
                }
                Err(batch) => {
                    to_squeeze_batch = Arc::new(batch);
                }
            }
        }
    }

    fn disk_entry_from_squeezed(array: &LiquidSqueezedArrayRef) -> CacheEntry {
        let constructor: fn(DataType) -> CacheEntry = match array.disk_backing() {
            SqueezedBacking::Liquid => CacheEntry::disk_liquid,
            SqueezedBacking::Arrow => CacheEntry::disk_arrow,
        };
        constructor(array.original_arrow_data_type())
    }

    async fn maybe_hydrate(
        &self,
        entry_id: &EntryID,
        cached: &CacheEntry,
        materialized: MaterializedEntry<'_>,
        expression: Option<&CacheExpression>,
    ) {
        let compressor = self.io_context.get_compressor(entry_id);
        if let Some(new_entry) = self.hydration_policy.hydrate(&HydrationRequest {
            entry_id: *entry_id,
            cached,
            materialized,
            expression,
            compressor,
        }) {
            let cached_type = CachedBatchType::from(cached);
            let new_type = CachedBatchType::from(&new_entry);
            self.trace(InternalEvent::Hydrate {
                entry: *entry_id,
                cached: cached_type,
                new: new_type,
            });
            self.insert_inner(*entry_id, new_entry).await;
        }
    }

    pub(crate) async fn read_arrow_array(
        &self,
        entry_id: &EntryID,
        selection: Option<&BooleanBuffer>,
        expression: Option<&CacheExpression>,
    ) -> Option<ArrayRef> {
        use arrow::array::BooleanArray;

        let batch = self.index.get(entry_id)?;
        self.cache_policy
            .notify_access(entry_id, CachedBatchType::from(batch.as_ref()));
        self.trace(InternalEvent::Read {
            entry: *entry_id,
            selection: selection.is_some(),
            expr: expression.cloned(),
            cached: CachedBatchType::from(batch.as_ref()),
        });

        match batch.as_ref() {
            CacheEntry::MemoryArrow(array) => match selection {
                Some(selection) => {
                    let selection_array = BooleanArray::new(selection.clone(), None);
                    arrow::compute::filter(array, &selection_array).ok()
                }
                None => Some(array.clone()),
            },
            CacheEntry::MemoryLiquid(array) => match selection {
                Some(selection) => Some(array.filter(selection)),
                None => Some(array.to_arrow_array()),
            },
            CacheEntry::DiskArrow(_) | CacheEntry::DiskLiquid(_) => {
                self.read_disk_array(batch.as_ref(), entry_id, expression, selection)
                    .await
            }
            CacheEntry::MemorySqueezedLiquid(array) => {
                self.read_squeezed_array(array, entry_id, expression, selection)
                    .await
            }
        }
    }

    async fn read_disk_array(
        &self,
        entry: &CacheEntry,
        entry_id: &EntryID,
        expression: Option<&CacheExpression>,
        selection: Option<&BooleanBuffer>,
    ) -> Option<ArrayRef> {
        match entry {
            CacheEntry::DiskArrow(data_type) => {
                if let Some(selection) = selection
                    && selection.count_set_bits() == 0
                {
                    return Some(arrow::array::new_empty_array(data_type));
                }
                let full_array = self.read_disk_arrow_array(entry_id).await;
                self.maybe_hydrate(
                    entry_id,
                    entry,
                    MaterializedEntry::Arrow(&full_array),
                    expression,
                )
                .await;
                match selection {
                    Some(selection) => {
                        let selection_array = BooleanArray::new(selection.clone(), None);
                        arrow::compute::filter(&full_array, &selection_array).ok()
                    }
                    None => Some(full_array),
                }
            }
            CacheEntry::DiskLiquid(data_type) => {
                if let Some(selection) = selection
                    && selection.count_set_bits() == 0
                {
                    return Some(arrow::array::new_empty_array(data_type));
                }
                let liquid = self.read_disk_liquid_array(entry_id).await;
                self.maybe_hydrate(
                    entry_id,
                    entry,
                    MaterializedEntry::Liquid(&liquid),
                    expression,
                )
                .await;
                match selection {
                    Some(selection) => Some(liquid.filter(selection)),
                    None => Some(liquid.to_arrow_array()),
                }
            }
            _ => unreachable!("Unexpected batch in read_disk_array"),
        }
    }

    async fn read_squeezed_array(
        &self,
        array: &LiquidSqueezedArrayRef,
        entry_id: &EntryID,
        expression: Option<&CacheExpression>,
        selection: Option<&BooleanBuffer>,
    ) -> Option<ArrayRef> {
        if let Some(array) = self.try_read_squeezed_date32_array(array, expression, selection) {
            self.observer.on_get_squeezed_success();
            self.trace(InternalEvent::ReadSqueezedData {
                entry: *entry_id,
                expression: expression.unwrap().clone(),
            });
            return Some(array);
        }

        if let Some(array) = self
            .try_read_squeezed_variant_array(array, entry_id, expression, selection)
            .await
        {
            self.observer.on_get_squeezed_success();
            self.trace(InternalEvent::ReadSqueezedData {
                entry: *entry_id,
                expression: expression.unwrap().clone(),
            });
            return Some(array);
        }

        self.observer.on_get_squeezed_success();
        let out = match selection {
            Some(selection) => array.filter(selection).await,
            None => array.to_arrow_array().await,
        };
        Some(out)
    }

    fn try_read_squeezed_date32_array(
        &self,
        array: &LiquidSqueezedArrayRef,
        expression: Option<&CacheExpression>,
        selection: Option<&BooleanBuffer>,
    ) -> Option<ArrayRef> {
        if let Some(CacheExpression::ExtractDate32 { field }) = expression
            && let Some(squeezed) = array.as_any().downcast_ref::<SqueezedDate32Array>()
            && squeezed.field() == *field
        {
            let component = Arc::new(squeezed.to_component_date32()) as ArrayRef;
            self.observer.on_hit_date32_expression();
            if let Some(selection) = selection {
                let selection_array = BooleanArray::new(selection.clone(), None);
                let filtered = arrow::compute::filter(&component, &selection_array).ok()?;
                return Some(filtered);
            }
            return Some(component);
        }
        None
    }

    async fn try_read_squeezed_variant_array(
        &self,
        array: &LiquidSqueezedArrayRef,
        entry_id: &EntryID,
        expression: Option<&CacheExpression>,
        selection: Option<&BooleanBuffer>,
    ) -> Option<ArrayRef> {
        let requests = expression.and_then(|expr| expr.variant_requests())?;
        let variant_squeezed = array
            .as_any()
            .downcast_ref::<VariantStructSqueezedArray>()?;
        let all_paths_present = requests
            .iter()
            .all(|request| variant_squeezed.contains_path(request.path()));

        let full_array = if !all_paths_present {
            let batch = CacheEntry::MemorySqueezedLiquid(array.clone());
            self.observer.on_get_squeezed_needs_io();
            let full_array = self.read_disk_arrow_array(entry_id).await;
            self.maybe_hydrate(
                entry_id,
                &batch,
                MaterializedEntry::Arrow(&full_array),
                expression,
            )
            .await;
            full_array
        } else {
            let requested_paths = requests.iter().map(|r| r.path());
            variant_squeezed
                .to_arrow_array_with_paths(requested_paths)
                .unwrap()
        };

        match selection {
            Some(selection) => {
                let selection_array = BooleanArray::new(selection.clone(), None);
                arrow::compute::filter(&full_array, &selection_array).ok()
            }
            None => Some(full_array),
        }
    }

    async fn write_batch_to_disk(&self, entry_id: EntryID, batch: &CacheEntry, bytes: Bytes) {
        self.trace(InternalEvent::IoWrite {
            entry: entry_id,
            kind: CachedBatchType::from(batch),
            bytes: bytes.len(),
        });
        let path = self.io_context.disk_path(&entry_id);
        let len = bytes.len();
        self.io_context.write_file(path, bytes).await.unwrap();
        self.budget.add_used_disk_bytes(len);
    }

    async fn read_disk_arrow_array(&self, entry_id: &EntryID) -> ArrayRef {
        let path = self.io_context.disk_path(entry_id);
        let bytes = self.io_context.read(path, None).await.expect("read failed");
        let cursor = std::io::Cursor::new(bytes.to_vec());
        let mut reader =
            arrow::ipc::reader::StreamReader::try_new(cursor, None).expect("create reader failed");
        let batch = reader.next().unwrap().expect("read batch failed");
        let array = batch.column(0).clone();
        self.trace(InternalEvent::IoReadArrow {
            entry: *entry_id,
            bytes: bytes.len(),
        });
        array
    }

    async fn read_disk_liquid_array(
        &self,
        entry_id: &EntryID,
    ) -> crate::liquid_array::LiquidArrayRef {
        let path = self.io_context.disk_path(entry_id);
        let bytes = self.io_context.read(path, None).await.expect("read failed");
        self.trace(InternalEvent::IoReadLiquid {
            entry: *entry_id,
            bytes: bytes.len(),
        });
        let compressor_states = self.io_context.get_compressor(entry_id);
        let compressor = compressor_states.fsst_compressor();

        (crate::liquid_array::ipc::read_from_bytes(
            bytes,
            &crate::liquid_array::ipc::LiquidIPCContext::new(compressor),
        )) as _
    }

    pub(crate) async fn eval_predicate_internal(
        &self,
        entry_id: &EntryID,
        selection_opt: Option<&BooleanBuffer>,
        predicate: &Arc<dyn PhysicalExpr>,
    ) -> Option<Result<BooleanArray, ArrayRef>> {
        use arrow::array::BooleanArray;

        self.observer.on_eval_predicate();
        let batch = self.index.get(entry_id)?;
        self.cache_policy
            .notify_access(entry_id, CachedBatchType::from(batch.as_ref()));
        self.trace(InternalEvent::EvalPredicate {
            entry: *entry_id,
            selection: selection_opt.is_some(),
            cached: CachedBatchType::from(batch.as_ref()),
        });

        match batch.as_ref() {
            CacheEntry::MemoryArrow(array) => {
                let mut owned = None;
                let selection = selection_opt.unwrap_or_else(|| {
                    owned = Some(BooleanBuffer::new_set(array.len()));
                    owned.as_ref().unwrap()
                });
                let selection_array = BooleanArray::new(selection.clone(), None);
                let filtered = arrow::compute::filter(array, &selection_array).ok()?;
                Some(Err(filtered))
            }
            entry @ CacheEntry::DiskArrow(_) => {
                let array = self.read_disk_arrow_array(entry_id).await;
                self.maybe_hydrate(entry_id, entry, MaterializedEntry::Arrow(&array), None)
                    .await;
                let mut owned = None;
                let selection = selection_opt.unwrap_or_else(|| {
                    owned = Some(BooleanBuffer::new_set(array.len()));
                    owned.as_ref().unwrap()
                });
                let selection_array = BooleanArray::new(selection.clone(), None);
                let filtered = arrow::compute::filter(&array, &selection_array).ok()?;
                Some(Err(filtered))
            }
            CacheEntry::MemoryLiquid(array) => {
                let mut owned = None;
                let selection = selection_opt.unwrap_or_else(|| {
                    owned = Some(BooleanBuffer::new_set(array.len()));
                    owned.as_ref().unwrap()
                });
                match array.try_eval_predicate(predicate, selection) {
                    Some(buf) => Some(Ok(buf)),
                    None => {
                        self.runtime_stats().incr_eval_predicate_on_liquid_failed();
                        let filtered = array.filter(selection);
                        Some(Err(filtered))
                    }
                }
            }
            entry @ CacheEntry::DiskLiquid(_) => {
                let liquid = self.read_disk_liquid_array(entry_id).await;
                self.maybe_hydrate(entry_id, entry, MaterializedEntry::Liquid(&liquid), None)
                    .await;
                let mut owned = None;
                let selection = selection_opt.unwrap_or_else(|| {
                    owned = Some(BooleanBuffer::new_set(liquid.len()));
                    owned.as_ref().unwrap()
                });
                match liquid.try_eval_predicate(predicate, selection) {
                    Some(buf) => Some(Ok(buf)),
                    None => {
                        self.runtime_stats().incr_eval_predicate_on_liquid_failed();
                        let filtered = liquid.filter(selection);
                        Some(Err(filtered))
                    }
                }
            }
            CacheEntry::MemorySqueezedLiquid(array) => {
                self.eval_predicate_on_squeezed(entry_id, array, selection_opt, predicate)
                    .await
            }
        }
    }

    async fn eval_predicate_on_squeezed(
        &self,
        _entry_id: &EntryID,
        array: &LiquidSqueezedArrayRef,
        selection_opt: Option<&BooleanBuffer>,
        predicate: &Arc<dyn PhysicalExpr>,
    ) -> Option<Result<BooleanArray, ArrayRef>> {
        let mut owned = None;
        let selection = selection_opt.unwrap_or_else(|| {
            owned = Some(BooleanBuffer::new_set(array.len()));
            owned.as_ref().unwrap()
        });
        match array.try_eval_predicate(predicate, selection).await {
            Some(buf) => Some(Ok(buf)),
            None => Some(Err(array.filter(selection).await)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cache::{
        CacheEntry, CacheExpression, CachePolicy, LiquidCacheBuilder, TranscodeSqueezeEvict,
        policies::LruPolicy,
        transcode_liquid_inner,
        utils::{
            LiquidCompressorStates, create_cache_store, create_test_array, create_test_arrow_array,
        },
    };
    use crate::liquid_array::{
        Date32Field, LiquidPrimitiveArray, LiquidSqueezedArrayRef, SqueezedDate32Array,
    };
    use crate::sync::thread;
    use arrow::array::{Array, ArrayRef, Date32Array, Int32Array};
    use arrow::datatypes::Date32Type;
    use std::future::Future;
    use std::sync::Arc;
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
        match retrieved1.as_ref() {
            CacheEntry::MemoryArrow(arr) => assert_eq!(arr.len(), 100),
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
        let squeezed: LiquidSqueezedArrayRef = Arc::new(squeezed);

        store
            .insert_inner(
                entry_id,
                CacheEntry::memory_squeezed_liquid(squeezed.clone()),
            )
            .await;

        let expr = Arc::new(CacheExpression::extract_date32(Date32Field::Year));
        let result = store
            .get(&entry_id)
            .with_expression_hint(expr)
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
            match store.index().get(&entry_id1).unwrap().as_ref() {
                CacheEntry::MemoryArrow(_) => {}
                other => panic!("Expected ArrowMemory, got {other:?}"),
            }

            store.insert_inner(entry_id2, create_test_array(800)).await;
            match store.index().get(&entry_id1).unwrap().as_ref() {
                CacheEntry::MemoryLiquid(_) => {}
                other => panic!("Expected LiquidMemory after eviction, got {other:?}"),
            }
        }
    }

    #[tokio::test]
    async fn test_concurrent_cache_operations() {
        concurrent_cache_operations().await;
    }

    #[cfg(feature = "shuttle")]
    #[test]
    fn shuttle_cache_operations() {
        crate::utils::shuttle_test(|| {
            block_on(concurrent_cache_operations());
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
        let storage = LiquidCacheBuilder::new()
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
        storage.flush_all_to_disk().await;
        let s2 = storage.stats();
        assert_eq!(s2.total_entries, 2);
        assert!(s2.disk_usage_bytes > 0);
        // In-memory usage should be reduced after moving to on-disk formats
        assert!(s2.memory_usage_bytes <= s.memory_usage_bytes);
    }

    #[tokio::test]
    async fn hydrate_disk_arrow_on_get_promotes_to_memory() {
        let store = create_cache_store(1 << 20, Box::new(LruPolicy::new()));
        let entry_id = EntryID::from(321usize);
        let array = create_test_arrow_array(8);

        store.insert(entry_id, array.clone()).await;
        store.flush_all_to_disk().await;
        {
            let entry = store.index().get(&entry_id).unwrap();
            assert!(matches!(entry.as_ref(), CacheEntry::DiskArrow(_)));
        }

        let result = store.get(&entry_id).await.expect("present");
        assert_eq!(result.as_ref(), array.as_ref());
        {
            let entry = store.index().get(&entry_id).unwrap();
            assert!(matches!(entry.as_ref(), CacheEntry::MemoryArrow(_)));
        }
    }

    #[tokio::test]
    async fn hydrate_disk_liquid_on_get_promotes_to_memory_liquid() {
        let store = create_cache_store(1 << 20, Box::new(LruPolicy::new()));
        let entry_id = EntryID::from(322usize);
        let arrow_array: ArrayRef = Arc::new(Int32Array::from(vec![1, 2, 3, 4]));
        let compressor = LiquidCompressorStates::new();
        let liquid = transcode_liquid_inner(&arrow_array, &compressor).unwrap();

        store
            .insert_inner(entry_id, CacheEntry::memory_liquid(liquid.clone()))
            .await;
        store.flush_all_to_disk().await;
        {
            let entry = store.index().get(&entry_id).unwrap();
            assert!(matches!(entry.as_ref(), CacheEntry::DiskLiquid(_)));
        }

        let result = store.get(&entry_id).await.expect("present");
        assert_eq!(result.as_ref(), arrow_array.as_ref());
        {
            let entry = store.index().get(&entry_id).unwrap();
            assert!(matches!(entry.as_ref(), CacheEntry::MemoryLiquid(_)));
        }
    }
}
