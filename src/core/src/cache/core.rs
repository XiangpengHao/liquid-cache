use arrow::array::cast::AsArray;
use arrow::array::{ArrayRef, BooleanArray};
use arrow::buffer::BooleanBuffer;
use arrow::record_batch::RecordBatch;
use arrow_schema::{Field, Schema};
use bytes::Bytes;
use futures::StreamExt;

use super::{
    budget::BudgetAccounting,
    builders::{EvaluatePredicate, Get, Insert},
    cached_batch::{CacheEntry, CachedBatchType},
    io_context::{EntryMetadata, entry_id_to_key},
    observer::{CacheTracer, InternalEvent, Observer},
    policies::{CachePolicy, HydrationPolicy, HydrationRequest, MaterializedEntry},
    utils::CacheConfig,
};
use crate::cache::policies::{EvictionOutcome, EvictionPolicy};
use crate::cache::utils::arrow_to_bytes;
use crate::cache::{CacheExpression, LiquidExpr, index::ArtIndex, utils::EntryID};
use crate::cache::{CacheFull, CacheStats, EventTrace};
use crate::sync::Arc;

// CacheStats and RuntimeStats moved to stats.rs

/// Cache storage for liquid cache.
///
/// # Identity
///
/// Every entry is cached for an *identity*: a `u64` naming the source the data
/// belongs to. Callers that pack several fields into a fixed-width
/// [`EntryID`] can have two sources compute the same key; the identity is how
/// the cache tells them apart. A lookup under any identity other than the one
/// an entry was cached for is a miss, and an insert replaces whatever held the
/// key. Callers whose keys are already unique can pass any constant.
///
/// Example (async read):
/// ```rust
/// use liquid_cache::cache::{LiquidCacheBuilder, EntryID};
/// use arrow::array::UInt64Array;
/// use std::sync::Arc;
///
/// tokio_test::block_on(async {
/// let storage = LiquidCacheBuilder::new().build().await;
///
/// let entry_id = EntryID::from(0);
/// let arrow_array = Arc::new(UInt64Array::from_iter_values(0..32));
/// // The identity names the source this data belongs to. One source here,
/// // so any constant will do.
/// storage.insert(entry_id, 0, arrow_array.clone()).await;
///
/// // Get the arrow array back asynchronously
/// let retrieved = storage.get(&entry_id, 0).await.unwrap();
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
    eviction_policy: Box<dyn EvictionPolicy>,
    observer: Arc<Observer>,
    metadata: Arc<dyn EntryMetadata>,
    store: t4::Store,
    evict_victims_concurrently: bool,
}

/// Outcome of [`LiquidCache::prefetch`].
pub enum PrefetchResult {
    /// A memory-form snapshot of the entry (Arrow or Liquid), ready to hand to a reader.
    Snapshot(Arc<CacheEntry>),
    /// The entry is not in the index, or its disk blob is gone.
    Absent,
}

/// Length of the identity trailer appended to every object written to disk.
const IDENTITY_TRAILER_LEN: usize = std::mem::size_of::<u64>();

/// Builder returned by [`LiquidCache::insert`] for configuring cache writes.
impl LiquidCache {
    /// Return current cache statistics: counts and resource usage.
    pub fn stats(&self) -> CacheStats {
        // Count entries by storage tier and format
        let total_entries = self.index.entry_count();

        let mut memory_arrow_entries = 0usize;
        let mut memory_liquid_entries = 0usize;
        let mut disk_liquid_entries = 0usize;
        let mut disk_arrow_entries = 0usize;

        let mut memory_arrow_bytes = 0usize;
        let mut memory_liquid_bytes = 0usize;

        self.index.for_each(|_, slot| match &slot.entry {
            CacheEntry::MemoryArrow(array) => {
                memory_arrow_entries += 1;
                memory_arrow_bytes += array.get_array_memory_size();
            }
            CacheEntry::MemoryLiquid(array) => {
                memory_liquid_entries += 1;
                memory_liquid_bytes += array.get_array_memory_size();
            }
            CacheEntry::DiskLiquid { .. } => disk_liquid_entries += 1,
            CacheEntry::DiskArrow { .. } => disk_arrow_entries += 1,
        });

        let memory_usage_bytes = self.budget.memory_usage_bytes();
        let disk_usage_bytes = self.budget.disk_usage_bytes();
        let runtime = self.observer.runtime_snapshot();

        CacheStats {
            total_entries,
            memory_arrow_entries,
            memory_liquid_entries,
            disk_liquid_entries,
            disk_arrow_entries,
            memory_arrow_bytes,
            memory_liquid_bytes,
            memory_usage_bytes,
            disk_usage_bytes,
            max_memory_bytes: self.config.max_memory_bytes(),
            max_disk_bytes: self.config.max_disk_bytes(),
            runtime,
        }
    }

    /// Insert a batch into the cache for `identity`.
    pub fn insert<'a>(
        self: &'a Arc<Self>,
        entry_id: EntryID,
        identity: u64,
        batch_to_cache: ArrayRef,
    ) -> Insert<'a> {
        Insert::new(self, entry_id, identity, batch_to_cache)
    }

    /// Create a [`Get`] builder for the provided entry, as cached for `identity`.
    pub fn get<'a>(&'a self, entry_id: &'a EntryID, identity: u64) -> Get<'a> {
        Get::new(self, entry_id, identity)
    }

    /// Create an [`EvaluatePredicate`] builder for evaluating predicates on cached data.
    pub fn eval_predicate<'a>(
        &'a self,
        entry_id: &'a EntryID,
        identity: u64,
        predicate: &'a LiquidExpr,
    ) -> EvaluatePredicate<'a> {
        EvaluatePredicate::new(self, entry_id, identity, predicate)
    }

    /// Prefetch an entry into a memory-form snapshot without recording an access.
    pub async fn prefetch(&self, entry_id: &EntryID, identity: u64) -> PrefetchResult {
        let Some(slot) = self.index.get_for(entry_id, identity) else {
            return PrefetchResult::Absent;
        };
        match &slot.entry {
            entry @ (CacheEntry::MemoryArrow(_) | CacheEntry::MemoryLiquid(_)) => {
                PrefetchResult::Snapshot(Arc::new(entry.clone()))
            }
            disk @ CacheEntry::DiskArrow { .. } => {
                let Some(array) = self.read_disk_arrow_array(entry_id, identity).await else {
                    return PrefetchResult::Absent;
                };
                self.maybe_hydrate(
                    entry_id,
                    identity,
                    disk,
                    MaterializedEntry::Arrow(&array),
                    None,
                )
                .await;
                PrefetchResult::Snapshot(Arc::new(CacheEntry::memory_arrow(array)))
            }
            disk @ CacheEntry::DiskLiquid { .. } => {
                let Some(array) = self.read_disk_liquid_array(entry_id, identity).await else {
                    return PrefetchResult::Absent;
                };
                self.maybe_hydrate(
                    entry_id,
                    identity,
                    disk,
                    MaterializedEntry::Liquid(&array),
                    None,
                )
                .await;
                PrefetchResult::Snapshot(Arc::new(CacheEntry::memory_liquid(array)))
            }
        }
    }

    /// Try to read a liquid array from the cache.
    /// Returns None if the cached data is not in liquid format.
    pub async fn try_read_liquid(
        &self,
        entry_id: &EntryID,
        identity: u64,
    ) -> Option<crate::liquid_array::LiquidArrayRef> {
        self.observer.on_try_read_liquid();
        self.trace(InternalEvent::TryReadLiquid { entry: *entry_id });
        let slot = self.index.get_for(entry_id, identity)?;
        self.cache_policy
            .notify_access(entry_id, CachedBatchType::from(&slot.entry));

        match &slot.entry {
            CacheEntry::MemoryLiquid(array) => Some(array.clone()),
            entry @ CacheEntry::DiskLiquid { .. } => {
                let liquid = self.read_disk_liquid_array(entry_id, identity).await?;
                self.maybe_hydrate(
                    entry_id,
                    identity,
                    entry,
                    MaterializedEntry::Liquid(&liquid),
                    None,
                )
                .await;
                Some(liquid)
            }
            CacheEntry::DiskArrow { .. } | CacheEntry::MemoryArrow(_) => None,
        }
    }

    /// Iterate over all entries in the cache, whichever identity holds them.
    /// No guarantees are made about the order of the entries.
    /// Isolation level: read-committed
    pub fn for_each_entry(&self, mut f: impl FnMut(&EntryID, &CacheEntry)) {
        self.index
            .for_each(|entry_id, slot| f(entry_id, &slot.entry));
    }

    /// Reset the cache.
    pub fn reset(&self) {
        self.index.reset();
        self.budget.reset_usage();
    }

    /// Check whether the cache holds a batch under `entry_id` for `identity`.
    pub fn contains(&self, entry_id: &EntryID, identity: u64) -> bool {
        self.index.contains(entry_id, identity)
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

    /// Add a lineage expression for an entry.
    pub fn add_lineage(&self, entry_id: &EntryID, expression: Arc<CacheExpression>) {
        self.metadata.add_lineage(entry_id, expression);
    }

    /// Flush all entries to disk.
    pub async fn flush_all_to_disk(&self) -> Result<(), CacheFull> {
        let mut entires = Vec::new();
        self.index.for_each(|entry_id, slot| {
            entires.push((*entry_id, slot.identity, slot.entry.clone()));
        });
        for (entry_id, identity, batch) in entires {
            match &batch {
                CacheEntry::MemoryArrow(array) => {
                    let bytes = arrow_to_bytes(array).expect("failed to convert arrow to bytes");
                    let disk_bytes = bytes.len();
                    match self
                        .write_batch_to_disk(entry_id, identity, &batch, bytes)
                        .await
                    {
                        Ok(()) => {
                            self.try_insert(
                                entry_id,
                                identity,
                                CacheEntry::disk_arrow(array.data_type().clone(), disk_bytes),
                            )
                            .expect("failed to insert disk arrow entry");
                        }
                        Err(CacheFull) => self.drop_memory_entry(entry_id, &batch),
                    }
                }
                CacheEntry::MemoryLiquid(liquid_array) => {
                    let liquid_bytes = liquid_array.to_bytes();
                    let disk_bytes = liquid_bytes.len();
                    match self
                        .write_batch_to_disk(entry_id, identity, &batch, Bytes::from(liquid_bytes))
                        .await
                    {
                        Ok(()) => {
                            self.try_insert(
                                entry_id,
                                identity,
                                CacheEntry::disk_liquid(
                                    liquid_array.original_arrow_data_type(),
                                    disk_bytes,
                                ),
                            )
                            .expect("failed to insert disk liquid entry");
                        }
                        Err(CacheFull) => self.drop_memory_entry(entry_id, &batch),
                    }
                }
                CacheEntry::DiskArrow { .. } | CacheEntry::DiskLiquid { .. } => {
                    // Already on disk, skip
                }
            }
        }
        Ok(())
    }
}

impl LiquidCache {
    /// returns the batch that was written to disk
    async fn write_in_memory_batch_to_disk(
        &self,
        entry_id: EntryID,
        identity: u64,
        batch: CacheEntry,
    ) -> Result<CacheEntry, CacheFull> {
        match &batch {
            batch @ CacheEntry::MemoryArrow(_) => {
                let outcome = self
                    .eviction_policy
                    .evict(batch, self.metadata.lineage(&entry_id).as_deref());
                let EvictionOutcome::Replace {
                    entry: new_batch,
                    bytes_to_write,
                } = outcome
                else {
                    unreachable!("memory Arrow eviction cannot remove entry");
                };
                if let Some(bytes_to_write) = bytes_to_write {
                    self.write_batch_to_disk(entry_id, identity, &new_batch, bytes_to_write)
                        .await?;
                }
                Ok(new_batch)
            }
            CacheEntry::MemoryLiquid(liquid_array) => {
                let liquid_bytes = Bytes::from(liquid_array.to_bytes());
                let disk_bytes = liquid_bytes.len();
                self.write_batch_to_disk(entry_id, identity, &batch, liquid_bytes)
                    .await?;
                Ok(CacheEntry::disk_liquid(
                    liquid_array.original_arrow_data_type(),
                    disk_bytes,
                ))
            }
            CacheEntry::DiskLiquid { .. } | CacheEntry::DiskArrow { .. } => {
                unreachable!("Unexpected batch in write_in_memory_batch_to_disk")
            }
        }
    }

    /// Insert a batch into the cache, it will run cache replacement policy until the batch is inserted.
    pub(crate) async fn insert_inner(
        &self,
        entry_id: EntryID,
        identity: u64,
        mut batch_to_cache: CacheEntry,
    ) -> Result<(), CacheFull> {
        loop {
            let Err(not_inserted) = self.try_insert(entry_id, identity, batch_to_cache) else {
                return Ok(());
            };
            self.trace(InternalEvent::InsertFailed {
                entry: entry_id,
                kind: CachedBatchType::from(&not_inserted),
            });

            let victims = self.cache_policy.find_memory_victim(8);
            if victims.is_empty() {
                // no advice, because the cache is already empty
                // this can happen if the entry to be inserted is too large, in that case,
                // we write it to disk
                let on_disk_batch = self
                    .write_in_memory_batch_to_disk(entry_id, identity, not_inserted)
                    .await?;
                batch_to_cache = on_disk_batch;
                continue;
            }
            self.evict_victims(victims).await?;

            batch_to_cache = not_inserted;
            crate::utils::yield_now_if_shuttle();
        }
    }

    /// Create a new instance of CacheStorage.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        batch_size: usize,
        max_memory_bytes: usize,
        max_disk_bytes: usize,
        eviction_policy: Box<dyn EvictionPolicy>,
        cache_policy: Box<dyn CachePolicy>,
        hydration_policy: Box<dyn HydrationPolicy>,
        metadata: Arc<dyn EntryMetadata>,
        store: t4::Store,
        evict_victims_concurrently: bool,
    ) -> Self {
        let config = CacheConfig::new(batch_size, max_memory_bytes, max_disk_bytes);
        let observer = Arc::new(Observer::new());
        Self {
            index: ArtIndex::new(),
            budget: BudgetAccounting::new(
                config.max_memory_bytes(),
                config.max_disk_bytes(),
                observer.clone(),
            ),
            config,
            cache_policy,
            hydration_policy,
            eviction_policy,
            observer,
            metadata,
            store,
            evict_victims_concurrently,
        }
    }

    fn try_insert(
        &self,
        entry_id: EntryID,
        identity: u64,
        to_insert: CacheEntry,
    ) -> Result<(), CacheEntry> {
        let new_memory_size = to_insert.memory_usage_bytes();
        let cached_batch_type = if let Some(slot) = self.index.get(&entry_id) {
            let old_memory_size = slot.entry.memory_usage_bytes();
            if self
                .budget
                .try_update_memory_usage(old_memory_size, new_memory_size)
                .is_err()
            {
                return Err(to_insert);
            }
            let batch_type = CachedBatchType::from(&to_insert);
            self.index.insert(&entry_id, identity, to_insert);
            batch_type
        } else {
            if self.budget.try_reserve_memory(new_memory_size).is_err() {
                return Err(to_insert);
            }
            let batch_type = CachedBatchType::from(&to_insert);
            self.index.insert(&entry_id, identity, to_insert);
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

    fn drop_memory_entry(&self, entry_id: EntryID, _expected: &CacheEntry) {
        let Some(removed) = self.index.remove(&entry_id) else {
            return;
        };
        assert!(
            matches!(
                &removed.entry,
                CacheEntry::MemoryArrow(_) | CacheEntry::MemoryLiquid(_)
            ),
            "flush should only drop memory entries"
        );
        self.budget
            .try_update_memory_usage(removed.entry.memory_usage_bytes(), 0)
            .expect("memory release cannot fail");
        self.cache_policy.notify_remove(&entry_id);
    }

    async fn remove_disk_entry(&self, entry_id: EntryID) {
        let Some(removed) = self.index.remove(&entry_id) else {
            return;
        };
        let disk_bytes = match &removed.entry {
            CacheEntry::DiskLiquid { disk_bytes, .. }
            | CacheEntry::DiskArrow { disk_bytes, .. } => *disk_bytes,
            _ => panic!("remove_disk_entry called for non-disk entry"),
        };
        self.store
            .remove(&entry_id_to_key(&entry_id))
            .await
            .expect("disk remove failed");
        self.budget.release_disk(disk_bytes + IDENTITY_TRAILER_LEN);
        self.cache_policy.notify_remove(&entry_id);
        self.trace(InternalEvent::DiskEvict {
            entry: entry_id,
            bytes: disk_bytes,
        });
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
    async fn evict_victims(&self, victims: Vec<EntryID>) -> Result<(), CacheFull> {
        self.trace(InternalEvent::EvictionBegin {
            victims: victims.clone(),
        });
        if self.evict_victims_concurrently {
            let results = futures::stream::iter(victims)
                .map(|victim| self.evict_victim_inner(victim))
                .buffer_unordered(usize::MAX)
                .collect::<Vec<_>>()
                .await;
            results.into_iter().collect::<Result<Vec<_>, _>>()?;
        } else {
            for victim in victims {
                self.evict_victim_inner(victim).await?;
            }
        }
        Ok(())
    }

    async fn evict_victim_inner(&self, victim: EntryID) -> Result<(), CacheFull> {
        let Some(slot) = self.index.get(&victim) else {
            return Ok(());
        };
        // The rewrite is made for the identity the victim was read under. If
        // another identity takes the key over meanwhile, the rewrite still
        // carries the old identity, so the new owner reads it as a miss.
        let identity = slot.identity;
        let mut victim_entry = slot.entry.clone();
        self.trace(InternalEvent::EvictionVictim { entry: victim });
        loop {
            let outcome = self
                .eviction_policy
                .evict(&victim_entry, self.metadata.lineage(&victim).as_deref());

            match outcome {
                EvictionOutcome::Replace {
                    entry: new_batch,
                    bytes_to_write,
                } => {
                    if let Some(bytes_to_write) = bytes_to_write {
                        self.write_batch_to_disk(victim, identity, &new_batch, bytes_to_write)
                            .await?;
                    }
                    match self.try_insert(victim, identity, new_batch) {
                        Ok(()) => {
                            break;
                        }
                        Err(batch) => {
                            victim_entry = batch;
                        }
                    }
                }
                EvictionOutcome::Remove => {
                    self.remove_disk_entry(victim).await;
                    break;
                }
            }
        }
        Ok(())
    }

    async fn maybe_hydrate(
        &self,
        entry_id: &EntryID,
        identity: u64,
        cached: &CacheEntry,
        materialized: MaterializedEntry<'_>,
        expression: Option<&CacheExpression>,
    ) {
        if let Some(new_entry) = self.hydration_policy.hydrate(&HydrationRequest {
            entry_id: *entry_id,
            cached,
            materialized,
            expression,
        }) {
            let cached_type = CachedBatchType::from(cached);
            let new_type = CachedBatchType::from(&new_entry);
            self.trace(InternalEvent::Hydrate {
                entry: *entry_id,
                cached: cached_type,
                new: new_type,
            });
            let _ = self.insert_inner(*entry_id, identity, new_entry).await;
        }
    }

    pub(crate) async fn read_arrow_array(
        &self,
        entry_id: &EntryID,
        identity: u64,
        selection: Option<&BooleanBuffer>,
        expression: Option<&CacheExpression>,
    ) -> Option<ArrayRef> {
        self.observer.on_get(selection.is_some());
        let slot = self.index.get_for(entry_id, identity)?;
        self.cache_policy
            .notify_access(entry_id, CachedBatchType::from(&slot.entry));
        self.read_entry_inner(entry_id, identity, &slot.entry, selection, expression)
            .await
    }

    /// Read an already-looked-up cache entry, cached for `identity`.
    ///
    /// `entry` is a value the caller already holds, so it is not checked
    /// against `identity`: the caller must have obtained it for that identity,
    /// e.g. from [`Self::prefetch`]. `identity` is checked when the entry's
    /// data has to be read from disk.
    pub async fn read_entry(
        &self,
        entry_id: &EntryID,
        identity: u64,
        entry: &CacheEntry,
        selection: Option<&BooleanBuffer>,
        expression: Option<&CacheExpression>,
    ) -> Option<ArrayRef> {
        self.observer.on_get(selection.is_some());
        self.read_entry_inner(entry_id, identity, entry, selection, expression)
            .await
    }

    async fn read_entry_inner(
        &self,
        entry_id: &EntryID,
        identity: u64,
        entry: &CacheEntry,
        selection: Option<&BooleanBuffer>,
        expression: Option<&CacheExpression>,
    ) -> Option<ArrayRef> {
        use arrow::array::BooleanArray;

        self.trace(InternalEvent::Read {
            entry: *entry_id,
            selection: selection.is_some(),
            expr: expression.cloned(),
            cached: CachedBatchType::from(entry),
        });

        match entry {
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
            CacheEntry::DiskArrow { .. } | CacheEntry::DiskLiquid { .. } => {
                self.read_disk_array(entry, entry_id, identity, expression, selection)
                    .await
            }
        }
    }

    async fn read_disk_array(
        &self,
        entry: &CacheEntry,
        entry_id: &EntryID,
        identity: u64,
        expression: Option<&CacheExpression>,
        selection: Option<&BooleanBuffer>,
    ) -> Option<ArrayRef> {
        match entry {
            CacheEntry::DiskArrow { data_type, .. } => {
                if let Some(selection) = selection
                    && selection.count_set_bits() == 0
                {
                    return Some(arrow::array::new_empty_array(data_type));
                }
                let full_array = self.read_disk_arrow_array(entry_id, identity).await?;
                self.maybe_hydrate(
                    entry_id,
                    identity,
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
            CacheEntry::DiskLiquid { data_type, .. } => {
                if let Some(selection) = selection
                    && selection.count_set_bits() == 0
                {
                    return Some(arrow::array::new_empty_array(data_type));
                }
                let liquid = self.read_disk_liquid_array(entry_id, identity).await?;
                self.maybe_hydrate(
                    entry_id,
                    identity,
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

    /// Write `bytes` to disk for `identity`, followed by the identity itself.
    ///
    /// Objects are keyed by entry id alone, so a write landing after the key
    /// changed hands can leave one identity's bytes where another's entry
    /// points. The trailer is what lets a read notice that and miss. It sits
    /// after the payload so the payload keeps the alignment it is read with.
    /// The disk budget is charged for the trailer as well as the payload.
    async fn write_batch_to_disk(
        &self,
        entry_id: EntryID,
        identity: u64,
        batch: &CacheEntry,
        bytes: Bytes,
    ) -> Result<(), CacheFull> {
        let len = bytes.len();
        loop {
            if self
                .budget
                .try_reserve_disk(len + IDENTITY_TRAILER_LEN)
                .is_ok()
            {
                break;
            }
            let victims = self.cache_policy.find_disk_victim(8);
            if victims.is_empty() {
                return Err(CacheFull);
            }
            for victim in victims {
                self.remove_disk_entry(victim).await;
            }
        }
        self.trace(InternalEvent::IoWrite {
            entry: entry_id,
            kind: CachedBatchType::from(batch),
            bytes: len,
        });
        let mut object = Vec::with_capacity(len + IDENTITY_TRAILER_LEN);
        object.extend_from_slice(&bytes);
        object.extend_from_slice(&identity.to_le_bytes());
        self.store
            .put(entry_id_to_key(&entry_id), object)
            .await
            .expect("write failed");
        Ok(())
    }

    /// The payload stored under `entry_id`, if it was written for `identity`.
    async fn read_disk_bytes(&self, entry_id: &EntryID, identity: u64) -> Option<Bytes> {
        let mut object = match self.store.get(&entry_id_to_key(entry_id)).await {
            Ok(object) => object,
            Err(t4::Error::NotFound) => return None,
            Err(error) => panic!("read failed: {error}"),
        };
        let payload_len = object.len().checked_sub(IDENTITY_TRAILER_LEN)?;
        let trailer: [u8; IDENTITY_TRAILER_LEN] = object[payload_len..].try_into().unwrap();
        if u64::from_le_bytes(trailer) != identity {
            return None;
        }
        object.truncate(payload_len);
        Some(Bytes::from(object))
    }

    async fn read_disk_arrow_array(&self, entry_id: &EntryID, identity: u64) -> Option<ArrayRef> {
        let bytes = self.read_disk_bytes(entry_id, identity).await?;
        let bytes_len = bytes.len();
        let cursor = std::io::Cursor::new(bytes);
        let mut reader =
            arrow::ipc::reader::StreamReader::try_new(cursor, None).expect("create reader failed");
        let batch = reader.next().unwrap().expect("read batch failed");
        let array = batch.column(0).clone();
        self.trace(InternalEvent::IoReadArrow {
            entry: *entry_id,
            bytes: bytes_len,
        });
        Some(array)
    }

    async fn read_disk_liquid_array(
        &self,
        entry_id: &EntryID,
        identity: u64,
    ) -> Option<crate::liquid_array::LiquidArrayRef> {
        let bytes = self.read_disk_bytes(entry_id, identity).await?;
        self.trace(InternalEvent::IoReadLiquid {
            entry: *entry_id,
            bytes: bytes.len(),
        });
        Some(Arc::new(crate::liquid_array::LiquidArray::from_bytes(
            bytes,
        )))
    }

    pub(crate) async fn eval_predicate_internal(
        &self,
        entry_id: &EntryID,
        identity: u64,
        selection_opt: Option<&BooleanBuffer>,
        predicate: &LiquidExpr,
    ) -> Option<BooleanArray> {
        self.observer.on_eval_predicate();
        let slot = self.index.get_for(entry_id, identity)?;
        self.cache_policy
            .notify_access(entry_id, CachedBatchType::from(&slot.entry));
        self.eval_predicate_on_entry_inner(
            entry_id,
            identity,
            &slot.entry,
            selection_opt,
            predicate,
        )
        .await
    }

    /// Evaluate a predicate on an already-looked-up cache entry, cached for `identity`.
    ///
    /// As with [`Self::read_entry`], `entry` itself is not checked against
    /// `identity`; only data read from disk is.
    pub async fn eval_predicate_on_entry(
        &self,
        entry_id: &EntryID,
        identity: u64,
        entry: &CacheEntry,
        selection_opt: Option<&BooleanBuffer>,
        predicate: &LiquidExpr,
    ) -> Option<BooleanArray> {
        self.observer.on_eval_predicate();
        self.eval_predicate_on_entry_inner(entry_id, identity, entry, selection_opt, predicate)
            .await
    }

    async fn eval_predicate_on_entry_inner(
        &self,
        entry_id: &EntryID,
        identity: u64,
        entry: &CacheEntry,
        selection_opt: Option<&BooleanBuffer>,
        predicate: &LiquidExpr,
    ) -> Option<BooleanArray> {
        self.trace(InternalEvent::EvalPredicate {
            entry: *entry_id,
            selection: selection_opt.is_some(),
            cached: CachedBatchType::from(entry),
        });

        match entry {
            CacheEntry::MemoryArrow(array) => {
                let mut owned = None;
                let selection = selection_opt.unwrap_or_else(|| {
                    owned = Some(BooleanBuffer::new_set(array.len()));
                    owned.as_ref().unwrap()
                });
                let selection_array = BooleanArray::new(selection.clone(), None);
                let filtered = arrow::compute::filter(array, &selection_array)
                    .expect("selection must match array length");
                Some(self.eval_predicate_on_array(filtered, predicate))
            }
            entry @ CacheEntry::DiskArrow { .. } => {
                let array = self.read_disk_arrow_array(entry_id, identity).await?;
                self.maybe_hydrate(
                    entry_id,
                    identity,
                    entry,
                    MaterializedEntry::Arrow(&array),
                    None,
                )
                .await;
                let mut owned = None;
                let selection = selection_opt.unwrap_or_else(|| {
                    owned = Some(BooleanBuffer::new_set(array.len()));
                    owned.as_ref().unwrap()
                });
                let selection_array = BooleanArray::new(selection.clone(), None);
                let filtered = arrow::compute::filter(&array, &selection_array)
                    .expect("selection must match array length");
                Some(self.eval_predicate_on_array(filtered, predicate))
            }
            CacheEntry::MemoryLiquid(array) => {
                let mut owned = None;
                let selection = selection_opt.unwrap_or_else(|| {
                    owned = Some(BooleanBuffer::new_set(array.len()));
                    owned.as_ref().unwrap()
                });
                Some(array.try_eval_predicate(predicate, selection))
            }
            entry @ CacheEntry::DiskLiquid { .. } => {
                let liquid = self.read_disk_liquid_array(entry_id, identity).await?;
                self.maybe_hydrate(
                    entry_id,
                    identity,
                    entry,
                    MaterializedEntry::Liquid(&liquid),
                    None,
                )
                .await;
                let mut owned = None;
                let selection = selection_opt.unwrap_or_else(|| {
                    owned = Some(BooleanBuffer::new_set(liquid.len()));
                    owned.as_ref().unwrap()
                });
                Some(liquid.try_eval_predicate(predicate, selection))
            }
        }
    }

    fn eval_predicate_on_array(&self, array: ArrayRef, predicate: &LiquidExpr) -> BooleanArray {
        let schema = Arc::new(Schema::new(vec![Field::new(
            "liquid_predicate_col",
            array.data_type().clone(),
            true,
        )]));
        let record_batch =
            RecordBatch::try_new(schema, vec![array]).expect("single-column predicate batch");
        let result = predicate
            .physical_expr()
            .evaluate(&record_batch)
            .expect("validated LiquidExpr must evaluate");
        let boolean_array = result
            .into_array(record_batch.num_rows())
            .expect("predicate output must be an array");
        boolean_array.as_boolean().clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cache::{
        CacheEntry, CachePolicy, LiquidCacheBuilder, LiquidPolicy, TranscodeEvict,
        utils::{arrow_to_bytes, create_cache_store, create_test_array, create_test_arrow_array},
    };
    use crate::sync::thread;
    use arrow::array::{Array, ArrayRef, Int32Array};
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
        fn find_memory_victim(&self, _cnt: usize) -> Vec<EntryID> {
            self.advice_count.fetch_add(1, Ordering::SeqCst);
            let id_to_use = self.target_id.unwrap();
            vec![id_to_use]
        }
    }

    #[tokio::test]
    async fn test_basic_cache_operations() {
        // Test basic insert, get, and size tracking in one test
        let budget_size = 10 * 1024;
        let store = create_cache_store(budget_size, Box::new(LiquidPolicy::new())).await;

        // 1. Initial budget should be empty
        assert_eq!(store.budget.memory_usage_bytes(), 0);

        // 2. Insert and verify first entry
        let entry_id1: EntryID = EntryID::from(1);
        let array1 = create_test_array(100);
        let size1 = array1.memory_usage_bytes();
        store.insert_inner(entry_id1, 0, array1).await.unwrap();

        // Verify budget usage and data correctness
        assert_eq!(store.budget.memory_usage_bytes(), size1);
        let retrieved1 = store.index().get(&entry_id1).unwrap();
        match &retrieved1.entry {
            CacheEntry::MemoryArrow(arr) => assert_eq!(arr.len(), 100),
            _ => panic!("Expected ArrowMemory"),
        }

        let entry_id2: EntryID = EntryID::from(2);
        let array2 = create_test_array(200);
        let size2 = array2.memory_usage_bytes();
        store.insert_inner(entry_id2, 0, array2).await.unwrap();

        assert_eq!(store.budget.memory_usage_bytes(), size1 + size2);

        let array3 = create_test_array(150);
        let size3 = array3.memory_usage_bytes();
        store.insert_inner(entry_id1, 0, array3).await.unwrap();

        assert_eq!(store.budget.memory_usage_bytes(), size3 + size2);
        assert!(store.index().get(&EntryID::from(999)).is_none());
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
            let store = create_cache_store(8000, Box::new(advisor)).await; // Small budget to force advice

            store
                .insert_inner(entry_id1, 0, create_test_array(800))
                .await
                .unwrap();
            match &store.index().get(&entry_id1).unwrap().entry {
                CacheEntry::MemoryArrow(_) => {}
                other => panic!("Expected ArrowMemory, got {other:?}"),
            }

            store
                .insert_inner(entry_id2, 0, create_test_array(800))
                .await
                .unwrap();
            match &store.index().get(&entry_id1).unwrap().entry {
                CacheEntry::MemoryLiquid(_) => {}
                other => panic!("Expected LiquidMemory after eviction, got {other:?}"),
            }
        }
    }

    #[tokio::test]
    async fn test_concurrent_cache_operations() {
        concurrent_cache_operations().await;
    }

    // #[cfg(feature = "shuttle")]
    // #[test]
    // fn shuttle_cache_operations() {
    //     crate::utils::shuttle_test(|| {
    //         block_on(concurrent_cache_operations());
    //     });
    // }

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
        let store = create_cache_store(budget_size, Box::new(LiquidPolicy::new())).await;

        let mut handles = vec![];
        for thread_id in 0..num_threads {
            let store = store.clone();
            handles.push(thread::spawn(move || {
                block_on(async {
                    for i in 0..ops_per_thread {
                        let unique_id = thread_id * ops_per_thread + i;
                        let entry_id: EntryID = EntryID::from(unique_id);
                        let array = create_test_arrow_array(100);
                        store.insert(entry_id, 0, array).await.unwrap();
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
            .with_max_memory_bytes(10 * 1024 * 1024)
            .with_eviction_policy(Box::new(TranscodeEvict))
            .build()
            .await;

        // Insert two small batches
        let arr1: ArrayRef = Arc::new(Int32Array::from_iter_values(0..64));
        let arr2: ArrayRef = Arc::new(Int32Array::from_iter_values(0..128));
        storage
            .insert(EntryID::from(1usize), 0, arr1)
            .await
            .unwrap();
        storage
            .insert(EntryID::from(2usize), 0, arr2)
            .await
            .unwrap();

        // Stats after insert: 2 entries, memory usage > 0, disk usage == 0
        let s = storage.stats();
        assert_eq!(s.total_entries, 2);
        assert!(s.memory_usage_bytes > 0);
        assert_eq!(s.disk_usage_bytes, 0);
        assert_eq!(s.max_memory_bytes, 10 * 1024 * 1024);

        // Flush to disk and verify memory usage drops and disk usage increases
        storage.flush_all_to_disk().await.unwrap();
        let s2 = storage.stats();
        assert_eq!(s2.total_entries, 2);
        assert!(s2.disk_usage_bytes > 0);
        // In-memory usage should be reduced after moving to on-disk formats
        assert!(s2.memory_usage_bytes <= s.memory_usage_bytes);
    }

    #[tokio::test]
    async fn hydrate_disk_arrow_on_get_promotes_to_memory() {
        let store = create_cache_store(1 << 20, Box::new(LiquidPolicy::new())).await;
        let entry_id = EntryID::from(321usize);
        let array = create_test_arrow_array(8);

        store.insert(entry_id, 0, array.clone()).await.unwrap();
        store.flush_all_to_disk().await.unwrap();
        {
            let entry = store.index().get(&entry_id).unwrap();
            assert!(matches!(&entry.entry, CacheEntry::DiskArrow { .. }));
        }

        let result = store.get(&entry_id, 0).await.expect("present");
        assert_eq!(result.as_ref(), array.as_ref());
        {
            let entry = store.index().get(&entry_id).unwrap();
            assert!(matches!(&entry.entry, CacheEntry::MemoryArrow(_)));
        }
    }

    #[tokio::test]
    async fn an_entry_is_served_only_to_its_identity() {
        let store = create_cache_store(1 << 20, Box::new(LiquidPolicy::new())).await;
        let entry_id = EntryID::from(321usize);
        let array = create_test_arrow_array(8);

        store.insert(entry_id, 1, array.clone()).await.unwrap();
        assert!(store.contains(&entry_id, 1));
        assert!(!store.contains(&entry_id, 2));
        assert!(store.get(&entry_id, 2).await.is_none());
        assert!(matches!(
            store.prefetch(&entry_id, 2).await,
            PrefetchResult::Absent
        ));

        store.flush_all_to_disk().await.unwrap();
        assert!(store.get(&entry_id, 2).await.is_none());
        let result = store.get(&entry_id, 1).await.expect("present");
        assert_eq!(result.as_ref(), array.as_ref());

        let replacement = create_test_arrow_array(16);
        store
            .insert(entry_id, 2, replacement.clone())
            .await
            .unwrap();
        assert!(!store.contains(&entry_id, 1));
        let result = store.get(&entry_id, 2).await.expect("present");
        assert_eq!(result.as_ref(), replacement.as_ref());
    }

    #[tokio::test]
    async fn disk_bytes_written_for_another_identity_are_a_miss() {
        let store = create_cache_store(1 << 20, Box::new(LiquidPolicy::new())).await;
        let entry_id = EntryID::from(321usize);
        let array = create_test_arrow_array(8);

        store.insert(entry_id, 1, array.clone()).await.unwrap();
        store.flush_all_to_disk().await.unwrap();
        let disk_entry = store.index().get(&entry_id).unwrap().entry.clone();
        assert!(matches!(disk_entry, CacheEntry::DiskArrow { .. }));

        // A record naming identity 2 over bytes identity 1 wrote: what a
        // rewrite landing after the key changed hands leaves behind.
        store.index().insert(&entry_id, 2, disk_entry);
        assert!(store.get(&entry_id, 2).await.is_none());
    }

    #[tokio::test]
    async fn missing_disk_blob_is_a_cache_miss() {
        let directory = tempfile::tempdir().unwrap();
        let store = t4::mount(directory.path().join("cache.t4")).await.unwrap();
        let cache = LiquidCacheBuilder::new()
            .with_store(store.clone())
            .build()
            .await;
        let id = EntryID::from(320usize);

        cache
            .insert(id, 0, create_test_arrow_array(8))
            .await
            .unwrap();
        cache.flush_all_to_disk().await.unwrap();
        store.remove(&entry_id_to_key(&id)).await.unwrap();

        assert!(cache.get(&id, 0).await.is_none());
    }

    #[tokio::test]
    async fn hydrate_disk_liquid_on_get_promotes_to_memory_liquid() {
        let store = create_cache_store(1 << 20, Box::new(LiquidPolicy::new())).await;
        let entry_id = EntryID::from(322usize);
        let arrow_array: ArrayRef = Arc::new(Int32Array::from(vec![1, 2, 3, 4]));
        let liquid =
            Arc::new(crate::liquid_array::LiquidArray::from_arrow_array(&arrow_array).unwrap());

        store
            .insert_inner(entry_id, 0, CacheEntry::memory_liquid(liquid.clone()))
            .await
            .unwrap();
        store.flush_all_to_disk().await.unwrap();
        {
            let entry = store.index().get(&entry_id).unwrap();
            assert!(matches!(&entry.entry, CacheEntry::DiskLiquid { .. }));
        }

        let result = store.get(&entry_id, 0).await.expect("present");
        assert_eq!(result.as_ref(), arrow_array.as_ref());
        {
            let entry = store.index().get(&entry_id).unwrap();
            assert!(matches!(&entry.entry, CacheEntry::MemoryLiquid(_)));
        }
    }

    #[tokio::test]
    async fn insert_returns_cache_full_when_memory_and_disk_are_saturated() {
        let cache = LiquidCacheBuilder::new()
            .with_max_memory_bytes(0)
            .with_max_disk_bytes(0)
            .with_eviction_policy(Box::new(TranscodeEvict))
            .build()
            .await;
        let array: ArrayRef = Arc::new(Int32Array::from_iter_values(0..16));

        let err = cache.insert(EntryID::from(900usize), 0, array).await;

        assert_eq!(err, Err(CacheFull));
        assert!(!cache.contains(&EntryID::from(900usize), 0));
    }

    #[tokio::test]
    async fn insert_until_disk_full_then_evicts_oldest_disk_entry() {
        let first_array: ArrayRef = Arc::new(Int32Array::from_iter_values(0..16));
        let second_array: ArrayRef = Arc::new(Int32Array::from_iter_values(16..32));
        let first_bytes = arrow_to_bytes(&first_array).unwrap().len() + IDENTITY_TRAILER_LEN;
        let second_bytes = arrow_to_bytes(&second_array).unwrap().len() + IDENTITY_TRAILER_LEN;
        let cache = LiquidCacheBuilder::new()
            .with_max_memory_bytes(1 << 20)
            .with_max_disk_bytes(first_bytes.max(second_bytes))
            .with_eviction_policy(Box::new(TranscodeEvict))
            .with_cache_policy(Box::new(LiquidPolicy::new()))
            .build()
            .await;

        let first = EntryID::from(910usize);
        let second = EntryID::from(911usize);
        cache.insert(first, 0, first_array).await.unwrap();
        cache.flush_all_to_disk().await.unwrap();
        assert!(cache.contains(&first, 0));

        cache.insert(second, 0, second_array).await.unwrap();
        cache.flush_all_to_disk().await.unwrap();

        assert!(!cache.contains(&first, 0));
        assert!(matches!(
            &cache.index().get(&second).unwrap().entry,
            CacheEntry::DiskArrow { .. }
        ));
    }

    #[tokio::test]
    async fn flush_all_to_disk_evicts_when_overflow() {
        let first_array: ArrayRef = Arc::new(Int32Array::from_iter_values(0..16));
        let second_array: ArrayRef = Arc::new(Int32Array::from_iter_values(16..32));
        let disk_bytes = arrow_to_bytes(&first_array).unwrap().len();
        let cache = LiquidCacheBuilder::new()
            .with_max_memory_bytes(1 << 20)
            .with_max_disk_bytes(disk_bytes)
            .with_eviction_policy(Box::new(TranscodeEvict))
            .with_cache_policy(Box::new(LiquidPolicy::new()))
            .build()
            .await;
        let first = EntryID::from(912usize);
        let second = EntryID::from(913usize);
        cache.insert(first, 0, first_array).await.unwrap();
        cache.flush_all_to_disk().await.unwrap();
        cache.insert(second, 0, second_array).await.unwrap();

        cache.flush_all_to_disk().await.unwrap();

        assert!(!cache.contains(&first, 0) || !cache.contains(&second, 0));
    }

    #[tokio::test]
    async fn disk_eviction_releases_budget() {
        let array: ArrayRef = Arc::new(Int32Array::from_iter_values(0..16));
        let disk_bytes = arrow_to_bytes(&array).unwrap().len() + IDENTITY_TRAILER_LEN;
        let cache = LiquidCacheBuilder::new()
            .with_max_memory_bytes(1 << 20)
            .with_max_disk_bytes(disk_bytes)
            .with_eviction_policy(Box::new(TranscodeEvict))
            .with_cache_policy(Box::new(LiquidPolicy::new()))
            .build()
            .await;
        let entry = EntryID::from(914usize);
        cache.insert(entry, 0, array).await.unwrap();
        cache.flush_all_to_disk().await.unwrap();
        let before = cache.stats().disk_usage_bytes;

        cache.remove_disk_entry(entry).await;

        assert_eq!(cache.stats().disk_usage_bytes, before - disk_bytes);
        assert!(!cache.contains(&entry, 0));
    }

    #[tokio::test]
    async fn flush_all_to_disk_drops_entry_on_unrecoverable_overflow() {
        let cache = LiquidCacheBuilder::new()
            .with_max_memory_bytes(1 << 20)
            .with_max_disk_bytes(0)
            .with_eviction_policy(Box::new(TranscodeEvict))
            .build()
            .await;
        let entry_id = EntryID::from(901usize);
        let array: ArrayRef = Arc::new(Int32Array::from_iter_values(0..16));
        cache.insert(entry_id, 0, array).await.unwrap();

        let result = cache.flush_all_to_disk().await;

        assert_eq!(result, Ok(()));
        assert!(!cache.contains(&entry_id, 0));
    }
}
