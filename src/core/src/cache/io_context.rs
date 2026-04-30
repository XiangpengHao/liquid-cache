use std::{fmt::Debug, ops::Range};

use ahash::AHashMap;
use bytes::Bytes;

use crate::sync::{Arc, RwLock};
use crate::{
    cache::{
        CacheExpression, Observer,
        observer::InternalEvent,
        utils::{EntryID, LiquidCompressorStates},
    },
    liquid_array::SqueezeIoHandler,
};

/// Per-entry metadata used by the cache.
///
/// This trait covers only the metadata side of the cache: where to find a
/// batch's compressor and squeeze hints. All actual byte IO goes through the
/// [`t4::Store`] held by the cache itself.
pub trait EntryMetadata: Debug + Send + Sync {
    /// Add a squeeze hint for an entry.
    fn add_squeeze_hint(&self, _entry_id: &EntryID, _expression: Arc<CacheExpression>) {
        // Do nothing by default
    }

    /// Get the squeeze hint for an entry.
    /// If None, the entry will be evicted to disk entirely.
    /// If Some, the entry will be squeezed according to the cache expressions previously recorded for this column.
    /// For example, if expression is ExtractDate32 { field: Date32Field::Year },
    /// the entry will be squeezed to a [crate::liquid_array::SqueezedDate32Array] with the year
    /// component (Date32 or Timestamp input).
    fn squeeze_hint(&self, _entry_id: &EntryID) -> Option<Arc<CacheExpression>> {
        None
    }

    /// Get the compressor for an entry.
    fn get_compressor(&self, entry_id: &EntryID) -> Arc<LiquidCompressorStates>;
}

/// Convert an [`EntryID`] to a t4 key (8-byte little-endian representation).
pub(crate) fn entry_id_to_key(entry_id: &EntryID) -> Vec<u8> {
    usize::from(*entry_id).to_le_bytes().to_vec()
}

/// A default implementation of [`CacheMetadata`].
///
/// All entries share a single [`LiquidCompressorStates`] and squeeze hints are
/// stored in a flat map keyed by [`EntryID`].
#[derive(Debug, Default)]
pub struct DefaultCacheMetadata {
    compressor_state: Arc<LiquidCompressorStates>,
    squeeze_hints: RwLock<AHashMap<EntryID, Arc<CacheExpression>>>,
}

impl DefaultCacheMetadata {
    /// Create a new instance of [`DefaultCacheMetadata`].
    pub fn new() -> Self {
        Self {
            compressor_state: Arc::new(LiquidCompressorStates::new()),
            squeeze_hints: RwLock::new(AHashMap::new()),
        }
    }
}

impl EntryMetadata for DefaultCacheMetadata {
    fn add_squeeze_hint(&self, entry_id: &EntryID, expression: Arc<CacheExpression>) {
        let mut guard = self.squeeze_hints.write().unwrap();
        guard.insert(*entry_id, expression);
    }

    fn squeeze_hint(&self, entry_id: &EntryID) -> Option<Arc<CacheExpression>> {
        let guard = self.squeeze_hints.read().unwrap();
        guard.get(entry_id).cloned()
    }

    fn get_compressor(&self, _entry_id: &EntryID) -> Arc<LiquidCompressorStates> {
        self.compressor_state.clone()
    }
}

/// A default implementation of [SqueezeIoHandler] backed by a [`t4::Store`].
#[derive(Debug)]
pub struct DefaultSqueezeIo {
    store: t4::Store,
    entry_id: EntryID,
    observer: Arc<Observer>,
}

impl DefaultSqueezeIo {
    /// Create a new instance of [DefaultSqueezeIo].
    pub fn new(store: t4::Store, entry_id: EntryID, observer: Arc<Observer>) -> Self {
        Self {
            store,
            entry_id,
            observer,
        }
    }
}

#[async_trait::async_trait]
impl SqueezeIoHandler for DefaultSqueezeIo {
    async fn read(&self, range: Option<Range<u64>>) -> std::io::Result<Bytes> {
        let key = entry_id_to_key(&self.entry_id);
        let bytes = match range {
            Some(range) => {
                let len = range.end - range.start;
                self.store
                    .get_range(&key, range.start, len)
                    .await
                    .map_err(|e| std::io::Error::other(e.to_string()))?
            }
            None => self
                .store
                .get(&key)
                .await
                .map_err(|e| std::io::Error::other(e.to_string()))?,
        };
        let bytes = Bytes::from(bytes);
        self.observer
            .record_internal(InternalEvent::IoReadSqueezedBacking {
                entry: self.entry_id,
                bytes: bytes.len(),
            });
        Ok(bytes)
    }

    fn tracing_decompress_count(&self, decompress_cnt: usize, total_cnt: usize) {
        self.observer
            .record_internal(InternalEvent::DecompressSqueezed {
                entry: self.entry_id,
                decompressed: decompress_cnt,
                total: total_cnt,
            });
    }

    fn trace_io_saved(&self) {
        self.observer.runtime_stats().incr_squeeze_io_saved();
    }
}

#[cfg(test)]
#[derive(Debug, Default)]
pub(crate) struct TestSqueezeIo {
    bytes: std::sync::Mutex<Option<Bytes>>,
    reads: std::sync::atomic::AtomicUsize,
}

#[cfg(test)]
impl TestSqueezeIo {
    pub(crate) fn set_bytes(&self, bytes: Bytes) {
        *self.bytes.lock().unwrap() = Some(bytes);
    }

    pub(crate) fn reads(&self) -> usize {
        self.reads.load(std::sync::atomic::Ordering::SeqCst)
    }

    pub(crate) fn reset_reads(&self) {
        self.reads.store(0, std::sync::atomic::Ordering::SeqCst);
    }
}

#[cfg(test)]
#[async_trait::async_trait]
impl SqueezeIoHandler for TestSqueezeIo {
    async fn read(&self, range: Option<Range<u64>>) -> std::io::Result<Bytes> {
        self.reads.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        let bytes = self
            .bytes
            .lock()
            .unwrap()
            .clone()
            .expect("test squeeze backing set");
        Ok(match range {
            Some(range) => bytes.slice(range.start as usize..range.end as usize),
            None => bytes,
        })
    }
}
