use std::fmt::Debug;

use ahash::AHashMap;

use crate::cache::{
    CacheExpression,
    utils::{EntryID, LiquidCompressorStates},
};
use crate::sync::{Arc, RwLock};

/// Per-entry metadata used by the cache.
///
/// This trait covers only the metadata side of the cache: where to find a
/// batch's compressor and lineage expressions. All actual byte IO goes through the
/// [`t4::Store`] held by the cache itself.
pub trait EntryMetadata: Debug + Send + Sync {
    /// Add a lineage expression for an entry.
    fn add_lineage(&self, _entry_id: &EntryID, _expression: Arc<CacheExpression>) {
        // Do nothing by default
    }

    /// Get the lineage expression for an entry.
    /// If None, the entry will be evicted to disk entirely.
    /// The expression records how the column is used by query plans and may inform
    /// encoding decisions without discarding any values.
    fn lineage(&self, _entry_id: &EntryID) -> Option<Arc<CacheExpression>> {
        None
    }

    /// Get the compressor for an entry.
    fn get_compressor(&self, entry_id: &EntryID) -> Arc<LiquidCompressorStates>;
}

/// Convert an [`EntryID`] to a t4 key (8-byte little-endian representation).
pub(crate) fn entry_id_to_key(entry_id: &EntryID) -> Vec<u8> {
    usize::from(*entry_id).to_le_bytes().to_vec()
}

/// A default implementation of [`EntryMetadata`].
///
/// All entries share a single [`LiquidCompressorStates`] and lineage expressions are
/// stored in a flat map keyed by [`EntryID`].
#[derive(Debug, Default)]
pub struct DefaultCacheMetadata {
    compressor_state: Arc<LiquidCompressorStates>,
    lineages: RwLock<AHashMap<EntryID, Arc<CacheExpression>>>,
}

impl DefaultCacheMetadata {
    /// Create a new instance of [`DefaultCacheMetadata`].
    pub fn new() -> Self {
        Self {
            compressor_state: Arc::new(LiquidCompressorStates::new()),
            lineages: RwLock::new(AHashMap::new()),
        }
    }
}

impl EntryMetadata for DefaultCacheMetadata {
    fn add_lineage(&self, entry_id: &EntryID, expression: Arc<CacheExpression>) {
        let mut guard = self.lineages.write().unwrap();
        guard.insert(*entry_id, expression);
    }

    fn lineage(&self, entry_id: &EntryID) -> Option<Arc<CacheExpression>> {
        let guard = self.lineages.read().unwrap();
        guard.get(entry_id).cloned()
    }

    fn get_compressor(&self, _entry_id: &EntryID) -> Arc<LiquidCompressorStates> {
        self.compressor_state.clone()
    }
}
