//! Policies for moving cache entries to cheaper storage under memory pressure.

use std::sync::Arc;

use arrow::array::Array;
use bytes::Bytes;

use crate::cache::{CacheExpression, cached_batch::CacheEntry, utils::arrow_to_bytes};
use crate::liquid_array::LiquidArray;

/// The next storage representation selected for a cache entry.
#[derive(Debug, Clone)]
pub enum EvictionOutcome {
    /// Replace the cache entry, optionally persisting these bytes first.
    Replace {
        /// Replacement cache entry.
        entry: CacheEntry,
        /// Bytes that must be persisted before installing the replacement.
        bytes_to_write: Option<Bytes>,
    },
    /// Remove an already-on-disk entry.
    Remove,
}

/// Chooses the next representation for an entry under memory pressure.
pub trait EvictionPolicy: std::fmt::Debug + Send + Sync {
    /// Move the entry one step toward cheaper storage; lineage is available for encoding decisions.
    fn evict(&self, entry: &CacheEntry, lineage: Option<&CacheExpression>) -> EvictionOutcome;
}

/// Evict memory entries directly to disk.
#[derive(Debug, Default, Clone)]
pub struct Evict;

impl EvictionPolicy for Evict {
    fn evict(&self, entry: &CacheEntry, _lineage: Option<&CacheExpression>) -> EvictionOutcome {
        persist(entry)
    }
}

/// Transcode Arrow to Liquid before eventually evicting it to disk.
#[derive(Debug, Default, Clone)]
pub struct TranscodeEvict;

impl EvictionPolicy for TranscodeEvict {
    fn evict(&self, entry: &CacheEntry, _lineage: Option<&CacheExpression>) -> EvictionOutcome {
        match entry {
            CacheEntry::MemoryArrow(array) => match LiquidArray::from_arrow_array(array) {
                Ok(liquid) => EvictionOutcome::Replace {
                    entry: CacheEntry::memory_liquid(Arc::new(liquid)),
                    bytes_to_write: None,
                },
                Err(_) => persist(entry),
            },
            _ => persist(entry),
        }
    }
}

fn persist(entry: &CacheEntry) -> EvictionOutcome {
    match entry {
        CacheEntry::MemoryArrow(array) => {
            let bytes = arrow_to_bytes(array).expect("failed to serialize Arrow array");
            EvictionOutcome::Replace {
                entry: CacheEntry::disk_arrow(array.data_type().clone(), bytes.len()),
                bytes_to_write: Some(bytes),
            }
        }
        CacheEntry::MemoryLiquid(array) => {
            let bytes = Bytes::from(array.to_bytes());
            EvictionOutcome::Replace {
                entry: CacheEntry::disk_liquid(array.original_arrow_data_type(), bytes.len()),
                bytes_to_write: Some(bytes),
            }
        }
        CacheEntry::DiskLiquid { .. } | CacheEntry::DiskArrow { .. } => EvictionOutcome::Remove,
    }
}
