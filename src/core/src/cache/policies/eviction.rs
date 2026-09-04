//! Policies for moving cache entries to cheaper storage under memory pressure.

use arrow::array::Array;
use bytes::Bytes;

use crate::cache::{
    CacheExpression, LiquidCompressorStates, cached_batch::CacheEntry,
    transcode_liquid_inner_with_hint, utils::arrow_to_bytes,
};

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
    /// Move the entry one step toward cheaper storage.
    fn evict(
        &self,
        entry: &CacheEntry,
        compressor: &LiquidCompressorStates,
        expression: Option<&CacheExpression>,
    ) -> EvictionOutcome;
}

/// Evict memory entries directly to disk.
#[derive(Debug, Default, Clone)]
pub struct Evict;

impl EvictionPolicy for Evict {
    fn evict(
        &self,
        entry: &CacheEntry,
        _compressor: &LiquidCompressorStates,
        _expression: Option<&CacheExpression>,
    ) -> EvictionOutcome {
        persist(entry)
    }
}

/// Transcode Arrow to Liquid before eventually evicting it to disk.
#[derive(Debug, Default, Clone)]
pub struct TranscodeEvict;

impl EvictionPolicy for TranscodeEvict {
    fn evict(
        &self,
        entry: &CacheEntry,
        compressor: &LiquidCompressorStates,
        expression: Option<&CacheExpression>,
    ) -> EvictionOutcome {
        match entry {
            CacheEntry::MemoryArrow(array) => {
                match transcode_liquid_inner_with_hint(array, compressor, expression) {
                    Ok(liquid) => EvictionOutcome::Replace {
                        entry: CacheEntry::memory_liquid(liquid),
                        bytes_to_write: None,
                    },
                    Err(_) => persist(entry),
                }
            }
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
