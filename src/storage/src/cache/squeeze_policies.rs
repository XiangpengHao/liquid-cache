//! Squeeze policies for liquid cache.

use bytes::Bytes;

use crate::cache::{
    LiquidCompressorStates, cached_data::CachedBatch, transcode_liquid_inner, utils::arrow_to_bytes,
};

/// What to do when we need to squeeze an entry?
pub trait SqueezePolicy: std::fmt::Debug + Send + Sync {
    /// Squeeze the entry.
    /// Returns the squeezed entry and the bytes that were used to store the entry on disk.
    fn squeeze(
        &self,
        entry: CachedBatch,
        compressor: &LiquidCompressorStates,
    ) -> (CachedBatch, Option<Bytes>);
}

/// Squeeze the entry to disk.
#[derive(Debug, Default)]
pub struct SqueezeToDiskPolicy;

impl SqueezePolicy for SqueezeToDiskPolicy {
    fn squeeze(
        &self,
        entry: CachedBatch,
        _compressor: &LiquidCompressorStates,
    ) -> (CachedBatch, Option<Bytes>) {
        match entry {
            CachedBatch::MemoryArrow(array) => {
                let bytes = arrow_to_bytes(&array).expect("failed to convert arrow to bytes");
                (CachedBatch::DiskArrow, Some(bytes))
            }
            CachedBatch::MemoryLiquid(liquid_array) => {
                let (hybrid_array, bytes) = match liquid_array.squeeze() {
                    Some(result) => result,
                    None => {
                        // not supported, evict to disk
                        let bytes = liquid_array.to_bytes();
                        let bytes = Bytes::from(bytes);
                        return (CachedBatch::DiskLiquid, Some(bytes));
                    }
                };
                (CachedBatch::MemoryHybridLiquid(hybrid_array), Some(bytes))
            }
            CachedBatch::MemoryHybridLiquid(_hybrid_array) => (CachedBatch::DiskLiquid, None),
            CachedBatch::DiskLiquid | CachedBatch::DiskArrow => (entry, None),
        }
    }
}

/// Squeeze the entry to liquid memory.
#[derive(Debug, Default)]
pub struct SqueezeToLiquidPolicy;

impl SqueezePolicy for SqueezeToLiquidPolicy {
    fn squeeze(
        &self,
        entry: CachedBatch,
        compressor: &LiquidCompressorStates,
    ) -> (CachedBatch, Option<Bytes>) {
        match entry {
            CachedBatch::MemoryArrow(array) => {
                let liquid_array = transcode_liquid_inner(&array, compressor).unwrap();
                (CachedBatch::MemoryLiquid(liquid_array), None)
            }
            CachedBatch::MemoryLiquid(liquid_array) => {
                let (hybrid_array, bytes) = match liquid_array.squeeze() {
                    Some(result) => result,
                    None => {
                        let bytes = liquid_array.to_bytes();
                        let bytes = Bytes::from(bytes);
                        return (CachedBatch::DiskLiquid, Some(bytes));
                    }
                };
                (CachedBatch::MemoryHybridLiquid(hybrid_array), Some(bytes))
            }
            CachedBatch::MemoryHybridLiquid(_hybrid_array) => (CachedBatch::DiskLiquid, None),
            CachedBatch::DiskLiquid | CachedBatch::DiskArrow => (entry, None),
        }
    }
}
