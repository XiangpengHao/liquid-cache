mod budget;
mod core;
pub mod policies;
mod tracer;
mod transcode;
mod utils;

use crate::liquid_array::LiquidArrayRef;
use arrow::array::ArrayRef;
pub use core::CacheStore;
pub use policies::CachePolicy;
use std::{
    fmt::Display,
    sync::{Arc, RwLock},
};
pub use transcode::transcode_liquid_inner;
pub use utils::{BatchID, CacheAdvice, CacheEntryID, ColumnAccessPath};

pub struct LiquidCompressorStates {
    fsst_compressor: RwLock<Option<Arc<fsst::Compressor>>>,
}

impl std::fmt::Debug for LiquidCompressorStates {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "EtcCompressorStates")
    }
}

impl Default for LiquidCompressorStates {
    fn default() -> Self {
        Self::new()
    }
}

impl LiquidCompressorStates {
    pub fn new() -> Self {
        Self {
            fsst_compressor: RwLock::new(None),
        }
    }

    pub fn fsst_compressor(&self) -> Option<Arc<fsst::Compressor>> {
        self.fsst_compressor.read().unwrap().clone()
    }
}

#[derive(Debug, Clone)]
pub enum CachedBatch {
    MemoryArrow(ArrayRef),
    MemoryLiquid(LiquidArrayRef),
    DiskLiquid,
    DiskArrow,
}

impl CachedBatch {
    pub fn memory_usage_bytes(&self) -> usize {
        match self {
            Self::MemoryArrow(array) => array.get_array_memory_size(),
            Self::MemoryLiquid(array) => array.get_array_memory_size(),
            Self::DiskLiquid => 0,
            Self::DiskArrow => 0,
        }
    }

    pub fn reference_count(&self) -> usize {
        match self {
            Self::MemoryArrow(array) => Arc::strong_count(array),
            Self::MemoryLiquid(array) => Arc::strong_count(array),
            Self::DiskLiquid => 0,
            Self::DiskArrow => 0,
        }
    }
}

impl Display for CachedBatch {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::MemoryArrow(_) => write!(f, "MemoryArrow"),
            Self::MemoryLiquid(_) => write!(f, "MemoryLiquid"),
            Self::DiskLiquid => write!(f, "DiskLiquid"),
            Self::DiskArrow => write!(f, "DiskArrow"),
        }
    }
}
