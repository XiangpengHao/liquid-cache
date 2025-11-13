//! Cached batch types.

use std::{fmt::Display, sync::Arc};

use arrow::array::ArrayRef;
use arrow_schema::DataType;

use crate::liquid_array::{LiquidArrayRef, LiquidHybridArrayRef};

/// Backing data for a cached entry.
#[derive(Debug, Clone)]
pub enum CachedData {
    /// Cached batch in memory as Arrow array.
    MemoryArrow(ArrayRef),
    /// Cached batch in memory as liquid array.
    MemoryLiquid(LiquidArrayRef),
    /// Cached batch in memory as hybrid liquid array.
    MemoryHybridLiquid(LiquidHybridArrayRef),
    /// Cached batch on disk as liquid array.
    DiskLiquid(DataType),
    /// Cached batch on disk as Arrow array.
    DiskArrow(DataType),
}

impl CachedData {
    /// Memory usage reported by the underlying representation.
    pub fn memory_usage_bytes(&self) -> usize {
        match self {
            Self::MemoryArrow(array) => array.get_array_memory_size(),
            Self::MemoryLiquid(array) => array.get_array_memory_size(),
            Self::MemoryHybridLiquid(array) => array.get_array_memory_size(),
            Self::DiskLiquid(_) | Self::DiskArrow(_) => 0,
        }
    }

    /// Reference count (if any) of the backing storage.
    pub fn reference_count(&self) -> usize {
        match self {
            Self::MemoryArrow(array) => Arc::strong_count(array),
            Self::MemoryLiquid(array) => Arc::strong_count(array),
            Self::MemoryHybridLiquid(array) => Arc::strong_count(array),
            Self::DiskLiquid(_) | Self::DiskArrow(_) => 0,
        }
    }
}

/// A cached entry.
#[derive(Debug, Clone)]
pub struct CacheEntry {
    data: CachedData,
}

impl CacheEntry {
    /// Construct a cached batch stored as an in-memory Arrow array.
    pub fn memory_arrow(array: ArrayRef) -> Self {
        Self::new(CachedData::MemoryArrow(array))
    }

    /// Construct a cached batch stored as an in-memory Liquid array.
    pub fn memory_liquid(array: LiquidArrayRef) -> Self {
        Self::new(CachedData::MemoryLiquid(array))
    }

    /// Construct a cached batch stored as an in-memory hybrid Liquid array.
    pub fn memory_hybrid_liquid(array: LiquidHybridArrayRef) -> Self {
        Self::new(CachedData::MemoryHybridLiquid(array))
    }

    /// Construct a cached batch stored on disk as Liquid bytes.
    pub fn disk_liquid(data_type: DataType) -> Self {
        Self::new(CachedData::DiskLiquid(data_type))
    }

    /// Construct a cached batch stored on disk as Arrow bytes.
    pub fn disk_arrow(data_type: DataType) -> Self {
        Self::new(CachedData::DiskArrow(data_type))
    }

    fn new(data: CachedData) -> Self {
        Self { data }
    }

    /// Rebuild an entry from stored data.
    pub(crate) fn from_data(data: CachedData) -> Self {
        Self { data }
    }

    /// Borrow the underlying data representation.
    pub fn data(&self) -> &CachedData {
        &self.data
    }

    /// Decompose the batch into its representation.
    pub(crate) fn into_data(self) -> CachedData {
        self.data
    }

    /// Get the memory usage of the cached batch.
    pub fn memory_usage_bytes(&self) -> usize {
        self.data.memory_usage_bytes()
    }

    /// Get the reference count of the cached batch.
    pub fn reference_count(&self) -> usize {
        self.data.reference_count()
    }
}

impl Display for CacheEntry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.data() {
            CachedData::MemoryArrow(_) => write!(f, "MemoryArrow"),
            CachedData::MemoryLiquid(_) => write!(f, "MemoryLiquid"),
            CachedData::MemoryHybridLiquid(_) => write!(f, "MemoryHybridLiquid"),
            CachedData::DiskLiquid(_) => write!(f, "DiskLiquid"),
            CachedData::DiskArrow(_) => write!(f, "DiskArrow"),
        }
    }
}

/// The type of the cached batch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CachedBatchType {
    /// Cached batch in memory as Arrow array.
    MemoryArrow,
    /// Cached batch in memory as liquid array.
    MemoryLiquid,
    /// Cached batch in memory as hybrid liquid array.
    MemoryHybridLiquid,
    /// Cached batch on disk as liquid array.
    DiskLiquid,
    /// Cached batch on disk as Arrow array.
    DiskArrow,
}

impl From<&CacheEntry> for CachedBatchType {
    fn from(batch: &CacheEntry) -> Self {
        match batch.data() {
            CachedData::MemoryArrow(_) => Self::MemoryArrow,
            CachedData::MemoryLiquid(_) => Self::MemoryLiquid,
            CachedData::MemoryHybridLiquid(_) => Self::MemoryHybridLiquid,
            CachedData::DiskLiquid(_) => Self::DiskLiquid,
            CachedData::DiskArrow(_) => Self::DiskArrow,
        }
    }
}
