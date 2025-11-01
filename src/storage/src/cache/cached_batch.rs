//! Cached batch types.

use std::{fmt::Display, sync::Arc};

use arrow::array::{ArrayRef, BooleanArray};
use arrow_schema::DataType;

use crate::liquid_array::{LiquidArrayRef, LiquidHybridArrayRef};

/// Cached batch.
#[derive(Debug, Clone)]
pub enum CachedBatch {
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

impl CachedBatch {
    /// Get the memory usage of the cached batch.
    pub fn memory_usage_bytes(&self) -> usize {
        match self {
            Self::MemoryArrow(array) => array.get_array_memory_size(),
            Self::MemoryLiquid(array) => array.get_array_memory_size(),
            Self::MemoryHybridLiquid(array) => array.get_array_memory_size(),
            Self::DiskLiquid(_) => 0,
            Self::DiskArrow(_) => 0,
        }
    }

    /// Get the reference count of the cached batch.
    pub fn reference_count(&self) -> usize {
        match self {
            Self::MemoryArrow(array) => Arc::strong_count(array),
            Self::MemoryLiquid(array) => Arc::strong_count(array),
            Self::MemoryHybridLiquid(array) => Arc::strong_count(array),
            Self::DiskLiquid(_) => 0,
            Self::DiskArrow(_) => 0,
        }
    }
}

impl Display for CachedBatch {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::MemoryArrow(_) => write!(f, "MemoryArrow"),
            Self::MemoryLiquid(_) => write!(f, "MemoryLiquid"),
            Self::MemoryHybridLiquid(_) => write!(f, "MemoryHybridLiquid"),
            Self::DiskLiquid(_) => write!(f, "DiskLiquid"),
            Self::DiskArrow(_) => write!(f, "DiskArrow"),
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

impl From<&CachedBatch> for CachedBatchType {
    fn from(batch: &CachedBatch) -> Self {
        match batch {
            CachedBatch::MemoryArrow(_) => Self::MemoryArrow,
            CachedBatch::MemoryLiquid(_) => Self::MemoryLiquid,
            CachedBatch::MemoryHybridLiquid(_) => Self::MemoryHybridLiquid,
            CachedBatch::DiskLiquid(_) => Self::DiskLiquid,
            CachedBatch::DiskArrow(_) => Self::DiskArrow,
        }
    }
}

/// The result of predicate pushdown.
#[derive(Debug, PartialEq)]
pub enum GetWithPredicateResult {
    /// The predicate is evaluated on the filtered data and the result is a boolean buffer.
    Evaluated(BooleanArray),

    /// The predicate is not evaluated but data is filtered.
    Filtered(ArrayRef),
}
