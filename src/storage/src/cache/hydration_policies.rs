//! Hydration policies decide whether and how to promote squeezed/on-disk entries back into memory.

use arrow::array::ArrayRef;

use crate::{
    cache::{CacheExpression, CachedBatchType, cached_batch::CacheEntry, utils::EntryID},
    liquid_array::LiquidArrayRef,
};

/// The materialized representation produced by a cache read.
#[derive(Debug, Clone)]
pub enum MaterializedEntry<'a> {
    /// Arrow array in memory.
    Arrow(&'a ArrayRef),
    /// Liquid array in memory.
    Liquid(&'a LiquidArrayRef),
}

/// Request context provided to a [`HydrationPolicy`].
#[derive(Debug, Clone)]
pub struct HydrationRequest<'a> {
    /// Cache key being materialized.
    pub entry_id: EntryID,
    /// The cached batch type before materialization (e.g., `DiskArrow`).
    pub cached_batch_type: CachedBatchType,
    /// The fully materialized entry produced by the read path.
    pub materialized: MaterializedEntry<'a>,
    /// Optional expression hint associated with the read.
    pub expression: Option<&'a CacheExpression>,
}

/// Decide if a materialized entry should be promoted back into memory.
pub trait HydrationPolicy: std::fmt::Debug + Send + Sync {
    /// Determine how to hydrate a cache entry that was just materialized.
    /// Return a new cache entry to insert if hydration is desired.
    fn decide(&self, request: &HydrationRequest<'_>) -> Option<CacheEntry>;
}

/// Default hydration policy: always keep a materialized cache miss in memory
/// by promoting along the path: disk -> squeezed -> liquid -> arrow.
#[derive(Debug, Default, Clone)]
pub struct AlwaysHydrate;

impl AlwaysHydrate {
    /// Create a new [`AlwaysHydrate`] policy.
    pub fn new() -> Self {
        Self
    }
}

impl HydrationPolicy for AlwaysHydrate {
    fn decide(&self, request: &HydrationRequest<'_>) -> Option<CacheEntry> {
        match (request.cached_batch_type, &request.materialized) {
            (CachedBatchType::DiskArrow, MaterializedEntry::Arrow(arr)) => {
                Some(CacheEntry::memory_arrow((*arr).clone()))
            }
            (CachedBatchType::DiskLiquid, MaterializedEntry::Liquid(liq)) => {
                Some(CacheEntry::memory_liquid((*liq).clone()))
            }
            (CachedBatchType::MemoryLiquid, _) => None,
            // When already squeezed/hybrid or liquid in memory, prefer promoting to Arrow if available.
            (CachedBatchType::MemoryHybridLiquid, MaterializedEntry::Arrow(arr)) => {
                Some(CacheEntry::memory_arrow((*arr).clone()))
            }
            (CachedBatchType::MemoryHybridLiquid, MaterializedEntry::Liquid(liq)) => {
                Some(CacheEntry::memory_liquid((*liq).clone()))
            }
            _ => None,
        }
    }
}

/// No hydration policy: never promote a materialized entry back into memory.
#[derive(Debug, Default, Clone)]
pub struct NoHydration;

impl NoHydration {
    /// Create a new [`NoHydration`] policy.
    pub fn new() -> Self {
        Self
    }
}

impl HydrationPolicy for NoHydration {
    fn decide(&self, _request: &HydrationRequest<'_>) -> Option<CacheEntry> {
        None
    }
}
