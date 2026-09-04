//! Policies for promoting on-disk entries back into memory.

use arrow::array::ArrayRef;

use crate::{
    cache::{CacheExpression, cached_batch::CacheEntry, utils::EntryID},
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

/// Context for deciding whether to retain a materialized disk entry.
#[derive(Debug, Clone)]
pub struct HydrationRequest<'a> {
    /// Cache key being materialized.
    pub entry_id: EntryID,
    /// The cached entry before materialization.
    pub cached: &'a CacheEntry,
    /// The fully materialized entry produced by the read path.
    pub materialized: MaterializedEntry<'a>,
    /// Lineage expression associated with the read, when available.
    pub expression: Option<&'a CacheExpression>,
}

/// Decide whether a materialized entry should be promoted back into memory.
pub trait HydrationPolicy: std::fmt::Debug + Send + Sync {
    /// Return a memory entry when hydration is desired.
    fn hydrate(&self, request: &HydrationRequest<'_>) -> Option<CacheEntry>;
}

/// Always retain materialized disk reads in memory.
#[derive(Debug, Default, Clone)]
pub struct AlwaysHydrate;

impl AlwaysHydrate {
    /// Create a new policy.
    pub fn new() -> Self {
        Self
    }
}

impl HydrationPolicy for AlwaysHydrate {
    fn hydrate(&self, request: &HydrationRequest<'_>) -> Option<CacheEntry> {
        match (&request.cached, &request.materialized) {
            (CacheEntry::DiskArrow { .. }, MaterializedEntry::Arrow(array)) => {
                Some(CacheEntry::memory_arrow((*array).clone()))
            }
            (CacheEntry::DiskLiquid { .. }, MaterializedEntry::Liquid(array)) => {
                Some(CacheEntry::memory_liquid((*array).clone()))
            }
            _ => None,
        }
    }
}

/// Never retain materialized disk reads in memory.
#[derive(Debug, Default, Clone)]
pub struct NoHydration;

impl NoHydration {
    /// Create a new policy.
    pub fn new() -> Self {
        Self
    }
}

impl HydrationPolicy for NoHydration {
    fn hydrate(&self, _request: &HydrationRequest<'_>) -> Option<CacheEntry> {
        None
    }
}
