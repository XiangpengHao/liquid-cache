//! Cache layer for liquid cache.

mod budget;
mod builders;
mod cached_batch;
mod core;
mod expressions;
mod index;
mod internal_tracing;
mod io_context;
pub mod policies;
mod stats;
mod tracer;
mod transcode;
mod utils;

pub use builders::{EvaluatePredicate, Get, Insert, LiquidCacheBuilder};
pub use cached_batch::{CacheEntry, CachedBatchType};
pub use core::LiquidCache;
pub use expressions::{CacheExpression, VariantRequest};
pub use internal_tracing::EventTrace;
pub use io_context::{DefaultIoContext, IoContext};
pub use policies::{
    AlwaysHydrate, CachePolicy, HydrationPolicy, HydrationRequest, LiquidPolicy, MaterializedEntry,
    NoHydration, SqueezePolicy, TranscodeSqueezeEvict,
};
pub use stats::{CacheStats, RuntimeStats, RuntimeStatsSnapshot};
pub use transcode::transcode_liquid_inner;
pub use utils::{EntryID, LiquidCompressorStates};

// Backwards-compatible module paths for existing imports.
/// Legacy path: re-export cache policy types under `cache::cache_policies`.
pub mod cache_policies {
    pub use super::policies::cache::*;
}

/// Legacy path: re-export hydration policy types under `cache::hydration_policies`.
pub mod hydration_policies {
    pub use super::policies::hydration::*;
}

/// Legacy path: re-export squeeze policy types under `cache::squeeze_policies`.
pub mod squeeze_policies {
    pub use super::policies::squeeze::*;
}

#[cfg(test)]
mod tests;
