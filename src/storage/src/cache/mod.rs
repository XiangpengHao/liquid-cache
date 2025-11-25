//! Cache layer for liquid cache.

mod budget;
pub mod cache_policies;
mod cached_batch;
mod core;
mod expressions;
pub mod hydration_policies;
mod index;
pub mod squeeze_policies;
mod stats;
mod tracer;
mod transcode;
mod utils;

pub use cache_policies::CachePolicy;
pub use cached_batch::{CacheEntry, CachedBatchType};
pub use core::{
    BlockingIoContext, DefaultIoContext, EvaluatePredicate, Get, Insert, IoContext, LiquidCache,
    LiquidCacheBuilder,
};
pub use expressions::{CacheExpression, ColumnID, ExpressionRegistry, VariantRequest};
pub use hydration_policies::{AlwaysHydrate, HydrationPolicy, HydrationRequest, MaterializedEntry};
pub use stats::{CacheStats, RuntimeStats, RuntimeStatsSnapshot};
pub use transcode::transcode_liquid_inner;
pub use utils::{EntryID, LiquidCompressorStates};
