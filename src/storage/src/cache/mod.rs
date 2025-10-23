//! Cache layer for liquid cache.

mod budget;
mod cached_batch;
mod core;
mod index;
#[cfg(target_os = "linux")]
mod io;
mod stats;
#[cfg(not(target_os = "linux"))]
mod io {}
pub mod cache_policies;
pub mod squeeze_policies;
mod tracer;
mod transcode;
mod utils;

pub use cache_policies::CachePolicy;
pub use cached_batch::{CachedBatch, CachedBatchType, GetWithPredicateResult};
pub use core::{CacheStorage, CacheStorageBuilder};
pub use core::{DefaultIoContext, IoContext};
pub use stats::{CacheStats, RuntimeStats, RuntimeStatsSnapshot};
pub use transcode::transcode_liquid_inner;
pub use utils::{EntryID, LiquidCompressorStates};
