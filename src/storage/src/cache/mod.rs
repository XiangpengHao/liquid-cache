//! Cache layer for liquid cache.

mod budget;
pub mod cache_policies;
mod cached_batch;
mod core;
mod index;
pub mod io_backend;
pub mod squeeze_policies;
mod stats;
mod tracer;
mod transcode;
mod utils;

pub use cache_policies::CachePolicy;
pub use cached_batch::{CachedBatch, CachedBatchType, GetWithPredicateResult};
pub use core::{BlockingIoContext, DefaultIoContext, IoContext};
pub use core::{CacheStorage, CacheStorageBuilder};
pub use stats::{CacheStats, RuntimeStats, RuntimeStatsSnapshot};
pub use transcode::transcode_liquid_inner;
pub use utils::{EntryID, LiquidCompressorStates};
