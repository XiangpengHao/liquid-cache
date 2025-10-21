//! Cache layer for liquid cache.

mod budget;
pub mod cached_data;
mod core;
mod index;
#[cfg(target_os = "linux")]
mod io;
mod stats;
#[cfg(not(target_os = "linux"))]
mod io {}
pub mod cache_policies;
#[cfg(target_os = "linux")]
pub mod io_backend;
pub mod io_state;
pub mod squeeze_policies;
mod tracer;
mod transcode;
mod utils;

pub use cache_policies::CachePolicy;
pub use core::{CacheStorage, CacheStorageBuilder};
pub use core::{DefaultIoContext, IoContext};
pub use stats::{CacheStats, RuntimeStats, RuntimeStatsSnapshot};
pub use transcode::transcode_liquid_inner;
pub use utils::{EntryID, LiquidCompressorStates};
