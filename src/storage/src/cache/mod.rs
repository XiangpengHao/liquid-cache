//! Cache layer for liquid cache.

mod budget;
pub mod cached_data;
mod core;
mod index;
#[cfg(target_os = "linux")]
mod io;
#[cfg(not(target_os = "linux"))]
mod io {}
pub mod policies;
mod tracer;
mod transcode;
mod utils;

pub use core::{CacheStats, CacheStorage, CacheStorageBuilder};
pub use core::{DefaultIoContext, IoContext};
pub use policies::CachePolicy;
pub use transcode::transcode_liquid_inner;
pub use utils::{CacheAdvice, EntryID, LiquidCompressorStates};
