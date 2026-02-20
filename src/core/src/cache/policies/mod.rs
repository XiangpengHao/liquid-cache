//! Policy modules for cache eviction, hydration, and squeezing.

pub mod cache;
pub mod hydration;
pub mod squeeze;

pub use cache::*;
pub use hydration::*;
pub use squeeze::*;
