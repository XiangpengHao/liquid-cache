//! Low level array primitives.
//! You should not use this module directly.
//! Instead, use `liquid_cache_server` or `liquid_cache_client` to interact with LiquidCache.
pub(super) mod bit_pack_array;
/// FSST dictionary backing used by byte-view arrays.
pub mod fsst_buffer;
pub use bit_pack_array::BitPackedArray;
pub use fsst_buffer::FsstArray;
