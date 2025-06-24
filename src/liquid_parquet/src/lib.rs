#![warn(missing_docs)]
#![doc = include_str!("../README.md")]

mod cache;
pub mod liquid_array;
mod reader;
mod sync;
pub use cache::policies;
pub use cache::lib;
pub use cache::{LiquidCache, LiquidCacheRef, LiquidCachedFileRef};
pub use reader::LiquidParquetSource;
pub(crate) mod utils;
pub use utils::{boolean_buffer_and_then, rewrite_data_source_plan};
mod inprocess;
#[cfg(test)]
mod tests;

pub use inprocess::LiquidCacheInProcessBuilder;

pub use liquid_cache_common as common;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Ord, PartialOrd)]
#[allow(unused)]
enum AblationStudyMode {
    FullDecoding = 0,
    SelectiveDecoding = 1,
    SelectiveWithLateMaterialization = 2,
    EvaluateOnEncodedData = 3,
    EvaluateOnPartialEncodedData = 4,
}

// This is deliberately made const to avoid the overhead of runtime branching.
const ABLATION_STUDY_MODE: AblationStudyMode = AblationStudyMode::EvaluateOnPartialEncodedData;
