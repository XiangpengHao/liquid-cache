#![warn(missing_docs)]
#![doc = include_str!("../README.md")]

pub mod liquid_array;
mod reader;
mod sync;
pub(crate) mod utils;

mod inprocess;
#[cfg(test)]
mod tests;

pub mod cache;
pub use cache::{LiquidCache, LiquidCacheRef};
pub use inprocess::LiquidCacheInProcessBuilder;
pub use liquid_cache_common as common;
pub use reader::LiquidParquetSource;
pub use utils::{boolean_buffer_and_then, extract_execution_metrics, rewrite_data_source_plan};

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
