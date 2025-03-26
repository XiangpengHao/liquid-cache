#![warn(missing_docs)]
#![cfg_attr(not(doctest), doc = include_str!(concat!("../", std::env!("CARGO_PKG_README"))))]

mod cache;
pub mod liquid_array;
mod reader;
pub use cache::{LiquidCache, LiquidCacheMode, LiquidCacheRef, LiquidCachedFileRef};
pub use reader::LiquidParquetFileFormat;
pub use reader::LiquidPredicate;
pub(crate) mod utils;

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
