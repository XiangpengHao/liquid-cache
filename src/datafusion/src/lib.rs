#![warn(missing_docs)]
#![doc = include_str!("../README.md")]

mod io;
pub mod optimizers;
mod reader;
mod sync;
pub(crate) mod utils;

pub mod cache;
pub use cache::{LiquidCacheParquet, LiquidCacheParquetRef};
pub use liquid_cache as storage;
pub use liquid_cache_common as common;
pub use reader::variant_udf::{VariantGetUdf, VariantPretty, VariantToJsonUdf};
pub use reader::{FilterCandidateBuilder, LiquidParquetSource, LiquidPredicate, LiquidRowFilter};
pub use reader::plantime::engagement_policy::{
    AlwaysEngagePolicy, CacheEngagementPolicy, DEFAULT_SELECTIVITY_THRESHOLD,
    EngagementContext, EngagementDecision, NeverEngagePolicy, SelectivityThresholdPolicy,
    default_engagement_policy,
};
pub use reader::plantime::source::pre_seed_metadata_cache;
pub use utils::{boolean_buffer_and_then, extract_execution_metrics};
