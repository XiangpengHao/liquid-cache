#![warn(missing_docs)]
#![doc = include_str!("../README.md")]

mod io;
pub mod optimizers;
mod reader;
mod sync;
pub(crate) mod utils;
pub use liquid_cache_storage::variant_schema;
mod variant_udf;

pub mod cache;
pub use cache::{LiquidCacheParquet, LiquidCacheParquetRef};
pub use liquid_cache_common as common;
pub use liquid_cache_storage as storage;
pub use reader::{FilterCandidateBuilder, LiquidParquetSource, LiquidPredicate, LiquidRowFilter};
pub use utils::{boolean_buffer_and_then, extract_execution_metrics};
pub use variant_udf::{VariantGetUdf, VariantPretty, VariantToJsonUdf};
