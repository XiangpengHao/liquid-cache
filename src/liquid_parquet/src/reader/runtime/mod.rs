use crate::reader::LiquidPredicate;
use in_memory_rg::InMemoryRowGroup;
pub(crate) use liquid_predicate::{extract_two_column_or, try_evaluate_predicate};
pub(crate) use parquet_bridge::ArrowReaderBuilderBridge;
pub(crate) use parquet_bridge::get_predicate_column_id;

mod in_memory_rg;
mod liquid_predicate;
mod liquid_stream;
mod parquet_bridge;
mod reader;
mod utils;

pub struct LiquidRowFilter {
    pub(crate) predicates: Vec<Box<LiquidPredicate>>,
}

impl LiquidRowFilter {
    pub fn new(predicates: Vec<Box<LiquidPredicate>>) -> Self {
        Self { predicates }
    }
}
