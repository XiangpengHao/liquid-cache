use crate::liquid_array::LiquidArrayRef;
use arrow::array::{BooleanArray, RecordBatch};
use arrow_schema::ArrowError;
use in_memory_rg::InMemoryRowGroup;
use parquet::arrow::arrow_reader::ArrowPredicate;
pub(crate) use parquet_bridge::ArrowReaderBuilderBridge;
use parquet_bridge::get_predicate_column_id;

mod in_memory_rg;
mod liquid_stream;
mod parquet_bridge;
mod reader;
mod utils;

/// A predicate that can be evaluated on a liquid array.
pub trait LiquidPredicate: ArrowPredicate {
    /// Evaluates the predicate on a liquid array.
    fn evaluate_liquid(&mut self, array: &LiquidArrayRef) -> Result<BooleanArray, ArrowError>;
    /// Evaluates the predicate on an arrow record batch.
    fn evaluate_arrow(&mut self, array: RecordBatch) -> Result<BooleanArray, ArrowError> {
        self.evaluate(array)
    }

    /// Returns the column ids of the predicate.
    fn predicate_column_ids(&self) -> Vec<usize> {
        let projection = self.projection();
        get_predicate_column_id(projection)
    }
}

pub struct LiquidRowFilter {
    pub(crate) predicates: Vec<Box<dyn LiquidPredicate>>,
}

impl LiquidRowFilter {
    pub fn new(predicates: Vec<Box<dyn LiquidPredicate>>) -> Self {
        Self { predicates }
    }
}
