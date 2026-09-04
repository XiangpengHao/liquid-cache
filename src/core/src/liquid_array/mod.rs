//! Vortex-backed encoded array used by LiquidCache.

mod array;

use std::sync::Arc;

use arrow::{
    array::{ArrayRef, BooleanArray, cast::AsArray},
    record_batch::RecordBatch,
};
use arrow_schema::{Field, Schema};

use crate::cache::LiquidExpr;

pub use array::LiquidArray;

/// A reference to a Liquid array.
pub type LiquidArrayRef = Arc<LiquidArray>;

/// Evaluate a validated Liquid predicate on an Arrow array.
pub fn eval_predicate_on_array(array: ArrayRef, predicate: &LiquidExpr) -> BooleanArray {
    let schema = Arc::new(Schema::new(vec![Field::new(
        "liquid_predicate_col",
        array.data_type().clone(),
        true,
    )]));
    let record_batch = RecordBatch::try_new(schema, vec![array]).expect("predicate input batch");
    let result = predicate
        .physical_expr()
        .evaluate(&record_batch)
        .expect("validated LiquidExpr must evaluate");
    let boolean_array = result
        .into_array(record_batch.num_rows())
        .expect("predicate output must be an array");
    boolean_array.as_boolean().clone()
}
