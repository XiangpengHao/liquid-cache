use std::sync::Arc;

use arrow::array::ArrayRef;
use arrow::array::{AsArray, BooleanArray};
use arrow::record_batch::RecordBatch;
use arrow_schema::{Field, Schema};
use datafusion::physical_plan::PhysicalExpr;

/// A trait for predicate expressions that can be evaluated on liquid arrays.
///
/// This decouples the core predicate API from DataFusion's `PhysicalExpr` while
/// still allowing fast-path downcasting for encoded-data evaluation.
pub trait LiquidExpr: std::fmt::Debug + Send + Sync {
    /// Access the underlying PhysicalExpr for fast-path downcasting.
    fn as_physical_expr(&self) -> &Arc<dyn PhysicalExpr>;

    /// Evaluate on a materialized Arrow array (fallback path).
    fn evaluate_arrow(&self, array: &ArrayRef) -> BooleanArray;
}

/// Default implementation of [`LiquidExpr`] that wraps a `PhysicalExpr` and a field.
#[derive(Debug)]
pub struct DefaultLiquidExpr {
    expr: Arc<dyn PhysicalExpr>,
    field: Arc<Field>,
}

impl DefaultLiquidExpr {
    /// Create a new `DefaultLiquidExpr`.
    pub fn new(expr: Arc<dyn PhysicalExpr>, field: Arc<Field>) -> Self {
        Self { expr, field }
    }
}

impl LiquidExpr for DefaultLiquidExpr {
    fn as_physical_expr(&self) -> &Arc<dyn PhysicalExpr> {
        &self.expr
    }

    fn evaluate_arrow(&self, array: &ArrayRef) -> BooleanArray {
        let schema = Arc::new(Schema::new(vec![(*self.field).clone()]));
        let batch = RecordBatch::try_new(schema, vec![array.clone()]).unwrap();
        let result = self.expr.evaluate(&batch).unwrap();
        result
            .into_array(batch.num_rows())
            .unwrap()
            .as_boolean()
            .clone()
    }
}
