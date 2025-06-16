use std::sync::Arc;

use arrow::array::{AsArray, BooleanArray};
use arrow::compute::kernels;
use arrow_schema::ArrowError;
use datafusion::common::ScalarValue;
use datafusion::logical_expr::{ColumnarValue, Operator};
use datafusion::physical_expr::PhysicalExpr;
use datafusion::physical_expr::expressions::{BinaryExpr, LikeExpr, Literal};
use datafusion::physical_expr_common::datum::apply_cmp;
use datafusion::physical_plan::expressions::Column;

use crate::liquid_array::{AsLiquidArray, LiquidArray, LiquidArrayRef};

fn get_string_needle(value: &ScalarValue) -> Option<&str> {
    if let ScalarValue::Utf8(Some(v)) = value {
        Some(v)
    } else if let ScalarValue::Dictionary(_, value) = value {
        if let ScalarValue::Utf8(Some(v)) = value.as_ref() {
            Some(v)
        } else {
            None
        }
    } else {
        None
    }
}

pub(crate) fn try_evaluate_predicate(
    expr: &Arc<dyn PhysicalExpr>,
    array: &LiquidArrayRef,
) -> Result<Option<BooleanArray>, ArrowError> {
    if let Some(binary_expr) = expr.as_any().downcast_ref::<BinaryExpr>() {
        if let Some(literal) = binary_expr.right().as_any().downcast_ref::<Literal>() {
            let op = binary_expr.op();
            if let Some(array) = array.as_string_array_opt() {
                if let Some(needle) = get_string_needle(literal.value()) {
                    if op == &Operator::Eq {
                        let result = array.compare_equals(needle);
                        return Ok(Some(result));
                    } else if op == &Operator::NotEq {
                        let result = array.compare_not_equals(needle);
                        return Ok(Some(result));
                    }
                }

                let dict_array = array.to_dict_arrow();
                let lhs = ColumnarValue::Array(Arc::new(dict_array));
                let rhs = ColumnarValue::Scalar(literal.value().clone());

                let result = match op {
                    Operator::NotEq => apply_cmp(&lhs, &rhs, kernels::cmp::neq),
                    Operator::Eq => apply_cmp(&lhs, &rhs, kernels::cmp::eq),
                    Operator::Lt => apply_cmp(&lhs, &rhs, kernels::cmp::lt),
                    Operator::LtEq => apply_cmp(&lhs, &rhs, kernels::cmp::lt_eq),
                    Operator::Gt => apply_cmp(&lhs, &rhs, kernels::cmp::gt),
                    Operator::GtEq => apply_cmp(&lhs, &rhs, kernels::cmp::gt_eq),
                    Operator::LikeMatch => apply_cmp(&lhs, &rhs, arrow::compute::like),
                    Operator::ILikeMatch => apply_cmp(&lhs, &rhs, arrow::compute::ilike),
                    Operator::NotLikeMatch => apply_cmp(&lhs, &rhs, arrow::compute::nlike),
                    Operator::NotILikeMatch => apply_cmp(&lhs, &rhs, arrow::compute::nilike),
                    _ => return Ok(None),
                };
                if let Ok(result) = result {
                    let filtered = result.into_array(array.len())?.as_boolean().clone();
                    return Ok(Some(filtered));
                }
            }
        }
    } else if let Some(like_expr) = expr.as_any().downcast_ref::<LikeExpr>()
        && like_expr
            .pattern()
            .as_any()
            .downcast_ref::<Literal>()
            .is_some()
        && let Some(literal) = like_expr.pattern().as_any().downcast_ref::<Literal>()
    {
        let arrow_dict = array.as_string().to_dict_arrow();

        let lhs = ColumnarValue::Array(Arc::new(arrow_dict));
        let rhs = ColumnarValue::Scalar(literal.value().clone());

        let result = match (like_expr.negated(), like_expr.case_insensitive()) {
            (false, false) => apply_cmp(&lhs, &rhs, arrow::compute::like),
            (true, false) => apply_cmp(&lhs, &rhs, arrow::compute::nlike),
            (false, true) => apply_cmp(&lhs, &rhs, arrow::compute::ilike),
            (true, true) => apply_cmp(&lhs, &rhs, arrow::compute::nilike),
        };
        if let Ok(result) = result {
            let filtered = result.into_array(array.len())?.as_boolean().clone();
            return Ok(Some(filtered));
        }
    }
    Ok(None)
}

/// Extract multiple column-literal expressions from a nested OR structure.
/// Returns a vector of (column_index, expression) pairs if all leaf expressions
/// are column-literal expressions connected by OR operators.
pub(crate) fn extract_multi_column_or(
    expr: &Arc<dyn PhysicalExpr>,
) -> Option<Vec<(usize, Arc<dyn PhysicalExpr>)>> {
    let mut result = Vec::new();

    fn collect_or_expressions(
        expr: &Arc<dyn PhysicalExpr>,
        result: &mut Vec<(usize, Arc<dyn PhysicalExpr>)>,
    ) -> bool {
        if let Some(binary) = expr.as_any().downcast_ref::<BinaryExpr>()
            && binary.op() == &Operator::Or
        {
            // Recursively collect from left and right
            return collect_or_expressions(binary.left(), result)
                && collect_or_expressions(binary.right(), result);
        }

        // Try to extract column-literal from this expression
        if let Some(column_literal) = extract_column_literal(expr) {
            result.push(column_literal);
            true
        } else {
            false
        }
    }

    if collect_or_expressions(expr, &mut result) && result.len() >= 2 {
        Some(result)
    } else {
        None
    }
}

fn extract_column_literal(expr: &Arc<dyn PhysicalExpr>) -> Option<(usize, Arc<dyn PhysicalExpr>)> {
    if let Some(binary) = expr.as_any().downcast_ref::<BinaryExpr>() {
        if binary.right().as_any().downcast_ref::<Literal>().is_some()
            && binary.left().as_any().downcast_ref::<Column>().is_some()
        {
            let column = binary.left().as_any().downcast_ref::<Column>().unwrap();
            return Some((column.index(), Arc::clone(expr)));
        }
    } else if let Some(like_expr) = expr.as_any().downcast_ref::<LikeExpr>()
        && like_expr
            .pattern()
            .as_any()
            .downcast_ref::<Literal>()
            .is_some()
        && like_expr.expr().as_any().downcast_ref::<Column>().is_some()
    {
        let column = like_expr.expr().as_any().downcast_ref::<Column>().unwrap();
        return Some((column.index(), Arc::clone(expr)));
    }
    None
}
