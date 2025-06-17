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
    } else if let ScalarValue::Utf8View(Some(v)) = value {
        Some(v)
    } else if let ScalarValue::LargeUtf8(Some(v)) = value {
        Some(v)
    } else if let ScalarValue::Dictionary(_, value) = value {
        get_string_needle(value.as_ref())
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

#[cfg(test)]
mod tests {
    use super::*;
    use datafusion::common::ScalarValue;
    use datafusion::logical_expr::Operator;
    use datafusion::physical_expr::expressions::{BinaryExpr, Literal};
    use datafusion::physical_plan::expressions::Column;

    #[test]
    fn test_extract_multi_column_or_valid_three_columns() {
        // Test case: a = 1 OR b = 2 OR c = 3
        // This should extract 3 column-literal pairs

        let expr_a: Arc<dyn PhysicalExpr> = Arc::new(BinaryExpr::new(
            Arc::new(Column::new("a", 0)),
            Operator::Eq,
            Arc::new(Literal::new(ScalarValue::Int32(Some(1)))),
        ));

        let expr_b: Arc<dyn PhysicalExpr> = Arc::new(BinaryExpr::new(
            Arc::new(Column::new("b", 1)),
            Operator::Eq,
            Arc::new(Literal::new(ScalarValue::Int32(Some(2)))),
        ));

        let expr_c: Arc<dyn PhysicalExpr> = Arc::new(BinaryExpr::new(
            Arc::new(Column::new("c", 2)),
            Operator::Eq,
            Arc::new(Literal::new(ScalarValue::Int32(Some(3)))),
        ));

        // Build nested OR: (a = 1 OR b = 2) OR c = 3
        let expr_ab = Arc::new(BinaryExpr::new(expr_a, Operator::Or, expr_b));
        let expr_final: Arc<dyn PhysicalExpr> =
            Arc::new(BinaryExpr::new(expr_ab, Operator::Or, expr_c));

        let result = extract_multi_column_or(&expr_final);

        assert!(result.is_some());
        let column_exprs = result.unwrap();
        assert_eq!(column_exprs.len(), 3);

        // Verify we got the correct column indices
        let mut column_indices: Vec<usize> = column_exprs.iter().map(|(idx, _)| *idx).collect();
        column_indices.sort();
        assert_eq!(column_indices, vec![0, 1, 2]);
    }

    #[test]
    fn test_extract_multi_column_or_invalid_expression() {
        // Test case: a + b = 5 (not a column-literal OR expression)
        // This should return None

        let add_expr: Arc<dyn PhysicalExpr> = Arc::new(BinaryExpr::new(
            Arc::new(Column::new("a", 0)),
            Operator::Plus,
            Arc::new(Column::new("b", 1)),
        ));

        let expr: Arc<dyn PhysicalExpr> = Arc::new(BinaryExpr::new(
            add_expr,
            Operator::Eq,
            Arc::new(Literal::new(ScalarValue::Int32(Some(5)))),
        ));

        let result = extract_multi_column_or(&expr);
        assert!(result.is_none());

        // Test case: Single column expression (a = 1)
        // This should return None because we need >= 2 columns
        let single_expr: Arc<dyn PhysicalExpr> = Arc::new(BinaryExpr::new(
            Arc::new(Column::new("a", 0)),
            Operator::Eq,
            Arc::new(Literal::new(ScalarValue::Int32(Some(1)))),
        ));

        let result = extract_multi_column_or(&single_expr);
        assert!(result.is_none());

        // Test case: Mixed valid and invalid OR (a = 1 OR (b + c))
        // This should return None because one branch is not column-literal
        let valid_expr: Arc<dyn PhysicalExpr> = Arc::new(BinaryExpr::new(
            Arc::new(Column::new("a", 0)),
            Operator::Eq,
            Arc::new(Literal::new(ScalarValue::Int32(Some(1)))),
        ));

        let invalid_expr: Arc<dyn PhysicalExpr> = Arc::new(BinaryExpr::new(
            Arc::new(Column::new("b", 1)),
            Operator::Plus,
            Arc::new(Column::new("c", 2)),
        ));

        let mixed_expr: Arc<dyn PhysicalExpr> =
            Arc::new(BinaryExpr::new(valid_expr, Operator::Or, invalid_expr));

        let result = extract_multi_column_or(&mixed_expr);
        assert!(result.is_none());
    }

    #[test]
    fn test_try_evaluate_predicate_string_equality() {
        use crate::liquid_array::LiquidByteArray;
        use arrow::array::StringViewArray;

        // Test the optimized string equality path
        let string_data = vec!["apple", "banana", "cherry", "apple", "grape"];
        let arrow_array = StringViewArray::from(string_data);
        let (_compressor, liquid_array) = LiquidByteArray::train_from_arrow_view(&arrow_array);
        let liquid_ref: LiquidArrayRef = Arc::new(liquid_array);

        // Create predicate: column = "apple"
        let expr: Arc<dyn PhysicalExpr> = Arc::new(BinaryExpr::new(
            Arc::new(Column::new("test_col", 0)),
            Operator::Eq,
            Arc::new(Literal::new(ScalarValue::Utf8(Some("apple".to_string())))),
        ));

        let result = try_evaluate_predicate(&expr, &liquid_ref).unwrap();
        assert!(result.is_some());

        let boolean_array = result.unwrap();
        assert_eq!(boolean_array.len(), 5);

        // Should match indices 0 and 3 (both "apple")
        assert_eq!(boolean_array.value(0), true); // "apple"
        assert_eq!(boolean_array.value(1), false); // "banana"
        assert_eq!(boolean_array.value(2), false); // "cherry"
        assert_eq!(boolean_array.value(3), true); // "apple"
        assert_eq!(boolean_array.value(4), false); // "grape"
    }

    #[test]
    fn test_try_evaluate_predicate_numeric_not_supported() {
        use crate::liquid_array::LiquidPrimitiveArray;
        use arrow::array::Int32Array;
        use arrow::datatypes::Int32Type;

        // Test that numeric comparisons are not supported and return None
        let numeric_data = vec![10, 20, 30, 15, 25];
        let arrow_array = Int32Array::from(numeric_data);
        let liquid_array = LiquidPrimitiveArray::<Int32Type>::from_arrow_array(arrow_array);
        let liquid_ref: LiquidArrayRef = Arc::new(liquid_array);

        // Create predicate: column > 20
        let expr: Arc<dyn PhysicalExpr> = Arc::new(BinaryExpr::new(
            Arc::new(Column::new("test_col", 0)),
            Operator::Gt,
            Arc::new(Literal::new(ScalarValue::Int32(Some(20)))),
        ));

        let result = try_evaluate_predicate(&expr, &liquid_ref).unwrap();
        // Numeric comparisons are not supported, should return None
        assert!(result.is_none());

        // Test other numeric operators that are also not supported
        let eq_expr: Arc<dyn PhysicalExpr> = Arc::new(BinaryExpr::new(
            Arc::new(Column::new("test_col", 0)),
            Operator::Eq,
            Arc::new(Literal::new(ScalarValue::Int32(Some(20)))),
        ));

        let result = try_evaluate_predicate(&eq_expr, &liquid_ref).unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn test_try_evaluate_predicate_unsupported_expression() {
        use crate::liquid_array::LiquidPrimitiveArray;
        use arrow::array::Int32Array;
        use arrow::datatypes::Int32Type;

        // Test unsupported expression types that should return None
        let numeric_data = vec![10, 20, 30, 15, 25];
        let arrow_array = Int32Array::from(numeric_data);
        let liquid_array = LiquidPrimitiveArray::<Int32Type>::from_arrow_array(arrow_array);
        let liquid_ref: LiquidArrayRef = Arc::new(liquid_array);

        // Create unsupported predicate: column + 5 (not column op literal)
        let add_expr: Arc<dyn PhysicalExpr> = Arc::new(BinaryExpr::new(
            Arc::new(Column::new("test_col", 0)),
            Operator::Plus,
            Arc::new(Literal::new(ScalarValue::Int32(Some(5)))),
        ));

        let result = try_evaluate_predicate(&add_expr, &liquid_ref).unwrap();
        assert!(result.is_none());

        // Test another unsupported case: literal op column (wrong order)
        let wrong_order_expr: Arc<dyn PhysicalExpr> = Arc::new(BinaryExpr::new(
            Arc::new(Literal::new(ScalarValue::Int32(Some(20)))),
            Operator::Eq,
            Arc::new(Column::new("test_col", 0)),
        ));

        let result = try_evaluate_predicate(&wrong_order_expr, &liquid_ref).unwrap();
        assert!(result.is_none());

        // Test column-column comparison (not column-literal)
        let col_col_expr: Arc<dyn PhysicalExpr> = Arc::new(BinaryExpr::new(
            Arc::new(Column::new("col1", 0)),
            Operator::Eq,
            Arc::new(Column::new("col2", 1)),
        ));

        let result = try_evaluate_predicate(&col_col_expr, &liquid_ref).unwrap();
        assert!(result.is_none());
    }
}
