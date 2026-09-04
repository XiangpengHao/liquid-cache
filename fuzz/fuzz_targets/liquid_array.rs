#![no_main]

use std::sync::Arc;

use arbitrary::Arbitrary;
use arrow::array::{Array, ArrayRef, StringViewArray};
use arrow::buffer::BooleanBuffer;
use bytes::Bytes;
use datafusion_common::ScalarValue;
use datafusion_expr_common::operator::Operator;
use datafusion_physical_expr::PhysicalExpr;
use datafusion_physical_expr::expressions::{BinaryExpr, Column, LikeExpr, Literal};
use libfuzzer_sys::fuzz_target;
use liquid_cache::cache::LiquidExpr;
use liquid_cache::liquid_array::{LiquidArray, eval_predicate_on_array};

#[derive(Arbitrary, Debug)]
enum FuzzOperator {
    Eq,
    NotEq,
    Lt,
    LtEq,
    Gt,
    GtEq,
    Like,
}

#[derive(Arbitrary, Debug)]
struct Input {
    values: Vec<Option<String>>,
    predicates: Vec<(String, FuzzOperator)>,
}

fuzz_target!(|input: Input| {
    let arrow: ArrayRef = Arc::new(StringViewArray::from_iter(
        input.values.iter().map(|value| value.as_deref()),
    ));
    let liquid = LiquidArray::from_arrow_array(&arrow).unwrap();
    assert_eq!(liquid.to_arrow_array().as_ref(), arrow.as_ref());

    let selection = BooleanBuffer::new_set(arrow.len());
    for (needle, operator) in input.predicates.iter().take(8) {
        let column: Arc<dyn PhysicalExpr> = Arc::new(Column::new("c", 0));
        let needle = match operator {
            FuzzOperator::Like => format!("%{needle}%"),
            _ => needle.clone(),
        };
        let literal: Arc<dyn PhysicalExpr> =
            Arc::new(Literal::new(ScalarValue::Utf8View(Some(needle))));
        let expr: Arc<dyn PhysicalExpr> = match operator {
            FuzzOperator::Like => Arc::new(LikeExpr::new(false, false, column, literal)),
            operator => Arc::new(BinaryExpr::new(column, binary_operator(operator), literal)),
        };
        let predicate = LiquidExpr::try_new(expr, arrow.data_type()).unwrap();
        assert_eq!(
            liquid.try_eval_predicate(&predicate, &selection),
            eval_predicate_on_array(arrow.clone(), &predicate)
        );
    }

    let decoded = LiquidArray::from_bytes(Bytes::from(liquid.to_bytes()));
    assert_eq!(decoded.to_arrow_array().as_ref(), arrow.as_ref());
});

fn binary_operator(operator: &FuzzOperator) -> Operator {
    match operator {
        FuzzOperator::Eq => Operator::Eq,
        FuzzOperator::NotEq => Operator::NotEq,
        FuzzOperator::Lt => Operator::Lt,
        FuzzOperator::LtEq => Operator::LtEq,
        FuzzOperator::Gt => Operator::Gt,
        FuzzOperator::GtEq => Operator::GtEq,
        FuzzOperator::Like => unreachable!(),
    }
}
