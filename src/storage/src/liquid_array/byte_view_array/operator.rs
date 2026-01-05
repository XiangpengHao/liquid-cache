use std::sync::Arc;

use datafusion::{
    logical_expr::Operator,
    physical_plan::{
        PhysicalExpr,
        expressions::{BinaryExpr, DynamicFilterPhysicalExpr, LikeExpr, Literal},
    },
    scalar::ScalarValue,
};

use crate::liquid_array::get_bytes_needle;

/// Supported ordering comparisons for byte views.
#[derive(Debug)]
pub enum Comparison {
    /// Less-than.
    Lt,
    /// Greater-than.
    Gt,
    /// Less-than or equal.
    LtEq,
    /// Greater-than or equal.
    GtEq,
}

/// Supported equality comparisons for byte views.
#[derive(Debug)]
pub enum Equality {
    /// Equal.
    Eq,
    /// Not equal.
    NotEq,
}

#[derive(Debug, PartialEq, Eq, Copy, Clone)]
/// Supported substring predicate kinds.
pub enum SubString {
    /// Contains a substring.
    Contains,
    /// Does not contain a substring.
    NotContains,
}

/// Supported operators for byte view predicates.
#[derive(Debug)]
pub enum ByteViewOperator {
    /// Ordering comparison.
    Comparison(Comparison),
    /// Equality comparison.
    Equality(Equality),
    /// Substring predicate.
    SubString(SubString),
}

impl ByteViewOperator {
    fn from_like_expr(like: &LikeExpr) -> Result<Self, UnsupportedOperator> {
        match (like.negated(), like.case_insensitive()) {
            (false, false) => Ok(ByteViewOperator::SubString(SubString::Contains)),
            (true, false) => Ok(ByteViewOperator::SubString(SubString::NotContains)),
            _ => Err(UnsupportedOperator),
        }
    }
}

pub struct UnsupportedOperator;

impl TryFrom<&Operator> for ByteViewOperator {
    type Error = UnsupportedOperator;

    fn try_from(operator: &Operator) -> Result<Self, UnsupportedOperator> {
        match operator {
            Operator::Eq => Ok(ByteViewOperator::Equality(Equality::Eq)),
            Operator::NotEq => Ok(ByteViewOperator::Equality(Equality::NotEq)),
            Operator::Lt => Ok(ByteViewOperator::Comparison(Comparison::Lt)),
            Operator::Gt => Ok(ByteViewOperator::Comparison(Comparison::Gt)),
            Operator::LtEq => Ok(ByteViewOperator::Comparison(Comparison::LtEq)),
            Operator::GtEq => Ok(ByteViewOperator::Comparison(Comparison::GtEq)),
            Operator::LikeMatch => Ok(ByteViewOperator::SubString(SubString::Contains)),
            Operator::NotLikeMatch => Ok(ByteViewOperator::SubString(SubString::NotContains)),
            _ => Err(UnsupportedOperator),
        }
    }
}

impl From<&ByteViewOperator> for Operator {
    fn from(byte_view_operator: &ByteViewOperator) -> Self {
        match byte_view_operator {
            ByteViewOperator::Comparison(comparison) => match comparison {
                Comparison::Lt => Operator::Lt,
                Comparison::Gt => Operator::Gt,
                Comparison::LtEq => Operator::LtEq,
                Comparison::GtEq => Operator::GtEq,
            },
            ByteViewOperator::Equality(equality) => match equality {
                Equality::Eq => Operator::Eq,
                Equality::NotEq => Operator::NotEq,
            },
            ByteViewOperator::SubString(substring) => match substring {
                SubString::Contains => Operator::LikeMatch,
                SubString::NotContains => Operator::NotLikeMatch,
            },
        }
    }
}

#[derive(Debug)]
pub(super) struct ByteViewExpression {
    op: ByteViewOperator,
    literal: Vec<u8>,
}

pub(super) enum UnsupportedExpression {
    Op,
    Expr,
    // This is frequently the case with dynamic filters
    Constant(bool),
}

impl From<UnsupportedOperator> for UnsupportedExpression {
    fn from(_op: UnsupportedOperator) -> Self {
        UnsupportedExpression::Op
    }
}

impl ByteViewExpression {
    pub(super) fn op(&self) -> &ByteViewOperator {
        &self.op
    }

    pub(super) fn literal(&self) -> &[u8] {
        &self.literal
    }
}

impl TryFrom<&Arc<dyn PhysicalExpr>> for ByteViewExpression {
    type Error = UnsupportedExpression;
    fn try_from(expr: &Arc<dyn PhysicalExpr>) -> Result<Self, UnsupportedExpression> {
        let expr = if let Some(dynamic_filter) =
            expr.as_any().downcast_ref::<DynamicFilterPhysicalExpr>()
        {
            dynamic_filter.current().unwrap()
        } else {
            expr.clone()
        };

        if let Some(literal) = expr.as_any().downcast_ref::<Literal>()
            && let ScalarValue::Boolean(Some(v)) = literal.value()
        {
            return Err(UnsupportedExpression::Constant(*v));
        }

        if let Some(binary_expr) = expr.as_any().downcast_ref::<BinaryExpr>() {
            if let Some(literal) = binary_expr.right().as_any().downcast_ref::<Literal>() {
                let op = binary_expr.op();
                let byte_view_operator = ByteViewOperator::try_from(op)?;
                let literal =
                    get_bytes_needle(literal.value()).ok_or(UnsupportedExpression::Expr)?;
                return Ok(ByteViewExpression {
                    op: byte_view_operator,
                    literal,
                });
            }
        }
        // Handle like expressions
        else if let Some(like_expr) = expr.as_any().downcast_ref::<LikeExpr>()
            && let Some(literal) = like_expr.pattern().as_any().downcast_ref::<Literal>()
        {
            let byte_view_operator = ByteViewOperator::from_like_expr(like_expr)?;
            let literal = get_bytes_needle(literal.value()).ok_or(UnsupportedExpression::Expr)?;
            return Ok(ByteViewExpression {
                op: byte_view_operator,
                literal,
            });
        }
        Err(UnsupportedExpression::Expr)
    }
}
