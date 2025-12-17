use arrow::array::{BooleanArray, cast::AsArray, types::UInt16Type};
use arrow::buffer::BooleanBuffer;
use datafusion::logical_expr::{ColumnarValue, Operator};
use datafusion::physical_expr_common::datum::apply_cmp;
use datafusion::physical_plan::PhysicalExpr;
use datafusion::physical_plan::expressions::{BinaryExpr, LikeExpr, Literal};
use std::sync::Arc;

use super::LiquidByteViewArray;
use crate::liquid_array::NeedsBacking;
use crate::liquid_array::byte_array::get_string_needle;
use crate::liquid_array::raw::fsst_buffer::FsstBacking;

pub(super) fn filter_inner<B: FsstBacking>(
    array: &LiquidByteViewArray<B>,
    filter: &BooleanBuffer,
) -> LiquidByteViewArray<B> {
    // Only filter the dictionary keys, not the offset views!
    // Offset views reference unique values in FSST buffer and should remain unchanged

    // Filter the dictionary keys using Arrow's built-in filter functionality
    let filter = BooleanArray::new(filter.clone(), None);
    let filtered_keys = arrow::compute::filter(&array.dictionary_keys, &filter).unwrap();
    let filtered_keys = filtered_keys.as_primitive::<UInt16Type>().clone();

    LiquidByteViewArray {
        dictionary_keys: filtered_keys,
        prefix_keys: array.prefix_keys.clone(),
        fsst_buffer: array.fsst_buffer.clone(),
        original_arrow_type: array.original_arrow_type,
        shared_prefix: array.shared_prefix.clone(),
    }
}

pub(super) fn try_eval_predicate_inner<B: FsstBacking>(
    expr: &Arc<dyn PhysicalExpr>,
    array: &LiquidByteViewArray<B>,
) -> Result<Option<BooleanArray>, NeedsBacking> {
    // Handle binary expressions (comparisons)
    if let Some(binary_expr) = expr.as_any().downcast_ref::<BinaryExpr>() {
        if let Some(literal) = binary_expr.right().as_any().downcast_ref::<Literal>() {
            let op = binary_expr.op();

            // Try to use string needle optimization first
            if let Some(needle) = get_string_needle(literal.value()) {
                let needle_bytes = needle.as_bytes();
                let result = array.compare_with(needle_bytes, op)?;
                return Ok(Some(result));
            }

            // Fallback to Arrow operations
            let dict_array = array.to_dict_arrow()?;
            let lhs = ColumnarValue::Array(Arc::new(dict_array));
            let rhs = ColumnarValue::Scalar(literal.value().clone());

            let result = match op {
                Operator::NotEq => apply_cmp(Operator::NotEq, &lhs, &rhs),
                Operator::Eq => apply_cmp(Operator::Eq, &lhs, &rhs),
                Operator::Lt => apply_cmp(Operator::Lt, &lhs, &rhs),
                Operator::LtEq => apply_cmp(Operator::LtEq, &lhs, &rhs),
                Operator::Gt => apply_cmp(Operator::Gt, &lhs, &rhs),
                Operator::GtEq => apply_cmp(Operator::GtEq, &lhs, &rhs),
                Operator::LikeMatch => apply_cmp(Operator::LikeMatch, &lhs, &rhs),
                Operator::ILikeMatch => apply_cmp(Operator::ILikeMatch, &lhs, &rhs),
                Operator::NotLikeMatch => apply_cmp(Operator::NotLikeMatch, &lhs, &rhs),
                Operator::NotILikeMatch => apply_cmp(Operator::NotILikeMatch, &lhs, &rhs),
                _ => return Ok(None),
            };
            if let Ok(result) = result {
                let filtered = result.into_array(array.len()).unwrap().as_boolean().clone();
                return Ok(Some(filtered));
            }
        }
    }
    // Handle like expressions
    else if let Some(like_expr) = expr.as_any().downcast_ref::<LikeExpr>()
        && like_expr
            .pattern()
            .as_any()
            .downcast_ref::<Literal>()
            .is_some()
        && let Some(literal) = like_expr.pattern().as_any().downcast_ref::<Literal>()
    {
        let arrow_dict = array.to_dict_arrow()?;

        let lhs = ColumnarValue::Array(Arc::new(arrow_dict));
        let rhs = ColumnarValue::Scalar(literal.value().clone());

        let result = match (like_expr.negated(), like_expr.case_insensitive()) {
            (false, false) => apply_cmp(Operator::LikeMatch, &lhs, &rhs),
            (true, false) => apply_cmp(Operator::NotLikeMatch, &lhs, &rhs),
            (false, true) => apply_cmp(Operator::ILikeMatch, &lhs, &rhs),
            (true, true) => apply_cmp(Operator::NotILikeMatch, &lhs, &rhs),
        };
        if let Ok(result) = result {
            let filtered = result.into_array(array.len()).unwrap().as_boolean().clone();
            return Ok(Some(filtered));
        }
    }
    Ok(None)
}

use std::fmt::Display;

/// Detailed memory usage of the byte view array
pub struct ByteViewArrayMemoryUsage {
    /// Memory usage of the dictionary key
    pub dictionary_key: usize,
    /// Memory usage of the offset views
    pub offsets: usize,
    /// Memory usage of the FSST buffer
    pub fsst_buffer: usize,
    /// Memory usage of the shared prefix
    pub shared_prefix: usize,
    /// Memory usage of the struct size
    pub struct_size: usize,
}

impl Display for ByteViewArrayMemoryUsage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ByteViewArrayMemoryUsage")
            .field("dictionary_key", &self.dictionary_key)
            .field("offsets", &self.offsets)
            .field("fsst_buffer", &self.fsst_buffer)
            .field("shared_prefix", &self.shared_prefix)
            .field("struct_size", &self.struct_size)
            .field("total", &self.total())
            .finish()
    }
}

impl ByteViewArrayMemoryUsage {
    /// Get the total memory usage of the byte view array
    pub fn total(&self) -> usize {
        self.dictionary_key
            + self.offsets
            + self.fsst_buffer
            + self.shared_prefix
            + self.struct_size
    }
}

impl std::ops::AddAssign for ByteViewArrayMemoryUsage {
    fn add_assign(&mut self, other: Self) {
        self.dictionary_key += other.dictionary_key;
        self.offsets += other.offsets;
        self.fsst_buffer += other.fsst_buffer;
        self.shared_prefix += other.shared_prefix;
        self.struct_size += other.struct_size;
    }
}
