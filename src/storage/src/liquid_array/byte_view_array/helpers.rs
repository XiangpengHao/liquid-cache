use arrow::array::{BooleanArray, cast::AsArray, types::UInt16Type};
use arrow::buffer::BooleanBuffer;
use datafusion::physical_plan::PhysicalExpr;
use std::sync::Arc;

use super::LiquidByteViewArray;
use super::operator::{ByteViewExpression, ByteViewOperator};
use crate::liquid_array::raw::FsstArray;
use crate::liquid_array::raw::fsst_buffer::{DiskBuffer, FsstBacking};

pub(super) fn filter_inner<B: FsstBacking>(
    array: &LiquidByteViewArray<B>,
    filter: &BooleanBuffer,
) -> LiquidByteViewArray<B> {
    // Only filter the dictionary keys, not the offsets!
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
        string_fingerprints: array.string_fingerprints.clone(),
    }
}

pub(super) fn try_eval_predicate_in_memory(
    expr: &Arc<dyn PhysicalExpr>,
    array: &LiquidByteViewArray<FsstArray>,
) -> Option<BooleanArray> {
    let expr = ByteViewExpression::try_from(expr).ok()?;
    let op = expr.op();
    let needle = expr.literal();
    Some(array.compare_with(needle, op))
}

pub(super) async fn try_eval_predicate_on_disk(
    expr: &Arc<dyn PhysicalExpr>,
    array: &LiquidByteViewArray<DiskBuffer>,
) -> Option<BooleanArray> {
    let expr = ByteViewExpression::try_from(expr).ok()?;
    let op = expr.op();
    let needle = expr.literal();

    if let ByteViewOperator::SubString(_substring_op) = op
        && array.string_fingerprints.as_ref().is_none()
    {
        return None;
    }
    let result = array.compare_with(needle, op).await;
    Some(result)
}

use std::fmt::Display;

/// Detailed memory usage of the byte view array
pub struct ByteViewArrayMemoryUsage {
    /// Memory usage of the dictionary key
    pub dictionary_key: usize,
    /// Memory usage of the compact offsets
    pub offsets: usize,
    /// Memory usage of the prefix keys
    pub prefix_keys: usize,
    /// Memory usage of the raw FSST buffer
    pub fsst_buffer: usize,
    /// Memory usage of the shared prefix
    pub shared_prefix: usize,
    /// Memory usage of the string fingerprints
    pub string_fingerprints: usize,
    /// Memory usage of the struct size
    pub struct_size: usize,
}

impl Display for ByteViewArrayMemoryUsage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ByteViewArrayMemoryUsage")
            .field("dictionary_key", &self.dictionary_key)
            .field("offsets", &self.offsets)
            .field("prefix_keys", &self.prefix_keys)
            .field("fsst_buffer", &self.fsst_buffer)
            .field("shared_prefix", &self.shared_prefix)
            .field("string_fingerprints", &self.string_fingerprints)
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
            + self.prefix_keys
            + self.fsst_buffer
            + self.shared_prefix
            + self.string_fingerprints
            + self.struct_size
    }
}

impl std::ops::AddAssign for ByteViewArrayMemoryUsage {
    fn add_assign(&mut self, other: Self) {
        self.dictionary_key += other.dictionary_key;
        self.offsets += other.offsets;
        self.prefix_keys += other.prefix_keys;
        self.fsst_buffer += other.fsst_buffer;
        self.shared_prefix += other.shared_prefix;
        self.string_fingerprints += other.string_fingerprints;
        self.struct_size += other.struct_size;
    }
}
