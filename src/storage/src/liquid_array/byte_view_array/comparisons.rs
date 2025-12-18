use arrow::array::DictionaryArray;
use arrow::array::{BinaryArray, BooleanArray, BooleanBuilder, cast::AsArray};
use arrow::datatypes::UInt16Type;
use datafusion::logical_expr::{ColumnarValue, Operator};
use datafusion::physical_expr_common::datum::apply_cmp;
use std::sync::Arc;
use std::vec;

use super::LiquidByteViewArray;
use crate::liquid_array::raw::FsstArray;
use crate::liquid_array::raw::fsst_buffer::{DiskBuffer, FsstBacking, PrefixKey};

impl LiquidByteViewArray<FsstArray> {
    /// Compare equality with a byte needle
    pub(super) fn compare_equals(&self, needle: &[u8]) -> BooleanArray {
        let (mut dict_results, ambiguous) = self.compare_equals_with_prefix(needle);

        // Resolve ambiguous candidates by selective decompression.
        if !ambiguous.is_empty() {
            let (values_buffer, offsets_buffer) =
                self.fsst_buffer.to_uncompressed_selected(&ambiguous);
            let binary_array =
                unsafe { BinaryArray::new_unchecked(offsets_buffer, values_buffer, None) };

            for (pos, &dict_index) in ambiguous.iter().enumerate() {
                if binary_array.value(pos) == needle {
                    dict_results[dict_index] = true;
                }
            }
        }

        // Map dict-level results to row-level mask.
        let mut builder = BooleanBuilder::with_capacity(self.dictionary_keys.len());
        for &dict_key in self.dictionary_keys.values().iter() {
            let matches = dict_results
                .get(dict_key as usize)
                .copied()
                .unwrap_or(false);
            builder.append_value(matches);
        }

        let mut mask = builder.finish();
        if let Some(nulls) = self.nulls() {
            let (values, _) = mask.into_parts();
            mask = BooleanArray::new(values, Some(nulls.clone()));
        }
        mask
    }

    /// Compare not equals with a byte needle
    fn compare_not_equals(&self, needle: &[u8]) -> BooleanArray {
        let result = self.compare_equals(needle);
        let (values, nulls) = result.into_parts();
        let values = !&values;
        BooleanArray::new(values, nulls)
    }

    /// Compare with prefix optimization and fallback to Arrow operations
    pub fn compare_with(&self, needle: &[u8], op: &Operator) -> BooleanArray {
        match op {
            // Handle equality operations with existing optimized methods
            Operator::Eq => self.compare_equals(needle),
            Operator::NotEq => self.compare_not_equals(needle),

            // Handle ordering operations with prefix optimization
            Operator::Lt | Operator::LtEq | Operator::Gt | Operator::GtEq => {
                self.compare_with_inner(needle, op)
            }

            // For other operations, fall back to Arrow operations
            _ => self.compare_with_arrow_fallback(needle, op),
        }
    }

    /// Prefix optimization for ordering operations
    pub(super) fn compare_with_inner(&self, needle: &[u8], op: &Operator) -> BooleanArray {
        let (mut dict_results, ambiguous) = self.compare_with_prefix(needle, op);

        // For values needing full comparison, load buffer and decompress
        if !ambiguous.is_empty() {
            let (values_buffer, offsets_buffer) =
                self.fsst_buffer.to_uncompressed_selected(&ambiguous);
            let binary_array =
                unsafe { BinaryArray::new_unchecked(offsets_buffer, values_buffer, None) };

            for (pos, &dict_index) in ambiguous.iter().enumerate() {
                let value_cmp = binary_array.value(pos).cmp(needle);
                let result = match (op, value_cmp) {
                    (Operator::Lt, std::cmp::Ordering::Less) => true,
                    (Operator::Lt, _) => false,
                    (Operator::LtEq, std::cmp::Ordering::Less | std::cmp::Ordering::Equal) => true,
                    (Operator::LtEq, _) => false,
                    (Operator::Gt, std::cmp::Ordering::Greater) => true,
                    (Operator::Gt, _) => false,
                    (Operator::GtEq, std::cmp::Ordering::Greater | std::cmp::Ordering::Equal) => {
                        true
                    }
                    (Operator::GtEq, _) => false,
                    _ => unreachable!(),
                };
                dict_results[dict_index] = result;
            }
        }

        // Map dictionary results to array results
        let mut result = BooleanArray::from(dict_results);
        // Preserve nulls from dictionary keys
        if let Some(nulls) = self.nulls() {
            let (values, _) = result.into_parts();
            result = BooleanArray::new(values, Some(nulls.clone()));
        }

        result
    }

    /// Fallback to Arrow operations for unsupported operations
    fn compare_with_arrow_fallback(&self, needle: &[u8], op: &Operator) -> BooleanArray {
        let dict_array = self.to_dict_arrow();
        compare_with_arrow_inner(dict_array, needle, op)
    }
}

impl LiquidByteViewArray<DiskBuffer> {
    pub(crate) async fn compare_with(&self, needle: &[u8], op: &Operator) -> BooleanArray {
        match op {
            // Handle equality operations with existing optimized methods
            Operator::Eq => self.compare_equals(needle).await,
            Operator::NotEq => self.compare_not_equals(needle).await,

            // Handle ordering operations with prefix optimization
            Operator::Lt | Operator::LtEq | Operator::Gt | Operator::GtEq => {
                self.compare_with_inner(needle, op).await
            }

            // For other operations, fall back to Arrow operations
            _ => self.compare_with_arrow_fallback(needle, op).await,
        }
    }

    /// Compare not equals with a byte needle
    async fn compare_not_equals(&self, needle: &[u8]) -> BooleanArray {
        let result = self.compare_equals(needle).await;
        let (values, nulls) = result.into_parts();
        let values = !&values;
        BooleanArray::new(values, nulls)
    }

    /// Compare equality with a byte needle
    pub(super) async fn compare_equals(&self, needle: &[u8]) -> BooleanArray {
        let (mut dict_results, ambiguous) = self.compare_equals_with_prefix(needle);
        // Resolve ambiguous candidates by selective decompression.
        if !ambiguous.is_empty() {
            let (values_buffer, offsets_buffer) =
                self.fsst_buffer.to_uncompressed_selected(&ambiguous).await;
            let binary_array =
                unsafe { BinaryArray::new_unchecked(offsets_buffer, values_buffer, None) };

            for (pos, &dict_index) in ambiguous.iter().enumerate() {
                if binary_array.value(pos) == needle {
                    dict_results[dict_index] = true;
                }
            }
        }

        // Map dict-level results to row-level mask.
        let mut builder = BooleanBuilder::with_capacity(self.dictionary_keys.len());
        for &dict_key in self.dictionary_keys.values().iter() {
            let matches = dict_results
                .get(dict_key as usize)
                .copied()
                .unwrap_or(false);
            builder.append_value(matches);
        }

        let mut mask = builder.finish();
        if let Some(nulls) = self.nulls() {
            let (values, _) = mask.into_parts();
            mask = BooleanArray::new(values, Some(nulls.clone()));
        }
        mask
    }

    /// Prefix optimization for ordering operations
    pub(super) async fn compare_with_inner(&self, needle: &[u8], op: &Operator) -> BooleanArray {
        let (mut dict_results, ambiguous) = self.compare_with_prefix(needle, op);

        // For values needing full comparison, load buffer and decompress
        if !ambiguous.is_empty() {
            let (values_buffer, offsets_buffer) =
                self.fsst_buffer.to_uncompressed_selected(&ambiguous).await;
            let binary_array =
                unsafe { BinaryArray::new_unchecked(offsets_buffer, values_buffer, None) };

            for (pos, &dict_index) in ambiguous.iter().enumerate() {
                let value_cmp = binary_array.value(pos).cmp(needle);
                let result = match (op, value_cmp) {
                    (Operator::Lt, std::cmp::Ordering::Less) => true,
                    (Operator::Lt, _) => false,
                    (Operator::LtEq, std::cmp::Ordering::Less | std::cmp::Ordering::Equal) => true,
                    (Operator::LtEq, _) => false,
                    (Operator::Gt, std::cmp::Ordering::Greater) => true,
                    (Operator::Gt, _) => false,
                    (Operator::GtEq, std::cmp::Ordering::Greater | std::cmp::Ordering::Equal) => {
                        true
                    }
                    (Operator::GtEq, _) => false,
                    _ => unreachable!(),
                };
                dict_results[dict_index] = result;
            }
        }

        // Map dictionary results to array results
        let mut result = BooleanArray::from(dict_results);

        // Preserve nulls from dictionary keys
        if let Some(nulls) = self.nulls() {
            let (values, _) = result.into_parts();
            result = BooleanArray::new(values, Some(nulls.clone()));
        }

        result
    }

    /// Fallback to Arrow operations for unsupported operations
    async fn compare_with_arrow_fallback(&self, needle: &[u8], op: &Operator) -> BooleanArray {
        let dict_array = self.to_dict_arrow().await;
        compare_with_arrow_inner(dict_array, needle, op)
    }
}

impl<B: FsstBacking> LiquidByteViewArray<B> {
    // returns a tuple of compare_results and ambiguous indices
    fn compare_with_prefix(&self, needle: &[u8], op: &Operator) -> (Vec<bool>, Vec<usize>) {
        // Try to short-circuit based on shared prefix comparison
        if let Some(result) = self.compare_with_shared_prefix(needle, op) {
            return (vec![result; self.dictionary_keys.len()], Vec::new());
        }

        let needle_suffix = &needle[self.shared_prefix.len()..];
        let num_unique = self.prefix_keys.len().saturating_sub(1);
        let mut dict_results = vec![false; num_unique];
        let mut ambiguous = Vec::new();

        // Try prefix comparison for each unique value
        for i in 0..num_unique {
            let prefix7 = self.prefix_keys[i].prefix7();

            // Compare prefix with needle_suffix
            let cmp_len = std::cmp::min(PrefixKey::prefix_len(), needle_suffix.len());
            let prefix_slice = &prefix7[..cmp_len];
            let needle_slice = &needle_suffix[..cmp_len];

            match prefix_slice.cmp(needle_slice) {
                std::cmp::Ordering::Less => {
                    // Prefix < needle, so full string < needle
                    match op {
                        Operator::Lt | Operator::LtEq => {
                            dict_results[i] = true;
                        }
                        Operator::Gt | Operator::GtEq => {
                            dict_results[i] = false;
                        }
                        _ => {
                            ambiguous.push(i);
                        }
                    };
                }
                std::cmp::Ordering::Greater => {
                    // Prefix > needle, so full string > needle
                    match op {
                        Operator::Lt | Operator::LtEq => {
                            dict_results[i] = false;
                        }
                        Operator::Gt | Operator::GtEq => {
                            dict_results[i] = true;
                        }
                        _ => {
                            ambiguous.push(i);
                        }
                    };
                }
                std::cmp::Ordering::Equal => {
                    ambiguous.push(i);
                }
            }
        }
        (dict_results, ambiguous)
    }

    // returns a tuple of compare_results and ambiguous indices
    fn compare_equals_with_prefix(&self, needle: &[u8]) -> (Vec<bool>, Vec<usize>) {
        let shared_prefix_len = self.shared_prefix.len();
        if needle.len() < shared_prefix_len || needle[..shared_prefix_len] != self.shared_prefix {
            return (vec![false; self.dictionary_keys.len()], Vec::new());
        }

        let needle_suffix = &needle[shared_prefix_len..];
        let needle_len = needle_suffix.len();
        let prefix_len = PrefixKey::prefix_len();

        let num_unique = self.prefix_keys.len().saturating_sub(1);
        let mut dict_results = vec![false; num_unique];
        let mut ambiguous = Vec::new();

        for (i, prefix_key) in self.prefix_keys.iter().enumerate().take(num_unique) {
            let known_len = if prefix_key.len_byte() == 255 {
                None
            } else {
                Some(prefix_key.len_byte() as usize)
            };

            // 1) Length gate
            match known_len {
                Some(l) => {
                    if l != needle_len {
                        continue;
                    }
                }
                None => {
                    if needle_len < 255 {
                        continue;
                    }
                }
            }

            // 2) Prefix classification
            match known_len {
                None => {
                    // Long strings: prefix match => need full comparison.
                    if prefix_key.prefix7()[..prefix_len] == needle_suffix[..prefix_len] {
                        ambiguous.push(i);
                    }
                }
                Some(l) if l <= prefix_len => {
                    // Small strings: exact compare on the known length.
                    if prefix_key.prefix7()[..l] == needle_suffix[..l] {
                        dict_results[i] = true;
                    }
                }
                Some(_l) => {
                    // Medium strings: prefix match => need full comparison.
                    if prefix_key.prefix7()[..prefix_len] == needle_suffix[..prefix_len] {
                        ambiguous.push(i);
                    }
                }
            }
        }
        (dict_results, ambiguous)
    }

    /// Check if shared prefix comparison can short-circuit the entire operation
    fn compare_with_shared_prefix(&self, needle: &[u8], op: &Operator) -> Option<bool> {
        let shared_prefix_len = self.shared_prefix.len();

        let needle_shared_len = std::cmp::min(needle.len(), shared_prefix_len);
        let shared_cmp = self.shared_prefix[..needle_shared_len].cmp(&needle[..needle_shared_len]);

        match (op, shared_cmp) {
            (Operator::Lt | Operator::LtEq, std::cmp::Ordering::Less) => Some(true),
            (Operator::Lt | Operator::LtEq, std::cmp::Ordering::Greater) => Some(false),
            (Operator::Gt | Operator::GtEq, std::cmp::Ordering::Greater) => Some(true),
            (Operator::Gt | Operator::GtEq, std::cmp::Ordering::Less) => Some(false),

            // Handle case where compared parts are equal but lengths differ
            (op, std::cmp::Ordering::Equal) => {
                if needle.len() < shared_prefix_len {
                    // All strings start with shared_prefix which is longer than needle
                    // So all strings > needle (for Gt/GtEq) or all strings not < needle (for Lt/LtEq)
                    match op {
                        Operator::Gt | Operator::GtEq => Some(true),
                        Operator::Lt => Some(false),
                        Operator::LtEq => {
                            // Only true if some string equals the needle exactly
                            // Since all strings start with shared_prefix (longer than needle), none can equal needle
                            Some(false)
                        }
                        _ => None,
                    }
                } else if needle.len() > shared_prefix_len {
                    // Needle is longer than shared prefix - can't determine from shared prefix alone
                    None
                } else {
                    // needle.len() == shared_prefix_len
                    // All strings start with exactly needle, so need to check if any string equals needle exactly
                    None
                }
            }

            _ => None,
        }
    }
}

fn compare_with_arrow_inner(
    dict_array: DictionaryArray<UInt16Type>,
    needle: &[u8],
    op: &Operator,
) -> BooleanArray {
    let needle_scalar = datafusion::common::ScalarValue::Binary(Some(needle.to_vec()));
    let lhs = ColumnarValue::Array(Arc::new(dict_array));
    let rhs = ColumnarValue::Scalar(needle_scalar);

    let result = match op {
        Operator::LikeMatch => apply_cmp(Operator::LikeMatch, &lhs, &rhs),
        Operator::ILikeMatch => apply_cmp(Operator::ILikeMatch, &lhs, &rhs),
        Operator::NotLikeMatch => apply_cmp(Operator::NotLikeMatch, &lhs, &rhs),
        Operator::NotILikeMatch => apply_cmp(Operator::NotILikeMatch, &lhs, &rhs),
        _ => {
            unreachable!()
        }
    };

    match result.expect("ArrowError") {
        ColumnarValue::Array(arr) => arr.as_boolean().clone(),
        ColumnarValue::Scalar(_) => unreachable!(),
    }
}
