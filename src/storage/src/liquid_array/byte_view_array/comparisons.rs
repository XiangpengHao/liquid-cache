use arrow::array::{BinaryArray, BooleanArray, BooleanBuilder, cast::AsArray};
use arrow::buffer::BooleanBuffer;
use datafusion::logical_expr::{ColumnarValue, Operator};
use datafusion::physical_expr_common::datum::apply_cmp;
use std::sync::Arc;

use super::LiquidByteViewArray;
use crate::liquid_array::NeedsBacking;
use crate::liquid_array::raw::fsst_buffer::{DiskBuffer, FsstBacking, PrefixKey};

impl<B: FsstBacking> LiquidByteViewArray<B> {
    /// Compare with prefix optimization and fallback to Arrow operations
    pub fn compare_with(&self, needle: &[u8], op: &Operator) -> Result<BooleanArray, NeedsBacking> {
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

    /// Compare equality with a byte needle
    pub(super) fn compare_equals(&self, needle: &[u8]) -> Result<BooleanArray, NeedsBacking> {
        let shared_prefix_len = self.shared_prefix.len();
        if needle.len() < shared_prefix_len || needle[..shared_prefix_len] != self.shared_prefix {
            return Ok(BooleanArray::new(
                BooleanBuffer::new_unset(self.dictionary_keys.len()),
                self.nulls().cloned(),
            ));
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

        // 3) Resolve ambiguous candidates by selective decompression.
        if !ambiguous.is_empty() {
            let (values_buffer, offsets_buffer) =
                self.fsst_buffer.to_uncompressed_selected(&ambiguous)?;
            let binary_array =
                unsafe { BinaryArray::new_unchecked(offsets_buffer, values_buffer, None) };

            for (pos, &dict_index) in ambiguous.iter().enumerate() {
                if binary_array.value(pos) == needle {
                    dict_results[dict_index] = true;
                }
            }
        }

        // 4) Map dict-level results to row-level mask.
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
        Ok(mask)
    }

    /// Compare not equals with a byte needle
    fn compare_not_equals(&self, needle: &[u8]) -> Result<BooleanArray, NeedsBacking> {
        let result = self.compare_equals(needle)?;
        let (values, nulls) = result.into_parts();
        let values = !&values;
        Ok(BooleanArray::new(values, nulls))
    }

    /// Check if shared prefix comparison can short-circuit the entire operation
    pub(super) fn try_shared_prefix_short_circuit(
        &self,
        needle: &[u8],
        op: &Operator,
    ) -> Option<BooleanArray> {
        let shared_prefix_len = self.shared_prefix.len();

        let needle_shared_len = std::cmp::min(needle.len(), shared_prefix_len);
        let shared_cmp = self.shared_prefix[..needle_shared_len].cmp(&needle[..needle_shared_len]);

        let all_true = || {
            let buffer = BooleanBuffer::new_set(self.dictionary_keys.len());
            BooleanArray::new(buffer, self.nulls().cloned())
        };

        let all_false = || {
            let buffer = BooleanBuffer::new_unset(self.dictionary_keys.len());
            BooleanArray::new(buffer, self.nulls().cloned())
        };

        match (op, shared_cmp) {
            (Operator::Lt | Operator::LtEq, std::cmp::Ordering::Less) => Some(all_true()),
            (Operator::Lt | Operator::LtEq, std::cmp::Ordering::Greater) => Some(all_false()),
            (Operator::Gt | Operator::GtEq, std::cmp::Ordering::Greater) => Some(all_true()),
            (Operator::Gt | Operator::GtEq, std::cmp::Ordering::Less) => Some(all_false()),

            // Handle case where compared parts are equal but lengths differ
            (op, std::cmp::Ordering::Equal) => {
                if needle.len() < shared_prefix_len {
                    // All strings start with shared_prefix which is longer than needle
                    // So all strings > needle (for Gt/GtEq) or all strings not < needle (for Lt/LtEq)
                    match op {
                        Operator::Gt | Operator::GtEq => Some(all_true()),
                        Operator::Lt => Some(all_false()),
                        Operator::LtEq => {
                            // Only true if some string equals the needle exactly
                            // Since all strings start with shared_prefix (longer than needle), none can equal needle
                            Some(all_false())
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

            // For all other operators that shouldn't be handled by this function
            _ => None,
        }
    }

    /// Prefix optimization for ordering operations
    pub(super) fn compare_with_inner(
        &self,
        needle: &[u8],
        op: &Operator,
    ) -> Result<BooleanArray, NeedsBacking> {
        // Try to short-circuit based on shared prefix comparison
        if let Some(result) = self.try_shared_prefix_short_circuit(needle, op) {
            return Ok(result);
        }

        let needle_suffix = &needle[self.shared_prefix.len()..];
        let num_unique = self.prefix_keys.len().saturating_sub(1);
        let mut dict_results = Vec::with_capacity(num_unique);
        let mut needs_full_comparison = Vec::new();

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
                    let result = match op {
                        Operator::Lt | Operator::LtEq => Some(true),
                        Operator::Gt | Operator::GtEq => Some(false),
                        _ => None,
                    };
                    dict_results.push(result);
                }
                std::cmp::Ordering::Greater => {
                    // Prefix > needle, so full string > needle
                    let result = match op {
                        Operator::Lt | Operator::LtEq => Some(false),
                        Operator::Gt | Operator::GtEq => Some(true),
                        _ => None,
                    };
                    dict_results.push(result);
                }
                std::cmp::Ordering::Equal => {
                    dict_results.push(None);
                    needs_full_comparison.push(i);
                }
            }
        }

        // For values needing full comparison, load buffer and decompress
        if !needs_full_comparison.is_empty() {
            let (values_buffer, offsets_buffer) = self
                .fsst_buffer
                .to_uncompressed_selected(&needs_full_comparison)?;
            let binary_array =
                unsafe { BinaryArray::new_unchecked(offsets_buffer, values_buffer, None) };

            for (pos, &dict_index) in needs_full_comparison.iter().enumerate() {
                let value_cmp = binary_array.value(pos).cmp(needle);
                let result = match (op, value_cmp) {
                    (Operator::Lt, std::cmp::Ordering::Less) => Some(true),
                    (Operator::Lt, _) => Some(false),
                    (Operator::LtEq, std::cmp::Ordering::Less | std::cmp::Ordering::Equal) => {
                        Some(true)
                    }
                    (Operator::LtEq, _) => Some(false),
                    (Operator::Gt, std::cmp::Ordering::Greater) => Some(true),
                    (Operator::Gt, _) => Some(false),
                    (Operator::GtEq, std::cmp::Ordering::Greater | std::cmp::Ordering::Equal) => {
                        Some(true)
                    }
                    (Operator::GtEq, _) => Some(false),
                    _ => None,
                };
                dict_results[dict_index] = result;
            }
        }

        // Map dictionary results to array results
        let mut builder = BooleanBuilder::with_capacity(self.dictionary_keys.len());
        for &dict_key in self.dictionary_keys.values().iter() {
            let matches = if dict_key as usize >= dict_results.len() {
                false
            } else {
                dict_results[dict_key as usize].unwrap_or(false)
            };
            builder.append_value(matches);
        }

        let mut result = builder.finish();
        // Preserve nulls from dictionary keys
        if let Some(nulls) = self.nulls() {
            let (values, _) = result.into_parts();
            result = BooleanArray::new(values, Some(nulls.clone()));
        }

        Ok(result)
    }

    /// Fallback to Arrow operations for unsupported operations
    fn compare_with_arrow_fallback(
        &self,
        needle: &[u8],
        op: &Operator,
    ) -> Result<BooleanArray, NeedsBacking> {
        let dict_array = self.to_dict_arrow()?;
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
            ColumnarValue::Array(arr) => Ok(arr.as_boolean().clone()),
            ColumnarValue::Scalar(_) => unreachable!(),
        }
    }
}

impl LiquidByteViewArray<DiskBuffer> {
    /// Equality using prefix first; if ambiguous, fall back to full buffer comparison
    pub(super) fn compare_equals_with_prefix(&self, needle: &[u8]) -> Option<BooleanArray> {
        // Quick shared prefix check identical to in-memory path
        let shared_prefix_len = self.shared_prefix.len();
        if needle.len() < shared_prefix_len || needle[..shared_prefix_len] != self.shared_prefix {
            return Some(BooleanArray::new(
                BooleanBuffer::new_unset(self.dictionary_keys.len()),
                self.nulls().cloned(),
            ));
        }

        let needle_suffix = &needle[shared_prefix_len..];
        let needle_len = needle_suffix.len();
        let prefix_len = PrefixKey::prefix_len();

        let num_unique = self.prefix_keys.len().saturating_sub(1);
        let mut dict_results = vec![false; num_unique];

        for (i, result) in dict_results.iter_mut().enumerate().take(num_unique) {
            let known_len = if self.prefix_keys[i].len_byte() == 255 {
                None
            } else {
                Some(self.prefix_keys[i].len_byte() as usize)
            };

            // 1) Length gate
            match known_len {
                Some(l) => {
                    if l != needle_len {
                        continue; // definitively not equal
                    }
                }
                None => {
                    if needle_len < 255 {
                        continue; // definitively not equal
                    }
                }
            }

            // 2) Compare by category
            match known_len {
                None => {
                    // Long strings: need IO if prefix matches
                    if self.prefix_keys[i].prefix7()[..prefix_len] == needle_suffix[..prefix_len] {
                        return None; // ambiguous, requires IO
                    }
                    // else definitively not equal, leave false
                }
                Some(l) if l <= prefix_len => {
                    // Small strings: exact compare on l bytes
                    if self.prefix_keys[i].prefix7()[..l] == needle_suffix[..l] {
                        *result = true; // definitive match
                    }
                }
                Some(_l) => {
                    // Medium strings: prefix compare; equal means ambiguous
                    if self.prefix_keys[i].prefix7()[..prefix_len] == needle_suffix[..prefix_len] {
                        return None; // ambiguous
                    }
                }
            }
        }

        // Map dict-level results to array-level mask
        let mut builder = BooleanBuilder::with_capacity(self.dictionary_keys.len());
        for &dict_key in self.dictionary_keys.values().iter() {
            let matches = dict_results[dict_key as usize];
            builder.append_value(matches);
        }
        let mut mask = builder.finish();
        if let Some(nulls) = self.nulls() {
            let (values, _) = mask.into_parts();
            mask = BooleanArray::new(values, Some(nulls.clone()));
        }
        Some(mask)
    }

    /// Ordering using prefixes only; returns None if ambiguity requires backing.
    pub(super) fn compare_ordering_with_prefix(
        &self,
        needle: &[u8],
        op: &Operator,
    ) -> Option<BooleanArray> {
        if !matches!(
            op,
            Operator::Lt | Operator::LtEq | Operator::Gt | Operator::GtEq
        ) {
            return None;
        }

        // Fast path: shared prefix can decide the whole array.
        if let Some(result) = self.try_shared_prefix_short_circuit(needle, op) {
            return Some(result);
        }

        let needle_suffix = &needle[self.shared_prefix.len()..];
        let num_unique = self.prefix_keys.len().saturating_sub(1);
        let mut dict_results = Vec::with_capacity(num_unique);

        for i in 0..num_unique {
            let prefix7 = self.prefix_keys[i].prefix7();

            let cmp_len = std::cmp::min(PrefixKey::prefix_len(), needle_suffix.len());
            let prefix_slice = &prefix7[..cmp_len];
            let needle_slice = &needle_suffix[..cmp_len];

            let result = match prefix_slice.cmp(needle_slice) {
                std::cmp::Ordering::Less => match op {
                    Operator::Lt | Operator::LtEq => Some(true),
                    Operator::Gt | Operator::GtEq => Some(false),
                    _ => None,
                },
                std::cmp::Ordering::Greater => match op {
                    Operator::Lt | Operator::LtEq => Some(false),
                    Operator::Gt | Operator::GtEq => Some(true),
                    _ => None,
                },
                std::cmp::Ordering::Equal => {
                    // Need full comparison to disambiguate.
                    return None;
                }
            };

            dict_results.push(result);
        }

        // Map dictionary-level decisions to row-level mask.
        let mut builder = BooleanBuilder::with_capacity(self.dictionary_keys.len());
        for &dict_key in self.dictionary_keys.values().iter() {
            let matches = if dict_key as usize >= dict_results.len() {
                false
            } else {
                dict_results[dict_key as usize].unwrap_or(false)
            };
            builder.append_value(matches);
        }

        let mut mask = builder.finish();
        if let Some(nulls) = self.nulls() {
            let (values, _) = mask.into_parts();
            mask = BooleanArray::new(values, Some(nulls.clone()));
        }
        Some(mask)
    }
}
