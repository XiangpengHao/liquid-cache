use arrow::array::{Array, DictionaryArray};
use arrow::array::{BinaryArray, BooleanArray, BooleanBufferBuilder, StringArray, cast::AsArray};
use arrow::datatypes::UInt16Type;
use arrow_schema::DataType;
use datafusion::common::ScalarValue;
use datafusion::logical_expr::{ColumnarValue, Operator};
use datafusion::physical_expr_common::datum::apply_cmp;
use fsst::Compressor;
use std::sync::Arc;
use std::vec;

use super::LiquidByteViewArray;
use super::fingerprint::{StringFingerprint, substring_pattern_bytes};
use crate::liquid_array::byte_view_array::operator::{self, ByteViewOperator};
use crate::liquid_array::raw::FsstArray;
use crate::liquid_array::raw::fsst_buffer::{DiskBuffer, FsstBacking, PrefixKey};

impl LiquidByteViewArray<FsstArray> {
    /// Compare equality with a byte needle
    pub(super) fn compare_equals(&self, needle: &[u8]) -> BooleanArray {
        let shared_prefix_len = self.shared_prefix.len();
        let num_unique = self.prefix_keys.len();
        if needle.len() < shared_prefix_len || needle[..shared_prefix_len] != self.shared_prefix {
            return self.map_dictionary_results_to_array_results(vec![false; num_unique]);
        }

        let needle_suffix = &needle[shared_prefix_len..];
        let needle_len = needle_suffix.len();
        let prefix_len = PrefixKey::prefix_len();
        let mut dict_results = vec![false; num_unique];

        if needle_len <= prefix_len {
            for (i, prefix_key) in self.prefix_keys.iter().enumerate().take(num_unique) {
                let known_len = if prefix_key.len_byte() == 255 {
                    None
                } else {
                    Some(prefix_key.len_byte() as usize)
                };
                if let Some(l) = known_len
                    && l == needle_len
                    && prefix_key.prefix7()[..l] == needle_suffix[..l]
                {
                    dict_results[i] = true;
                }
            }

            return self.map_dictionary_results_to_array_results(dict_results);
        }

        let compressed_needle = compress_needle(self.fsst_buffer.compressor(), needle);

        for (i, prefix_key) in self.prefix_keys.iter().enumerate().take(num_unique) {
            let known_len = if prefix_key.len_byte() == 255 {
                None
            } else {
                Some(prefix_key.len_byte() as usize)
            };

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

            if prefix_key.prefix7()[..prefix_len] == needle_suffix[..prefix_len] {
                let compressed_value = self.fsst_buffer.get_compressed_slice(i);
                if compressed_value == compressed_needle.as_slice() {
                    dict_results[i] = true;
                }
            }
        }

        self.map_dictionary_results_to_array_results(dict_results)
    }

    /// Compare not equals with a byte needle
    fn compare_not_equals(&self, needle: &[u8]) -> BooleanArray {
        let result = self.compare_equals(needle);
        let (values, nulls) = result.into_parts();
        let values = !&values;
        BooleanArray::new(values, nulls)
    }

    /// Compare with prefix optimization and fallback to Arrow operations
    pub fn compare_with(&self, needle: &[u8], op: &ByteViewOperator) -> BooleanArray {
        match op {
            ByteViewOperator::Comparison(cmp) => self.compare_with_inner(needle, cmp),
            ByteViewOperator::Equality(operator::Equality::Eq) => self.compare_equals(needle),
            ByteViewOperator::Equality(operator::Equality::NotEq) => {
                self.compare_not_equals(needle)
            }
            ByteViewOperator::SubString(_) => self.compare_with_arrow_fallback(needle, op),
        }
    }

    /// Prefix optimization for ordering operations
    pub(super) fn compare_with_inner(
        &self,
        needle: &[u8],
        op: &operator::Comparison,
    ) -> BooleanArray {
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
                    (operator::Comparison::Lt, std::cmp::Ordering::Less) => true,
                    (operator::Comparison::Lt, _) => false,
                    (
                        operator::Comparison::LtEq,
                        std::cmp::Ordering::Less | std::cmp::Ordering::Equal,
                    ) => true,
                    (operator::Comparison::LtEq, _) => false,
                    (operator::Comparison::Gt, std::cmp::Ordering::Greater) => true,
                    (operator::Comparison::Gt, _) => false,
                    (
                        operator::Comparison::GtEq,
                        std::cmp::Ordering::Greater | std::cmp::Ordering::Equal,
                    ) => true,
                    (operator::Comparison::GtEq, _) => false,
                };
                dict_results[dict_index] = result;
            }
        }

        self.map_dictionary_results_to_array_results(dict_results)
    }

    /// Fallback to Arrow operations for unsupported operations
    fn compare_with_arrow_fallback(&self, needle: &[u8], op: &ByteViewOperator) -> BooleanArray {
        let dict_array = self.to_dict_arrow();
        compare_with_arrow_inner(dict_array, needle, op)
    }
}

impl LiquidByteViewArray<DiskBuffer> {
    pub(crate) async fn compare_with(&self, needle: &[u8], op: &ByteViewOperator) -> BooleanArray {
        match op {
            ByteViewOperator::Equality(operator::Equality::Eq) => self.compare_equals(needle).await,
            ByteViewOperator::Equality(operator::Equality::NotEq) => {
                self.compare_not_equals(needle).await
            }
            ByteViewOperator::Comparison(op) => self.compare_with_inner(needle, op).await,
            ByteViewOperator::SubString(op) => {
                let pattern = substring_pattern_bytes(needle).expect("Invalid substring pattern");
                let fingerprints = self
                    .string_fingerprints
                    .as_ref()
                    .expect("Fingerprints not initialized");
                self.compare_like_substring(pattern, *op, fingerprints)
                    .await
            }
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
        if !ambiguous.is_empty() {
            let bytes = self
                .fsst_buffer
                .squeeze_io()
                .read(Some(self.fsst_buffer.disk_range()))
                .await
                .expect("read squeezed backing");
            let hydrated = LiquidByteViewArray::<FsstArray>::from_bytes(
                bytes,
                self.fsst_buffer.compressor_arc(),
            );
            let compressed_needle = compress_needle(hydrated.fsst_buffer.compressor(), needle);

            for &dict_index in ambiguous.iter() {
                let compressed_value = hydrated.fsst_buffer.get_compressed_slice(dict_index);
                if compressed_value == compressed_needle.as_slice() {
                    dict_results[dict_index] = true;
                }
            }
        }

        self.map_dictionary_results_to_array_results(dict_results)
    }

    /// Prefix optimization for ordering operations
    pub(super) async fn compare_with_inner(
        &self,
        needle: &[u8],
        op: &operator::Comparison,
    ) -> BooleanArray {
        let (mut dict_results, ambiguous) = self.compare_with_prefix(needle, op);

        // For values needing full comparison, load buffer and decompress
        if !ambiguous.is_empty() {
            let (values_buffer, offsets_buffer) =
                self.fsst_buffer.to_uncompressed_selected(&ambiguous).await;
            let binary_array =
                unsafe { BinaryArray::new_unchecked(offsets_buffer, values_buffer, None) };

            for (pos, &dict_index) in ambiguous.iter().enumerate() {
                let value_cmp = bytes_cmp_short_auto(binary_array.value(pos), needle);
                let result = match (op, value_cmp) {
                    (operator::Comparison::Lt, std::cmp::Ordering::Less) => true,
                    (operator::Comparison::Lt, _) => false,
                    (
                        operator::Comparison::LtEq,
                        std::cmp::Ordering::Less | std::cmp::Ordering::Equal,
                    ) => true,
                    (operator::Comparison::LtEq, _) => false,
                    (operator::Comparison::Gt, std::cmp::Ordering::Greater) => true,
                    (operator::Comparison::Gt, _) => false,
                    (
                        operator::Comparison::GtEq,
                        std::cmp::Ordering::Greater | std::cmp::Ordering::Equal,
                    ) => true,
                    (operator::Comparison::GtEq, _) => false,
                };
                dict_results[dict_index] = result;
            }
        }

        self.map_dictionary_results_to_array_results(dict_results)
    }

    pub(super) async fn compare_like_substring(
        &self,
        needle: &[u8],
        operator: operator::SubString,
        fingerprints: &Arc<[u32]>,
    ) -> BooleanArray {
        let needle_fp = StringFingerprint::from_bytes(needle);

        let mut dict_results = vec![false; fingerprints.len()];
        let mut ambiguous = Vec::new();

        for (index, &bits) in fingerprints.iter().enumerate() {
            if StringFingerprint::from_bits(bits).might_contain(needle_fp) {
                ambiguous.push(index);
            }
        }

        if !ambiguous.is_empty() {
            let (values_buffer, offsets_buffer) =
                self.fsst_buffer.to_uncompressed_selected(&ambiguous).await;
            // Safety: the offsets and values are valid because they are from fsst buffer, which already checked utf-8.
            let values = unsafe { StringArray::new_unchecked(offsets_buffer, values_buffer, None) };
            let pattern = std::str::from_utf8(needle).ok().unwrap();
            let pattern = format!("%{}%", pattern);

            let lhs = ColumnarValue::Array(Arc::new(values));
            let rhs = ColumnarValue::Scalar(ScalarValue::Utf8(Some(pattern)));
            let result = apply_cmp(Operator::LikeMatch, &lhs, &rhs).unwrap();
            let result = result.into_array(ambiguous.len()).unwrap();
            let matches = result.as_boolean();

            for (pos, &dict_index) in ambiguous.iter().enumerate() {
                if !matches.is_null(pos) && matches.value(pos) {
                    dict_results[dict_index] = true;
                }
            }
        }

        if operator == operator::SubString::NotContains {
            for value in &mut dict_results {
                *value = !*value;
            }
        }

        self.map_dictionary_results_to_array_results(dict_results)
    }
}

impl<B: FsstBacking> LiquidByteViewArray<B> {
    fn map_dictionary_results_to_array_results(&self, dict_results: Vec<bool>) -> BooleanArray {
        let len = self.dictionary_keys.len();
        let mut builder = BooleanBufferBuilder::new(len);
        builder.advance(len);
        for (index, &dict_key) in self.dictionary_keys.values().iter().enumerate() {
            let dict_index = dict_key as usize;
            debug_assert!(dict_index < dict_results.len());
            // Safety: dictionary keys are within the dictionary value range.
            if unsafe { *dict_results.get_unchecked(dict_index) } {
                builder.set_bit(index, true);
            }
        }

        let values = builder.finish();
        if let Some(nulls) = self.nulls() {
            BooleanArray::new(values, Some(nulls.clone()))
        } else {
            BooleanArray::new(values, None)
        }
    }

    // returns a tuple of compare_results and ambiguous indices
    #[inline(never)]
    fn compare_with_prefix(
        &self,
        needle: &[u8],
        op: &operator::Comparison,
    ) -> (Vec<bool>, Vec<usize>) {
        // Try to short-circuit based on shared prefix comparison
        if let Some(result) = self.compare_with_shared_prefix(needle, op) {
            return (vec![result; self.dictionary_keys.len()], Vec::new());
        }

        let needle_suffix = &needle[self.shared_prefix.len()..];
        let num_unique = self.prefix_keys.len();
        let mut dict_results = vec![false; num_unique];
        let mut ambiguous = Vec::new();

        let cmp_len = needle_suffix.len().min(PrefixKey::prefix_len());
        if cmp_len == 0 {
            ambiguous.extend(0..num_unique);
            return (dict_results, ambiguous);
        }

        for (i, prefix_key) in self.prefix_keys.iter().enumerate() {
            let ordering = bytes_cmp_short(prefix_key.prefix7(), needle_suffix, cmp_len);
            match ordering {
                std::cmp::Ordering::Less => match op {
                    operator::Comparison::Lt | operator::Comparison::LtEq => {
                        dict_results[i] = true;
                    }
                    operator::Comparison::Gt | operator::Comparison::GtEq => {
                        dict_results[i] = false;
                    }
                },
                std::cmp::Ordering::Greater => match op {
                    operator::Comparison::Lt | operator::Comparison::LtEq => {
                        dict_results[i] = false;
                    }
                    operator::Comparison::Gt | operator::Comparison::GtEq => {
                        dict_results[i] = true;
                    }
                },
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
        let num_unique = self.prefix_keys.len();
        if needle.len() < shared_prefix_len || needle[..shared_prefix_len] != self.shared_prefix {
            return (vec![false; num_unique], Vec::new());
        }

        let needle_suffix = &needle[shared_prefix_len..];
        let needle_len = needle_suffix.len();
        let prefix_len = PrefixKey::prefix_len();

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
    fn compare_with_shared_prefix(&self, needle: &[u8], op: &operator::Comparison) -> Option<bool> {
        let shared_prefix_len = self.shared_prefix.len();

        let needle_shared_len = std::cmp::min(needle.len(), shared_prefix_len);
        let shared_cmp = self.shared_prefix[..needle_shared_len].cmp(&needle[..needle_shared_len]);
        match (op, shared_cmp) {
            (operator::Comparison::Lt | operator::Comparison::LtEq, std::cmp::Ordering::Less) => {
                Some(true)
            }
            (
                operator::Comparison::Lt | operator::Comparison::LtEq,
                std::cmp::Ordering::Greater,
            ) => Some(false),
            (
                operator::Comparison::Gt | operator::Comparison::GtEq,
                std::cmp::Ordering::Greater,
            ) => Some(true),
            (operator::Comparison::Gt | operator::Comparison::GtEq, std::cmp::Ordering::Less) => {
                Some(false)
            }
            (_, std::cmp::Ordering::Equal) => {
                if needle.len() < shared_prefix_len {
                    match op {
                        operator::Comparison::Gt | operator::Comparison::GtEq => Some(true),
                        operator::Comparison::Lt => Some(false),
                        operator::Comparison::LtEq => Some(false),
                    }
                } else {
                    None
                }
            }
        }
    }
}

fn compare_with_arrow_inner(
    dict_array: DictionaryArray<UInt16Type>,
    needle: &[u8],
    op: &ByteViewOperator,
) -> BooleanArray {
    let needle_scalar = match dict_array.values().data_type() {
        DataType::Utf8 => ScalarValue::Utf8(Some(
            std::str::from_utf8(needle)
                .expect("utf8 needle")
                .to_string(),
        )),
        DataType::Utf8View => ScalarValue::Utf8View(Some(
            std::str::from_utf8(needle)
                .expect("utf8 needle")
                .to_string(),
        )),
        DataType::LargeUtf8 => ScalarValue::LargeUtf8(Some(
            std::str::from_utf8(needle)
                .expect("utf8 needle")
                .to_string(),
        )),
        DataType::Binary => ScalarValue::Binary(Some(needle.to_vec())),
        DataType::BinaryView => ScalarValue::BinaryView(Some(needle.to_vec())),
        DataType::LargeBinary => ScalarValue::LargeBinary(Some(needle.to_vec())),
        _ => ScalarValue::Binary(Some(needle.to_vec())),
    };
    let lhs = ColumnarValue::Array(Arc::new(dict_array));
    let rhs = ColumnarValue::Scalar(needle_scalar);
    let op = Operator::from(op);
    let result = apply_cmp(op, &lhs, &rhs);

    match result.expect("ArrowError") {
        ColumnarValue::Array(arr) => arr.as_boolean().clone(),
        ColumnarValue::Scalar(_) => unreachable!(),
    }
}

fn compress_needle(compressor: &Compressor, needle: &[u8]) -> Vec<u8> {
    let mut compressed = Vec::with_capacity(needle.len().saturating_mul(2));
    unsafe {
        compressor.compress_into(needle, &mut compressed);
    }
    compressed
}

fn bytes_cmp_const<const N: usize>(left: &[u8; N], right: &[u8; N]) -> std::cmp::Ordering {
    left.cmp(right)
}

fn bytes_cmp_short(left: &[u8], right: &[u8], len: usize) -> std::cmp::Ordering {
    match len {
        0 => std::cmp::Ordering::Equal,
        1 => bytes_cmp_const::<1>(
            &left[..1].try_into().unwrap(),
            &right[..1].try_into().unwrap(),
        ),
        2 => bytes_cmp_const::<2>(
            &left[..2].try_into().unwrap(),
            &right[..2].try_into().unwrap(),
        ),
        3 => bytes_cmp_const::<3>(
            &left[..3].try_into().unwrap(),
            &right[..3].try_into().unwrap(),
        ),
        4 => bytes_cmp_const::<4>(
            &left[..4].try_into().unwrap(),
            &right[..4].try_into().unwrap(),
        ),
        5 => bytes_cmp_const::<5>(
            &left[..5].try_into().unwrap(),
            &right[..5].try_into().unwrap(),
        ),
        6 => bytes_cmp_const::<6>(
            &left[..6].try_into().unwrap(),
            &right[..6].try_into().unwrap(),
        ),
        7 => bytes_cmp_const::<7>(
            &left[..7].try_into().unwrap(),
            &right[..7].try_into().unwrap(),
        ),
        _ => left[..len].cmp(&right[..len]),
    }
}

fn bytes_cmp_short_auto(left: &[u8], right: &[u8]) -> std::cmp::Ordering {
    let len = left.len().min(right.len());
    let ordering = bytes_cmp_short(left, right, len);
    if ordering == std::cmp::Ordering::Equal {
        left.len().cmp(&right.len())
    } else {
        ordering
    }
}
