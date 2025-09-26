use std::sync::Arc;

use arrow::{
    array::BooleanBufferBuilder,
    buffer::{BooleanBuffer, MutableBuffer},
};
use datafusion::{
    common::tree_node::{Transformed, TreeNode, TreeNodeRecursion},
    datasource::{physical_plan::ParquetSource, source::DataSource, source::DataSourceExec},
    physical_plan::{ExecutionPlan, metrics::MetricValue},
};
use liquid_cache_common::rpc::ExecutionMetricsResponse;
use parquet::arrow::arrow_reader::RowSelector;

use crate::{LiquidParquetSource, cache::LiquidCacheRef};

fn boolean_buffer_and_then_fallback(left: &BooleanBuffer, right: &BooleanBuffer) -> BooleanBuffer {
    debug_assert_eq!(
        left.count_set_bits(),
        right.len(),
        "the right selection must have the same number of set bits as the left selection"
    );

    if left.len() == right.len() {
        debug_assert_eq!(left.count_set_bits(), left.len());
        return right.clone();
    }

    let mut buffer = MutableBuffer::from_len_zeroed(left.values().len());
    buffer.copy_from_slice(left.values());
    let mut builder = BooleanBufferBuilder::new_from_buffer(buffer, left.len());

    let mut other_bits = right.iter();

    for bit_idx in left.set_indices() {
        let predicate = other_bits
            .next()
            .expect("Mismatch in set bits between self and other");
        if !predicate {
            builder.set_bit(bit_idx, false);
        }
    }

    builder.finish()
}

/// Combines this [`BooleanBuffer`] with another using logical AND on the selected bits.
///
/// Unlike intersection, the `other` [`BooleanBuffer`] must have exactly as many **set bits** as `self`,
/// i.e., self.count_set_bits() == other.len().
///
/// This method will keep only the bits in `self` that are also set in `other`
/// at the positions corresponding to `self`'s set bits.
/// For example:
/// left:   NNYYYNNYYNYN
/// right:    YNY  NY N
/// result: NNYNYNNNYNNN
///
/// Optimized version of `boolean_buffer_and_then` using BMI2 PDEP instructions.
/// This function performs the same operation but uses bit manipulation instructions
/// for better performance on supported x86_64 CPUs.
pub fn boolean_buffer_and_then(left: &BooleanBuffer, right: &BooleanBuffer) -> BooleanBuffer {
    debug_assert_eq!(
        left.count_set_bits(),
        right.len(),
        "the right selection must have the same number of set bits as the left selection"
    );

    if left.len() == right.len() {
        debug_assert_eq!(left.count_set_bits(), left.len());
        return right.clone();
    }

    // Fast path for BMI2 support on x86_64
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("bmi2") {
            return unsafe { boolean_buffer_and_then_bmi2(left, right) };
        }
    }

    boolean_buffer_and_then_fallback(left, right)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "bmi2")]
unsafe fn boolean_buffer_and_then_bmi2(
    left: &BooleanBuffer,
    right: &BooleanBuffer,
) -> BooleanBuffer {
    use core::arch::x86_64::_pdep_u64;

    debug_assert_eq!(left.count_set_bits(), right.len());

    let bit_len = left.len();
    let byte_len = bit_len.div_ceil(8);
    let left_ptr = left.values().as_ptr();
    let right_ptr = right.values().as_ptr();

    let mut out = MutableBuffer::from_len_zeroed(byte_len);
    let out_ptr = out.as_mut_ptr();

    let full_words = byte_len / 8;
    let mut right_bit_idx = 0; // how many bits we have processed from right

    for word_idx in 0..full_words {
        let left_word =
            unsafe { core::ptr::read_unaligned(left_ptr.add(word_idx * 8) as *const u64) };

        if left_word == 0 {
            continue;
        }

        let need = left_word.count_ones();

        // Absolute byte & bit offset of the first needed bit inside `right`.
        let rb_byte = right_bit_idx / 8;
        let rb_bit = (right_bit_idx & 7) as u32;

        // We load two u64 words and shift/mask them to avoid branches and loops.
        let mut r_bits =
            unsafe { core::ptr::read_unaligned(right_ptr.add(rb_byte) as *const u64) } >> rb_bit;
        if rb_bit != 0 {
            let next =
                unsafe { core::ptr::read_unaligned(right_ptr.add(rb_byte + 8) as *const u64) };
            r_bits |= next << (64 - rb_bit);
        }

        // Mask off the high garbage.
        r_bits &= 1u64.unbounded_shl(need).wrapping_sub(1);

        // The PDEP instruction: https://www.felixcloutier.com/x86/pdep
        // It takes left_word as the mask, and deposit the packed bits into the sparse positions of `left_word`.
        let result = _pdep_u64(r_bits, left_word);

        unsafe {
            core::ptr::write_unaligned(out_ptr.add(word_idx * 8) as *mut u64, result);
        }

        right_bit_idx += need as usize;
    }

    // Handle remaining bits that are less than 64 bits
    let tail_bits = bit_len & 63;
    if tail_bits != 0 {
        let mut mask = 0u64;
        for bit in 0..tail_bits {
            let byte = unsafe { *left_ptr.add(full_words * 8 + (bit / 8)) };
            mask |= (((byte >> (bit & 7)) & 1) as u64) << bit;
        }

        if mask != 0 {
            let need = mask.count_ones();

            let rb_byte = right_bit_idx / 8;
            let rb_bit = (right_bit_idx & 7) as u32;

            let mut r_bits =
                unsafe { core::ptr::read_unaligned(right_ptr.add(rb_byte) as *const u64) }
                    >> rb_bit;
            if rb_bit != 0 {
                let next =
                    unsafe { core::ptr::read_unaligned(right_ptr.add(rb_byte + 8) as *const u64) };
                r_bits |= next << (64 - rb_bit);
            }

            r_bits &= 1u64.unbounded_shl(need).wrapping_sub(1);

            let result = _pdep_u64(r_bits, mask);

            let tail_bytes = tail_bits.div_ceil(8);
            unsafe {
                core::ptr::copy_nonoverlapping(
                    &result.to_le_bytes()[0],
                    out_ptr.add(full_words * 8),
                    tail_bytes,
                );
            }
        }
    }

    BooleanBuffer::new(out.into(), 0, bit_len)
}

pub(super) fn row_selector_to_boolean_buffer(selection: &[RowSelector]) -> BooleanBuffer {
    let mut buffer = BooleanBufferBuilder::new(8192);
    for selector in selection.iter() {
        if selector.skip {
            buffer.append_n(selector.row_count, false);
        } else {
            buffer.append_n(selector.row_count, true);
        }
    }
    buffer.finish()
}

/// Extracts execution metrics from a physical execution plan.
///
/// This function traverses the plan tree to find all DataSourceExec nodes and aggregates
/// their metrics, including processing time and bytes scanned. It can be used by both
/// client-server and in-process benchmarks to get consistent metrics.
///
/// # Arguments
///
/// * `plan` - The execution plan to extract metrics from
/// * `liquid_cache` - Optional reference to the liquid cache for memory usage calculation
///
/// # Returns
///
/// An `ExecutionMetricsResponse` containing:
/// - `pushdown_eval_time`: Time spent processing data (in milliseconds)
/// - `cache_memory_usage`: Total cache memory usage including bytes scanned
/// - `liquid_cache_usage`: Memory usage of the liquid cache specifically
pub fn extract_execution_metrics(
    plan: &Arc<dyn ExecutionPlan>,
    liquid_cache: Option<&LiquidCacheRef>,
) -> ExecutionMetricsResponse {
    // Traverse the plan tree to find all DataSourceExec nodes and collect their metrics
    let mut time_elapsed_processing_millis = 0;
    let mut bytes_scanned = 0;

    let _ = plan.apply(|node| {
        let any_plan = node.as_any();
        if let Some(data_source_exec) = any_plan.downcast_ref::<DataSourceExec>()
            && let Some(metrics) = data_source_exec.metrics()
        {
            let aggregated_metrics = metrics
                .aggregate_by_name()
                .sorted_for_display()
                .timestamps_removed();

            for metric in aggregated_metrics.iter() {
                if let MetricValue::Time { name, time } = metric.value()
                    && name == "time_elapsed_processing"
                {
                    time_elapsed_processing_millis += time.value() / 1_000_000;
                } else if let MetricValue::Count { name, count } = metric.value()
                    && name == "bytes_scanned"
                {
                    bytes_scanned += count.value();
                }
            }
        }
        Ok(TreeNodeRecursion::Continue)
    });

    let liquid_cache_usage = liquid_cache
        .map(|cache| cache.compute_memory_usage_bytes())
        .unwrap_or(0);
    let cache_memory_usage = liquid_cache_usage + bytes_scanned as u64;

    ExecutionMetricsResponse {
        pushdown_eval_time: time_elapsed_processing_millis as u64,
        cache_memory_usage,
        liquid_cache_usage,
    }
}

/// Rewrite the data source plan to use liquid cache.
pub fn rewrite_data_source_plan(
    plan: Arc<dyn ExecutionPlan>,
    cache: &LiquidCacheRef,
) -> Arc<dyn ExecutionPlan> {
    let rewritten = plan
        .transform_up(|node| {
            let any_plan = node.as_any();
            if let Some(data_source_exec) = any_plan.downcast_ref::<DataSourceExec>() {
                if let Some((file_scan_config, parquet_source)) =
                    data_source_exec.downcast_to_file_source::<ParquetSource>()
                {
                    let new_source = LiquidParquetSource::from_parquet_source(
                        parquet_source.clone(),
                        file_scan_config.file_schema.clone(),
                        cache.clone(),
                    );
                    let mut new_config = file_scan_config.clone();
                    new_config.file_source = Arc::new(new_source);
                    let new_file_source: Arc<dyn DataSource> = Arc::new(new_config);
                    let new_plan = Arc::new(DataSourceExec::new(new_file_source));

                    return Ok(Transformed::new(
                        new_plan,
                        true,
                        TreeNodeRecursion::Continue,
                    ));
                }

                return Ok(Transformed::no(node));
            }
            Ok(Transformed::no(node))
        })
        .unwrap();
    rewritten.data
}

#[cfg(test)]
mod tests {
    use crate::cache::LiquidCache;

    use super::*;
    use datafusion::{datasource::physical_plan::FileScanConfig, prelude::SessionContext};
    use liquid_cache_storage::{
        cache::squeeze_policies::TranscodeSqueezeEvict, cache_policies::FiloPolicy,
    };
    use std::{path::PathBuf, sync::Arc};

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_boolean_buffer_and_then_bmi2_large() {
        use super::boolean_buffer_and_then_bmi2;

        // Test with larger buffer (more than 64 bits)
        let size = 128;
        let mut left_builder = BooleanBufferBuilder::new(size);
        let mut right_bits = Vec::new();

        // Create a pattern where every 3rd bit is set in left
        for i in 0..size {
            let is_set = i.is_multiple_of(3);
            left_builder.append(is_set);
            if is_set {
                // For right buffer, alternate between true/false
                right_bits.push(right_bits.len().is_multiple_of(2));
            }
        }
        let left = left_builder.finish();

        let mut right_builder = BooleanBufferBuilder::new(right_bits.len());
        for bit in right_bits {
            right_builder.append(bit);
        }
        let right = right_builder.finish();

        let result_bmi2 = unsafe { boolean_buffer_and_then_bmi2(&left, &right) };
        let result_orig = boolean_buffer_and_then_fallback(&left, &right);

        assert_eq!(result_bmi2.len(), result_orig.len());
        assert_eq!(result_bmi2.len(), size);

        // Verify they produce the same result
        for i in 0..size {
            assert_eq!(
                result_bmi2.value(i),
                result_orig.value(i),
                "Mismatch at position {i}"
            );
        }
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_boolean_buffer_and_then_bmi2_edge_cases() {
        use super::boolean_buffer_and_then_bmi2;

        // Test case: all bits set in left, alternating pattern in right
        let mut left_builder = BooleanBufferBuilder::new(16);
        for _ in 0..16 {
            left_builder.append(true);
        }
        let left = left_builder.finish();

        let mut right_builder = BooleanBufferBuilder::new(16);
        for i in 0..16 {
            right_builder.append(i % 2 == 0);
        }
        let right = right_builder.finish();

        let result_bmi2 = unsafe { boolean_buffer_and_then_bmi2(&left, &right) };
        let result_orig = boolean_buffer_and_then_fallback(&left, &right);

        assert_eq!(result_bmi2.len(), result_orig.len());
        for i in 0..16 {
            assert_eq!(
                result_bmi2.value(i),
                result_orig.value(i),
                "Mismatch at position {i}"
            );
            // Should be true for even indices, false for odd
            assert_eq!(result_bmi2.value(i), i.is_multiple_of(2));
        }

        // Test case: no bits set in left
        let mut left_empty_builder = BooleanBufferBuilder::new(8);
        for _ in 0..8 {
            left_empty_builder.append(false);
        }
        let left_empty = left_empty_builder.finish();
        let right_empty = BooleanBufferBuilder::new(0).finish();

        let result_bmi2_empty = unsafe { boolean_buffer_and_then_bmi2(&left_empty, &right_empty) };
        let result_orig_empty = boolean_buffer_and_then_fallback(&left_empty, &right_empty);

        assert_eq!(result_bmi2_empty.len(), result_orig_empty.len());
        assert_eq!(result_bmi2_empty.len(), 8);
        for i in 0..8 {
            assert!(!result_bmi2_empty.value(i));
            assert!(!result_orig_empty.value(i));
        }
    }

    fn rewrite_plan_inner(plan: Arc<dyn ExecutionPlan>) {
        let expected_schema = plan.schema();
        let liquid_cache = Arc::new(LiquidCache::new(
            8192,
            1000000,
            PathBuf::from("test"),
            Box::new(FiloPolicy::new()),
            Box::new(TranscodeSqueezeEvict),
        ));
        let rewritten = rewrite_data_source_plan(plan, &liquid_cache);

        rewritten
            .apply(|node| {
                if let Some(plan) = node.as_any().downcast_ref::<DataSourceExec>() {
                    let data_source = plan.data_source();
                    let any_source = data_source.as_any();
                    let source = any_source.downcast_ref::<FileScanConfig>().unwrap();
                    let file_source = source.file_source();
                    let any_file_source = file_source.as_any();
                    let _parquet_source = any_file_source
                        .downcast_ref::<LiquidParquetSource>()
                        .unwrap();
                    let schema = source.file_schema.as_ref();
                    assert_eq!(schema, expected_schema.as_ref());
                }
                Ok(TreeNodeRecursion::Continue)
            })
            .unwrap();
    }

    #[tokio::test]
    async fn test_plan_rewrite() {
        let ctx = SessionContext::new();
        ctx.register_parquet(
            "nano_hits",
            "../../examples/nano_hits.parquet",
            Default::default(),
        )
        .await
        .unwrap();
        let df = ctx
            .sql("SELECT * FROM nano_hits WHERE \"URL\" like 'https://%' limit 10")
            .await
            .unwrap();
        let plan = df.create_physical_plan().await.unwrap();
        rewrite_plan_inner(plan.clone());
    }
}
