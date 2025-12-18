//! LiquidByteViewArray

use arrow::array::BooleanArray;
use arrow::array::{
    Array, ArrayRef, BinaryArray, DictionaryArray, StringArray, UInt16Array, UInt32Array,
    types::UInt16Type,
};
use arrow::buffer::{BooleanBuffer, NullBuffer};
use arrow::compute::{cast, sort_to_indices};
use arrow_schema::DataType;
use bytes::Bytes;
use datafusion::logical_expr::Operator;
use datafusion::physical_plan::PhysicalExpr;
use datafusion::physical_plan::expressions::{BinaryExpr, DynamicFilterPhysicalExpr, Literal};
use std::any::Any;
use std::sync::Arc;

#[cfg(test)]
use std::cell::Cell;

use crate::cache::CacheExpression;
use crate::liquid_array::byte_array::ArrowByteType;
use crate::liquid_array::byte_view_array::serialization::ByteViewArrayHeader;
use crate::liquid_array::ipc::LiquidIPCHeader;
use crate::liquid_array::raw::FsstArray;
use crate::liquid_array::raw::fsst_buffer::{DiskBuffer, FsstBacking, PrefixKey};
use crate::liquid_array::{
    LiquidArray, LiquidDataType, LiquidSqueezedArray, LiquidSqueezedArrayRef, NeedsBacking,
    SqueezeIoHandler, SqueezeResult, get_string_needle,
};

// Declare submodules
mod comparisons;
mod conversions;
mod helpers;
mod serialization;

#[cfg(test)]
mod tests;

pub use helpers::ByteViewArrayMemoryUsage;

#[cfg(test)]
thread_local! {
    static DISK_READ_COUNTER: Cell<usize> = const { Cell::new(0)};
    static FULL_DATA_COMPARISON_COUNTER: Cell<usize> = const { Cell::new(0)};
}

#[cfg(test)]
fn get_disk_read_counter() -> usize {
    DISK_READ_COUNTER.with(|counter| counter.get())
}

#[cfg(test)]
fn reset_disk_read_counter() {
    DISK_READ_COUNTER.with(|counter| counter.set(0));
}

/// An array that stores strings using the FSST view format with compact offset compression:
/// - Dictionary keys with 2-byte keys stored in memory
/// - Compact offset views with variable-size offsets (1, 2, or 4 bytes) and 7-byte prefixes stored in memory
/// - FSST buffer can be stored in memory or on disk
///
/// # Initialization
///
/// The recommended way to create a `LiquidByteViewArray` is using the `from_*_array` constructors
/// which build a compact (offset + prefix key) representation directly from Arrow inputs.
///
/// ```rust,ignore
/// let liquid_array = LiquidByteViewArray::from_string_array(&input, compressor);
/// ```
///
/// Data access flow:
/// 1. Use dictionary key to index into compact offset views buffer
/// 2. Reconstruct actual offset from linear regression (predicted + residual)
/// 3. Use prefix from offset views for quick comparisons to avoid decompression when possible
/// 4. Decompress bytes from FSST buffer to get the full value when needed
#[derive(Clone)]
pub struct LiquidByteViewArray<B: FsstBacking> {
    /// Dictionary keys (u16) - one per array element, using Arrow's UInt16Array for zero-copy
    pub(super) dictionary_keys: UInt16Array,
    /// Per-value prefix keys (prefix7 + len metadata), includes the final sentinel entry.
    pub(super) prefix_keys: Arc<[PrefixKey]>,
    /// FSST-compressed buffer (can be in memory or on disk)
    pub(super) fsst_buffer: B,
    /// Used to convert back to the original arrow type
    pub(super) original_arrow_type: ArrowByteType,
    /// Shared prefix across all strings in the array
    pub(super) shared_prefix: Vec<u8>,
}

impl<B: FsstBacking> std::fmt::Debug for LiquidByteViewArray<B> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LiquidByteViewArray")
            .field("dictionary_keys", &self.dictionary_keys)
            .field("prefix_keys", &self.prefix_keys)
            .field("fsst_buffer", &self.fsst_buffer)
            .field("original_arrow_type", &self.original_arrow_type)
            .field("shared_prefix", &self.shared_prefix)
            .finish()
    }
}

impl<B: FsstBacking> LiquidByteViewArray<B> {
    /// Convert to Arrow DictionaryArray
    pub fn to_dict_arrow(&self) -> Result<DictionaryArray<UInt16Type>, NeedsBacking> {
        let keys_array = self.dictionary_keys.clone();

        let (values_buffer, offsets_buffer) = self.fsst_buffer.to_uncompressed()?;

        let values = if self.original_arrow_type == ArrowByteType::Utf8
            || self.original_arrow_type == ArrowByteType::Utf8View
            || self.original_arrow_type == ArrowByteType::Dict16Utf8
        {
            let string_array =
                unsafe { StringArray::new_unchecked(offsets_buffer, values_buffer, None) };
            Arc::new(string_array) as ArrayRef
        } else {
            let binary_array =
                unsafe { BinaryArray::new_unchecked(offsets_buffer, values_buffer, None) };
            Arc::new(binary_array) as ArrayRef
        };

        Ok(unsafe { DictionaryArray::<UInt16Type>::new_unchecked(keys_array, values) })
    }

    /// Convert to Arrow array with original type
    pub fn to_arrow_array(&self) -> Result<ArrayRef, NeedsBacking> {
        let dict = self.to_dict_arrow()?;
        Ok(cast(&dict, &self.original_arrow_type.to_arrow_type()).unwrap())
    }

    /// Get the nulls buffer
    pub fn nulls(&self) -> Option<&NullBuffer> {
        self.dictionary_keys.nulls()
    }

    /// Get detailed memory usage of the byte view array
    pub fn get_detailed_memory_usage(&self) -> ByteViewArrayMemoryUsage {
        ByteViewArrayMemoryUsage {
            dictionary_key: self.dictionary_keys.get_array_memory_size(),
            offsets: self.prefix_keys.len() * std::mem::size_of::<PrefixKey>(),
            fsst_buffer: self.fsst_buffer.get_array_memory_size(),
            shared_prefix: self.shared_prefix.len(),
            struct_size: std::mem::size_of::<Self>(),
        }
    }

    /// Sort the array and return indices that would sort the array
    /// This implements efficient sorting as described in the design document:
    /// 1. First sort the dictionary using prefixes to delay decompression
    /// 2. If decompression is needed, decompress the entire array at once
    /// 3. Use dictionary ranks to sort the final keys
    pub fn sort_to_indices(&self) -> Result<UInt32Array, NeedsBacking> {
        // if distinct ratio is more than 10%, use arrow sort.
        if self.prefix_keys.len() > (self.dictionary_keys.len() / 10) {
            let array = self.to_dict_arrow()?;
            let sorted_array = sort_to_indices(&array, None, None).unwrap();
            Ok(sorted_array)
        } else {
            self.sort_to_indices_inner()
        }
    }

    /// Check if the FSST buffer is currently stored on disk
    pub fn is_fsst_buffer_on_disk(&self) -> bool {
        self.fsst_buffer.get_array_memory_size() == 0 && self.fsst_buffer.uncompressed_bytes() > 0
    }

    /// Check if the FSST buffer is currently stored in memory
    pub fn is_fsst_buffer_in_memory(&self) -> bool {
        !self.is_fsst_buffer_on_disk()
    }

    /// Get the length of the array
    pub fn len(&self) -> usize {
        self.dictionary_keys.len()
    }

    /// Is the array empty?
    pub fn is_empty(&self) -> bool {
        self.dictionary_keys.is_empty()
    }

    /// Get disk read count for testing
    #[cfg(test)]
    pub fn get_disk_read_count(&self) -> usize {
        get_disk_read_counter()
    }

    /// Reset disk read count for testing
    #[cfg(test)]
    pub fn reset_disk_read_count(&self) {
        reset_disk_read_counter()
    }

    fn sort_to_indices_inner(&self) -> Result<UInt32Array, NeedsBacking> {
        // Step 1: Get dictionary ranks using prefix optimization
        let dict_ranks = self.get_dictionary_ranks()?;

        // Step 2: Partition array indices into nulls and non-nulls, then sort non-nulls
        let mut non_null_indices = Vec::with_capacity(self.dictionary_keys.len());

        let mut array_indices = Vec::with_capacity(self.dictionary_keys.len());
        for array_idx in 0..self.dictionary_keys.len() as u32 {
            if self.dictionary_keys.is_null(array_idx as usize) {
                array_indices.push(array_idx);
            } else {
                non_null_indices.push(array_idx);
            }
        }

        // Sort non-null indices by their dictionary ranks
        non_null_indices.sort_unstable_by_key(|&array_idx| unsafe {
            let dict_key = self.dictionary_keys.value_unchecked(array_idx as usize);
            dict_ranks.get_unchecked(dict_key as usize)
        });

        array_indices.extend(non_null_indices);

        Ok(UInt32Array::from(array_indices))
    }

    /// Get dictionary ranks using prefix optimization and lazy decompression
    /// Returns a mapping from dictionary key to its rank in sorted order
    fn get_dictionary_ranks(&self) -> Result<Vec<u16>, NeedsBacking> {
        let num_unique = self.prefix_keys.len().saturating_sub(1);
        let mut dict_indices: Vec<u32> = (0..num_unique as u32).collect();

        let decompressed = {
            let (values_buffer, offsets_buffer) = self.fsst_buffer.to_uncompressed()?;
            unsafe { BinaryArray::new_unchecked(offsets_buffer, values_buffer, None) }
        };

        // Sort using prefix optimization first, then full strings when needed
        dict_indices.sort_unstable_by(|&a, &b| unsafe {
            // First try prefix comparison - no need to include shared_prefix since all strings have it
            let prefix_a = self.prefix_keys[a as usize].prefix7();
            let prefix_b = self.prefix_keys[b as usize].prefix7();

            let prefix_cmp = prefix_a.cmp(prefix_b);

            if prefix_cmp != std::cmp::Ordering::Equal {
                // Prefix comparison is sufficient
                prefix_cmp
            } else {
                // Prefixes are equal, need full string comparison
                let string_a = decompressed.value_unchecked(a as usize);
                let string_b = decompressed.value_unchecked(b as usize);
                string_a.cmp(string_b)
            }
        });

        // Convert sorted indices to rank mapping
        let mut dict_ranks = vec![0u16; dict_indices.len()];
        for (rank, dict_key) in dict_indices.into_iter().enumerate() {
            dict_ranks[dict_key as usize] = rank as u16;
        }

        Ok(dict_ranks)
    }
}

impl LiquidArray for LiquidByteViewArray<FsstArray> {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn get_array_memory_size(&self) -> usize {
        self.get_detailed_memory_usage().total()
    }

    fn len(&self) -> usize {
        self.dictionary_keys.len()
    }

    #[inline]
    fn to_arrow_array(&self) -> ArrayRef {
        let dict = self.to_arrow_array().expect("InMemoryFsstBuffer");
        Arc::new(dict)
    }

    fn to_best_arrow_array(&self) -> ArrayRef {
        let dict = self.to_dict_arrow().expect("InMemoryFsstBuffer");
        Arc::new(dict)
    }

    fn try_eval_predicate(
        &self,
        expr: &Arc<dyn PhysicalExpr>,
        filter: &BooleanBuffer,
    ) -> Option<BooleanArray> {
        let filtered = helpers::filter_inner(self, filter);
        helpers::try_eval_predicate_inner(expr, &filtered).expect("InMemoryFsstBuffer")
    }

    fn to_bytes(&self) -> Vec<u8> {
        self.to_bytes_inner().expect("InMemoryFsstBuffer")
    }

    fn original_arrow_data_type(&self) -> DataType {
        self.original_arrow_type.to_arrow_type()
    }

    fn data_type(&self) -> LiquidDataType {
        LiquidDataType::ByteViewArray
    }

    fn squeeze(
        &self,
        _io: Arc<dyn SqueezeIoHandler>,
        squeeze_hint: Option<&CacheExpression>,
    ) -> Option<(LiquidSqueezedArrayRef, Bytes)> {
        squeeze_hint?;

        // Serialize full IPC bytes first
        let bytes = match self.to_bytes_inner() {
            Ok(b) => b,
            Err(_) => return None,
        };

        const IPC_HEADER_SIZE: usize = LiquidIPCHeader::size();
        const VIEW_HEADER_SIZE: usize = ByteViewArrayHeader::size();
        let header_size = IPC_HEADER_SIZE + VIEW_HEADER_SIZE;

        assert!(bytes.len() >= header_size);

        let fsst_size = u32::from_le_bytes(
            bytes[IPC_HEADER_SIZE + 12..IPC_HEADER_SIZE + 16]
                .try_into()
                .unwrap(),
        ) as usize;

        // FSST starts at the first 8-byte boundary after headers
        let fsst_start = (header_size + 7) & !7;
        let fsst_end = fsst_start + fsst_size;

        assert!(fsst_end <= bytes.len());

        // Build the hybrid (disk-backed FSST) view
        let disk = DiskBuffer::new(self.fsst_buffer.uncompressed_bytes());
        let hybrid = LiquidByteViewArray::<DiskBuffer> {
            dictionary_keys: self.dictionary_keys.clone(),
            prefix_keys: self.prefix_keys.clone(),
            fsst_buffer: disk,
            original_arrow_type: self.original_arrow_type,
            shared_prefix: self.shared_prefix.clone(),
        };

        let bytes = Bytes::from(bytes);
        Some((Arc::new(hybrid) as LiquidSqueezedArrayRef, bytes))
    }
}

#[async_trait::async_trait]
impl LiquidSqueezedArray for LiquidByteViewArray<DiskBuffer> {
    /// Get the underlying any type.
    fn as_any(&self) -> &dyn Any {
        self
    }

    /// Get the memory size of the Liquid array.
    fn get_array_memory_size(&self) -> usize {
        self.get_detailed_memory_usage().total()
    }

    /// Get the length of the Liquid array.
    fn len(&self) -> usize {
        self.dictionary_keys.len()
    }

    /// Check if the Liquid array is empty.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Convert the Liquid array to an Arrow array.
    async fn to_arrow_array(&self) -> SqueezeResult<ArrayRef> {
        LiquidByteViewArray::<DiskBuffer>::to_arrow_array(self)
    }

    /// Get the logical data type of the Liquid array.
    fn data_type(&self) -> LiquidDataType {
        LiquidDataType::ByteViewArray
    }

    fn original_arrow_data_type(&self) -> DataType {
        self.original_arrow_type.to_arrow_type()
    }

    /// Serialize the Liquid array to a byte array.
    async fn to_bytes(&self) -> SqueezeResult<Vec<u8>> {
        Err(NeedsBacking)
    }

    /// Filter the Liquid array with a boolean array and return an **arrow array**.
    async fn filter(&self, selection: &BooleanBuffer) -> SqueezeResult<ArrayRef> {
        let select_any = selection.count_set_bits() > 0;
        if !select_any {
            return Ok(arrow::array::new_empty_array(
                &self.original_arrow_data_type(),
            ));
        }
        let filtered = helpers::filter_inner(self, selection);
        filtered.to_best_arrow_array().await
    }

    /// Try to evaluate a predicate on the Liquid array with a filter.
    /// Returns `Ok(None)` if the predicate is not supported.
    ///
    /// Note that the filter is a boolean buffer, not a boolean array, i.e., filter can't be nullable.
    /// The returned boolean mask is nullable if the the original array is nullable.
    async fn try_eval_predicate(
        &self,
        expr: &Arc<dyn PhysicalExpr>,
        filter: &BooleanBuffer,
    ) -> SqueezeResult<Option<BooleanArray>> {
        // Reuse generic filter path first to reduce input rows if any
        let filtered = helpers::filter_inner(self, filter);

        let expr = if let Some(dynamic_filter) =
            expr.as_any().downcast_ref::<DynamicFilterPhysicalExpr>()
        {
            dynamic_filter.current().expect("DynamicFilterPhysicalExpr")
        } else {
            expr.clone()
        };

        // Handle binary expressions with prefix-only optimizations
        if let Some(binary_expr) = expr.as_any().downcast_ref::<BinaryExpr>()
            && let Some(literal) = binary_expr.right().as_any().downcast_ref::<Literal>()
        {
            let op = binary_expr.op();
            if let Some(needle) = get_string_needle(literal.value()) {
                match op {
                    Operator::Eq | Operator::NotEq => {
                        // Try prefix-based equality. On ambiguity, bubble up NeedsBacking for fallback.
                        let eq_mask = filtered
                            .compare_equals_with_prefix(needle.as_bytes())
                            .ok_or(NeedsBacking)?;
                        if matches!(op, Operator::Eq) {
                            return Ok(Some(eq_mask));
                        } else {
                            let (values, nulls) = eq_mask.into_parts();
                            return Ok(Some(BooleanArray::new(!&values, nulls)));
                        }
                    }
                    Operator::Lt | Operator::LtEq | Operator::Gt | Operator::GtEq => {
                        // Prefix-only ordering; ambiguous cases ask for backing.
                        let ord_mask = filtered
                            .compare_ordering_with_prefix(needle.as_bytes(), op)
                            .ok_or(NeedsBacking)?;
                        return Ok(Some(ord_mask));
                    }
                    _ => {}
                }
            }
        }

        Ok(None)
    }
}
