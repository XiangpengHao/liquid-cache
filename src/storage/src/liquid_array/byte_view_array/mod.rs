//! LiquidByteViewArray

use arrow::array::BooleanArray;
use arrow::array::{
    Array, ArrayRef, BinaryArray, DictionaryArray, StringArray, UInt16Array, types::UInt16Type,
};
use arrow::buffer::{BooleanBuffer, NullBuffer};
use arrow::compute::cast;
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
use crate::liquid_array::raw::FsstArray;
use crate::liquid_array::raw::fsst_buffer::{DiskBuffer, FsstBacking, PrefixKey};
use crate::liquid_array::{
    LiquidArray, LiquidDataType, LiquidSqueezedArray, LiquidSqueezedArrayRef, NeedsBacking,
    SqueezeIoHandler, get_string_needle,
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
        io: Arc<dyn SqueezeIoHandler>,
        squeeze_hint: Option<&CacheExpression>,
    ) -> Option<(LiquidSqueezedArrayRef, Bytes)> {
        squeeze_hint?;

        // Serialize full IPC bytes first
        let bytes = match self.to_bytes_inner() {
            Ok(b) => b,
            Err(_) => return None,
        };

        // Build the hybrid (disk-backed FSST) view
        let disk_range = 0u64..(bytes.len() as u64);
        let compressor = self.fsst_buffer.compressor_arc();
        let disk = DiskBuffer::new(
            self.fsst_buffer.uncompressed_bytes(),
            io,
            disk_range,
            compressor,
        );
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
    async fn to_arrow_array(&self) -> ArrayRef {
        let bytes = self
            .fsst_buffer
            .squeeze_io()
            .read(Some(self.fsst_buffer.disk_range()))
            .await
            .expect("read squeezed backing");
        let hydrated =
            LiquidByteViewArray::<FsstArray>::from_bytes(bytes, self.fsst_buffer.compressor_arc());
        LiquidByteViewArray::<FsstArray>::to_arrow_array(&hydrated).expect("hydrate byte view")
    }

    /// Get the logical data type of the Liquid array.
    fn data_type(&self) -> LiquidDataType {
        LiquidDataType::ByteViewArray
    }

    fn original_arrow_data_type(&self) -> DataType {
        self.original_arrow_type.to_arrow_type()
    }

    /// Filter the Liquid array with a boolean array and return an **arrow array**.
    async fn filter(&self, selection: &BooleanBuffer) -> ArrayRef {
        let select_any = selection.count_set_bits() > 0;
        if !select_any {
            return arrow::array::new_empty_array(&self.original_arrow_data_type());
        }
        let bytes = self
            .fsst_buffer
            .squeeze_io()
            .read(Some(self.fsst_buffer.disk_range()))
            .await
            .expect("read squeezed backing");
        let hydrated =
            LiquidByteViewArray::<FsstArray>::from_bytes(bytes, self.fsst_buffer.compressor_arc());
        let arrow = hydrated.to_arrow_array().unwrap();
        let selection_array = BooleanArray::new(selection.clone(), None);
        arrow::compute::filter(&arrow, &selection_array).unwrap()
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
    ) -> Option<BooleanArray> {
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
                        // Try prefix-based equality. On ambiguity, hydrate and retry.
                        let Some(eq_mask) = filtered.compare_equals_with_prefix(needle.as_bytes())
                        else {
                            return self.try_eval_predicate_fully(&expr, filter).await;
                        };
                        if matches!(op, Operator::Eq) {
                            return Some(eq_mask);
                        } else {
                            let (values, nulls) = eq_mask.into_parts();
                            return Some(BooleanArray::new(!&values, nulls));
                        }
                    }
                    Operator::Lt | Operator::LtEq | Operator::Gt | Operator::GtEq => {
                        let Some(ord_mask) =
                            filtered.compare_ordering_with_prefix(needle.as_bytes(), op)
                        else {
                            return self.try_eval_predicate_fully(&expr, filter).await;
                        };
                        return Some(ord_mask);
                    }
                    _ => {}
                }
            }
        }

        self.try_eval_predicate_fully(&expr, filter).await
    }
}

impl LiquidByteViewArray<DiskBuffer> {
    async fn try_eval_predicate_fully(
        &self,
        expr: &Arc<dyn PhysicalExpr>,
        filter: &BooleanBuffer,
    ) -> Option<BooleanArray> {
        let bytes = self
            .fsst_buffer
            .squeeze_io()
            .read(Some(self.fsst_buffer.disk_range()))
            .await
            .ok()?;
        let hydrated =
            LiquidByteViewArray::<FsstArray>::from_bytes(bytes, self.fsst_buffer.compressor_arc());
        let hydrated_filtered = helpers::filter_inner(&hydrated, filter);
        helpers::try_eval_predicate_inner(expr, &hydrated_filtered)
            .expect("hydrate byte view backing")
    }
}
