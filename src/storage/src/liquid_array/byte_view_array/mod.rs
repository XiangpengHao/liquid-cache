//! LiquidByteViewArray

use arrow::array::BooleanArray;
use arrow::array::{
    Array, ArrayRef, BinaryArray, DictionaryArray, StringArray, UInt16Array, types::UInt16Type,
};
use arrow::buffer::{BooleanBuffer, Buffer, NullBuffer, OffsetBuffer};
use arrow::compute::cast;
use arrow_schema::DataType;
use bytes::Bytes;
use datafusion::physical_plan::PhysicalExpr;
use datafusion::physical_plan::expressions::DynamicFilterPhysicalExpr;
use std::any::Any;
use std::sync::Arc;

#[cfg(test)]
use std::cell::Cell;

use crate::cache::CacheExpression;
use crate::liquid_array::byte_array::ArrowByteType;
use crate::liquid_array::byte_view_array::fingerprint::build_fingerprints;
use crate::liquid_array::raw::FsstArray;
use crate::liquid_array::raw::fsst_buffer::{DiskBuffer, FsstBacking, PrefixKey};
use crate::liquid_array::{
    LiquidArray, LiquidDataType, LiquidSqueezedArray, LiquidSqueezedArrayRef, SqueezeIoHandler,
};

// Declare submodules
mod comparisons;
mod conversions;
mod fingerprint;
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

/// An array that stores strings using the FSST format with compact offsets:
/// - Dictionary keys with 2-byte keys stored in memory
/// - Compact offsets with variable-size residuals (1, 2, or 4 bytes) stored in memory
/// - Per-value prefix keys (7-byte prefix + len) stored in memory
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
/// 1. Use dictionary key to index into compact offsets buffer
/// 2. Reconstruct actual offset from linear regression (predicted + residual)
/// 3. Use prefix keys for quick comparisons to avoid decompression when possible
/// 4. Decompress bytes from FSST buffer to get the full value when needed
#[derive(Clone)]
pub struct LiquidByteViewArray<B: FsstBacking> {
    /// Dictionary keys (u16) - one per array element, using Arrow's UInt16Array for zero-copy
    pub(super) dictionary_keys: UInt16Array,
    /// Per-value prefix keys (prefix7 + len metadata).
    pub(super) prefix_keys: Arc<[PrefixKey]>,
    /// FSST-compressed buffer (can be in memory or on disk)
    pub(super) fsst_buffer: B,
    /// Used to convert back to the original arrow type
    pub(super) original_arrow_type: ArrowByteType,
    /// Shared prefix across all strings in the array
    pub(super) shared_prefix: Vec<u8>,
    /// Optional per-dictionary string fingerprints (32 bins).
    pub(super) string_fingerprints: Option<Arc<[u32]>>,
}

impl<B: FsstBacking> std::fmt::Debug for LiquidByteViewArray<B> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LiquidByteViewArray")
            .field("dictionary_keys", &self.dictionary_keys)
            .field("prefix_keys", &self.prefix_keys)
            .field("fsst_buffer", &self.fsst_buffer)
            .field("original_arrow_type", &self.original_arrow_type)
            .field("shared_prefix", &self.shared_prefix)
            .field("string_fingerprints", &self.string_fingerprints)
            .finish()
    }
}

impl<B: FsstBacking> LiquidByteViewArray<B> {
    /// Convert to Arrow DictionaryArray
    fn to_dict_arrow_inner(
        &self,
        values_buffer: Buffer,
        offsets_buffer: OffsetBuffer<i32>,
    ) -> DictionaryArray<UInt16Type> {
        let keys_array = self.dictionary_keys.clone();
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

        unsafe { DictionaryArray::<UInt16Type>::new_unchecked(keys_array, values) }
    }

    /// Get the nulls buffer
    pub fn nulls(&self) -> Option<&NullBuffer> {
        self.dictionary_keys.nulls()
    }

    /// Get detailed memory usage of the byte view array
    pub fn get_detailed_memory_usage(&self) -> ByteViewArrayMemoryUsage {
        let fingerprint_bytes = self
            .string_fingerprints
            .as_ref()
            .map(|fingerprints| fingerprints.len() * std::mem::size_of::<u32>())
            .unwrap_or(0);
        ByteViewArrayMemoryUsage {
            dictionary_key: self.dictionary_keys.get_array_memory_size(),
            offsets: self.fsst_buffer.compact_offsets_memory_usage(),
            prefix_keys: self.prefix_keys.len() * std::mem::size_of::<PrefixKey>(),
            fsst_buffer: self.fsst_buffer.raw_memory_usage(),
            shared_prefix: self.shared_prefix.len(),
            string_fingerprints: fingerprint_bytes,
            struct_size: std::mem::size_of::<Self>(),
        }
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

impl LiquidByteViewArray<FsstArray> {
    /// Convert to Arrow DictionaryArray
    pub fn to_dict_arrow(&self) -> DictionaryArray<UInt16Type> {
        let (values_buffer, offsets_buffer) = self.fsst_buffer.to_uncompressed();
        self.to_dict_arrow_inner(values_buffer, offsets_buffer)
    }

    /// Convert to Arrow array with original type
    pub fn to_arrow_array(&self) -> ArrayRef {
        let dict = self.to_dict_arrow();
        cast(&dict, &self.original_arrow_type.to_arrow_type()).unwrap()
    }

    /// Check if the FSST buffer is currently stored on disk
    pub fn is_fsst_buffer_on_disk(&self) -> bool {
        false
    }
}

impl LiquidByteViewArray<DiskBuffer> {
    /// Check if the FSST buffer is currently stored on disk
    pub fn is_fsst_buffer_on_disk(&self) -> bool {
        true
    }

    /// Convert to Arrow DictionaryArray
    pub async fn to_dict_arrow(&self) -> DictionaryArray<UInt16Type> {
        let (values_buffer, offsets_buffer) = self.fsst_buffer.to_uncompressed().await;
        self.to_dict_arrow_inner(values_buffer, offsets_buffer)
    }

    /// Convert to Arrow array with original type
    pub async fn to_arrow_array(&self) -> ArrayRef {
        let dict = self.to_dict_arrow().await;
        cast(&dict, &self.original_arrow_type.to_arrow_type()).unwrap()
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
        let dict = self.to_arrow_array();
        Arc::new(dict)
    }

    fn to_best_arrow_array(&self) -> ArrayRef {
        let dict = self.to_dict_arrow();
        Arc::new(dict)
    }

    fn try_eval_predicate(
        &self,
        expr: &Arc<dyn PhysicalExpr>,
        filter: &BooleanBuffer,
    ) -> Option<BooleanArray> {
        let filtered = helpers::filter_inner(self, filter);
        helpers::try_eval_predicate_in_memory(expr, &filtered)
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

        let string_fingerprints = if matches!(squeeze_hint, Some(CacheExpression::SubstringSearch))
        {
            let (values_buffer, offsets_buffer) = self.fsst_buffer.to_uncompressed();
            Some(build_fingerprints(&values_buffer, &offsets_buffer))
        } else {
            None
        };

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
            string_fingerprints,
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
        LiquidByteViewArray::<FsstArray>::to_arrow_array(&hydrated)
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
        let arrow = hydrated.to_arrow_array();
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

        helpers::try_eval_predicate_on_disk(&expr, &filtered).await
    }
}
