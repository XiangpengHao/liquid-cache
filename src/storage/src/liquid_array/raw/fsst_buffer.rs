use arrow::{
    array::{
        ArrayDataBuilder, Decimal128Array, Decimal256Array, GenericByteArray, OffsetBufferBuilder,
    },
    buffer::{Buffer, OffsetBuffer},
    datatypes::ByteArrayType,
};
use bytes;
use fsst::{Compressor, Decompressor, Symbol};
use std::io::Result;
use std::io::{Error, ErrorKind};
use std::sync::Arc;

use crate::liquid_array::{NeedsBacking, SqueezeResult};

use crate::liquid_array::fix_len_byte_array::ArrowFixedLenByteArrayType;

mod sealed {
    pub trait Sealed {}
}

/// Raw FSST buffer that stores compressed data using Arrow Buffer.
/// Offsets are managed externally as a `u32` slice (including the final sentinel offset).
#[derive(Clone)]
pub(crate) struct RawFsstBuffer {
    values: Buffer,
    uncompressed_bytes: usize,
}

impl std::fmt::Debug for RawFsstBuffer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RawFsstBuffer")
            .field("values_len", &self.values.len())
            .field("uncompressed_bytes", &self.uncompressed_bytes)
            .finish()
    }
}

impl RawFsstBuffer {
    pub(crate) fn from_parts(values: Buffer, uncompressed_bytes: usize) -> Self {
        Self {
            values,
            uncompressed_bytes,
        }
    }

    /// Create RawFsstBuffer from an iterator of byte slices.
    /// Returns the buffer and a vector of byte offsets (including the final sentinel).
    pub(crate) fn from_byte_slices<I, T>(
        iter: I,
        compressor: Arc<Compressor>,
        compress_buffer: &mut Vec<u8>,
    ) -> (Self, Vec<u32>)
    where
        I: Iterator<Item = Option<T>>,
        T: AsRef<[u8]>,
    {
        let mut values_buffer = Vec::new();
        let mut offsets = Vec::new();
        let mut uncompressed_len = 0;

        offsets.push(0u32);
        for item in iter {
            if let Some(bytes) = item {
                let bytes = bytes.as_ref();
                uncompressed_len += bytes.len();

                compress_buffer.clear();
                // `fsst::Compressor::compress_into` requires capacity for the worst-case expansion
                // (all bytes escaped) which is `2 * plaintext_len`.
                compress_buffer.reserve(bytes.len().saturating_mul(2));
                unsafe {
                    compressor.compress_into(bytes, compress_buffer);
                }

                values_buffer.extend_from_slice(compress_buffer);
            }
            offsets.push(values_buffer.len() as u32);
        }

        values_buffer.shrink_to_fit();
        let values_buffer = Buffer::from(values_buffer);
        let raw_buffer = Self::from_parts(values_buffer, uncompressed_len);

        (raw_buffer, offsets)
    }

    pub(crate) fn to_uncompressed(
        &self,
        decompressor: &Decompressor<'_>,
        offsets: &[u32],
    ) -> (Buffer, OffsetBuffer<i32>) {
        let mut value_buffer: Vec<u8> = Vec::with_capacity(self.uncompressed_bytes + 8);
        let num_values = offsets.len().saturating_sub(1);
        let mut out_offsets: OffsetBufferBuilder<i32> = OffsetBufferBuilder::new(num_values);

        for i in 0..num_values {
            let start_offset = offsets[i];
            let end_offset = offsets[i + 1];

            if start_offset != end_offset {
                let compressed_slice = self.get_compressed_slice(start_offset, end_offset);
                let decompressed_len = decompressor
                    .decompress_into(compressed_slice, value_buffer.spare_capacity_mut());

                let new_len = value_buffer.len() + decompressed_len;
                debug_assert!(new_len <= value_buffer.capacity());
                unsafe {
                    value_buffer.set_len(new_len);
                }
                out_offsets.push_length(decompressed_len);
            } else {
                out_offsets.push_length(0);
            }
        }

        let buffer = Buffer::from(value_buffer);
        (buffer, out_offsets.finish())
    }

    /// Get compressed data slice using byte offsets.
    pub(crate) fn get_compressed_slice(&self, start_offset: u32, end_offset: u32) -> &[u8] {
        let start = start_offset as usize;
        let end = end_offset as usize;
        debug_assert!(end <= self.values.len(), "Offset out of bounds");
        debug_assert!(start <= end, "Invalid offset range");
        &self.values.as_slice()[start..end]
    }

    pub(crate) fn values_len(&self) -> usize {
        self.values.len()
    }

    pub(crate) fn get_memory_size(&self) -> usize {
        self.values.len() + std::mem::size_of::<Self>()
    }

    pub(crate) fn to_bytes(&self) -> Vec<u8> {
        let mut buffer = Vec::with_capacity(self.values.len() + 12);
        buffer.extend_from_slice(&(self.uncompressed_bytes as u64).to_le_bytes());
        buffer.extend_from_slice(&(self.values.len() as u32).to_le_bytes());
        buffer.extend_from_slice(self.values.as_slice());
        buffer
    }

    pub(crate) fn uncompressed_bytes(&self) -> usize {
        self.uncompressed_bytes
    }

    pub(crate) fn from_bytes(bytes: bytes::Bytes) -> Self {
        let uncompressed_bytes = u64::from_le_bytes(bytes[0..8].try_into().unwrap()) as usize;
        let values_len = u32::from_le_bytes(bytes[8..12].try_into().unwrap()) as usize;
        let values = bytes.slice(12..12 + values_len);
        let values = Buffer::from(values);
        Self::from_parts(values, uncompressed_bytes)
    }
}

/// PrefixKey stores a small suffix fingerprint (prefix bytes + length metadata).
#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub(crate) struct PrefixKey {
    prefix7: [u8; 7],
    /// Suffix length in bytes (after shared prefix), or 255 if >= 255 / unknown.
    len: u8,
}

impl PrefixKey {
    pub(crate) const fn prefix_len() -> usize {
        7
    }

    /// Construct from the full suffix bytes (after shared prefix).
    /// Embeds up to `prefix_len()` bytes into `prefix7` and stores length (or 255 if >=255).
    pub(crate) fn new(suffix_bytes: &[u8]) -> Self {
        let mut prefix7 = [0u8; 7];
        let copy_len = std::cmp::min(Self::prefix_len(), suffix_bytes.len());
        if copy_len > 0 {
            prefix7[..copy_len].copy_from_slice(&suffix_bytes[..copy_len]);
        }
        let len = if suffix_bytes.len() >= 255 {
            255u8
        } else {
            suffix_bytes.len() as u8
        };
        Self { prefix7, len }
    }

    /// Construct directly from stored parts (used by deserialization only)
    pub(crate) fn from_parts(prefix7: [u8; 7], len: u8) -> Self {
        Self { prefix7, len }
    }

    #[inline]
    pub(crate) fn prefix7(&self) -> &[u8; 7] {
        &self.prefix7
    }

    #[inline]
    pub(crate) fn len_byte(&self) -> u8 {
        self.len
    }

    #[cfg(test)]
    pub(crate) fn known_suffix_len(&self) -> Option<usize> {
        if self.len == 255 {
            None
        } else {
            Some(self.len as usize)
        }
    }
}

const _: () = if std::mem::size_of::<PrefixKey>() != 8 {
    panic!("PrefixKey must be 8 bytes")
};

#[derive(Debug, Clone, Copy)]
struct CompactOffsetHeader {
    slope: i32,
    intercept: i32,
    offset_bytes: u8, // 1, 2, or 4 bytes per residual
}

#[derive(Debug, Clone)]
enum OffsetResiduals {
    One(Arc<[i8]>),
    Two(Arc<[i16]>),
    Four(Arc<[i32]>),
}

impl OffsetResiduals {
    fn len(&self) -> usize {
        match self {
            Self::One(values) => values.len(),
            Self::Two(values) => values.len(),
            Self::Four(values) => values.len(),
        }
    }

    #[cfg(test)]
    fn bytes_per(&self) -> usize {
        match self {
            Self::One(_) => 1,
            Self::Two(_) => 2,
            Self::Four(_) => 4,
        }
    }

    fn get_i32(&self, index: usize) -> i32 {
        match self {
            Self::One(values) => values[index] as i32,
            Self::Two(values) => values[index] as i32,
            Self::Four(values) => values[index],
        }
    }
}

/// Compact offset index for FSST dictionary values (includes the final sentinel offset).
#[derive(Debug, Clone)]
pub(crate) struct CompactOffsets {
    header: CompactOffsetHeader,
    residuals: OffsetResiduals,
}

// Proper least-squares linear regression
fn fit_line(offsets: &[u32]) -> (i32, i32) {
    let n = offsets.len();
    if n <= 1 {
        return (0, offsets.first().copied().unwrap_or(0) as i32);
    }

    let n_f64 = n as f64;

    // Sum of indices: 0 + 1 + 2 + ... + (n-1) = n*(n-1)/2
    let sum_x = (n * (n - 1) / 2) as f64;

    // Sum of offsets
    let sum_y: f64 = offsets.iter().map(|&o| o as f64).sum();

    // Sum of (index * offset)
    let sum_xy: f64 = offsets
        .iter()
        .enumerate()
        .map(|(i, &o)| i as f64 * o as f64)
        .sum();

    // Sum of index squared: 0² + 1² + 2² + ... + (n-1)² = n*(n-1)*(2n-1)/6
    let sum_x_sq = (n * (n - 1) * (2 * n - 1) / 6) as f64;

    // Least squares formulas
    let slope = (n_f64 * sum_xy - sum_x * sum_y) / (n_f64 * sum_x_sq - sum_x * sum_x);
    let intercept = (sum_y - slope * sum_x) / n_f64;

    (slope.round() as i32, intercept.round() as i32)
}

impl CompactOffsets {
    pub(crate) fn from_offsets(offsets: &[u32]) -> Self {
        if offsets.is_empty() {
            return Self {
                header: CompactOffsetHeader {
                    slope: 0,
                    intercept: 0,
                    offset_bytes: 1,
                },
                residuals: OffsetResiduals::One(Arc::new([])),
            };
        }

        let (slope, intercept) = fit_line(offsets);

        let mut offset_residuals: Vec<i32> = Vec::with_capacity(offsets.len());
        let mut min_residual = i32::MAX;
        let mut max_residual = i32::MIN;
        for (index, &offset) in offsets.iter().enumerate() {
            let predicted = slope * index as i32 + intercept;
            let residual = offset as i32 - predicted;
            offset_residuals.push(residual);
            min_residual = min_residual.min(residual);
            max_residual = max_residual.max(residual);
        }

        let offset_bytes = if min_residual >= i8::MIN as i32 && max_residual <= i8::MAX as i32 {
            1
        } else if min_residual >= i16::MIN as i32 && max_residual <= i16::MAX as i32 {
            2
        } else {
            4
        };

        let residuals = match offset_bytes {
            1 => OffsetResiduals::One(
                offset_residuals
                    .iter()
                    .map(|&r| r as i8)
                    .collect::<Vec<_>>()
                    .into(),
            ),
            2 => OffsetResiduals::Two(
                offset_residuals
                    .iter()
                    .map(|&r| r as i16)
                    .collect::<Vec<_>>()
                    .into(),
            ),
            4 => OffsetResiduals::Four(offset_residuals.into()),
            _ => unreachable!("offset_bytes must be 1, 2, or 4"),
        };

        Self {
            header: CompactOffsetHeader {
                slope,
                intercept,
                offset_bytes,
            },
            residuals,
        }
    }

    pub(crate) fn len(&self) -> usize {
        self.residuals.len()
    }

    pub(crate) fn get_offset(&self, index: usize) -> u32 {
        let predicted = self.header.slope * index as i32 + self.header.intercept;
        (predicted + self.residuals.get_i32(index)) as u32
    }

    pub(crate) fn offsets(&self) -> Vec<u32> {
        (0..self.len()).map(|i| self.get_offset(i)).collect()
    }

    pub(crate) fn memory_usage(&self) -> usize {
        let header_size = std::mem::size_of::<CompactOffsetHeader>();
        let residuals_size = match &self.residuals {
            OffsetResiduals::One(values) => values.len() * std::mem::size_of::<i8>(),
            OffsetResiduals::Two(values) => values.len() * std::mem::size_of::<i16>(),
            OffsetResiduals::Four(values) => values.len() * std::mem::size_of::<i32>(),
        };
        header_size + residuals_size
    }

    pub(crate) fn write_interleaved(&self, prefixes: &[PrefixKey], out: &mut Vec<u8>) {
        debug_assert_eq!(
            self.len(),
            prefixes.len(),
            "residuals and prefixes must have the same length"
        );

        out.extend_from_slice(&self.header.slope.to_le_bytes());
        out.extend_from_slice(&self.header.intercept.to_le_bytes());
        out.push(self.header.offset_bytes);

        match &self.residuals {
            OffsetResiduals::One(residuals) => {
                for (residual, prefix) in residuals.iter().zip(prefixes.iter()) {
                    out.push(*residual as u8);
                    out.extend_from_slice(prefix.prefix7());
                    out.push(prefix.len_byte());
                }
            }
            OffsetResiduals::Two(residuals) => {
                for (residual, prefix) in residuals.iter().zip(prefixes.iter()) {
                    out.extend_from_slice(&residual.to_le_bytes());
                    out.extend_from_slice(prefix.prefix7());
                    out.push(prefix.len_byte());
                }
            }
            OffsetResiduals::Four(residuals) => {
                for (residual, prefix) in residuals.iter().zip(prefixes.iter()) {
                    out.extend_from_slice(&residual.to_le_bytes());
                    out.extend_from_slice(prefix.prefix7());
                    out.push(prefix.len_byte());
                }
            }
        }
    }
}

pub(crate) fn empty_compact_offsets() -> CompactOffsets {
    CompactOffsets::from_offsets(&[])
}

/// Decode interleaved compact offsets + prefix keys from bytes.
pub(crate) fn decode_offset_views(bytes: &[u8]) -> (CompactOffsets, Arc<[PrefixKey]>) {
    if bytes.len() < 9 {
        panic!("CompactOffsets requires at least 9 bytes for header");
    }

    let slope = i32::from_le_bytes(bytes[0..4].try_into().unwrap());
    let intercept = i32::from_le_bytes(bytes[4..8].try_into().unwrap());
    let offset_bytes = bytes[8] as usize;
    if !matches!(offset_bytes, 1 | 2 | 4) {
        panic!("Invalid offset_bytes value: {}", offset_bytes);
    }

    let header = CompactOffsetHeader {
        slope,
        intercept,
        offset_bytes: offset_bytes as u8,
    };

    let payload = &bytes[9..];
    let entry_size = offset_bytes + std::mem::size_of::<PrefixKey>();
    if !payload.len().is_multiple_of(entry_size) {
        panic!("Invalid payload size for CompactOffsets");
    }
    let count = payload.len() / entry_size;

    let mut prefixes = Vec::with_capacity(count);
    match offset_bytes {
        1 => {
            let mut residuals = Vec::with_capacity(count);
            for i in 0..count {
                let base = i * entry_size;
                let residual = payload[base] as i8;
                let mut prefix7 = [0u8; 7];
                prefix7.copy_from_slice(&payload[base + 1..base + 8]);
                let len = payload[base + 8];
                residuals.push(residual);
                prefixes.push(PrefixKey::from_parts(prefix7, len));
            }
            (
                CompactOffsets {
                    header,
                    residuals: OffsetResiduals::One(residuals.into()),
                },
                prefixes.into(),
            )
        }
        2 => {
            let mut residuals = Vec::with_capacity(count);
            for i in 0..count {
                let base = i * entry_size;
                let residual = i16::from_le_bytes(payload[base..base + 2].try_into().unwrap());
                let mut prefix7 = [0u8; 7];
                prefix7.copy_from_slice(&payload[base + 2..base + 9]);
                let len = payload[base + 9];
                residuals.push(residual);
                prefixes.push(PrefixKey::from_parts(prefix7, len));
            }
            (
                CompactOffsets {
                    header,
                    residuals: OffsetResiduals::Two(residuals.into()),
                },
                prefixes.into(),
            )
        }
        4 => {
            let mut residuals = Vec::with_capacity(count);
            for i in 0..count {
                let base = i * entry_size;
                let residual = i32::from_le_bytes(payload[base..base + 4].try_into().unwrap());
                let mut prefix7 = [0u8; 7];
                prefix7.copy_from_slice(&payload[base + 4..base + 11]);
                let len = payload[base + 11];
                residuals.push(residual);
                prefixes.push(PrefixKey::from_parts(prefix7, len));
            }
            (
                CompactOffsets {
                    header,
                    residuals: OffsetResiduals::Four(residuals.into()),
                },
                prefixes.into(),
            )
        }
        _ => unreachable!("validated offset_bytes"),
    }
}

const SYMBOL_SIZE_BYTES: usize = std::mem::size_of::<Symbol>();

pub(crate) fn train_compressor<'a, I>(iter: I) -> Compressor
where
    I: Iterator<Item = &'a [u8]>,
{
    let strings: Vec<&[u8]> = iter.collect();
    fsst::Compressor::train(&strings)
}

/// In-memory FSST dictionary buffer that bundles compressed bytes, compact offsets, and the
/// compressor needed to (de)compress values.
#[derive(Clone)]
pub struct FsstArray {
    compressor: Arc<Compressor>,
    raw: Arc<RawFsstBuffer>,
    compact_offsets: CompactOffsets,
}

impl std::fmt::Debug for FsstArray {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FsstBuffer")
            .field("raw", &self.raw)
            .field("compact_offsets", &"<CompactOffsets>")
            .field("compressor", &"<Compressor>")
            .finish()
    }
}

impl FsstArray {
    pub(crate) fn new(
        raw: Arc<RawFsstBuffer>,
        compact_offsets: CompactOffsets,
        compressor: Arc<Compressor>,
    ) -> Self {
        Self {
            compressor,
            raw,
            compact_offsets,
        }
    }

    pub(crate) fn from_byte_offsets(
        raw: Arc<RawFsstBuffer>,
        byte_offsets: &[u32],
        compressor: Arc<Compressor>,
    ) -> Self {
        Self::new(raw, CompactOffsets::from_offsets(byte_offsets), compressor)
    }

    pub(crate) fn raw_to_bytes(&self) -> Vec<u8> {
        self.raw.to_bytes()
    }

    pub(crate) fn write_offset_views(&self, prefixes: &[PrefixKey], out: &mut Vec<u8>) {
        self.compact_offsets.write_interleaved(prefixes, out)
    }

    /// Trains a compressor on a sequence of strings.
    pub fn train_compressor<'a>(input: impl Iterator<Item = &'a [u8]>) -> Compressor {
        train_compressor(input)
    }

    /// Creates a new FSST buffer from a GenericByteArray and a compressor.
    pub fn from_byte_array_with_compressor<T: ByteArrayType>(
        input: &GenericByteArray<T>,
        compressor: Arc<Compressor>,
    ) -> Self {
        let iter = input.iter();
        let mut compress_buffer = Vec::with_capacity(2 * 1024 * 1024);
        let (raw, offsets) =
            RawFsstBuffer::from_byte_slices(iter, compressor.clone(), &mut compress_buffer);
        Self::from_byte_offsets(Arc::new(raw), &offsets, compressor)
    }

    /// Creates a new FSST buffer from a Decimal128Array and a compressor.
    pub fn from_decimal128_array_with_compressor(
        array: &Decimal128Array,
        compressor: Arc<Compressor>,
    ) -> Self {
        let iter = array.iter().map(|v| v.map(|v| v.to_le_bytes()));
        let mut compress_buffer = Vec::with_capacity(64);
        let (raw, offsets) =
            RawFsstBuffer::from_byte_slices(iter, compressor.clone(), &mut compress_buffer);
        Self::from_byte_offsets(Arc::new(raw), &offsets, compressor)
    }

    /// Creates a new FSST buffer from a Decimal256Array and a compressor.
    pub fn from_decimal256_array_with_compressor(
        array: &Decimal256Array,
        compressor: Arc<Compressor>,
    ) -> Self {
        let iter = array.iter().map(|v| v.map(|v| v.to_le_bytes()));
        let mut compress_buffer = Vec::with_capacity(128);
        let (raw, offsets) =
            RawFsstBuffer::from_byte_slices(iter, compressor.clone(), &mut compress_buffer);
        Self::from_byte_offsets(Arc::new(raw), &offsets, compressor)
    }

    /// Decompress all values and return Arrow buffers (values + offsets).
    pub fn to_uncompressed(&self) -> SqueezeResult<(Buffer, OffsetBuffer<i32>)> {
        <Self as FsstBacking>::to_uncompressed(self)
    }

    /// Get a compressed value slice by index.
    pub fn get_compressed_slice(&self, index: usize) -> SqueezeResult<&[u8]> {
        <Self as FsstBacking>::get_compressed_slice(self, index)
    }

    /// Decompress selected values into Arrow buffers, preserving selection order (and duplicates).
    pub fn to_uncompressed_selected(
        &self,
        selected: &[usize],
    ) -> SqueezeResult<(Buffer, OffsetBuffer<i32>)> {
        <Self as FsstBacking>::to_uncompressed_selected(self, selected)
    }

    /// Returns the uncompressed byte size of this buffer.
    pub fn uncompressed_bytes(&self) -> usize {
        <Self as FsstBacking>::uncompressed_bytes(self)
    }

    /// Returns the in-memory size of this buffer.
    pub fn get_array_memory_size(&self) -> usize {
        <Self as FsstBacking>::get_array_memory_size(self)
    }

    /// Returns the number of values in this buffer.
    #[allow(clippy::len_without_is_empty)]
    pub fn len(&self) -> usize {
        self.compact_offsets.len().saturating_sub(1)
    }

    /// Returns a decompressor for this buffer.
    pub fn decompressor(&self) -> Decompressor<'_> {
        self.compressor.decompressor()
    }

    /// Returns a reference to the compressor.
    pub fn compressor(&self) -> &Compressor {
        &self.compressor
    }

    /// Serializes this FSST buffer (raw bytes + compact offsets) to `out`.
    pub fn to_bytes(&self, out: &mut Vec<u8>) {
        out.extend_from_slice(&self.raw.to_bytes());
        self.compact_offsets.write_residuals(out);
    }

    /// Deserializes a FSST buffer from the `to_bytes()` format.
    pub fn from_bytes(bytes: bytes::Bytes, compressor: Arc<Compressor>) -> Self {
        if bytes.len() < 12 {
            panic!("Input buffer too small for RawFsstBuffer header");
        }

        let raw_values_len = u32::from_le_bytes(bytes[8..12].try_into().unwrap()) as usize;
        let raw_len = 12 + raw_values_len;
        if raw_len > bytes.len() {
            panic!("RawFsstBuffer extends beyond input buffer");
        }

        let raw = RawFsstBuffer::from_bytes(bytes.slice(0..raw_len));
        let compact = decode_compact_offsets(&bytes[raw_len..]);

        if compact.len() > 0 {
            let last = compact.get_offset(compact.len().saturating_sub(1)) as usize;
            debug_assert_eq!(
                last,
                raw.values_len(),
                "offsets must end at raw values length"
            );
        }

        Self::new(Arc::new(raw), compact, compressor)
    }

    /// Decompress all values into an Arrow byte array.
    pub fn to_arrow_byte_array<T: ByteArrayType<Offset = i32>>(&self) -> GenericByteArray<T> {
        let (value_buffer, offsets) = self
            .to_uncompressed()
            .expect("in-memory FSST buffer must have backing");
        unsafe { GenericByteArray::<T>::new_unchecked(offsets, value_buffer, None) }
    }

    fn decompress_as_fixed_size_binary(&self, value_width: usize) -> Vec<u8> {
        let decompressor = self.compressor.decompressor();
        let mut value_buffer: Vec<u8> = Vec::with_capacity(self.len() * value_width + 8);

        for i in 0..self.len() {
            let compressed = self
                .get_compressed_slice(i)
                .expect("in-memory FSST buffer must have backing");
            let required = decompressor.max_decompression_capacity(compressed) + 8;
            value_buffer.reserve(required);
            let len = decompressor.decompress_into(compressed, value_buffer.spare_capacity_mut());
            debug_assert!(len == value_width);
            let new_len = value_buffer.len() + len;
            unsafe {
                value_buffer.set_len(new_len);
            }
        }
        value_buffer
    }

    fn to_decimal_array_inner(&self, data_type: &ArrowFixedLenByteArrayType) -> Buffer {
        let value_width = data_type.value_width();
        Buffer::from(self.decompress_as_fixed_size_binary(value_width))
    }

    /// Converts this FSST buffer to a Decimal128Array.
    pub fn to_decimal128_array(&self, data_type: &ArrowFixedLenByteArrayType) -> Decimal128Array {
        let value_buffer = self.to_decimal_array_inner(data_type);
        let array_builder = ArrayDataBuilder::new(data_type.into())
            .len(self.len())
            .add_buffer(value_buffer);
        let array_data = unsafe { array_builder.build_unchecked() };
        Decimal128Array::from(array_data)
    }

    /// Converts this FSST buffer to a Decimal256Array.
    pub fn to_decimal256_array(&self, data_type: &ArrowFixedLenByteArrayType) -> Decimal256Array {
        let value_buffer = self.to_decimal_array_inner(data_type);
        let array_builder = ArrayDataBuilder::new(data_type.into())
            .len(self.len())
            .add_buffer(value_buffer);
        let array_data = unsafe { array_builder.build_unchecked() };
        Decimal256Array::from(array_data)
    }

    #[cfg(test)]
    pub(crate) fn offsets_len(&self) -> usize {
        self.compact_offsets.len()
    }

    #[cfg(test)]
    pub(crate) fn offset_bytes(&self) -> u8 {
        self.compact_offsets.header.offset_bytes
    }

    #[cfg(test)]
    pub(crate) fn offsets(&self) -> Vec<u32> {
        self.compact_offsets.offsets()
    }
}

/// Disk buffer for FSST buffer.
#[derive(Debug, Clone)]
pub struct DiskBuffer {
    uncompressed_bytes: usize,
}

impl DiskBuffer {
    pub(crate) fn new(uncompressed_bytes: usize) -> Self {
        Self { uncompressed_bytes }
    }
}

/// FSST backing store for `LiquidByteViewArray` (in-memory or disk-only handle).
pub trait FsstBacking: std::fmt::Debug + Clone + sealed::Sealed {
    /// Decompress all dictionary values and return Arrow buffers (values + offsets).
    fn to_uncompressed(&self) -> SqueezeResult<(Buffer, OffsetBuffer<i32>)>;

    /// Get a compressed dictionary value slice by dictionary index.
    fn get_compressed_slice(&self, dict_index: usize) -> SqueezeResult<&[u8]>;

    /// Decompress a subset of dictionary values, preserving `selected` order (and duplicates).
    fn to_uncompressed_selected(
        &self,
        selected: &[usize],
    ) -> SqueezeResult<(Buffer, OffsetBuffer<i32>)>;

    /// Get the uncompressed bytes of the FSST buffer (used for sizing / squeeze bookkeeping).
    fn uncompressed_bytes(&self) -> usize;

    /// Get the in-memory size of the FSST backing (raw bytes + any in-memory indices).
    fn get_array_memory_size(&self) -> usize;
}

impl sealed::Sealed for FsstArray {}
impl sealed::Sealed for DiskBuffer {}

impl FsstBacking for FsstArray {
    fn to_uncompressed(&self) -> SqueezeResult<(Buffer, OffsetBuffer<i32>)> {
        let offsets = self.compact_offsets.offsets();
        Ok(self
            .raw
            .to_uncompressed(&self.compressor.decompressor(), &offsets))
    }

    fn get_compressed_slice(&self, dict_index: usize) -> SqueezeResult<&[u8]> {
        let start_offset = self.compact_offsets.get_offset(dict_index);
        let end_offset = self.compact_offsets.get_offset(dict_index + 1);
        Ok(self.raw.get_compressed_slice(start_offset, end_offset))
    }

    fn to_uncompressed_selected(
        &self,
        selected: &[usize],
    ) -> SqueezeResult<(Buffer, OffsetBuffer<i32>)> {
        let decompressor = self.compressor.decompressor();
        let mut value_buffer: Vec<u8> = Vec::new();
        let mut out_offsets: OffsetBufferBuilder<i32> = OffsetBufferBuilder::new(selected.len());

        for &dict_index in selected {
            let start_offset = self.compact_offsets.get_offset(dict_index);
            let end_offset = self.compact_offsets.get_offset(dict_index + 1);

            if start_offset == end_offset {
                out_offsets.push_length(0);
                continue;
            }

            let compressed_value = self.raw.get_compressed_slice(start_offset, end_offset);
            let required = decompressor.max_decompression_capacity(compressed_value) + 7;
            value_buffer.reserve(required);
            let decompressed_len =
                decompressor.decompress_into(compressed_value, value_buffer.spare_capacity_mut());
            let new_len = value_buffer.len() + decompressed_len;
            debug_assert!(new_len <= value_buffer.capacity());
            unsafe {
                value_buffer.set_len(new_len);
            }
            out_offsets.push_length(decompressed_len);
        }

        Ok((Buffer::from(value_buffer), out_offsets.finish()))
    }

    fn uncompressed_bytes(&self) -> usize {
        self.raw.uncompressed_bytes()
    }

    fn get_array_memory_size(&self) -> usize {
        self.raw.get_memory_size()
            + self.compact_offsets.memory_usage()
            + std::mem::size_of::<Self>()
    }
}

impl FsstBacking for DiskBuffer {
    fn to_uncompressed(&self) -> SqueezeResult<(Buffer, OffsetBuffer<i32>)> {
        Err(NeedsBacking)
    }

    fn get_compressed_slice(&self, _dict_index: usize) -> SqueezeResult<&[u8]> {
        Err(NeedsBacking)
    }

    fn to_uncompressed_selected(
        &self,
        _selected: &[usize],
    ) -> SqueezeResult<(Buffer, OffsetBuffer<i32>)> {
        Err(NeedsBacking)
    }

    fn uncompressed_bytes(&self) -> usize {
        self.uncompressed_bytes
    }

    fn get_array_memory_size(&self) -> usize {
        0
    }
}

impl CompactOffsets {
    fn write_residuals(&self, out: &mut Vec<u8>) {
        out.extend_from_slice(&self.header.slope.to_le_bytes());
        out.extend_from_slice(&self.header.intercept.to_le_bytes());
        out.push(self.header.offset_bytes);

        match &self.residuals {
            OffsetResiduals::One(residuals) => {
                out.extend(residuals.iter().map(|r| *r as u8));
            }
            OffsetResiduals::Two(residuals) => {
                for r in residuals.iter() {
                    out.extend_from_slice(&r.to_le_bytes());
                }
            }
            OffsetResiduals::Four(residuals) => {
                for r in residuals.iter() {
                    out.extend_from_slice(&r.to_le_bytes());
                }
            }
        }
    }
}

fn decode_compact_offsets(bytes: &[u8]) -> CompactOffsets {
    if bytes.len() < 9 {
        panic!("CompactOffsets requires at least 9 bytes for header");
    }

    let slope = i32::from_le_bytes(bytes[0..4].try_into().unwrap());
    let intercept = i32::from_le_bytes(bytes[4..8].try_into().unwrap());
    let offset_bytes = bytes[8] as usize;
    if !matches!(offset_bytes, 1 | 2 | 4) {
        panic!("Invalid offset_bytes value: {}", offset_bytes);
    }

    let header = CompactOffsetHeader {
        slope,
        intercept,
        offset_bytes: offset_bytes as u8,
    };

    let payload = &bytes[9..];
    if !payload.len().is_multiple_of(offset_bytes) {
        panic!("Invalid payload size for CompactOffsets");
    }
    let count = payload.len() / offset_bytes;

    match offset_bytes {
        1 => {
            let residuals: Arc<[i8]> = payload.iter().map(|b| *b as i8).collect::<Vec<_>>().into();
            CompactOffsets {
                header,
                residuals: OffsetResiduals::One(residuals),
            }
        }
        2 => {
            let mut residuals = Vec::with_capacity(count);
            for i in 0..count {
                let base = i * 2;
                residuals.push(i16::from_le_bytes(
                    payload[base..base + 2].try_into().unwrap(),
                ));
            }
            CompactOffsets {
                header,
                residuals: OffsetResiduals::Two(residuals.into()),
            }
        }
        4 => {
            let mut residuals = Vec::with_capacity(count);
            for i in 0..count {
                let base = i * 4;
                residuals.push(i32::from_le_bytes(
                    payload[base..base + 4].try_into().unwrap(),
                ));
            }
            CompactOffsets {
                header,
                residuals: OffsetResiduals::Four(residuals.into()),
            }
        }
        _ => unreachable!("validated offset_bytes"),
    }
}

/// Saves symbol table from the compressor to a buffer.
///
/// Format:
/// 1. The first byte is the length of the symbol table as a u8.
/// 2. The next bytes are the lengths of each symbol as u8.
/// 3. The next bytes are the symbols as u64.
pub fn save_symbol_table(compressor: Arc<Compressor>, buffer: &mut Vec<u8>) -> Result<()> {
    let symbols = compressor.symbol_table();
    let symbols_lengths = compressor.symbol_lengths();

    if symbols.len() != symbols_lengths.len() {
        return Err(Error::new(
            ErrorKind::InvalidInput,
            "Symbol table and symbol lengths have different lengths",
        ));
    }

    if symbols.len() > u8::MAX as usize {
        return Err(Error::new(
            ErrorKind::InvalidInput,
            "Symbol table too large",
        ));
    }

    buffer.push(symbols.len() as u8);

    for &len in symbols_lengths.iter() {
        buffer.push(len);
    }

    for sym in symbols.iter() {
        buffer.extend_from_slice(&sym.to_u64().to_le_bytes());
    }

    Ok(())
}

/// Loads symbol table from a buffer saved by `save_symbol_table`.
pub fn load_symbol_table(data: bytes::Bytes) -> Result<Compressor> {
    if data.is_empty() {
        return Err(Error::new(ErrorKind::InvalidInput, "Empty symbol table"));
    }

    let symbol_count = data[0] as usize;
    let lengths_start = 1;
    let lengths_end = lengths_start + symbol_count;
    if lengths_end > data.len() {
        return Err(Error::new(
            ErrorKind::InvalidInput,
            "Buffer too small for symbol lengths",
        ));
    }

    let lengths = &data[lengths_start..lengths_end];
    let symbols_start = lengths_end;
    let symbols_end = symbols_start + symbol_count * SYMBOL_SIZE_BYTES;
    if symbols_end > data.len() {
        return Err(Error::new(
            ErrorKind::InvalidInput,
            "Buffer too small for symbols",
        ));
    }

    let mut symbols = Vec::with_capacity(symbol_count);
    for i in 0..symbol_count {
        let base = symbols_start + i * SYMBOL_SIZE_BYTES;
        let bytes: [u8; SYMBOL_SIZE_BYTES] =
            data[base..base + SYMBOL_SIZE_BYTES].try_into().unwrap();
        symbols.push(Symbol::from_slice(&bytes));
    }

    Ok(fsst::Compressor::rebuild_from(symbols, lengths))
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::{
        array::{Array, Decimal128Builder, StringBuilder},
        datatypes::DataType,
    };

    #[test]
    fn test_compact_offset_view_round_trip() {
        // Test 1: Small offsets (should use OneByte variant)
        let small_offsets = vec![100u32, 105, 110];
        let small_prefixes = vec![
            PrefixKey::new(b"hello"),
            PrefixKey::new(b"world"),
            PrefixKey::new(b"test"),
        ];
        test_round_trip(&small_offsets, &small_prefixes, "small offsets");

        // Test 2: Medium offsets (should use TwoBytes variant)
        let medium_offsets = vec![1000u32, 2000, 3000];
        let medium_prefixes = vec![
            PrefixKey::new(b"medium1"),
            PrefixKey::new(b"medium2"),
            PrefixKey::new(b"medium3"),
        ];
        test_round_trip(&medium_offsets, &medium_prefixes, "medium offsets");

        // Test 3: Large offsets (should use FourBytes variant)
        let large_offsets = vec![100000u32, 200000, 300000];
        let large_prefixes = vec![
            PrefixKey::new(b"large1"),
            PrefixKey::new(b"large2"),
            PrefixKey::new(b"large3"),
        ];
        test_round_trip(&large_offsets, &large_prefixes, "large offsets");

        // Test 4: Mixed scenario with varying prefix lengths
        let mixed_offsets = vec![1000u32, 1010, 1020, 1030, 1040];
        let mixed_prefixes = vec![
            PrefixKey::new(b"a"),             // 1 byte prefix
            PrefixKey::new(b"abcdef"),        // 6 byte prefix
            PrefixKey::new(b"abcdefg"),       // 7 byte prefix (max)
            PrefixKey::new(b"abcdefgh"),      // 8 bytes (7 stored + len)
            PrefixKey::new(&vec![b'x'; 300]), // 300 bytes (long string, len=255)
        ];
        test_round_trip(&mixed_offsets, &mixed_prefixes, "mixed scenarios");

        // Test 5: Edge case - empty offsets
        let empty_offsets: Vec<u32> = vec![];
        let empty_prefixes: Vec<PrefixKey> = vec![];
        test_round_trip(&empty_offsets, &empty_prefixes, "empty offsets");

        // Test 6: Single offset
        let single_offset = vec![42u32];
        let single_prefixes = vec![PrefixKey::new(b"single")];
        test_round_trip(&single_offset, &single_prefixes, "single offset");
    }

    fn test_round_trip(offsets: &[u32], prefixes: &[PrefixKey], test_name: &str) {
        let compact_offsets = CompactOffsets::from_offsets(offsets);

        assert_eq!(
            offsets.len(),
            compact_offsets.len(),
            "Length mismatch in {}",
            test_name
        );
        for (i, offset) in offsets.iter().enumerate() {
            assert_eq!(
                compact_offsets.get_offset(i),
                *offset,
                "Offset mismatch at index {} in {}",
                i,
                test_name
            );
        }

        let mut bytes = Vec::new();
        compact_offsets.write_interleaved(prefixes, &mut bytes);
        let (reconstructed, reconstructed_prefixes) = decode_offset_views(&bytes);

        assert_eq!(
            offsets.len(),
            reconstructed.len(),
            "Reconstructed length mismatch in {}",
            test_name
        );
        assert_eq!(
            prefixes.len(),
            reconstructed_prefixes.len(),
            "Reconstructed prefix length mismatch in {}",
            test_name
        );
        for i in 0..offsets.len() {
            assert_eq!(offsets[i], reconstructed.get_offset(i));
            assert_eq!(prefixes[i].prefix7(), reconstructed_prefixes[i].prefix7());
            assert_eq!(prefixes[i].len_byte(), reconstructed_prefixes[i].len_byte());
        }
    }

    #[test]
    fn test_compact_offset_view_memory_efficiency() {
        // test that compaction actually saves memory
        let offsets = vec![1000u32, 1010, 1020, 1030];
        let prefixes = [
            PrefixKey::new(b"test1"),
            PrefixKey::new(b"test2"),
            PrefixKey::new(b"test3"),
            PrefixKey::new(b"test4"),
        ];

        let original_size =
            offsets.len() * (std::mem::size_of::<u32>() + std::mem::size_of::<PrefixKey>());
        let compact_offsets = CompactOffsets::from_offsets(&offsets);
        let compact_size =
            compact_offsets.memory_usage() + prefixes.len() * std::mem::size_of::<PrefixKey>();

        // for this test case, we should see some savings due to using smaller residuals
        assert!(
            compact_size <= original_size,
            "Compact representation should not be larger"
        );
    }

    #[test]
    fn test_compact_offset_view_struct_methods() {
        let key = PrefixKey::from_parts([1, 2, 3, 4, 5, 6, 7], 15);
        assert_eq!(key.prefix7(), &[1, 2, 3, 4, 5, 6, 7]);
        assert_eq!(key.len_byte(), 15);
        assert_eq!(key.known_suffix_len(), Some(15));

        let unknown = PrefixKey::from_parts([7, 6, 5, 4, 3, 2, 1], 255);
        assert_eq!(unknown.len_byte(), 255);
        assert_eq!(unknown.known_suffix_len(), None);

        let r1 = OffsetResiduals::One(vec![-42i8, 7].into());
        assert_eq!(r1.bytes_per(), 1);
        assert_eq!(r1.get_i32(0), -42);

        let r2 = OffsetResiduals::Two(vec![12345i16].into());
        assert_eq!(r2.bytes_per(), 2);
        assert_eq!(r2.get_i32(0), 12345);

        let r4 = OffsetResiduals::Four(vec![-1000000i32].into());
        assert_eq!(r4.bytes_per(), 4);
        assert_eq!(r4.get_i32(0), -1000000);

        assert_eq!(PrefixKey::prefix_len(), 7);
    }

    #[test]
    fn test_compact_offset_view_group_from_bytes_errors() {
        // Test with insufficient bytes for header
        let short_bytes = vec![1, 2, 3]; // only 3 bytes, need at least 9
        let result = std::panic::catch_unwind(|| decode_offset_views(&short_bytes));
        assert!(result.is_err(), "Should panic with insufficient bytes");

        // Test with invalid offset_bytes value
        let mut invalid_header = vec![0; 9];
        invalid_header[8] = 3; // invalid offset_bytes (should be 1, 2, or 4)
        let result = std::panic::catch_unwind(|| decode_offset_views(&invalid_header));
        assert!(result.is_err(), "Should panic with invalid offset_bytes");

        // Test with misaligned residual data for OneByte variant
        let mut misaligned_one_byte = vec![0; 9 + 8]; // header + incomplete residual
        misaligned_one_byte[8] = 1; // offset_bytes = 1
        let result = std::panic::catch_unwind(|| decode_offset_views(&misaligned_one_byte));
        assert!(
            result.is_err(),
            "Should panic with misaligned OneByte residuals"
        );

        // Test with misaligned residual data for TwoBytes variant
        let mut misaligned_two_bytes = vec![0; 9 + 9]; // header + incomplete residual
        misaligned_two_bytes[8] = 2; // offset_bytes = 2
        let result = std::panic::catch_unwind(|| decode_offset_views(&misaligned_two_bytes));
        assert!(
            result.is_err(),
            "Should panic with misaligned TwoBytes residuals"
        );

        // Test with misaligned residual data for FourBytes variant
        let mut misaligned_four_bytes = vec![0; 9 + 11]; // header + incomplete residual
        misaligned_four_bytes[8] = 4; // offset_bytes = 4
        let result = std::panic::catch_unwind(|| decode_offset_views(&misaligned_four_bytes));
        assert!(
            result.is_err(),
            "Should panic with misaligned FourBytes residuals"
        );
    }

    #[test]
    fn test_compact_offset_view_group_from_bytes_valid() {
        // Test OneByte variant roundtrip
        let offsets = vec![100u32, 101];
        let prefixes = vec![
            PrefixKey::from_parts([1, 2, 3, 4, 5, 6, 7], 10),
            PrefixKey::from_parts([7, 6, 5, 4, 3, 2, 1], 20),
        ];
        let original = CompactOffsets::from_offsets(&offsets);

        let mut bytes = Vec::new();
        original.write_interleaved(&prefixes, &mut bytes);

        let (reconstructed, reconstructed_prefixes) = decode_offset_views(&bytes);

        // Verify they match
        assert_eq!(offsets.len(), reconstructed.len());
        for i in 0..offsets.len() {
            assert_eq!(offsets[i], reconstructed.get_offset(i));
            assert_eq!(prefixes[i].prefix7(), reconstructed_prefixes[i].prefix7());
            assert_eq!(prefixes[i].len_byte(), reconstructed_prefixes[i].len_byte());
        }
    }

    #[test]
    fn test_fsst_buffer_bytes_roundtrip() {
        let mut builder = StringBuilder::new();
        for i in 0..1000 {
            builder.append_value(format!("test string value {i}"));
        }
        let original = builder.finish();

        let compressor =
            FsstArray::train_compressor(original.iter().flat_map(|s| s.map(|s| s.as_bytes())));
        let compressor_arc = Arc::new(compressor);
        let original_fsst =
            FsstArray::from_byte_array_with_compressor(&original, compressor_arc.clone());

        let mut buffer = Vec::new();
        original_fsst.to_bytes(&mut buffer);

        let bytes = bytes::Bytes::from(buffer);
        let deserialized = FsstArray::from_bytes(bytes, compressor_arc);

        let original_strings = original_fsst.to_arrow_byte_array::<arrow::datatypes::Utf8Type>();
        let deserialized_strings = deserialized.to_arrow_byte_array::<arrow::datatypes::Utf8Type>();
        assert_eq!(original_strings.len(), deserialized_strings.len());
        for (orig, deser) in original_strings.iter().zip(deserialized_strings.iter()) {
            assert_eq!(orig, deser);
        }
    }

    #[test]
    fn test_decimal_compression_smoke() {
        let mut builder = Decimal128Builder::new().with_data_type(DataType::Decimal128(10, 2));
        for i in 0..4096 {
            builder.append_value(i128::from_le_bytes([(i % 16) as u8; 16]));
        }
        let original = builder.finish();
        let original_size = original.get_array_memory_size();

        let values = original
            .iter()
            .filter_map(|v| v.map(|v| v.to_le_bytes()))
            .collect::<Vec<_>>();
        let compressor = FsstArray::train_compressor(values.iter().map(|b| b.as_slice()));
        let compressor_arc = Arc::new(compressor);

        let fsst = FsstArray::from_decimal128_array_with_compressor(&original, compressor_arc);
        let compressed_size = fsst.get_array_memory_size();
        assert!(compressed_size < original_size);
    }

    #[test]
    fn test_save_and_load_symbol_table_roundtrip() {
        let mut builder = StringBuilder::new();
        for i in 0..1000 {
            builder.append_value(format!("hello world {i}"));
        }
        let original = builder.finish();

        let compressor =
            FsstArray::train_compressor(original.iter().flat_map(|s| s.map(|s| s.as_bytes())));
        let compressor_arc = Arc::new(compressor);

        let mut bytes = Vec::new();
        save_symbol_table(compressor_arc.clone(), &mut bytes).unwrap();
        let reloaded = load_symbol_table(bytes::Bytes::from(bytes)).unwrap();

        let fsst_original = FsstArray::from_byte_array_with_compressor(&original, compressor_arc);
        let fsst_reloaded =
            FsstArray::from_byte_array_with_compressor(&original, Arc::new(reloaded));

        let a = fsst_original.to_arrow_byte_array::<arrow::datatypes::Utf8Type>();
        let b = fsst_reloaded.to_arrow_byte_array::<arrow::datatypes::Utf8Type>();
        assert_eq!(a, b);
    }
}
