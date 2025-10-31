//! LiquidByteViewArray

use arrow::array::BinaryViewArray;
use arrow::array::{
    Array, ArrayAccessor, ArrayIter, ArrayRef, BinaryArray, BooleanArray, BooleanBuilder,
    DictionaryArray, GenericByteArray, StringArray, StringViewArray, UInt16Array, UInt32Array,
    cast::AsArray, types::UInt16Type,
};
use arrow::buffer::{BooleanBuffer, NullBuffer};
use arrow::compute::{cast, kernels, sort_to_indices};
use arrow::datatypes::ByteArrayType;
use arrow_schema::DataType;
use bytes::Bytes;
use datafusion::logical_expr::{ColumnarValue, Operator};
use datafusion::physical_expr_common::datum::apply_cmp;
use datafusion::physical_plan::PhysicalExpr;
use datafusion::physical_plan::expressions::{BinaryExpr, LikeExpr, Literal};
use fsst::Compressor;
use std::any::Any;
use std::fmt::Display;
use std::ops::Range;
use std::sync::Arc;

#[cfg(test)]
use std::cell::Cell;

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

use super::{
    LiquidArray, LiquidArrayRef, LiquidDataType, LiquidHybridArray,
    byte_array::{ArrowByteType, get_string_needle},
};
use crate::liquid_array::ipc::LiquidIPCHeader;
use crate::liquid_array::raw::BitPackedArray;
use crate::liquid_array::raw::fsst_array::{RawFsstBuffer, train_compressor};
use crate::liquid_array::{IoRange, LiquidHybridArrayRef};
use crate::utils::CheckedDictionaryArray;

// Header for LiquidByteViewArray serialization
#[repr(C)]
struct ByteViewArrayHeader {
    keys_size: u32,
    offset_views_size: u32,
    shared_prefix_size: u32,
    fsst_raw_size: u32,
}

impl ByteViewArrayHeader {
    const fn size() -> usize {
        const _: () =
            assert!(std::mem::size_of::<ByteViewArrayHeader>() == ByteViewArrayHeader::size());
        16
    }

    fn to_bytes(&self) -> [u8; Self::size()] {
        let mut bytes = [0u8; Self::size()];
        bytes[0..4].copy_from_slice(&self.keys_size.to_le_bytes());
        bytes[4..8].copy_from_slice(&self.offset_views_size.to_le_bytes());
        bytes[8..12].copy_from_slice(&self.shared_prefix_size.to_le_bytes());
        bytes[12..16].copy_from_slice(&self.fsst_raw_size.to_le_bytes());
        bytes
    }

    fn from_bytes(bytes: &[u8]) -> Self {
        if bytes.len() < Self::size() {
            panic!(
                "value too small for ByteViewArrayHeader, expected at least {} bytes, got {}",
                Self::size(),
                bytes.len()
            );
        }
        let keys_size = u32::from_le_bytes(bytes[0..4].try_into().unwrap());
        let offset_views_size = u32::from_le_bytes(bytes[4..8].try_into().unwrap());
        let shared_prefix_size = u32::from_le_bytes(bytes[8..12].try_into().unwrap());
        let fsst_raw_size = u32::from_le_bytes(bytes[12..16].try_into().unwrap());
        Self {
            keys_size,
            offset_views_size,
            shared_prefix_size,
            fsst_raw_size,
        }
    }
}

fn align_up_8(len: usize) -> usize {
    (len + 7) & !7
}

impl<B: FsstBuffer> LiquidByteViewArray<B> {
    /*
    Serialized LiquidByteViewArray Memory Layout:

    +--------------------------------------------------+
    | LiquidIPCHeader (16 bytes)                       |
    +--------------------------------------------------+
    | ByteViewArrayHeader (16 bytes)                   |  // keys_size, offsets_size, prefix_size, fsst_size
    +--------------------------------------------------+
    | Padding (to 8-byte alignment)                    |
    +--------------------------------------------------+
    | RawFsstBuffer bytes                              |
    +--------------------------------------------------+
    | Padding (to 8-byte alignment)                    |
    +--------------------------------------------------+
    | BitPackedArray Data (dictionary_keys)            |
    +--------------------------------------------------+
    | [BitPackedArray Header & Values]                 |
    +--------------------------------------------------+
    | Padding (to 8-byte alignment)                    |
    +--------------------------------------------------+
    | CompactOffsetViewGroup bytes (header + residuals)|
    +--------------------------------------------------+
    | Padding (to 8-byte alignment)                    |
    +--------------------------------------------------+
    | Shared prefix bytes                              |
    +--------------------------------------------------+
    */
    pub(crate) fn to_bytes_inner(&self) -> Result<Vec<u8>, IoRange> {
        let header_size = LiquidIPCHeader::size() + ByteViewArrayHeader::size();
        let mut result = Vec::with_capacity(header_size + 1024);
        result.resize(header_size, 0);

        // A) Align and serialize RawFsstBuffer first (near the start)
        while !result.len().is_multiple_of(8) {
            result.push(0);
        }
        let fsst_start = result.len();
        let fsst_raw_bytes = {
            let raw = self.fsst_buffer.get_fsst_buffer()?;
            raw.to_bytes()
        };
        result.extend_from_slice(&fsst_raw_bytes);
        let fsst_raw_size = result.len() - fsst_start;

        // B) Alignment before keys
        while !result.len().is_multiple_of(8) {
            result.push(0);
        }

        // C) Serialize dictionary keys
        let keys_start = result.len();
        {
            use std::num::NonZero;
            let bit_packed = BitPackedArray::<UInt16Type>::from_primitive(
                self.dictionary_keys.clone(),
                NonZero::new(16).unwrap(),
            );
            bit_packed.to_bytes(&mut result);
        }
        let keys_size = result.len() - keys_start;

        // D) Alignment before offset views
        while !result.len().is_multiple_of(8) {
            result.push(0);
        }

        // e) Serialize compact offset views (header + residuals)
        let offsets_start = result.len();
        {
            let header = self.compact_offset_views.header();
            // serialize header
            result.extend_from_slice(&header.slope.to_le_bytes());
            result.extend_from_slice(&header.intercept.to_le_bytes());
            result.push(header.offset_bytes);
            
            // serialize residuals based on type
            match &self.compact_offset_views {
                CompactOffsetViewGroup::OneByte { residuals, .. } => {
                    for residual in residuals.iter() {
                        result.push(residual.offset_residual() as u8);
                        result.extend_from_slice(residual.prefix7());
                        result.push(residual.len_byte());
                    }
                },
                CompactOffsetViewGroup::TwoBytes { residuals, .. } => {
                    for residual in residuals.iter() {
                        result.extend_from_slice(&residual.offset_residual().to_le_bytes());
                        result.extend_from_slice(residual.prefix7());
                        result.push(residual.len_byte());
                    }
                },
                CompactOffsetViewGroup::FourBytes { residuals, .. } => {
                    for residual in residuals.iter() {
                        result.extend_from_slice(&residual.offset_residual().to_le_bytes());
                        result.extend_from_slice(residual.prefix7());
                        result.push(residual.len_byte());
                    }
                },
            }
        }
        let offset_views_size = result.len() - offsets_start;

        // F) Alignment before shared prefix
        while !result.len().is_multiple_of(8) {
            result.push(0);
        }

        // G) Serialize shared prefix
        let prefix_start = result.len();
        result.extend_from_slice(&self.shared_prefix);
        let shared_prefix_size = result.len() - prefix_start;

        // Prepare headers
        let ipc = LiquidIPCHeader::new(
            LiquidDataType::ByteViewArray as u16,
            self.original_arrow_type as u16,
        );
        let view_header = ByteViewArrayHeader {
            keys_size: keys_size as u32,
            offset_views_size: offset_views_size as u32,
            shared_prefix_size: shared_prefix_size as u32,
            fsst_raw_size: fsst_raw_size as u32,
        };

        // Write headers into reserved space at start
        let header_slice = &mut result[0..header_size];
        header_slice[0..LiquidIPCHeader::size()].copy_from_slice(&ipc.to_bytes());
        header_slice[LiquidIPCHeader::size()..header_size].copy_from_slice(&view_header.to_bytes());

        Ok(result)
    }

    /// Create LiquidByteViewArray from parts
    pub(crate) fn from_parts(
        dictionary_keys: UInt16Array,
        offset_views: &[OffsetView],
        fsst_buffer: B,
        original_arrow_type: ArrowByteType,
        shared_prefix: Vec<u8>,
        compressor: Arc<Compressor>,
    ) -> Self {
        let offset_views = Arc::<[OffsetView]>::from(offset_views.to_vec());

        let compact_offset_views = CompactOffsetViewGroup::from_offset_views(&offset_views);
        
        Self {
            dictionary_keys,
            compact_offset_views,
            fsst_buffer,
            original_arrow_type,
            shared_prefix,
            compressor,
        }
    }

    pub fn offset_views(&self) -> Vec<OffsetView> {
        let mut offset_views: Vec<OffsetView> = Vec::new();
        
        let header = self.compact_offset_views.header();

        match &self.compact_offset_views {
            CompactOffsetViewGroup::OneByte { residuals, .. } => {
                for (index, residual_offset) in residuals.iter().enumerate() {
                    let predicted = header.slope * index as i32 + header.intercept;
                    let offset = (predicted + residual_offset.offset_residual() as i32) as u32;
                    let offset_view = OffsetView {
                        offset,
                        prefix7: *residual_offset.prefix7(),
                        len: residual_offset.len_byte(),
                    };
                    offset_views.push(offset_view);
                }
            },
            CompactOffsetViewGroup::TwoBytes { residuals, .. } => {
                for (index, residual_offset) in residuals.iter().enumerate() {
                    let predicted = header.slope * index as i32 + header.intercept;
                    let offset = (predicted + residual_offset.offset_residual() as i32) as u32;
                    let offset_view = OffsetView {
                        offset,
                        prefix7: *residual_offset.prefix7(),
                        len: residual_offset.len_byte(),
                    };
                    offset_views.push(offset_view);
                }
            },
            CompactOffsetViewGroup::FourBytes { residuals, .. } => {
                for (index, residual_offset) in residuals.iter().enumerate() {
                    let predicted = header.slope * index as i32 + header.intercept;
                    let offset = (predicted + residual_offset.offset_residual() as i32) as u32;
                    let offset_view = OffsetView {
                        offset,
                        prefix7: *residual_offset.prefix7(),
                        len: residual_offset.len_byte(),
                    };
                    offset_views.push(offset_view);
                }
            },
        }

        offset_views
    }
}

impl LiquidByteViewArray<MemoryBuffer> {
    /// Deserialize a LiquidByteViewArray from bytes.
    pub fn from_bytes(
        bytes: Bytes,
        compressor: Arc<Compressor>,
    ) -> LiquidByteViewArray<MemoryBuffer> {
        // 0) Read IPC header and our view header
        let ipc = LiquidIPCHeader::from_bytes(&bytes);
        let original_arrow_type = ArrowByteType::from(ipc.physical_type_id);
        let header_size = LiquidIPCHeader::size() + ByteViewArrayHeader::size();
        let view_header =
            ByteViewArrayHeader::from_bytes(&bytes[LiquidIPCHeader::size()..header_size]);

        let mut cursor = header_size;

        // A) Align and read FSST raw buffer first
        cursor = align_up_8(cursor);
        let fsst_end = cursor + view_header.fsst_raw_size as usize;
        if fsst_end > bytes.len() {
            panic!("FSST raw buffer extends beyond input buffer");
        }
        let fsst_raw = bytes.slice(cursor..fsst_end);
        let raw_buffer = RawFsstBuffer::from_bytes(fsst_raw);
        cursor = fsst_end;

        // B) Align and read keys
        cursor = align_up_8(cursor);
        let keys_end = cursor + view_header.keys_size as usize;
        if keys_end > bytes.len() {
            panic!("Keys data extends beyond input buffer");
        }
        let keys_data = bytes.slice(cursor..keys_end);
        let bit_packed = BitPackedArray::<UInt16Type>::from_bytes(keys_data);
        let dictionary_keys = bit_packed.to_primitive();
        cursor = keys_end;

        // C) Align and read offset views
        cursor = align_up_8(cursor);
        let offsets_end = cursor + view_header.offset_views_size as usize;
        if offsets_end > bytes.len() {
            panic!("Offset views data extends beyond input buffer");
        }

        // deserialize compact offset view directly
        let compact_offset_views = if view_header.offset_views_size > 0 {
            let chunk = bytes.slice(cursor..offsets_end);
            CompactOffsetViewGroup::from_bytes(&chunk)
        } else {
            CompactOffsetViewGroup::OneByte {
                header: CompactOffsetViewHeader {
                    slope: 0,
                    intercept: 0,
                    offset_bytes: 1,
                },
                residuals: Arc::new([]),
            }
        };
        cursor = offsets_end;

        // D) Align and read shared prefix
        cursor = align_up_8(cursor);
        let prefix_end = cursor + view_header.shared_prefix_size as usize;
        if prefix_end > bytes.len() {
            panic!("Shared prefix data extends beyond input buffer");
        }
        let shared_prefix = bytes[cursor..prefix_end].to_vec();

        LiquidByteViewArray {
            dictionary_keys,
            compact_offset_views,
            fsst_buffer: MemoryBuffer::new(Arc::new(raw_buffer)),
            original_arrow_type,
            shared_prefix,
            compressor,
        }
    }
}

#[derive(Debug, Clone, Copy)]
#[repr(C)]
// TODO: change it back to pub(crate)
pub struct OffsetView {
    offset: u32,
    prefix7: [u8; 7],
    len: u8,
}


#[derive(Debug, Clone, Copy)]
#[repr(C, packed)]
struct CompactOffsetViewHeader {
    slope: i32,    
    intercept: i32,
    offset_bytes: u8,         // 1, 2, or 4 bytes per offset
}

#[derive(Debug, Clone, Copy)]
#[repr(C, packed)]
struct CompactOffsetView<T> {
    offset_residual: T,
    prefix7: [u8; 7],
    len: u8
}

type CompactOffsetViewOneByte = CompactOffsetView<i8>;
type CompactOffsetViewTwoBytes = CompactOffsetView<i16>;
type CompactOffsetViewFourBytes = CompactOffsetView<i32>;

#[derive(Debug, Clone)]
enum CompactOffsetViewGroup {
    OneByte {
        header: CompactOffsetViewHeader,
        residuals: Arc<[CompactOffsetViewOneByte]>,
    },
    TwoBytes {
        header: CompactOffsetViewHeader,
        residuals: Arc<[CompactOffsetViewTwoBytes]>
    },
    FourBytes {
        header: CompactOffsetViewHeader,
        residuals: Arc<[CompactOffsetViewFourBytes]>
    },
}

const _: () = if std::mem::size_of::<OffsetView>() != 12 {
    panic!("OffsetView must be 12 bytes")
};

// Proper least-squares linear regression
fn fit_line(offsets: &[u32]) -> (i32, i32) {
    let n = offsets.len();
    if n <= 1 {
        return (0, offsets.get(0).copied().unwrap_or(0) as i32);
    }
    
    let n_f64 = n as f64;
    
    // Sum of indices: 0 + 1 + 2 + ... + (n-1) = n*(n-1)/2
    let sum_x = (n * (n - 1) / 2) as f64;
    
    // Sum of offsets
    let sum_y: f64 = offsets.iter().map(|&o| o as f64).sum();
    
    // Sum of (index * offset)
    let sum_xy: f64 = offsets.iter().enumerate()
        .map(|(i, &o)| i as f64 * o as f64)
        .sum();
    
    // Sum of index squared: 0² + 1² + 2² + ... + (n-1)² = n*(n-1)*(2n-1)/6
    let sum_x_sq = (n * (n - 1) * (2 * n - 1) / 6) as f64;
    
    // Least squares formulas
    let slope = (n_f64 * sum_xy - sum_x * sum_y) / (n_f64 * sum_x_sq - sum_x * sum_x);
    let intercept = (sum_y - slope * sum_x) / n_f64;
    
    (slope.round() as i32, intercept.round() as i32)
}

impl CompactOffsetViewGroup {
    fn from_offset_views(offset_views: &[OffsetView]) -> Self {
        if offset_views.is_empty() {
            return Self::OneByte {
                header: CompactOffsetViewHeader {
                    slope: 0,
                    intercept: 0,
                    offset_bytes: 1,
                },
                residuals: Arc::new([]),
            };
        }

        let offsets: Vec<u32> = offset_views.iter().map(|offset_view| offset_view.offset()).collect();

        let min_offset = *offsets.iter().min().unwrap();
        let max_offset = *offsets.iter().max().unwrap();

        // // simple linear regression: slope = (max - min) / (n - 1)
        // let slope = if offset_views.len() > 1 {
        //     (max_offset as i32 - min_offset as i32) / (offset_views.len() - 1) as i32
        // } else {
        //     0
        // };
        // let intercept = min_offset as i32;

        let (slope, intercept) = fit_line(&offsets);

        // calculate residuals
        let mut offset_residuals: Vec<i32> = Vec::new();
        let mut min_residual = i32::MAX;
        let mut max_residual = i32::MIN;
        for (index, &offset) in offsets.iter().enumerate() {
            let predicted = slope * index as i32 + intercept;
            offset_residuals.push(offset as i32 - predicted);
            min_residual = min_residual.min(*offset_residuals.last().unwrap());
            max_residual = max_residual.max(*offset_residuals.last().unwrap());
        }

        assert!(min_residual <= max_residual);

        // determine bytes needed for residuals
        let offset_bytes = if min_residual >= i8::MIN as i32 && max_residual <= i8::MAX as i32 {
            1
        } else if min_residual >= i16::MIN as i32 && max_residual <= i16::MAX as i32 {
            2
        } else {
            4
        };

        let header = CompactOffsetViewHeader {
            slope,    
            intercept,
            offset_bytes,
        };

        match header.offset_bytes {
            1 => {
                let mut residuals = Vec::new();
                for (index, offset) in offset_views.iter().enumerate() {
                    let compact_residual = CompactOffsetViewOneByte {
                        offset_residual: *offset_residuals.get(index).unwrap() as i8,
                        prefix7: offset.prefix7().clone(),
                        len: offset.len_byte(),
                    };
                    residuals.push(compact_residual);
                }
                Self::OneByte {
                    header,
                    residuals: residuals.into(),
                }
            },
            2 => {
                let mut residuals = Vec::new();
                for (index, offset) in offset_views.iter().enumerate() {
                    let compact_residual = CompactOffsetViewTwoBytes {
                        offset_residual: *offset_residuals.get(index).unwrap() as i16,
                        prefix7: offset.prefix7().clone(),
                        len: offset.len_byte(),
                    };
                    residuals.push(compact_residual);
                }
                    
                Self::TwoBytes {
                    header,
                    residuals: residuals.into(),
                }
            },
            4 => {
                let mut residuals = Vec::new();
                for (index, offset) in offset_views.iter().enumerate() {
                    let compact_residual = CompactOffsetViewFourBytes {
                        offset_residual: *offset_residuals.get(index).unwrap() as i32,
                        prefix7: offset.prefix7().clone(),
                        len: offset.len_byte(),
                    };
                    residuals.push(compact_residual);
                }
                Self::FourBytes {
                    header,
                    residuals: residuals.into(),
                }
            },
            _ => panic!("Invalid offset_bytes value"),
        }
    }

    fn header(&self) -> &CompactOffsetViewHeader {
        match self {
            Self::OneByte { header, .. } => header,
            Self::TwoBytes { header, .. } => header,
            Self::FourBytes { header, .. } => header,
        }
    }

    fn len(&self) -> usize {
        match self {
            Self::OneByte { residuals, .. } => residuals.len(),
            Self::TwoBytes { residuals, .. } => residuals.len(),
            Self::FourBytes { residuals, .. } => residuals.len(),
        }
    }

    fn get_offset(&self, index: usize) -> u32 {
        let header = self.header();
        match self {
            Self::OneByte { residuals, .. } => {
                let residual = &residuals[index];
                let predicted = header.slope * index as i32 + header.intercept;
                (predicted + residual.offset_residual as i32) as u32
            },
            Self::TwoBytes { residuals, .. } => {
                let residual = &residuals[index];
                let predicted = header.slope * index as i32 + header.intercept;
                (predicted + residual.offset_residual as i32) as u32
            },
            Self::FourBytes { residuals, .. } => {
                let residual = &residuals[index];
                let predicted = header.slope * index as i32 + header.intercept;
                (predicted + residual.offset_residual) as u32
            },
        }
    }

    fn get_prefix7(&self, index: usize) -> &[u8; 7] {
        match self {
            Self::OneByte { residuals, .. } => &residuals[index].prefix7,
            Self::TwoBytes { residuals, .. } => &residuals[index].prefix7,
            Self::FourBytes { residuals, .. } => &residuals[index].prefix7,
        }
    }

    fn get_len_byte(&self, index: usize) -> u8 {
        match self {
            Self::OneByte { residuals, .. } => residuals[index].len,
            Self::TwoBytes { residuals, .. } => residuals[index].len,
            Self::FourBytes { residuals, .. } => residuals[index].len,
        }
    }

    fn memory_usage(&self) -> usize {
        let header_size = std::mem::size_of::<CompactOffsetViewHeader>();
        let residuals_size = match self {
            Self::OneByte { residuals, .. } => residuals.len() * std::mem::size_of::<CompactOffsetViewOneByte>(),
            Self::TwoBytes { residuals, .. } => residuals.len() * std::mem::size_of::<CompactOffsetViewTwoBytes>(),
            Self::FourBytes { residuals, .. } => residuals.len() * std::mem::size_of::<CompactOffsetViewFourBytes>(),
        };
        
        header_size + residuals_size
    }

    fn from_bytes(bytes: &[u8]) -> Self {
        if bytes.len() < 9 {
            panic!("CompactOffsetViewGroup requires at least 9 bytes for header");
        }

        // read header
        let slope = i32::from_le_bytes(bytes[0..4].try_into().unwrap());
        let intercept = i32::from_le_bytes(bytes[4..8].try_into().unwrap());
        let offset_bytes = bytes[8];

        let header = CompactOffsetViewHeader {
            slope,
            intercept,
            offset_bytes,
        };

        let residuals_data = &bytes[9..];

        match offset_bytes {
            1 => {
                let residual_size = std::mem::size_of::<CompactOffsetViewOneByte>();
                if residuals_data.len() % residual_size != 0 {
                    panic!("Invalid residuals data size for OneByte variant");
                }
                let count = residuals_data.len() / residual_size;
                let mut residuals = Vec::with_capacity(count);
                
                for i in 0..count {
                    let base = i * residual_size;
                    let offset_residual = residuals_data[base] as i8;
                    let mut prefix7 = [0u8; 7];
                    prefix7.copy_from_slice(&residuals_data[base + 1..base + 8]);
                    let len = residuals_data[base + 8];
                    
                    residuals.push(CompactOffsetViewOneByte {
                        offset_residual,
                        prefix7,
                        len,
                    });
                }
                
                Self::OneByte {
                    header,
                    residuals: residuals.into(),
                }
            },
            2 => {
                let residual_size = std::mem::size_of::<CompactOffsetViewTwoBytes>();
                if residuals_data.len() % residual_size != 0 {
                    panic!("Invalid residuals data size for TwoBytes variant");
                }
                let count = residuals_data.len() / residual_size;
                let mut residuals = Vec::with_capacity(count);
                
                for i in 0..count {
                    let base = i * residual_size;
                    let offset_residual = i16::from_le_bytes(residuals_data[base..base + 2].try_into().unwrap());
                    let mut prefix7 = [0u8; 7];
                    prefix7.copy_from_slice(&residuals_data[base + 2..base + 9]);
                    let len = residuals_data[base + 9];
                    
                    residuals.push(CompactOffsetViewTwoBytes {
                        offset_residual,
                        prefix7,
                        len,
                    });
                }
                
                Self::TwoBytes {
                    header,
                    residuals: residuals.into(),
                }
            },
            4 => {
                let residual_size = std::mem::size_of::<CompactOffsetViewFourBytes>();
                if residuals_data.len() % residual_size != 0 {
                    panic!("Invalid residuals data size for FourBytes variant");
                }
                let count = residuals_data.len() / residual_size;
                let mut residuals = Vec::with_capacity(count);
                
                for i in 0..count {
                    let base = i * residual_size;
                    let offset_residual = i32::from_le_bytes(residuals_data[base..base + 4].try_into().unwrap());
                    let mut prefix7 = [0u8; 7];
                    prefix7.copy_from_slice(&residuals_data[base + 4..base + 11]);
                    let len = residuals_data[base + 11];
                    
                    residuals.push(CompactOffsetViewFourBytes {
                        offset_residual,
                        prefix7,
                        len,
                    });
                }
                
                Self::FourBytes {
                    header,
                    residuals: residuals.into(),
                }
            },
            _ => panic!("Invalid offset_bytes value: {}", offset_bytes),
        }
    }
}

impl OffsetView {
    /// Construct from offset and the full suffix bytes (after shared prefix).
    /// Embeds up to `prefix_len()` bytes into `prefix7` and stores length (or 255 if >=255).
    pub fn new(offset: u32, suffix_bytes: &[u8]) -> Self {
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
        Self {
            offset,
            prefix7,
            len,
        }
    }

    /// Construct directly from stored parts (used by deserialization only)
    pub fn from_parts(offset: u32, prefix7: [u8; 7], len: u8) -> Self {
        Self {
            offset,
            prefix7,
            len,
        }
    }

    pub fn offset(&self) -> u32 {
        self.offset
    }

    /// Returns the 7-byte content prefix stored in the view
    #[inline]
    pub fn prefix7(&self) -> &[u8; 7] {
        &self.prefix7
    }

    /// Returns Some(length) if known (<255), otherwise None for unknown (>=255)
    #[allow(dead_code)]
    #[inline]
    pub fn known_suffix_len(&self) -> Option<usize> {
        if self.len == 255 {
            None
        } else {
            Some(self.len as usize)
        }
    }

    #[inline]
    pub fn len_byte(&self) -> u8 {
        self.len
    }

    #[inline]
    pub const fn prefix_len() -> usize {
        7
    }
}

impl<T> CompactOffsetView<T> {
    
    #[inline]
    pub fn offset_residual(&self) -> T
    where 
        T: Copy,
    {
        self.offset_residual
    }

    #[allow(dead_code)]
    pub fn known_suffix_len(&self) -> Option<usize> {
        if self.len == 255 {
            None
        } else {
            Some(self.len as usize)
        }
    }

    #[inline]
    pub fn prefix7(&self) -> &[u8; 7] {
        &self.prefix7
    }

    #[inline]
    pub fn len_byte(&self) -> u8 {
        self.len
    }

    #[allow(dead_code)]
    #[inline]
    pub const fn prefix_len() -> usize {
        7
    }
}

/// Memory buffer for FSST buffer
#[derive(Debug, Clone)]
pub struct MemoryBuffer {
    buffer: Arc<RawFsstBuffer>,
}

impl MemoryBuffer {
    fn new(raw_buffer: Arc<RawFsstBuffer>) -> Self {
        Self { buffer: raw_buffer }
    }
}

/// Disk buffer for FSST buffer
#[derive(Debug, Clone)]
pub struct DiskBuffer {
    uncompressed_bytes: usize,
    disk_range: Range<u64>,
}

impl DiskBuffer {
    fn new(uncompressed_bytes: usize, disk_range: Range<u64>) -> Self {
        Self {
            uncompressed_bytes,
            disk_range,
        }
    }
}

mod sealed {
    pub trait Sealed {}
}

/// Trait for FSST buffer - can be in memory or on disk
pub trait FsstBuffer: std::fmt::Debug + Clone + sealed::Sealed {
    /// Get the raw FSST buffer, loading from disk if necessary
    fn get_fsst_buffer(&self) -> Result<Arc<RawFsstBuffer>, IoRange>;

    /// Get the memory size of the FSST buffer
    fn get_array_memory_size(&self) -> usize;

    /// Get the uncompressed bytes of the FSST buffer
    fn uncompressed_bytes(&self) -> usize;
}

impl sealed::Sealed for MemoryBuffer {}
impl sealed::Sealed for DiskBuffer {}

impl FsstBuffer for MemoryBuffer {
    fn get_fsst_buffer(&self) -> Result<Arc<RawFsstBuffer>, IoRange> {
        Ok(self.buffer.clone())
    }

    fn get_array_memory_size(&self) -> usize {
        self.buffer.get_memory_size()
    }

    fn uncompressed_bytes(&self) -> usize {
        self.buffer.uncompressed_bytes()
    }
}

impl FsstBuffer for DiskBuffer {
    fn get_fsst_buffer(&self) -> Result<Arc<RawFsstBuffer>, IoRange> {
        Err(IoRange {
            range: self.disk_range.clone(),
        })
    }

    fn get_array_memory_size(&self) -> usize {
        0
    }

    fn uncompressed_bytes(&self) -> usize {
        self.uncompressed_bytes
    }
}

impl LiquidArray for LiquidByteViewArray<MemoryBuffer> {
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

    fn filter(&self, selection: &BooleanBuffer) -> LiquidArrayRef {
        let filtered = filter_inner(self, selection);
        Arc::new(filtered)
    }

    fn try_eval_predicate(
        &self,
        expr: &Arc<dyn PhysicalExpr>,
        filter: &BooleanBuffer,
    ) -> Option<BooleanArray> {
        let filtered = filter_inner(self, filter);
        try_eval_predicate_inner(expr, &filtered).expect("InMemoryFsstBuffer")
    }

    fn to_bytes(&self) -> Vec<u8> {
        self.to_bytes_inner().expect("InMemoryFsstBuffer")
    }

    fn data_type(&self) -> LiquidDataType {
        LiquidDataType::ByteViewArray
    }

    fn squeeze(&self) -> Option<(LiquidHybridArrayRef, bytes::Bytes)> {
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

        let disk_range = (fsst_start as u64)..(fsst_end as u64);

        // Build the hybrid (disk-backed FSST) view
        let disk = DiskBuffer::new(self.fsst_buffer.uncompressed_bytes(), disk_range);
        let hybrid = LiquidByteViewArray::<DiskBuffer> {
            dictionary_keys: self.dictionary_keys.clone(),
            compact_offset_views: self.compact_offset_views.clone(),
            fsst_buffer: disk,
            original_arrow_type: self.original_arrow_type,
            shared_prefix: self.shared_prefix.clone(),
            compressor: self.compressor.clone(),
        };

        let bytes = bytes::Bytes::from(bytes);
        Some((Arc::new(hybrid) as LiquidHybridArrayRef, bytes))
    }
}

impl LiquidHybridArray for LiquidByteViewArray<DiskBuffer> {
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
    fn to_arrow_array(&self) -> Result<ArrayRef, IoRange> {
        self.to_arrow_array()
    }

    /// Convert the Liquid array to an Arrow array.
    /// Except that it will pick the best encoding for the arrow array.
    /// Meaning that it may not obey the data type of the original arrow array.
    fn to_best_arrow_array(&self) -> Result<ArrayRef, IoRange> {
        self.to_arrow_array()
    }

    /// Get the logical data type of the Liquid array.
    fn data_type(&self) -> LiquidDataType {
        LiquidDataType::ByteViewArray
    }

    /// Serialize the Liquid array to a byte array.
    fn to_bytes(&self) -> Result<Vec<u8>, IoRange> {
        self.to_bytes_inner()
    }

    /// Filter the Liquid array with a boolean buffer.
    fn filter(&self, selection: &BooleanBuffer) -> Result<LiquidHybridArrayRef, IoRange> {
        Ok(Arc::new(filter_inner(self, selection)))
    }

    /// Filter the Liquid array with a boolean array and return an **arrow array**.
    fn filter_to_arrow(&self, selection: &BooleanBuffer) -> Result<ArrayRef, IoRange> {
        let filtered = self.filter(selection)?;
        filtered.to_best_arrow_array()
    }

    /// Try to evaluate a predicate on the Liquid array with a filter.
    /// Returns `Ok(None)` if the predicate is not supported.
    ///
    /// Note that the filter is a boolean buffer, not a boolean array, i.e., filter can't be nullable.
    /// The returned boolean mask is nullable if the the original array is nullable.
    fn try_eval_predicate(
        &self,
        expr: &Arc<dyn PhysicalExpr>,
        filter: &BooleanBuffer,
    ) -> Result<Option<BooleanArray>, IoRange> {
        // Reuse generic filter path first to reduce input rows if any
        let filtered = filter_inner(self, filter);

        // Handle binary expressions (equality/inequality) with prefix optimization
        if let Some(binary_expr) = expr.as_any().downcast_ref::<BinaryExpr>()
            && let Some(literal) = binary_expr.right().as_any().downcast_ref::<Literal>()
        {
            let op = binary_expr.op();
            if matches!(op, Operator::Eq | Operator::NotEq)
                && let Some(needle) = get_string_needle(literal.value())
            {
                // Try prefix-based equality. On ambiguity, bubble up IoRange for fallback.
                let eq_mask = filtered
                    .compare_equals_with_prefix(needle.as_bytes())
                    .ok_or_else(|| self.to_liquid())?;
                if matches!(op, Operator::Eq) {
                    return Ok(Some(eq_mask));
                } else {
                    let (values, nulls) = eq_mask.into_parts();
                    return Ok(Some(BooleanArray::new(!&values, nulls)));
                }
            }
        }

        Ok(None)
    }

    /// Feed IO data to the `LiquidHybridArray`.
    /// Returns the in-memory `LiquidArray`.
    /// This is the bridge from hybrid array to in-memory array.
    fn soak(&self, data: bytes::Bytes) -> LiquidArrayRef {
        // `data` is the raw FSST buffer bytes (no IPC header)
        let buffer = MemoryBuffer::new(Arc::new(RawFsstBuffer::from_bytes(data)));

        let in_memory_array = LiquidByteViewArray::<MemoryBuffer> {
            dictionary_keys: self.dictionary_keys.clone(),
            compact_offset_views: self.compact_offset_views.clone(),
            fsst_buffer: buffer,
            original_arrow_type: self.original_arrow_type,
            shared_prefix: self.shared_prefix.clone(),
            compressor: self.compressor.clone(),
        };
        Arc::new(in_memory_array)
    }

    fn to_liquid(&self) -> IoRange {
        IoRange {
            range: self.fsst_buffer.disk_range.clone(),
        }
    }
}

impl LiquidByteViewArray<DiskBuffer> {
    /// Equality using prefix first; if ambiguous, fall back to full buffer comparison
    fn compare_equals_with_prefix(&self, needle: &[u8]) -> Option<BooleanArray> {
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
        let prefix_len = OffsetView::prefix_len();

        let num_unique = self.compact_offset_views.len().saturating_sub(1);
        let mut dict_results = vec![false; num_unique];

        for i in 0..num_unique {
            let known_len = if self.compact_offset_views.get_len_byte(i) == 255 {
                None
            } else {
                Some(self.compact_offset_views.get_len_byte(i) as usize)
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
                            if self.compact_offset_views.get_prefix7(i)[..prefix_len] == needle_suffix[..prefix_len] {
                                return None; // ambiguous, requires IO
                            }
                            // else definitively not equal, leave false
                        }
                        Some(l) if l <= prefix_len => {
                            // Small strings: exact compare on l bytes
                            if self.compact_offset_views.get_prefix7(i)[..l] == needle_suffix[..l] {
                                dict_results[i] = true; // definitive match
                            }
                        }
                        Some(_l) => {
                            // Medium strings: prefix compare; equal means ambiguous
                            if self.compact_offset_views.get_prefix7(i)[..prefix_len] == needle_suffix[..prefix_len] {
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
}

fn filter_inner<B: FsstBuffer>(
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
        compact_offset_views: array.compact_offset_views.clone(), // Keep original offset views - they reference unique values
        fsst_buffer: array.fsst_buffer.clone(),
        original_arrow_type: array.original_arrow_type,
        shared_prefix: array.shared_prefix.clone(),
        compressor: array.compressor.clone(),
    }
}

fn try_eval_predicate_inner<B: FsstBuffer>(
    expr: &Arc<dyn PhysicalExpr>,
    array: &LiquidByteViewArray<B>,
) -> Result<Option<BooleanArray>, IoRange> {
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
                Operator::NotEq => apply_cmp(&lhs, &rhs, kernels::cmp::neq),
                Operator::Eq => apply_cmp(&lhs, &rhs, kernels::cmp::eq),
                Operator::Lt => apply_cmp(&lhs, &rhs, kernels::cmp::lt),
                Operator::LtEq => apply_cmp(&lhs, &rhs, kernels::cmp::lt_eq),
                Operator::Gt => apply_cmp(&lhs, &rhs, kernels::cmp::gt),
                Operator::GtEq => apply_cmp(&lhs, &rhs, kernels::cmp::gt_eq),
                Operator::LikeMatch => apply_cmp(&lhs, &rhs, arrow::compute::like),
                Operator::ILikeMatch => apply_cmp(&lhs, &rhs, arrow::compute::ilike),
                Operator::NotLikeMatch => apply_cmp(&lhs, &rhs, arrow::compute::nlike),
                Operator::NotILikeMatch => apply_cmp(&lhs, &rhs, arrow::compute::nilike),
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
            (false, false) => apply_cmp(&lhs, &rhs, arrow::compute::like),
            (true, false) => apply_cmp(&lhs, &rhs, arrow::compute::nlike),
            (false, true) => apply_cmp(&lhs, &rhs, arrow::compute::ilike),
            (true, true) => apply_cmp(&lhs, &rhs, arrow::compute::nilike),
        };
        if let Ok(result) = result {
            let filtered = result.into_array(array.len()).unwrap().as_boolean().clone();
            return Ok(Some(filtered));
        }
    }
    Ok(None)
}

/// An array that stores strings using the FSST view format:
/// - Dictionary keys with 2-byte keys stored in memory
/// - Offset views with 4-byte offsets and 8-byte prefixes stored in memory
/// - FSST buffer can be stored in memory or on disk
///
/// Data access flow:
/// 1. Use dictionary key to index into offset_views buffer
/// 2. Use offset from offset_views to read the corresponding bytes from FSST buffer
/// 3. Use prefix from offset_views for quick comparisons to avoid decompression when possible
/// 4. Decompress bytes from FSST buffer to get the full value when needed
#[derive(Clone)]
pub struct LiquidByteViewArray<B: FsstBuffer> {
    /// Dictionary keys (u16) - one per array element, using Arrow's UInt16Array for zero-copy
    dictionary_keys: UInt16Array,
    /// Offset views containing offset (u32) and prefix (8 bytes) - one per unique value
    /// Stored as `Arc<[CompactOffsetView]>` for cheap clones when passing across layers (e.g., soak/squeeze).
    compact_offset_views: CompactOffsetViewGroup,
    // offset_views: Arc<[OffsetView]>,
    /// FSST-compressed buffer (can be in memory or on disk)
    fsst_buffer: B,
    /// Used to convert back to the original arrow type
    original_arrow_type: ArrowByteType,
    /// Shared prefix across all strings in the array
    shared_prefix: Vec<u8>,
    /// Compressor for decompression
    compressor: Arc<Compressor>,
}

impl<B: FsstBuffer> std::fmt::Debug for LiquidByteViewArray<B> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LiquidByteViewArray")
            .field("dictionary_keys", &self.dictionary_keys)
            .field("compact_offset_views", &self.compact_offset_views)
            .field("fsst_buffer", &self.fsst_buffer)
            .field("original_arrow_type", &self.original_arrow_type)
            .field("shared_prefix", &self.shared_prefix)
            .field("compressor", &"<Compressor>")
            .finish()
    }
}

impl<B: FsstBuffer> LiquidByteViewArray<B> {
    /// Create a LiquidByteViewArray from an Arrow StringViewArray
    pub fn from_string_view_array(
        array: &StringViewArray,
        compressor: Arc<Compressor>,
    ) -> LiquidByteViewArray<MemoryBuffer> {
        Self::from_view_array_inner(array, compressor, ArrowByteType::Utf8View)
    }

    /// Create a LiquidByteViewArray from an Arrow BinaryViewArray
    pub fn from_binary_view_array(
        array: &BinaryViewArray,
        compressor: Arc<Compressor>,
    ) -> LiquidByteViewArray<MemoryBuffer> {
        Self::from_view_array_inner(array, compressor, ArrowByteType::BinaryView)
    }

    /// Create a LiquidByteViewArray from an Arrow StringArray
    pub fn from_string_array(
        array: &StringArray,
        compressor: Arc<Compressor>,
    ) -> LiquidByteViewArray<MemoryBuffer> {
        Self::from_byte_array_inner(array, compressor, ArrowByteType::Utf8)
    }

    /// Create a LiquidByteViewArray from an Arrow BinaryArray
    pub fn from_binary_array(
        array: &BinaryArray,
        compressor: Arc<Compressor>,
    ) -> LiquidByteViewArray<MemoryBuffer> {
        Self::from_byte_array_inner(array, compressor, ArrowByteType::Binary)
    }

    /// Train a compressor from an Arrow StringViewArray
    pub fn train_from_string_view(
        array: &StringViewArray,
    ) -> (Arc<Compressor>, LiquidByteViewArray<MemoryBuffer>) {
        let compressor = Self::train_compressor(array.iter());
        (
            compressor.clone(),
            Self::from_view_array_inner(array, compressor, ArrowByteType::Utf8View),
        )
    }

    /// Train a compressor from an Arrow BinaryViewArray
    pub fn train_from_binary_view(
        array: &BinaryViewArray,
    ) -> (Arc<Compressor>, LiquidByteViewArray<MemoryBuffer>) {
        let compressor = Self::train_compressor_bytes(array.iter());
        (
            compressor.clone(),
            Self::from_view_array_inner(array, compressor, ArrowByteType::BinaryView),
        )
    }

    /// Train a compressor from an Arrow ByteArray.
    pub fn train_from_arrow<T: ByteArrayType>(
        array: &GenericByteArray<T>,
    ) -> (Arc<Compressor>, LiquidByteViewArray<MemoryBuffer>) {
        let dict = CheckedDictionaryArray::from_byte_array::<T>(array);
        let value_type = dict.as_ref().values().data_type();

        let compressor = if value_type == &DataType::Utf8 {
            Self::train_compressor(dict.as_ref().values().as_string::<i32>().iter())
        } else {
            Self::train_compressor_bytes(dict.as_ref().values().as_binary::<i32>().iter())
        };
        (
            compressor.clone(),
            Self::from_dict_array_inner(
                dict,
                compressor,
                ArrowByteType::from_arrow_type(&T::DATA_TYPE),
            ),
        )
    }

    /// Only used when the dictionary is read from a trusted parquet reader,
    /// which reads a trusted parquet file, written by a trusted writer.
    ///
    /// # Safety
    /// The caller must ensure that the values in the dictionary are unique.
    pub unsafe fn from_unique_dict_array(
        array: &DictionaryArray<UInt16Type>,
        compressor: Arc<Compressor>,
    ) -> LiquidByteViewArray<MemoryBuffer> {
        let arrow_type = ArrowByteType::from_arrow_type(array.values().data_type());
        Self::from_dict_array_inner(
            unsafe { CheckedDictionaryArray::new_unchecked_i_know_what_i_am_doing(array) },
            compressor,
            arrow_type,
        )
    }

    /// Train a compressor from an Arrow DictionaryArray.
    pub fn train_from_arrow_dict(
        array: &DictionaryArray<UInt16Type>,
    ) -> (Arc<Compressor>, LiquidByteViewArray<MemoryBuffer>) {
        if array.values().data_type() == &DataType::Utf8 {
            let values = array.values().as_string::<i32>();

            let compressor = Self::train_compressor(values.iter());
            (
                compressor.clone(),
                Self::from_dict_array_inner(
                    CheckedDictionaryArray::new_checked(array),
                    compressor,
                    ArrowByteType::Dict16Utf8,
                ),
            )
        } else if array.values().data_type() == &DataType::Binary {
            let values = array.values().as_binary::<i32>();
            let compressor = Self::train_compressor_bytes(values.iter());
            (
                compressor.clone(),
                Self::from_dict_array_inner(
                    CheckedDictionaryArray::new_checked(array),
                    compressor,
                    ArrowByteType::Dict16Binary,
                ),
            )
        } else {
            panic!("Unsupported dictionary type: {:?}", array.data_type())
        }
    }

    /// Train a compressor from an iterator of strings
    pub fn train_compressor<'a, T: ArrayAccessor<Item = &'a str>>(
        array: ArrayIter<T>,
    ) -> Arc<Compressor> {
        Arc::new(train_compressor(
            array.filter_map(|s| s.as_ref().map(|s| s.as_bytes())),
        ))
    }

    /// Train a compressor from an iterator of byte arrays
    pub fn train_compressor_bytes<'a, T: ArrayAccessor<Item = &'a [u8]>>(
        array: ArrayIter<T>,
    ) -> Arc<Compressor> {
        Arc::new(train_compressor(
            array.filter_map(|s| s.as_ref().map(|s| *s)),
        ))
    }

    /// Convert to Arrow DictionaryArray
    pub fn to_dict_arrow(&self) -> Result<DictionaryArray<UInt16Type>, IoRange> {
        let keys_array = self.dictionary_keys.clone();

        // Convert raw FSST buffer to values using our offset views
        let raw_buffer = self.fsst_buffer.get_fsst_buffer()?;

        let offset_views = self.offset_views();
        let (values_buffer, offsets_buffer) =
            raw_buffer.to_uncompressed(&self.compressor.decompressor(), &offset_views);

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
    pub fn to_arrow_array(&self) -> Result<ArrayRef, IoRange> {
        let dict = self.to_dict_arrow()?;
        Ok(cast(&dict, &self.original_arrow_type.to_arrow_type()).unwrap())
    }

    /// Get the nulls buffer
    pub fn nulls(&self) -> Option<&NullBuffer> {
        self.dictionary_keys.nulls()
    }

    /// Compare with prefix optimization and fallback to Arrow operations
    pub fn compare_with(&self, needle: &[u8], op: &Operator) -> Result<BooleanArray, IoRange> {
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

    /// Get detailed memory usage of the byte view array
    pub fn get_detailed_memory_usage(&self) -> ByteViewArrayMemoryUsage {
        ByteViewArrayMemoryUsage {
            dictionary_key: self.dictionary_keys.get_array_memory_size(),
            offsets: self.compact_offset_views.memory_usage(),  
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
    pub fn sort_to_indices(&self) -> Result<UInt32Array, IoRange> {
        // if distinct ratio is more than 10%, use arrow sort.
        if self.compact_offset_views.len() > (self.dictionary_keys.len() / 10) {
            let array = self.to_dict_arrow()?;
            let sorted_array = sort_to_indices(&array, None, None).unwrap();
            Ok(sorted_array)
        } else {
            self.sort_to_indices_inner()
        }
    }

    /// Check if the FSST buffer is currently stored on disk
    pub fn is_fsst_buffer_on_disk(&self) -> bool {
        self.fsst_buffer.get_fsst_buffer().is_err()
    }

    /// Check if the FSST buffer is currently stored in memory
    pub fn is_fsst_buffer_in_memory(&self) -> bool {
        self.fsst_buffer.get_fsst_buffer().is_ok()
    }

    /// Get the length of the array
    pub fn len(&self) -> usize {
        self.dictionary_keys.len()
    }

    /// Is the array empty?
    pub fn is_empty(&self) -> bool {
        self.dictionary_keys.is_empty()
    }
}

impl<B: FsstBuffer> LiquidByteViewArray<B> {
    /// Generic implementation for view arrays (StringViewArray and BinaryViewArray)
    fn from_view_array_inner<T>(
        array: &T,
        compressor: Arc<Compressor>,
        arrow_type: ArrowByteType,
    ) -> LiquidByteViewArray<MemoryBuffer>
    where
        T: Array + 'static,
    {
        // Convert view array to CheckedDictionaryArray using existing infrastructure
        let dict = if let Some(string_view) = array.as_any().downcast_ref::<StringViewArray>() {
            CheckedDictionaryArray::from_string_view_array(string_view)
        } else if let Some(binary_view) = array.as_any().downcast_ref::<BinaryViewArray>() {
            CheckedDictionaryArray::from_binary_view_array(binary_view)
        } else {
            panic!("Unsupported view array type")
        };

        Self::from_dict_array_inner(dict, compressor, arrow_type)
    }

    fn from_byte_array_inner<T: ByteArrayType>(
        array: &GenericByteArray<T>,
        compressor: Arc<Compressor>,
        arrow_type: ArrowByteType,
    ) -> LiquidByteViewArray<MemoryBuffer> {
        let dict = CheckedDictionaryArray::from_byte_array::<T>(array);
        Self::from_dict_array_inner(dict, compressor, arrow_type)
    }

    /// Core implementation that converts a CheckedDictionaryArray to LiquidByteViewArray
    fn from_dict_array_inner(
        dict: CheckedDictionaryArray,
        compressor: Arc<Compressor>,
        arrow_type: ArrowByteType,
    ) -> LiquidByteViewArray<MemoryBuffer> {
        let (keys, values) = dict.as_ref().clone().into_parts();

        // Calculate shared prefix directly from values array without intermediate allocations
        let shared_prefix = if values.is_empty() {
            Vec::new()
        } else {
            // Get first value as initial candidate for shared prefix
            let first_value_bytes = if let Some(string_values) = values.as_string_opt::<i32>() {
                string_values.value(0).as_bytes()
            } else if let Some(binary_values) = values.as_binary_opt::<i32>() {
                binary_values.value(0)
            } else {
                panic!("Unsupported dictionary value type")
            };

            let mut shared_prefix = first_value_bytes.to_vec();

            // Compare with remaining values and truncate shared prefix
            for i in 1..values.len() {
                let value_bytes = if let Some(string_values) = values.as_string_opt::<i32>() {
                    string_values.value(i).as_bytes()
                } else if let Some(binary_values) = values.as_binary_opt::<i32>() {
                    binary_values.value(i)
                } else {
                    panic!("Unsupported dictionary value type")
                };

                let common_len = shared_prefix
                    .iter()
                    .zip(value_bytes.iter())
                    .take_while(|(a, b)| a == b)
                    .count();
                shared_prefix.truncate(common_len);

                // Early exit if no common prefix
                if shared_prefix.is_empty() {
                    break;
                }
            }

            shared_prefix
        };

        let shared_prefix_len = shared_prefix.len();

        // Create offset views with prefixes - one per unique value in dictionary
        let mut offset_views = Vec::with_capacity(values.len());

        let mut compress_buffer = Vec::with_capacity(1024 * 1024 * 2);

        // Create the raw buffer and get the byte offsets
        let (raw_fsst_buffer, byte_offsets) =
            if let Some(string_values) = values.as_string_opt::<i32>() {
                RawFsstBuffer::from_byte_slices(
                    string_values.iter().map(|s| s.map(|s| s.as_bytes())),
                    compressor.clone(),
                    &mut compress_buffer,
                )
            } else if let Some(binary_values) = values.as_binary_opt::<i32>() {
                RawFsstBuffer::from_byte_slices(
                    binary_values.iter(),
                    compressor.clone(),
                    &mut compress_buffer,
                )
            } else {
                panic!("Unsupported dictionary value type")
            };

        for (i, byte_offset) in byte_offsets.iter().enumerate().take(values.len()) {
            let value_bytes = if let Some(string_values) = values.as_string_opt::<i32>() {
                string_values.value(i).as_bytes()
            } else if let Some(binary_values) = values.as_binary_opt::<i32>() {
                binary_values.value(i)
            } else {
                panic!("Unsupported dictionary value type")
            };

            // Build OffsetView from the suffix bytes directly to avoid leaking encoding details
            let remaining_bytes = if shared_prefix_len < value_bytes.len() {
                &value_bytes[shared_prefix_len..]
            } else {
                &[]
            };

            offset_views.push(OffsetView::new(*byte_offset, remaining_bytes));
        }

        assert_eq!(values.len(), byte_offsets.len() - 1);
        offset_views.push(OffsetView::from_parts(
            byte_offsets[values.len()],
            [0u8; 7],
            0,
        ));

        LiquidByteViewArray::from_parts(
            keys,
            &offset_views,
            MemoryBuffer {
                buffer: Arc::new(raw_fsst_buffer),
            },
            arrow_type,
            shared_prefix,
            compressor,
        )
    }

    /// Compare equality with a byte needle
    fn compare_equals(&self, needle: &[u8]) -> Result<BooleanArray, IoRange> {
        // Fast path 1: Check shared prefix
        let shared_prefix_len = self.shared_prefix.len();
        if needle.len() < shared_prefix_len || needle[..shared_prefix_len] != self.shared_prefix {
            return Ok(BooleanArray::new(
                BooleanBuffer::new_unset(self.dictionary_keys.len()),
                self.nulls().cloned(),
            ));
        }

        let raw_buffer = self.fsst_buffer.get_fsst_buffer()?;
        Ok(self.compare_equals_with_raw_buffer(needle, &raw_buffer))
    }

    fn compare_equals_with_raw_buffer(
        &self,
        needle: &[u8],
        raw_buffer: &RawFsstBuffer,
    ) -> BooleanArray {
        let compressed_needle = self.compressor.compress(needle);

        // Find the matching dictionary value (early exit since values are unique)
        let num_unique = self.compact_offset_views.len().saturating_sub(1);
        let mut matching_dict_key = None;

        for i in 0..num_unique {
            let start_offset = self.compact_offset_views.get_offset(i);
            let end_offset = self.compact_offset_views.get_offset(i + 1);

            let compressed_value = raw_buffer.get_compressed_slice(start_offset, end_offset);
            if compressed_value == compressed_needle.as_slice() {
                matching_dict_key = Some(i as u16);
                break; // Early exit - dictionary values are unique
            }
        }
        let Some(matching_dict_key) = matching_dict_key else {
            return BooleanArray::new(
                BooleanBuffer::new_unset(self.dictionary_keys.len()),
                self.nulls().cloned(),
            );
        };

        let to_compare = UInt16Array::new_scalar(matching_dict_key);
        arrow::compute::kernels::cmp::eq(&self.dictionary_keys, &to_compare).unwrap()
    }

    /// Compare not equals with a byte needle
    fn compare_not_equals(&self, needle: &[u8]) -> Result<BooleanArray, IoRange> {
        let result = self.compare_equals(needle)?;
        let (values, nulls) = result.into_parts();
        let values = !&values;
        Ok(BooleanArray::new(values, nulls))
    }

    /// Check if shared prefix comparison can short-circuit the entire operation
    fn try_shared_prefix_short_circuit(
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
    fn compare_with_inner(&self, needle: &[u8], op: &Operator) -> Result<BooleanArray, IoRange> {
        // Try to short-circuit based on shared prefix comparison
        if let Some(result) = self.try_shared_prefix_short_circuit(needle, op) {
            return Ok(result);
        }

        let needle_suffix = &needle[self.shared_prefix.len()..];
        let num_unique = self.compact_offset_views.len().saturating_sub(1);
        let mut dict_results = Vec::with_capacity(num_unique);
        let mut needs_full_comparison = Vec::new();

        // Try prefix comparison for each unique value
        for i in 0..num_unique {
            let prefix7 = self.compact_offset_views.get_prefix7(i);

            // Compare prefix with needle_suffix
            let cmp_len = std::cmp::min(OffsetView::prefix_len(), needle_suffix.len());
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
            let raw_buffer = self.fsst_buffer.get_fsst_buffer()?;

            let mut decompressed_buffer = Vec::with_capacity(1024 * 1024 * 2);
            for &i in &needs_full_comparison {
                let start_offset = self.compact_offset_views.get_offset(i);
                let end_offset = self.compact_offset_views.get_offset(i + 1);

                let compressed_value = raw_buffer.get_compressed_slice(start_offset, end_offset);
                decompressed_buffer.clear();
                let decompressed_len = unsafe {
                    let slice = std::slice::from_raw_parts_mut(
                        decompressed_buffer.as_mut_ptr() as *mut std::mem::MaybeUninit<u8>,
                        decompressed_buffer.capacity(),
                    );
                    self.compressor
                        .decompressor()
                        .decompress_into(compressed_value, slice)
                };
                unsafe {
                    decompressed_buffer.set_len(decompressed_len);
                }

                let value_cmp = decompressed_buffer.as_slice().cmp(needle);
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
                dict_results[i] = result;
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
    ) -> Result<BooleanArray, IoRange> {
        let dict_array = self.to_dict_arrow()?;
        let needle_scalar = datafusion::common::ScalarValue::Binary(Some(needle.to_vec()));
        let lhs = ColumnarValue::Array(Arc::new(dict_array));
        let rhs = ColumnarValue::Scalar(needle_scalar);

        let result = match op {
            Operator::LikeMatch => apply_cmp(&lhs, &rhs, arrow::compute::like),
            Operator::ILikeMatch => apply_cmp(&lhs, &rhs, arrow::compute::ilike),
            Operator::NotLikeMatch => apply_cmp(&lhs, &rhs, arrow::compute::nlike),
            Operator::NotILikeMatch => apply_cmp(&lhs, &rhs, arrow::compute::nilike),
            _ => {
                unreachable!()
            }
        };

        match result.expect("ArrowError") {
            ColumnarValue::Array(arr) => Ok(arr.as_boolean().clone()),
            ColumnarValue::Scalar(_) => unreachable!(),
        }
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

    fn sort_to_indices_inner(&self) -> Result<UInt32Array, IoRange> {
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
    fn get_dictionary_ranks(&self) -> Result<Vec<u16>, IoRange> {
        let num_unique = self.compact_offset_views.len().saturating_sub(1);
        let mut dict_indices: Vec<u32> = (0..num_unique as u32).collect();

        let mut decompressed: Option<BinaryArray> = None;
        let raw_buffer = self.fsst_buffer.get_fsst_buffer()?;

        // Sort using prefix optimization first, then full strings when needed
        dict_indices.sort_unstable_by(|&a, &b| unsafe {
            // First try prefix comparison - no need to include shared_prefix since all strings have it
            let prefix_a = self.compact_offset_views.get_prefix7(a as usize);
            let prefix_b = self.compact_offset_views.get_prefix7(b as usize);

            let prefix_cmp = prefix_a.cmp(prefix_b);

            if prefix_cmp != std::cmp::Ordering::Equal {
                // Prefix comparison is sufficient
                prefix_cmp
            } else {
                // Prefixes are equal, need full string comparison
                // This will trigger decompression on first call if needed
                match &decompressed {
                    Some(decompressed) => {
                        let string_a = decompressed.value_unchecked(a as usize);
                        let string_b = decompressed.value_unchecked(b as usize);
                        string_a.cmp(string_b)
                    }
                    None => {
                        let offset_views = self.offset_views();
                        let (values_buffer, offsets_buffer) = raw_buffer
                            .to_uncompressed(&self.compressor.decompressor(), &offset_views);

                        let binary_array =
                            BinaryArray::new_unchecked(offsets_buffer, values_buffer, None);

                        let string_a = binary_array.value(a as usize);
                        let string_b = binary_array.value(b as usize);
                        let rt = string_a.cmp(string_b);
                        decompressed = Some(binary_array);
                        rt
                    }
                }
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

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::Array;
    use rand::{Rng, SeedableRng};

    #[test]
    fn test_dictionary_view_structure() {
        // Test OffsetView structure
        let offset_view = OffsetView::from_parts(1024, [1, 2, 3, 4, 5, 6, 7], 7);
        assert_eq!(offset_view.offset(), 1024);
        assert_eq!(offset_view.prefix7(), &[1, 2, 3, 4, 5, 6, 7]);

        // Test UInt16Array creation (dictionary keys are now stored directly in UInt16Array)
        let keys = UInt16Array::from(vec![42, 100, 255]);
        assert_eq!(keys.value(0), 42);
        assert_eq!(keys.value(1), 100);
        assert_eq!(keys.value(2), 255);
    }

    #[test]
    fn test_prefix_extraction() {
        let input = StringArray::from(vec!["hello", "world", "test"]);
        let compressor = LiquidByteViewArray::<MemoryBuffer>::train_compressor(input.iter());
        let liquid_array =
            LiquidByteViewArray::<MemoryBuffer>::from_string_array(&input, compressor);

        // With no shared prefix, the offset view prefixes should be the original strings (truncated to 7 bytes)
        assert_eq!(liquid_array.shared_prefix, Vec::<u8>::new());
        assert_eq!(liquid_array.compact_offset_views.get_prefix7(0), b"hello\0\0");
        assert_eq!(liquid_array.compact_offset_views.get_prefix7(1), b"world\0\0");
        assert_eq!(liquid_array.compact_offset_views.get_prefix7(2), b"test\0\0\0");
    }

    #[test]
    fn test_shared_prefix_functionality() {
        // Test case with shared prefix
        let input = StringArray::from(vec![
            "hello_world",
            "hello_rust",
            "hello_test",
            "hello_code",
        ]);
        let compressor = LiquidByteViewArray::<MemoryBuffer>::train_compressor(input.iter());
        let liquid_array =
            LiquidByteViewArray::<MemoryBuffer>::from_string_array(&input, compressor);

        // Should extract "hello_" as shared prefix
        assert_eq!(liquid_array.shared_prefix, b"hello_");

        // Offset view prefixes (7 bytes) and lengths
        assert_eq!(liquid_array.compact_offset_views.get_prefix7(0), b"world\0\0");
        assert_eq!(liquid_array.compact_offset_views.get_prefix7(1), b"rust\0\0\0");
        assert_eq!(liquid_array.compact_offset_views.get_prefix7(2), b"test\0\0\0");
        assert_eq!(liquid_array.compact_offset_views.get_prefix7(3), b"code\0\0\0");

        // Test roundtrip - should reconstruct original strings correctly
        let output = liquid_array.to_arrow_array().unwrap();
        assert_eq!(&input, output.as_string::<i32>());

        // Test comparison with shared prefix optimization
        let result = liquid_array.compare_equals(b"hello_rust").unwrap();
        let expected = BooleanArray::from(vec![false, true, false, false]);
        assert_eq!(result, expected);

        // Test comparison that doesn't match shared prefix
        let result = liquid_array.compare_equals(b"goodbye_world").unwrap();
        let expected = BooleanArray::from(vec![false, false, false, false]);
        assert_eq!(result, expected);

        // Test partial shared prefix match
        let result = liquid_array.compare_equals(b"hello_").unwrap();
        let expected = BooleanArray::from(vec![false, false, false, false]);
        assert_eq!(result, expected);
    }

    #[test]
    fn test_shared_prefix_with_short_strings() {
        // Test case: short strings that fit entirely in the 6-byte prefix
        let input = StringArray::from(vec!["abc", "abcde", "abcdef", "abcdefg"]);
        let compressor = LiquidByteViewArray::<MemoryBuffer>::train_compressor(input.iter());
        let liquid_array =
            LiquidByteViewArray::<MemoryBuffer>::from_string_array(&input, compressor);

        // Should extract "abc" as shared prefix
        assert_eq!(liquid_array.shared_prefix, b"abc");

        // Offset view prefixes should be the remaining parts after shared prefix (7 bytes)
        assert_eq!(liquid_array.compact_offset_views.get_prefix7(0), &[0u8; 7]); // empty after "abc"
        assert_eq!(liquid_array.compact_offset_views.get_prefix7(1), b"de\0\0\0\0\0"); // "de" after "abc"
        assert_eq!(liquid_array.compact_offset_views.get_prefix7(2), b"def\0\0\0\0"); // "def" after "abc"
        assert_eq!(liquid_array.compact_offset_views.get_prefix7(3), b"defg\0\0\0"); // "defg" after "abc"

        // Test roundtrip
        let output = liquid_array.to_arrow_array().unwrap();
        assert_eq!(&input, output.as_string::<i32>());

        // Test equality comparisons with short strings
        let result = liquid_array.compare_equals(b"abc").unwrap();
        let expected = BooleanArray::from(vec![true, false, false, false]);
        assert_eq!(result, expected);

        let result = liquid_array.compare_equals(b"abcde").unwrap();
        let expected = BooleanArray::from(vec![false, true, false, false]);
        assert_eq!(result, expected);

        // Test ordering comparisons that can be resolved by shared prefix
        let result = liquid_array.compare_with(b"ab", &Operator::Gt).unwrap();
        let expected = BooleanArray::from(vec![true, true, true, true]); // All start with "abc" > "ab"
        assert_eq!(result, expected);

        let result = liquid_array.compare_with(b"abcd", &Operator::Lt).unwrap();
        let expected = BooleanArray::from(vec![true, false, false, false]); // Only "abc" < "abcd"
        assert_eq!(result, expected);
    }

    #[test]
    fn test_shared_prefix_contains_complete_strings() {
        // Test case: shared prefix completely contains some strings
        let input = StringArray::from(vec!["data", "database", "data_entry", "data_", "datatype"]);
        let compressor = LiquidByteViewArray::<MemoryBuffer>::train_compressor(input.iter());
        let liquid_array =
            LiquidByteViewArray::<MemoryBuffer>::from_string_array(&input, compressor);

        // Should extract "data" as shared prefix
        assert_eq!(liquid_array.shared_prefix, b"data");

        // Offset view prefixes should be the remaining parts (7 bytes)
        assert_eq!(liquid_array.compact_offset_views.get_prefix7(0), &[0u8; 7]); // "data" - empty remainder
        assert_eq!(liquid_array.compact_offset_views.get_prefix7(1), b"base\0\0\0"); // "database" - "base" remainder
        assert_eq!(liquid_array.compact_offset_views.get_prefix7(2), b"_entry\0"); // "data_entry" - "_entry" remainder
        assert_eq!(liquid_array.compact_offset_views.get_prefix7(3), b"_\0\0\0\0\0\0"); // "data_" - "_" remainder
        assert_eq!(liquid_array.compact_offset_views.get_prefix7(4), b"type\0\0\0"); // "datatype" - "type" remainder

        // Test roundtrip
        let output = liquid_array.to_arrow_array().unwrap();
        assert_eq!(&input, output.as_string::<i32>());

        // Test equality with exact shared prefix
        let result = liquid_array.compare_equals(b"data").unwrap();
        let expected = BooleanArray::from(vec![true, false, false, false, false]);
        assert_eq!(result, expected);

        // Test comparisons where shared prefix helps
        let result = liquid_array.compare_with(b"dat", &Operator::Gt).unwrap();
        let expected = BooleanArray::from(vec![true, true, true, true, true]); // All > "dat"
        assert_eq!(result, expected);

        let result = liquid_array.compare_with(b"datab", &Operator::Lt).unwrap();
        let expected = BooleanArray::from(vec![true, false, true, true, false]); // "data", "data_entry", and "data_" < "datab"
        assert_eq!(result, expected);

        // Test comparison with needle shorter than shared prefix
        let result = liquid_array.compare_with(b"da", &Operator::Gt).unwrap();
        let expected = BooleanArray::from(vec![true, true, true, true, true]); // All > "da"
        assert_eq!(result, expected);

        // Test comparison with needle equal to shared prefix
        let result = liquid_array.compare_with(b"data", &Operator::GtEq).unwrap();
        let expected = BooleanArray::from(vec![true, true, true, true, true]); // All >= "data"
        assert_eq!(result, expected);

        let result = liquid_array.compare_with(b"data", &Operator::Gt).unwrap();
        let expected = BooleanArray::from(vec![false, true, true, true, true]); // All except exact "data" > "data"
        assert_eq!(result, expected);
    }

    #[test]
    fn test_shared_prefix_corner_case() {
        let input = StringArray::from(vec!["data", "database", "data_entry", "data_", "datatype"]);
        let compressor = LiquidByteViewArray::<MemoryBuffer>::train_compressor(input.iter());
        let liquid_array =
            LiquidByteViewArray::<MemoryBuffer>::from_string_array(&input, compressor);
        let result = liquid_array.compare_with(b"data", &Operator::GtEq).unwrap();
        let expected = BooleanArray::from(vec![true, true, true, true, true]); // All >= "data"
        assert_eq!(result, expected);
    }

    #[test]
    fn test_shared_prefix_edge_cases() {
        // Test case 1: All strings are the same (full shared prefix)
        let input = StringArray::from(vec!["identical", "identical", "identical"]);
        let compressor = LiquidByteViewArray::<MemoryBuffer>::train_compressor(input.iter());
        let liquid_array =
            LiquidByteViewArray::<MemoryBuffer>::from_string_array(&input, compressor);

        assert_eq!(liquid_array.shared_prefix, b"identical");
        // All offset view prefixes should be empty
        for i in 0..liquid_array.compact_offset_views.len() {
            assert_eq!(liquid_array.compact_offset_views.get_prefix7(i), &[0u8; 7]);
        }

        // Test roundtrip
        let output = liquid_array.to_arrow_array().unwrap();
        assert_eq!(&input, output.as_string::<i32>());

        // Test case 2: One string is a prefix of others
        let input = StringArray::from(vec!["hello", "hello_world", "hello_test"]);
        let compressor = LiquidByteViewArray::<MemoryBuffer>::train_compressor(input.iter());
        let liquid_array =
            LiquidByteViewArray::<MemoryBuffer>::from_string_array(&input, compressor);

        assert_eq!(liquid_array.shared_prefix, b"hello");
        assert_eq!(liquid_array.compact_offset_views.get_prefix7(0), &[0u8; 7]); // empty after "hello"
        assert_eq!(liquid_array.compact_offset_views.get_prefix7(1), b"_world\0");
        assert_eq!(liquid_array.compact_offset_views.get_prefix7(2), b"_test\0\0");

        // Test roundtrip
        let output = liquid_array.to_arrow_array().unwrap();
        assert_eq!(&input, output.as_string::<i32>());

        // Test case 3: Empty string in array (should limit shared prefix)
        let input = StringArray::from(vec!["", "hello", "hello_world"]);
        let compressor = LiquidByteViewArray::<MemoryBuffer>::train_compressor(input.iter());
        let liquid_array =
            LiquidByteViewArray::<MemoryBuffer>::from_string_array(&input, compressor);

        assert_eq!(liquid_array.shared_prefix, Vec::<u8>::new()); // empty shared prefix
        assert_eq!(liquid_array.compact_offset_views.get_prefix7(0), &[0u8; 7]);
        assert_eq!(liquid_array.compact_offset_views.get_prefix7(1), b"hello\0\0");
        assert_eq!(liquid_array.compact_offset_views.get_prefix7(2), b"hello_w"); // "hello_world" truncated to 7 bytes

        // Test roundtrip
        let output = liquid_array.to_arrow_array().unwrap();
        assert_eq!(&input, output.as_string::<i32>());
    }

    #[test]
    fn test_memory_layout() {
        let input = StringArray::from(vec!["hello", "world", "test"]);
        let compressor = LiquidByteViewArray::<MemoryBuffer>::train_compressor(input.iter());
        let liquid_array =
            LiquidByteViewArray::<MemoryBuffer>::from_string_array(&input, compressor);

        // Verify memory layout components
        assert_eq!(liquid_array.dictionary_keys.len(), 3);
        assert_eq!(liquid_array.compact_offset_views.len(), 4);
        assert!(liquid_array.nulls().is_none());
        let _raw_buffer = liquid_array.fsst_buffer.get_fsst_buffer().unwrap();
    }

    fn check_filter_result(input: &StringArray, filter: BooleanBuffer) {
        let compressor = LiquidByteViewArray::<MemoryBuffer>::train_compressor(input.iter());
        let liquid_array =
            LiquidByteViewArray::<MemoryBuffer>::from_string_array(input, compressor);
        let filtered = liquid_array.filter(&filter);
        let output = filtered.to_arrow_array();
        let expected = {
            let selection = BooleanArray::new(filter.clone(), None);
            let arrow_filtered = arrow::compute::filter(&input, &selection).unwrap();
            arrow_filtered.as_string::<i32>().clone()
        };
        assert_eq!(output.as_ref(), &expected);
    }

    #[test]
    fn test_filter_functionality() {
        let input = StringArray::from(vec![
            Some("hello"),
            Some("test"),
            None,
            Some("test"),
            None,
            Some("test"),
            Some("rust"),
        ]);
        let mut seeded_rng = rand::rngs::StdRng::seed_from_u64(42);
        for _i in 0..100 {
            let filter =
                BooleanBuffer::from_iter((0..input.len()).map(|_| seeded_rng.random::<bool>()));
            check_filter_result(&input, filter);
        }
    }

    #[test]
    fn test_memory_efficiency() {
        let input = StringArray::from(vec!["hello", "world", "hello", "world", "hello"]);
        let compressor = LiquidByteViewArray::<MemoryBuffer>::train_compressor(input.iter());
        let liquid_array =
            LiquidByteViewArray::<MemoryBuffer>::from_string_array(&input, compressor);

        // Verify that dictionary views store unique values efficiently
        assert_eq!(liquid_array.dictionary_keys.len(), 5);

        // Verify that FSST buffer contains unique values
        let dict = liquid_array.to_dict_arrow().unwrap();
        assert_eq!(dict.values().len(), 2); // Only "hello" and "world"
    }

    #[test]
    fn test_to_best_arrow_array() {
        let input = StringArray::from(vec!["hello", "world", "test"]);
        let compressor = LiquidByteViewArray::<MemoryBuffer>::train_compressor(input.iter());
        let liquid_array =
            LiquidByteViewArray::<MemoryBuffer>::from_string_array(&input, compressor);

        let best_array = liquid_array.to_best_arrow_array();
        let dict_array = best_array.as_dictionary::<UInt16Type>();

        // Should return dictionary array as the best encoding
        assert_eq!(dict_array.len(), 3);
        assert_eq!(dict_array.values().len(), 3); // Three unique values
    }

    #[test]
    fn test_data_type() {
        let input = StringArray::from(vec!["hello", "world"]);
        let compressor = LiquidByteViewArray::<MemoryBuffer>::train_compressor(input.iter());
        let liquid_array =
            LiquidByteViewArray::<MemoryBuffer>::from_string_array(&input, compressor);

        // Just verify we can get the data type without errors
        let data_type = liquid_array.data_type();
        assert!(matches!(data_type, LiquidDataType::ByteViewArray));
    }

    #[test]
    fn test_compare_with_prefix_optimization_fast_path() {
        // Test case 1: Prefix comparison can decide most results without decompression
        // Uses strings with distinct prefixes to test the fast path
        let input = StringArray::from(vec![
            "apple123",  // prefix: "apple\0"
            "banana456", // prefix: "banana"
            "cherry789", // prefix: "cherry"
            "apple999",  // prefix: "apple\0" (same as first)
            "zebra000",  // prefix: "zebra\0"
        ]);

        let compressor = LiquidByteViewArray::<MemoryBuffer>::train_compressor(input.iter());
        let liquid_array =
            LiquidByteViewArray::<MemoryBuffer>::from_string_array(&input, compressor);

        // Test Lt with needle "car" (prefix: "car\0\0\0")
        // Expected: "apple123" < "car" => true, "banana456" < "car" => true, others false
        let result = liquid_array
            .compare_with_inner(b"car", &Operator::Lt)
            .unwrap();
        let expected = BooleanArray::from(vec![true, true, false, true, false]);
        assert_eq!(result, expected);

        // Test Gt with needle "dog" (prefix: "dog\0\0\0")
        // Expected: only "zebra000" > "dog" => true
        let result = liquid_array
            .compare_with_inner(b"dog", &Operator::Gt)
            .unwrap();
        let expected = BooleanArray::from(vec![false, false, false, false, true]);
        assert_eq!(result, expected);

        // Test GtEq with needle "apple" (prefix: "apple\0")
        // Expected: all except "apple123" and "apple999" need decompression, others by prefix
        let result = liquid_array
            .compare_with_inner(b"apple", &Operator::GtEq)
            .unwrap();
        let expected = BooleanArray::from(vec![true, true, true, true, true]);
        assert_eq!(result, expected);
    }

    #[test]
    fn test_compare_with_prefix_optimization_decompression_path() {
        // Test case 2: Cases where prefix comparison is inconclusive and requires decompression
        // Uses strings with identical prefixes but different suffixes
        let input = StringArray::from(vec![
            "prefix_aaa", // prefix: "prefix"
            "prefix_bbb", // prefix: "prefix" (same prefix)
            "prefix_ccc", // prefix: "prefix" (same prefix)
            "prefix_abc", // prefix: "prefix" (same prefix)
            "different",  // prefix: "differ" (different prefix)
        ]);

        let compressor = LiquidByteViewArray::<MemoryBuffer>::train_compressor(input.iter());
        let liquid_array =
            LiquidByteViewArray::<MemoryBuffer>::from_string_array(&input, compressor);

        // Test Lt with needle "prefix_b" - this will require decompression for prefix matches
        // Expected: "prefix_aaa" < "prefix_b" => true, "prefix_bbb" < "prefix_b" => false, etc.
        let result = liquid_array
            .compare_with_inner(b"prefix_b", &Operator::Lt)
            .unwrap();
        let expected = BooleanArray::from(vec![true, false, false, true, true]);
        assert_eq!(result, expected);

        // Test LtEq with needle "prefix_bbb" - exact match case with decompression
        let result = liquid_array
            .compare_with_inner(b"prefix_bbb", &Operator::LtEq)
            .unwrap();
        let expected = BooleanArray::from(vec![true, true, false, true, true]);
        assert_eq!(result, expected);

        // Test Gt with needle "prefix_abc" - requires decompression for prefix matches
        let result = liquid_array
            .compare_with_inner(b"prefix_abc", &Operator::Gt)
            .unwrap();
        let expected = BooleanArray::from(vec![false, true, true, false, false]);
        assert_eq!(result, expected);
    }

    #[test]
    fn test_compare_with_prefix_optimization_edge_cases_and_nulls() {
        // Test case 3: Edge cases including nulls, empty strings, and boundary conditions
        let input = StringArray::from(vec![
            Some(""),           // Empty string (prefix: "\0\0\0\0\0\0")
            None,               // Null value
            Some("a"),          // Single character (prefix: "a\0\0\0\0\0")
            Some("abcdef"),     // Exactly 6 chars (prefix: "abcdef")
            Some("abcdefghij"), // Longer than 6 chars (prefix: "abcdef")
            Some("abcdeg"),     // Differs at position 5 (prefix: "abcdeg")
        ]);

        let compressor = LiquidByteViewArray::<MemoryBuffer>::train_compressor(input.iter());
        let liquid_array =
            LiquidByteViewArray::<MemoryBuffer>::from_string_array(&input, compressor);

        // Test Lt with empty string needle - should test null handling
        let result = liquid_array.compare_with_inner(b"", &Operator::Lt).unwrap();
        let expected = BooleanArray::from(vec![
            Some(false),
            None,
            Some(false),
            Some(false),
            Some(false),
            Some(false),
        ]);
        assert_eq!(result, expected);

        // Test Gt with needle "abcdef" - tests exact prefix match requiring decompression
        let result = liquid_array
            .compare_with_inner(b"abcdef", &Operator::Gt)
            .unwrap();
        let expected = BooleanArray::from(vec![
            Some(false),
            None,
            Some(false),
            Some(false),
            Some(true),
            Some(true),
        ]);
        assert_eq!(result, expected);

        // Test LtEq with needle "b" - tests single character comparisons
        let result = liquid_array
            .compare_with_inner(b"b", &Operator::LtEq)
            .unwrap();
        let expected = BooleanArray::from(vec![
            Some(true),
            None,
            Some(true),
            Some(true),
            Some(true),
            Some(true),
        ]);
        assert_eq!(result, expected);

        // Test GtEq with needle "abcdeg" - tests decompression when prefix exactly matches needle prefix
        let result = liquid_array
            .compare_with_inner(b"abcdeg", &Operator::GtEq)
            .unwrap();
        // b"" >= b"abcdeg" => false
        // null => null
        // b"a" >= b"abcdeg" => false
        // b"abcdef" >= b"abcdeg" => false (because b"abcdef" < b"abcdeg" since 'f' < 'g')
        // b"abcdefghij" >= b"abcdeg" => false (because b"abcdefghij" < b"abcdeg" since 'f' < 'g')
        // b"abcdeg" >= b"abcdeg" => true (exact match)
        let expected = BooleanArray::from(vec![
            Some(false),
            None,
            Some(false),
            Some(false),
            Some(false),
            Some(true),
        ]);
        assert_eq!(result, expected);
    }

    #[test]
    fn test_compare_with_prefix_optimization_utf8_and_binary() {
        // Test case 4: UTF-8 encoded strings and binary data comparisons
        // This demonstrates the advantage of byte-level comparison
        let input = StringArray::from(vec![
            "café",   // UTF-8: [99, 97, 102, 195, 169]
            "naïve",  // UTF-8: [110, 97, 195, 175, 118, 101]
            "résumé", // UTF-8: [114, 195, 169, 115, 117, 109, 195, 169]
            "hello",  // ASCII: [104, 101, 108, 108, 111]
            "世界",   // UTF-8: [228, 184, 150, 231, 149, 140]
        ]);

        let compressor = LiquidByteViewArray::<MemoryBuffer>::train_compressor(input.iter());
        let liquid_array =
            LiquidByteViewArray::<MemoryBuffer>::from_string_array(&input, compressor);

        // Test Lt with UTF-8 needle "naïve" (UTF-8: [110, 97, 195, 175, 118, 101])
        // Expected: "café" < "naïve" => true (99 < 110), "hello" < "naïve" => true, others false
        let naive_bytes = "naïve".as_bytes(); // [110, 97, 195, 175, 118, 101]
        let result = liquid_array
            .compare_with_inner(naive_bytes, &Operator::Lt)
            .unwrap();
        let expected = BooleanArray::from(vec![true, false, false, true, false]);
        assert_eq!(result, expected);

        // Test Gt with UTF-8 needle "café" (UTF-8: [99, 97, 102, 195, 169])
        // Expected: strings with first byte > 99 should be true
        let cafe_bytes = "café".as_bytes(); // [99, 97, 102, 195, 169]
        let result = liquid_array
            .compare_with_inner(cafe_bytes, &Operator::Gt)
            .unwrap();
        let expected = BooleanArray::from(vec![false, true, true, true, true]);
        assert_eq!(result, expected);

        // Test LtEq with Chinese characters "世界" (UTF-8: [228, 184, 150, 231, 149, 140])
        // Expected: only strings with first byte <= 228 should be true, but since 228 is quite high,
        // most Latin characters will be true
        let world_bytes = "世界".as_bytes(); // [228, 184, 150, 231, 149, 140]
        let result = liquid_array
            .compare_with_inner(world_bytes, &Operator::LtEq)
            .unwrap();
        let expected = BooleanArray::from(vec![true, true, true, true, true]);
        assert_eq!(result, expected);

        // Test exact equality with "résumé" using GtEq and LtEq to verify byte-level precision
        let resume_bytes = "résumé".as_bytes(); // [114, 195, 169, 115, 117, 109, 195, 169]
        let gte_result = liquid_array
            .compare_with_inner(resume_bytes, &Operator::GtEq)
            .unwrap();
        let lte_result = liquid_array
            .compare_with_inner(resume_bytes, &Operator::LtEq)
            .unwrap();

        // Check GtEq and LtEq results separately
        // GtEq: "café"(99) >= "résumé"(114) => false, "naïve"(110) >= "résumé"(114) => false,
        //       "résumé"(114) >= "résumé"(114) => true, "hello"(104) >= "résumé"(114) => false,
        //       "世界"(228) >= "résumé"(114) => true
        let gte_expected = BooleanArray::from(vec![false, false, true, false, true]);
        // LtEq: all strings with first byte <= 114 should be true, "世界"(228) should be false
        let lte_expected = BooleanArray::from(vec![true, true, true, true, false]);
        assert_eq!(gte_result, gte_expected);
        assert_eq!(lte_result, lte_expected);
    }

    fn sort_to_indices_test(input: Vec<Option<&str>>, expected_idx: Vec<u32>) {
        let input = StringArray::from(input);
        let compressor = LiquidByteViewArray::<MemoryBuffer>::train_compressor(input.iter());
        let liquid_array =
            LiquidByteViewArray::<MemoryBuffer>::from_string_array(&input, compressor);
        let indices = liquid_array.sort_to_indices().unwrap();
        assert_eq!(indices, UInt32Array::from(expected_idx));
    }

    #[test]
    fn test_sort_to_indices_basic() {
        sort_to_indices_test(
            vec![Some("zebra"), Some("apple"), Some("banana"), Some("cherry")],
            vec![1, 2, 3, 0],
        );

        // with nulls
        sort_to_indices_test(
            vec![
                Some("zebra"),
                None,
                Some("apple"),
                Some("banana"),
                None,
                Some("cherry"),
            ],
            // Expected order: null(1), null(4), apple(2), banana(3), cherry(5), zebra(0)
            vec![1, 4, 2, 3, 5, 0],
        );

        // shared prefix
        sort_to_indices_test(
            vec![
                Some("prefix_zebra"),
                Some("prefix_apple"),
                Some("prefix_banana"),
                Some("prefix_cherry"),
            ],
            vec![1, 2, 3, 0],
        );

        // with duplicate values
        sort_to_indices_test(
            vec![
                Some("apple"),
                Some("banana"),
                Some("apple"),
                Some("cherry"),
                Some("banana"),
            ],
            vec![0, 2, 1, 4, 3],
        );
    }

    #[test]
    fn test_sort_to_indices_test_full_data_comparison() {
        sort_to_indices_test(
            vec![
                Some("pre_apple_abc"),
                Some("pre_banana_abc"),
                Some("pre_apple_def"),
            ],
            vec![0, 2, 1],
        );

        sort_to_indices_test(
            vec![
                Some("pre_apple_abc"),
                Some("pre_banana_abc"),
                Some("pre_appll_abc"),
            ],
            vec![0, 2, 1],
        );
    }

    fn test_compare_equals(input: StringArray, needle: &[u8], expected: BooleanArray) {
        let compressor = LiquidByteViewArray::<MemoryBuffer>::train_compressor(input.iter());
        let liquid_array =
            LiquidByteViewArray::<MemoryBuffer>::from_string_array(&input, compressor);
        let result = liquid_array.compare_equals(needle).unwrap();
        assert_eq!(result, expected);
    }

    #[test]
    fn test_compare_equals_on_disk() {
        let input = StringArray::from(vec![
            Some("apple_orange"),
            None,
            Some("apple_orange_long_string"),
            Some("apple_b"),
            Some("apple_oo_long_string"),
            Some("apple_b"),
            Some("apple"),
        ]);
        test_compare_equals(
            input.clone(),
            b"apple",
            BooleanArray::from(vec![
                Some(false),
                None,
                Some(false),
                Some(false),
                Some(false),
                Some(false),
                Some(true),
            ]),
        );
        test_compare_equals(
            input.clone(),
            b"",
            BooleanArray::from(vec![
                Some(false),
                None,
                Some(false),
                Some(false),
                Some(false),
                Some(false),
                Some(false),
            ]),
        );
        test_compare_equals(
            input.clone(),
            b"apple_b",
            BooleanArray::from(vec![
                Some(false),
                None,
                Some(false),
                Some(true),
                Some(false),
                Some(true),
                Some(false),
            ]),
        );
        test_compare_equals(
            input.clone(),
            b"apple_oo_long_string",
            BooleanArray::from(vec![
                Some(false),
                None,
                Some(false),
                Some(false),
                Some(true),
                Some(false),
                Some(false),
            ]),
        );
    }

    #[test]
    fn test_compare_equals_with_prefix_decidable_and_ambiguous() {
        // Craft values so we cover:
        // - decidable by small length (<= prefix_len): exact length + content match
        // - ambiguous for medium length (>prefix_len and <255): same first 7 bytes, length equal
        // - ambiguous for long length (>=255): same first 7 bytes, unknown stored len

        // Build strings with a shared prefix to exercise suffix logic
        let short_match = "pre_abc"; // after shared prefix: len<=7
        let medium_value = "pre_1234567X"; // suffix len 8, first 7 are 1234567
        let long_suffix: String = std::iter::repeat_n('a', 260).collect();
        let long_value = format!("pre_{long_suffix}"); // suffix len >= 255

        let input = StringArray::from(vec![
            Some(short_match),         // index 0
            Some(medium_value),        // index 1
            Some(long_value.as_str()), // index 2
        ]);

        let compressor = LiquidByteViewArray::<MemoryBuffer>::train_compressor(input.iter());
        let in_mem = LiquidByteViewArray::<MemoryBuffer>::from_string_array(&input, compressor);

        // Squeeze to disk-backed so we exercise compare_equals_with_prefix in DiskBuffer path
        let (hybrid, _bytes) = in_mem.squeeze().unwrap();
        let disk_view = hybrid
            .as_any()
            .downcast_ref::<LiquidByteViewArray<DiskBuffer>>()
            .unwrap()
            .clone();

        // 1) Decidable case: needle equals short_match exactly
        let decidable = disk_view.compare_equals_with_prefix(short_match.as_bytes());
        assert!(decidable.is_some(), "should be decidable without IO");
        let arr = decidable.unwrap();
        assert_eq!(arr, BooleanArray::from(vec![true, false, false]));

        // 2) Ambiguous medium: construct needle with same first 7 bytes as medium_value's suffix
        // medium_value is pre_1234567X, needle is pre_1234567Y (same 7, differs at 8th)
        let medium_needle = b"pre_1234567Y".to_vec();
        let ambiguous_medium = disk_view.compare_equals_with_prefix(&medium_needle);
        assert!(
            ambiguous_medium.is_none(),
            "medium case should be ambiguous"
        );

        // 3) Ambiguous long: build a needle with len>=255 and same first 7 suffix bytes as long_value
        let long_needle = {
            let mut s = String::from("pre_");
            s.push_str(&std::iter::repeat_n('a', 300).collect::<String>());
            s.into_bytes()
        };
        let ambiguous_long = disk_view.compare_equals_with_prefix(&long_needle);
        assert!(ambiguous_long.is_none(), "long case should be ambiguous");
    }

    #[test]
    fn test_compact_offset_view_round_trip() {
        // Test 1: Small offsets (should use OneByte variant)
        let small_offsets = vec![
            OffsetView::new(100, b"hello"),
            OffsetView::new(105, b"world"),
            OffsetView::new(110, b"test"),
        ];
        test_round_trip(&small_offsets, "small offsets");

        // Test 2: Medium offsets (should use TwoBytes variant)
        let medium_offsets = vec![
            OffsetView::new(1000, b"medium1"),
            OffsetView::new(2000, b"medium2"),
            OffsetView::new(3000, b"medium3"),
        ];
        test_round_trip(&medium_offsets, "medium offsets");

        // Test 3: Large offsets (should use FourBytes variant)
        let large_offsets = vec![
            OffsetView::new(100000, b"large1"),
            OffsetView::new(200000, b"large2"),
            OffsetView::new(300000, b"large3"),
        ];
        test_round_trip(&large_offsets, "large offsets");

        // Test 4: Mixed scenario with varying prefix lengths
        let mixed_offsets = vec![
            OffsetView::new(1000, b"a"),                    // 1 byte prefix
            OffsetView::new(1010, b"abcdef"),              // 6 byte prefix
            OffsetView::new(1020, b"abcdefg"),             // 7 byte prefix (max)
            OffsetView::new(1030, b"abcdefgh"),            // 8 bytes (7 stored + len)
            OffsetView::new(1040, &vec![b'x'; 300]),       // 300 bytes (long string, len=255)
        ];
        test_round_trip(&mixed_offsets, "mixed scenarios");

        // Test 5: Edge case - empty offsets
        let empty_offsets = vec![];
        test_round_trip(&empty_offsets, "empty offsets");

        // Test 6: Single offset
        let single_offset = vec![OffsetView::new(42, b"single")];
        test_round_trip(&single_offset, "single offset");
    }

    fn test_round_trip(original_offsets: &[OffsetView], test_name: &str) {
        // convert to compact representation
        let compact = CompactOffsetViewGroup::from_offset_views(original_offsets);
        
        // convert back to offset views
        let recovered_offsets = convert_compact_to_offset_views(&compact);
        
        // verify they match
        assert_eq!(
            original_offsets.len(), 
            recovered_offsets.len(),
            "Length mismatch in {}", test_name
        );
        
        for (i, (original, recovered)) in original_offsets.iter().zip(&recovered_offsets).enumerate() {
            assert_eq!(
                original.offset(), 
                recovered.offset(),
                "Offset mismatch at index {} in {}", i, test_name
            );
            assert_eq!(
                original.prefix7(), 
                recovered.prefix7(),
                "Prefix mismatch at index {} in {}", i, test_name
            );
            assert_eq!(
                original.len_byte(), 
                recovered.len_byte(),
                "Length byte mismatch at index {} in {}", i, test_name
            );
        }
    }

    fn convert_compact_to_offset_views(compact: &CompactOffsetViewGroup) -> Vec<OffsetView> {
        let mut result = Vec::new();
        for i in 0..compact.len() {
            let offset = compact.get_offset(i);
            let prefix7 = *compact.get_prefix7(i);
            let len_byte = compact.get_len_byte(i);
            result.push(OffsetView::from_parts(offset, prefix7, len_byte));
        }
        result
    }

    #[test]
    fn test_compact_offset_view_memory_efficiency() {
        // test that compaction actually saves memory
        let offsets = vec![
            OffsetView::new(1000, b"test1"),
            OffsetView::new(1010, b"test2"),
            OffsetView::new(1020, b"test3"),
            OffsetView::new(1030, b"test4"),
        ];
        
        let original_size = offsets.len() * std::mem::size_of::<OffsetView>();
        let compact = CompactOffsetViewGroup::from_offset_views(&offsets);
        let compact_size = compact.memory_usage();
        
        // for this test case, we should see some savings due to using smaller residuals
        assert!(compact_size <= original_size, "Compact representation should not be larger");
    }

    // Benchmark tests for v2 offset compression improvements
    fn generate_mixed_size_strings(count: usize, seed: u64) -> Vec<String> {
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        let mut strings = Vec::with_capacity(count);
        
        for _ in 0..count {
            let size_type = rng.random_range(0..4);
            let string = match size_type {
                0 => {
                    // Very short strings (1-3 chars) - stress test prefix optimization
                    let len = rng.random_range(1..=3);
                    (0..len).map(|_| rng.random_range(b'a'..=b'z') as char).collect()
                }
                1 => {
                    // Medium strings (50-200 chars) - test offset compression
                    let len = rng.random_range(50..=200);
                    (0..len).map(|_| rng.random_range(b'a'..=b'z') as char).collect()
                }
                2 => {
                    // Long strings (1000-5000 chars) - stress test linear regression
                    let len = rng.random_range(1000..=5000);
                    (0..len).map(|_| rng.random_range(b'a'..=b'z') as char).collect()
                }
                _ => {
                    // Very long strings (10k+ chars) - edge case for offset compression
                    let len = rng.random_range(10000..=50000);
                    "x".repeat(len)
                }
            };
            strings.push(string);
        }
        
        strings
    }

    fn generate_zipf_strings(count: usize, base_strings: &[&str], seed: u64) -> Vec<String> {
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        let mut strings = Vec::with_capacity(count);
        
        // Simple Zipf-like distribution: first few strings are much more common
        for _ in 0..count {
            let zipf_choice = rng.random_range(0..100);
            let base_idx = if zipf_choice < 50 {
                0 // 50% chance of first string
            } else if zipf_choice < 75 {
                1 // 25% chance of second string  
            } else if zipf_choice < 87 {
                2 // 12% chance of third string
            } else {
                rng.random_range(3..base_strings.len()) // remaining strings split rest
            };
            
            let base = base_strings[base_idx];
            
            // Add variations to create realistic patterns
            let variation = rng.random_range(0..4);
            let string = match variation {
                0 => base.to_string(), // Exact duplicate
                1 => format!("{}_{}", base, rng.random_range(1000..9999)), // Common suffix
                2 => format!("{}/{}", base, rng.random_range(100..999)), // Path-like
                _ => format!("prefix_{}", base), // Common prefix
            };
            strings.push(string);
        }
        
        strings
    }

    #[test]
    fn test_mixed_size_compression() {
        let strings = generate_mixed_size_strings(16384, 42);
        let input = StringArray::from(strings.clone());
        
        let compressor = LiquidByteViewArray::<MemoryBuffer>::train_compressor(input.iter());
        let liquid_array = LiquidByteViewArray::<MemoryBuffer>::from_string_array(&input, compressor);
        
        // Verify correctness
        let output = liquid_array.to_arrow_array().unwrap();
        assert_eq!(&input, output.as_string::<i32>());
        
        // Check memory efficiency
        let original_size = input.get_array_memory_size();
        let compressed_size = liquid_array.get_array_memory_size();
        let compression_ratio = original_size as f64 / compressed_size as f64;
        
        assert!(compression_ratio > 1.0, "Should achieve compression on mixed size data");
    }

    #[test]
    fn test_zipf_patterns() {
        // Real-world string patterns
        let base_patterns = &[
            "error", "warning", "info", "debug",
            "user", "admin", "guest", 
            "GET", "POST", "PUT", "DELETE",
            "success", "failure", "pending",
            "/api/v1", "/api/v2", "/health",
        ];
        
        let strings = generate_zipf_strings(16384, base_patterns, 123);
        let input = StringArray::from(strings.clone());
        
        let compressor = LiquidByteViewArray::<MemoryBuffer>::train_compressor(input.iter());
        let liquid_array = LiquidByteViewArray::<MemoryBuffer>::from_string_array(&input, compressor);
        
        // Verify correctness
        let output = liquid_array.to_arrow_array().unwrap();
        assert_eq!(&input, output.as_string::<i32>());
        
        // Check compression effectiveness
        let memory_usage = liquid_array.get_detailed_memory_usage();
        let original_size = input.get_array_memory_size();
        let total_compressed = memory_usage.total();
        let compression_ratio = original_size as f64 / total_compressed as f64;
        
        println!("Zipf test - Original: {}, Compressed: {}, Ratio: {:.2}x", 
                 original_size, total_compressed, compression_ratio);
        
        // Zipf should compress very well due to duplication
        assert!(compression_ratio > 2.0, "Zipf patterns should achieve good compression");
    }

    #[test] 
    fn test_offset_stress() {
        let mut strings = Vec::with_capacity(16384);
        
        // Create strings with problematic offset patterns
        for i in 0..16384 {
            let string = match i % 8 {
                0 => "a".to_string(), // tiny
                1 => "x".repeat(1000 + (i % 100)), // variable medium
                2 => "b".to_string(), // tiny
                3 => "y".repeat(5000 + (i % 1000)), // variable large
                4 => "c".to_string(), // tiny
                5 => "medium".repeat(50 + (i % 20)), // variable medium
                6 => "huge".repeat(2000 + (i % 500)), // variable huge
                _ => format!("string_{}", i), // varied length based on number
            };
            strings.push(string);
        }
        
        let input = StringArray::from(strings);
        let compressor = LiquidByteViewArray::<MemoryBuffer>::train_compressor(input.iter());
        let liquid_array = LiquidByteViewArray::<MemoryBuffer>::from_string_array(&input, compressor);
        
        // Verify correctness
        let output = liquid_array.to_arrow_array().unwrap();
        assert_eq!(&input, output.as_string::<i32>());
        
        // Test offset compression handles the stress case
        let offset_views = liquid_array.offset_views();
        
        // Verify offsets are monotonic
        for i in 1..offset_views.len() {
            assert!(offset_views[i].offset() >= offset_views[i-1].offset(), 
                   "Offsets should be monotonic");
        }
    }
}
