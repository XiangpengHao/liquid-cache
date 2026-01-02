use arrow::array::types::UInt16Type;
use bytes::Bytes;
use fsst::Compressor;
use std::sync::Arc;

use super::LiquidByteViewArray;
use crate::liquid_array::byte_array::ArrowByteType;
use crate::liquid_array::ipc::LiquidIPCHeader;
use crate::liquid_array::raw::BitPackedArray;
use crate::liquid_array::raw::fsst_buffer::{
    FsstArray, PrefixKey, RawFsstBuffer, decode_compact_offsets, empty_compact_offsets,
};
use crate::liquid_array::{LiquidDataType, SqueezeResult};

// Header for LiquidByteViewArray serialization
#[repr(C)]
pub(super) struct ByteViewArrayHeader {
    pub(super) keys_size: u32,
    pub(super) compact_offsets_size: u32,
    pub(super) shared_prefix_size: u32,
    pub(super) fsst_raw_size: u32,
}

impl ByteViewArrayHeader {
    pub(super) const fn size() -> usize {
        const _: () =
            assert!(std::mem::size_of::<ByteViewArrayHeader>() == ByteViewArrayHeader::size());
        16
    }

    pub(super) fn to_bytes(&self) -> [u8; Self::size()] {
        let mut bytes = [0u8; Self::size()];
        bytes[0..4].copy_from_slice(&self.keys_size.to_le_bytes());
        bytes[4..8].copy_from_slice(&self.compact_offsets_size.to_le_bytes());
        bytes[8..12].copy_from_slice(&self.shared_prefix_size.to_le_bytes());
        bytes[12..16].copy_from_slice(&self.fsst_raw_size.to_le_bytes());
        bytes
    }

    pub(super) fn from_bytes(bytes: &[u8]) -> Self {
        if bytes.len() < Self::size() {
            panic!(
                "value too small for ByteViewArrayHeader, expected at least {} bytes, got {}",
                Self::size(),
                bytes.len()
            );
        }
        let keys_size = u32::from_le_bytes(bytes[0..4].try_into().unwrap());
        let compact_offsets_size = u32::from_le_bytes(bytes[4..8].try_into().unwrap());
        let shared_prefix_size = u32::from_le_bytes(bytes[8..12].try_into().unwrap());
        let fsst_raw_size = u32::from_le_bytes(bytes[12..16].try_into().unwrap());
        Self {
            keys_size,
            compact_offsets_size,
            shared_prefix_size,
            fsst_raw_size,
        }
    }
}

pub(super) fn align_up_8(len: usize) -> usize {
    (len + 7) & !7
}

fn decode_prefix_keys(bytes: &[u8]) -> Arc<[PrefixKey]> {
    let entry_size = std::mem::size_of::<PrefixKey>();
    if bytes.len() % entry_size != 0 {
        panic!("Invalid prefix keys size");
    }
    if bytes.is_empty() {
        return Arc::<[PrefixKey]>::from([]);
    }
    let mut keys = Vec::with_capacity(bytes.len() / entry_size);
    for chunk in bytes.chunks_exact(entry_size) {
        let mut prefix7 = [0u8; 7];
        prefix7.copy_from_slice(&chunk[..7]);
        let len = chunk[7];
        keys.push(PrefixKey::from_parts(prefix7, len));
    }
    keys.into()
}

impl LiquidByteViewArray<FsstArray> {
    /*
    Serialized LiquidByteViewArray Memory Layout:

    +--------------------------------------------------+
    | LiquidIPCHeader (16 bytes)                       |
    +--------------------------------------------------+
    | ByteViewArrayHeader (16 bytes)                   |  // keys_size, compact_offsets_size, shared_prefix_size, fsst_size
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
    | Compact offsets bytes (header + residuals)       |
    +--------------------------------------------------+
    | Padding (to 8-byte alignment)                    |
    +--------------------------------------------------+
    | Prefix keys bytes (prefix7 + len)                |
    +--------------------------------------------------+
    | Padding (to 8-byte alignment)                    |
    +--------------------------------------------------+
    | Shared prefix bytes                              |
    +--------------------------------------------------+
    */
    pub(crate) fn to_bytes_inner(&self) -> SqueezeResult<Vec<u8>> {
        let header_size = LiquidIPCHeader::size() + ByteViewArrayHeader::size();
        let mut result = Vec::with_capacity(header_size + 1024);
        result.resize(header_size, 0);

        // A) Align and serialize RawFsstBuffer first (near the start)
        while !result.len().is_multiple_of(8) {
            result.push(0);
        }
        let fsst_start = result.len();
        let fsst_raw_bytes = self.fsst_buffer.raw_to_bytes();
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

        // D) Alignment before compact offsets
        while !result.len().is_multiple_of(8) {
            result.push(0);
        }

        // E) Serialize compact offsets (header + residuals)
        let offsets_start = result.len();
        self.fsst_buffer.write_compact_offsets(&mut result);
        let compact_offsets_size = result.len() - offsets_start;

        // F) Alignment before prefix keys
        while !result.len().is_multiple_of(8) {
            result.push(0);
        }

        // G) Serialize prefix keys (prefix7 + len)
        for prefix in self.prefix_keys.iter() {
            result.extend_from_slice(prefix.prefix7());
            result.push(prefix.len_byte());
        }

        // H) Alignment before shared prefix
        while !result.len().is_multiple_of(8) {
            result.push(0);
        }

        // I) Serialize shared prefix
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
            compact_offsets_size: compact_offsets_size as u32,
            shared_prefix_size: shared_prefix_size as u32,
            fsst_raw_size: fsst_raw_size as u32,
        };

        // Write headers into reserved space at start
        let header_slice = &mut result[0..header_size];
        header_slice[0..LiquidIPCHeader::size()].copy_from_slice(&ipc.to_bytes());
        header_slice[LiquidIPCHeader::size()..header_size].copy_from_slice(&view_header.to_bytes());

        Ok(result)
    }

    /// Deserialize a LiquidByteViewArray from bytes.
    pub fn from_bytes(bytes: Bytes, compressor: Arc<Compressor>) -> LiquidByteViewArray<FsstArray> {
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

        // C) Align and read compact offsets
        cursor = align_up_8(cursor);
        let offsets_end = cursor + view_header.compact_offsets_size as usize;
        if offsets_end > bytes.len() {
            panic!("Compact offsets data extends beyond input buffer");
        }

        // Deserialize compact offsets.
        let compact_offsets = if view_header.compact_offsets_size > 0 {
            let chunk = bytes.slice(cursor..offsets_end);
            decode_compact_offsets(chunk.as_ref())
        } else {
            empty_compact_offsets()
        };
        cursor = offsets_end;

        // D) Align and read prefix keys
        cursor = align_up_8(cursor);
        let prefix_count = compact_offsets.len().saturating_sub(1);
        let prefix_keys_size = prefix_count * std::mem::size_of::<PrefixKey>();
        let prefix_keys_end = cursor + prefix_keys_size;
        if prefix_keys_end > bytes.len() {
            panic!("Prefix keys data extends beyond input buffer");
        }
        let prefix_keys = if prefix_keys_size > 0 {
            decode_prefix_keys(&bytes[cursor..prefix_keys_end])
        } else {
            Arc::<[PrefixKey]>::from([])
        };
        cursor = prefix_keys_end;

        // E) Align and read shared prefix
        cursor = align_up_8(cursor);
        let prefix_end = cursor + view_header.shared_prefix_size as usize;
        if prefix_end > bytes.len() {
            panic!("Shared prefix data extends beyond input buffer");
        }
        let shared_prefix = bytes[cursor..prefix_end].to_vec();

        LiquidByteViewArray {
            dictionary_keys,
            prefix_keys,
            fsst_buffer: FsstArray::new(Arc::new(raw_buffer), compact_offsets, compressor),
            original_arrow_type,
            shared_prefix,
            string_fingerprints: None,
        }
    }
}
