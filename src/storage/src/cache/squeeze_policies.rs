//! Squeeze policies for liquid cache.

use bytes::Bytes;

use crate::cache::{
    LiquidCompressorStates, cached_batch::CachedBatch, transcode_liquid_inner, utils::arrow_to_bytes,
};

/// What to do when we need to squeeze an entry?
pub trait SqueezePolicy: std::fmt::Debug + Send + Sync {
    /// Squeeze the entry.
    /// Returns the squeezed entry and the bytes that were used to store the entry on disk.
    fn squeeze(
        &self,
        entry: CachedBatch,
        compressor: &LiquidCompressorStates,
    ) -> (CachedBatch, Option<Bytes>);
}

/// Squeeze the entry to disk.
#[derive(Debug, Default, Clone)]
pub struct Evict;

impl SqueezePolicy for Evict {
    fn squeeze(
        &self,
        entry: CachedBatch,
        _compressor: &LiquidCompressorStates,
    ) -> (CachedBatch, Option<Bytes>) {
        match entry {
            CachedBatch::MemoryArrow(array) => {
                let bytes = arrow_to_bytes(&array).expect("failed to convert arrow to bytes");
                (
                    CachedBatch::DiskArrow(array.data_type().clone()),
                    Some(bytes),
                )
            }
            CachedBatch::MemoryLiquid(liquid_array) => {
                let (hybrid_array, bytes) = match liquid_array.squeeze() {
                    Some(result) => result,
                    None => {
                        // not supported, evict to disk
                        let bytes = liquid_array.to_bytes();
                        let bytes = Bytes::from(bytes);
                        return (
                            CachedBatch::DiskLiquid(liquid_array.original_arrow_data_type()),
                            Some(bytes),
                        );
                    }
                };
                (CachedBatch::MemoryHybridLiquid(hybrid_array), Some(bytes))
            }
            CachedBatch::MemoryHybridLiquid(hybrid_array) => (
                CachedBatch::DiskLiquid(hybrid_array.original_arrow_data_type()),
                None,
            ),
            CachedBatch::DiskLiquid(_) | CachedBatch::DiskArrow(_) => (entry, None),
        }
    }
}

/// Squeeze the entry to liquid memory.
#[derive(Debug, Default, Clone)]
pub struct TranscodeSqueezeEvict;

impl SqueezePolicy for TranscodeSqueezeEvict {
    fn squeeze(
        &self,
        entry: CachedBatch,
        compressor: &LiquidCompressorStates,
    ) -> (CachedBatch, Option<Bytes>) {
        match entry {
            CachedBatch::MemoryArrow(array) => {
                let liquid_array = transcode_liquid_inner(&array, compressor).unwrap();
                (CachedBatch::MemoryLiquid(liquid_array), None)
            }
            CachedBatch::MemoryLiquid(liquid_array) => {
                let (hybrid_array, bytes) = match liquid_array.squeeze() {
                    Some(result) => result,
                    None => {
                        let bytes = liquid_array.to_bytes();
                        let bytes = Bytes::from(bytes);
                        return (
                            CachedBatch::DiskLiquid(liquid_array.original_arrow_data_type()),
                            Some(bytes),
                        );
                    }
                };
                (CachedBatch::MemoryHybridLiquid(hybrid_array), Some(bytes))
            }
            CachedBatch::MemoryHybridLiquid(hybrid_array) => {
                // the full data of hybrid array is already on disk
                (
                    CachedBatch::DiskLiquid(hybrid_array.original_arrow_data_type()),
                    None,
                )
            }
            CachedBatch::DiskLiquid(_) | CachedBatch::DiskArrow(_) => (entry, None),
        }
    }
}

/// Squeeze the entry to liquid memory, but don't convert to hybrid.
#[derive(Debug, Default, Clone)]
pub struct TranscodeEvict;

impl SqueezePolicy for TranscodeEvict {
    fn squeeze(
        &self,
        entry: CachedBatch,
        compressor: &LiquidCompressorStates,
    ) -> (CachedBatch, Option<Bytes>) {
        match entry {
            CachedBatch::MemoryArrow(array) => {
                let liquid_array = transcode_liquid_inner(&array, compressor).unwrap();
                (CachedBatch::MemoryLiquid(liquid_array), None)
            }
            CachedBatch::MemoryLiquid(liquid_array) => {
                let bytes = liquid_array.to_bytes();
                let bytes = Bytes::from(bytes);
                (
                    CachedBatch::DiskLiquid(liquid_array.original_arrow_data_type()),
                    Some(bytes),
                )
            }
            CachedBatch::MemoryHybridLiquid(hybrid_array) => (
                CachedBatch::DiskLiquid(hybrid_array.original_arrow_data_type()),
                None,
            ),
            CachedBatch::DiskLiquid(_) | CachedBatch::DiskArrow(_) => (entry, None),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cache::cached_batch::CachedBatch;
    use arrow::array::{ArrayRef, Int32Array, StringArray};
    use arrow_schema::DataType;
    use std::sync::Arc;

    fn int_array(n: i32) -> ArrayRef {
        Arc::new(Int32Array::from_iter_values(0..n))
    }

    fn decode_arrow(bytes: &Bytes) -> ArrayRef {
        let cursor = std::io::Cursor::new(bytes.to_vec());
        let mut reader =
            arrow::ipc::reader::StreamReader::try_new(cursor, None).expect("arrow stream");
        let batch = reader
            .next()
            .expect("non-empty stream")
            .expect("read stream");
        batch.column(0).clone()
    }

    #[test]
    fn test_squeeze_to_disk_policy() {
        let disk = Evict;
        let states = LiquidCompressorStates::new();

        // MemoryArrow -> DiskArrow + bytes (Arrow IPC)
        let arr = int_array(8);
        let (new_batch, bytes) = disk.squeeze(CachedBatch::MemoryArrow(arr.clone()), &states);
        match (new_batch, bytes) {
            (CachedBatch::DiskArrow(DataType::Int32), Some(b)) => {
                let decoded = decode_arrow(&b);
                assert_eq!(decoded.as_ref(), arr.as_ref());
            }
            other => panic!("unexpected: {other:?}"),
        }

        // MemoryLiquid (strings) -> MemoryHybridLiquid + bytes
        let strings = Arc::new(StringArray::from(vec!["a", "b", "a"])) as ArrayRef;
        let liquid = transcode_liquid_inner(&strings, &states).unwrap();
        let (new_batch, bytes) = disk.squeeze(CachedBatch::MemoryLiquid(liquid.clone()), &states);
        match (new_batch, bytes) {
            (CachedBatch::MemoryHybridLiquid(_), Some(b)) => {
                assert!(!b.is_empty());
            }
            other => panic!("unexpected: {other:?}"),
        }

        // MemoryHybridLiquid -> DiskLiquid, no extra bytes
        let hybrid = match liquid.squeeze() {
            Some((h, _b)) => h,
            None => panic!("squeeze should succeed for byte-view"),
        };
        let (new_batch, bytes) = disk.squeeze(CachedBatch::MemoryHybridLiquid(hybrid), &states);
        match (new_batch, bytes) {
            (CachedBatch::DiskLiquid(_data_type), None) => {}
            other => panic!("unexpected: {other:?}"),
        }

        // Disk* -> unchanged, no bytes
        let (b1, w1) = disk.squeeze(CachedBatch::DiskArrow(DataType::Utf8), &states);
        assert!(matches!(b1, CachedBatch::DiskArrow(DataType::Utf8)) && w1.is_none());
        let (b2, w2) = disk.squeeze(CachedBatch::DiskLiquid(DataType::Utf8), &states);
        assert!(matches!(b2, CachedBatch::DiskLiquid(DataType::Utf8)) && w2.is_none());
    }

    #[test]
    fn test_squeeze_to_liquid_policy() {
        let to_liquid = TranscodeSqueezeEvict;
        let states = LiquidCompressorStates::new();

        // MemoryArrow -> MemoryLiquid, no bytes
        let arr = int_array(8);
        let (new_batch, bytes) = to_liquid.squeeze(CachedBatch::MemoryArrow(arr.clone()), &states);
        assert!(bytes.is_none());
        match new_batch {
            CachedBatch::MemoryLiquid(liq) => {
                assert_eq!(liq.to_arrow_array().as_ref(), arr.as_ref());
            }
            other => panic!("unexpected: {other:?}"),
        }

        // MemoryLiquid (strings) -> MemoryHybridLiquid + bytes
        let strings = Arc::new(StringArray::from(vec!["x", "y", "x"])) as ArrayRef;
        let liquid = transcode_liquid_inner(&strings, &states).unwrap();
        let (new_batch, bytes) = to_liquid.squeeze(CachedBatch::MemoryLiquid(liquid), &states);
        match (new_batch, bytes) {
            (CachedBatch::MemoryHybridLiquid(_), Some(b)) => assert!(!b.is_empty()),
            other => panic!("unexpected: {other:?}"),
        }

        // MemoryHybridLiquid -> DiskLiquid, no bytes
        let strings = Arc::new(StringArray::from(vec!["m", "n"])) as ArrayRef;
        let liquid = transcode_liquid_inner(&strings, &states).unwrap();
        let hybrid = liquid.squeeze().unwrap().0;
        let (new_batch, bytes) =
            to_liquid.squeeze(CachedBatch::MemoryHybridLiquid(hybrid), &states);
        match (new_batch, bytes) {
            (CachedBatch::DiskLiquid(DataType::Utf8), None) => {}
            other => panic!("unexpected: {other:?}"),
        }

        // Disk* -> unchanged
        let (b1, w1) = to_liquid.squeeze(CachedBatch::DiskArrow(DataType::Utf8), &states);
        assert!(matches!(b1, CachedBatch::DiskArrow(DataType::Utf8)) && w1.is_none());
        let (b2, w2) = to_liquid.squeeze(CachedBatch::DiskLiquid(DataType::Utf8), &states);
        assert!(matches!(b2, CachedBatch::DiskLiquid(DataType::Utf8)) && w2.is_none());
    }
}
