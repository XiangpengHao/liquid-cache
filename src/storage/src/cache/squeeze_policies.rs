//! Squeeze policies for liquid cache.

use bytes::Bytes;

use crate::cache::{
    LiquidCompressorStates,
    cached_batch::{CacheEntry, CachedData},
    transcode_liquid_inner,
    utils::arrow_to_bytes,
};

/// What to do when we need to squeeze an entry?
pub trait SqueezePolicy: std::fmt::Debug + Send + Sync {
    /// Squeeze the entry.
    /// Returns the squeezed entry and the bytes that were used to store the entry on disk.
    fn squeeze(
        &self,
        entry: CacheEntry,
        compressor: &LiquidCompressorStates,
    ) -> (CacheEntry, Option<Bytes>);
}

/// Squeeze the entry to disk.
#[derive(Debug, Default, Clone)]
pub struct Evict;

impl SqueezePolicy for Evict {
    fn squeeze(
        &self,
        entry: CacheEntry,
        _compressor: &LiquidCompressorStates,
    ) -> (CacheEntry, Option<Bytes>) {
        let (data, tracker) = entry.into_parts();

        match data {
            CachedData::MemoryArrow(array) => {
                let bytes = arrow_to_bytes(&array).expect("failed to convert arrow to bytes");
                (
                    CacheEntry::with_expression_tracker(
                        CachedData::DiskArrow(array.data_type().clone()),
                        tracker,
                    ),
                    Some(bytes),
                )
            }
            CachedData::MemoryLiquid(liquid_array) => {
                let disk_data = liquid_array.to_bytes();
                (
                    CacheEntry::with_expression_tracker(
                        CachedData::DiskLiquid(liquid_array.original_arrow_data_type()),
                        tracker,
                    ),
                    Some(Bytes::from(disk_data)),
                )
            }
            CachedData::MemoryHybridLiquid(hybrid_array) => (
                CacheEntry::with_expression_tracker(
                    CachedData::DiskLiquid(hybrid_array.original_arrow_data_type()),
                    tracker,
                ),
                None,
            ),
            CachedData::DiskLiquid(_) | CachedData::DiskArrow(_) => {
                (CacheEntry::with_expression_tracker(data, tracker), None)
            }
        }
    }
}

/// Squeeze the entry to liquid memory.
#[derive(Debug, Default, Clone)]
pub struct TranscodeSqueezeEvict;

impl SqueezePolicy for TranscodeSqueezeEvict {
    fn squeeze(
        &self,
        entry: CacheEntry,
        compressor: &LiquidCompressorStates,
    ) -> (CacheEntry, Option<Bytes>) {
        let (data, tracker) = entry.into_parts();
        let squeeze_hint = tracker.majority_expression();

        match data {
            CachedData::MemoryArrow(array) => match transcode_liquid_inner(&array, compressor) {
                Ok(liquid_array) => (
                    CacheEntry::with_expression_tracker(
                        CachedData::MemoryLiquid(liquid_array),
                        tracker,
                    ),
                    None,
                ),
                Err(_) => {
                    let bytes = arrow_to_bytes(&array).expect("failed to convert arrow to bytes");
                    (
                        CacheEntry::with_expression_tracker(
                            CachedData::DiskArrow(array.data_type().clone()),
                            tracker,
                        ),
                        Some(bytes),
                    )
                }
            },
            CachedData::MemoryLiquid(liquid_array) => {
                let (hybrid_array, bytes) = match liquid_array.squeeze(squeeze_hint.as_ref()) {
                    Some(result) => result,
                    None => {
                        let bytes = Bytes::from(liquid_array.to_bytes());
                        return (
                            CacheEntry::with_expression_tracker(
                                CachedData::DiskLiquid(liquid_array.original_arrow_data_type()),
                                tracker,
                            ),
                            Some(bytes),
                        );
                    }
                };
                (
                    CacheEntry::with_expression_tracker(
                        CachedData::MemoryHybridLiquid(hybrid_array),
                        tracker,
                    ),
                    Some(bytes),
                )
            }
            CachedData::MemoryHybridLiquid(hybrid_array) => (
                CacheEntry::with_expression_tracker(
                    CachedData::DiskLiquid(hybrid_array.original_arrow_data_type()),
                    tracker,
                ),
                None,
            ),
            CachedData::DiskLiquid(_) | CachedData::DiskArrow(_) => {
                (CacheEntry::with_expression_tracker(data, tracker), None)
            }
        }
    }
}

/// Squeeze the entry to liquid memory, but don't convert to hybrid.
#[derive(Debug, Default, Clone)]
pub struct TranscodeEvict;

impl SqueezePolicy for TranscodeEvict {
    fn squeeze(
        &self,
        entry: CacheEntry,
        compressor: &LiquidCompressorStates,
    ) -> (CacheEntry, Option<Bytes>) {
        let (data, tracker) = entry.into_parts();

        match data {
            CachedData::MemoryArrow(array) => match transcode_liquid_inner(&array, compressor) {
                Ok(liquid_array) => (
                    CacheEntry::with_expression_tracker(
                        CachedData::MemoryLiquid(liquid_array),
                        tracker,
                    ),
                    None,
                ),
                Err(_) => {
                    let bytes = arrow_to_bytes(&array).expect("failed to convert arrow to bytes");
                    (
                        CacheEntry::with_expression_tracker(
                            CachedData::DiskArrow(array.data_type().clone()),
                            tracker,
                        ),
                        Some(bytes),
                    )
                }
            },
            CachedData::MemoryLiquid(liquid_array) => {
                let bytes = Bytes::from(liquid_array.to_bytes());
                (
                    CacheEntry::with_expression_tracker(
                        CachedData::DiskLiquid(liquid_array.original_arrow_data_type()),
                        tracker,
                    ),
                    Some(bytes),
                )
            }
            CachedData::MemoryHybridLiquid(hybrid_array) => (
                CacheEntry::with_expression_tracker(
                    CachedData::DiskLiquid(hybrid_array.original_arrow_data_type()),
                    tracker,
                ),
                None,
            ),
            CachedData::DiskLiquid(_) | CachedData::DiskArrow(_) => {
                (CacheEntry::with_expression_tracker(data, tracker), None)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cache::cached_batch::CacheEntry;
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
        let (new_batch, bytes) = disk.squeeze(CacheEntry::memory_arrow(arr.clone()), &states);
        let (data, _) = new_batch.into_parts();
        match (data, bytes) {
            (CachedData::DiskArrow(dt), Some(b)) => {
                assert_eq!(dt, DataType::Int32);
                let decoded = decode_arrow(&b);
                assert_eq!(decoded.as_ref(), arr.as_ref());
            }
            other => panic!("unexpected: {other:?}"),
        }

        // MemoryLiquid (strings) -> MemoryHybridLiquid + bytes
        let strings = Arc::new(StringArray::from(vec!["a", "b", "a"])) as ArrayRef;
        let liquid = transcode_liquid_inner(&strings, &states).unwrap();
        let (new_batch, bytes) = disk.squeeze(CacheEntry::memory_liquid(liquid.clone()), &states);
        let (data, _) = new_batch.into_parts();
        match (data, bytes) {
            (CachedData::DiskLiquid(_), Some(b)) => {
                assert!(!b.is_empty());
            }
            other => panic!("unexpected: {other:?}"),
        }

        // MemoryHybridLiquid -> DiskLiquid, no extra bytes
        let hybrid = match liquid.squeeze(None) {
            Some((h, _b)) => h,
            None => panic!("squeeze should succeed for byte-view"),
        };
        let (new_batch, bytes) = disk.squeeze(CacheEntry::memory_hybrid_liquid(hybrid), &states);
        let (data, _) = new_batch.into_parts();
        match (data, bytes) {
            (CachedData::DiskLiquid(_data_type), None) => {}
            other => panic!("unexpected: {other:?}"),
        }

        // Disk* -> unchanged, no bytes
        let (b1, w1) = disk.squeeze(CacheEntry::disk_arrow(DataType::Utf8), &states);
        assert!(matches!(b1.data(), CachedData::DiskArrow(DataType::Utf8)) && w1.is_none());
        let (b2, w2) = disk.squeeze(CacheEntry::disk_liquid(DataType::Utf8), &states);
        assert!(matches!(b2.data(), CachedData::DiskLiquid(DataType::Utf8)) && w2.is_none());
    }

    #[test]
    fn test_squeeze_to_liquid_policy() {
        let to_liquid = TranscodeSqueezeEvict;
        let states = LiquidCompressorStates::new();

        // MemoryArrow -> MemoryLiquid, no bytes
        let arr = int_array(8);
        let (new_batch, bytes) = to_liquid.squeeze(CacheEntry::memory_arrow(arr.clone()), &states);
        assert!(bytes.is_none());
        match new_batch.data() {
            CachedData::MemoryLiquid(liq) => {
                assert_eq!(liq.to_arrow_array().as_ref(), arr.as_ref());
            }
            other => panic!("unexpected: {other:?}"),
        }

        // MemoryLiquid (strings) -> MemoryHybridLiquid + bytes
        let strings = Arc::new(StringArray::from(vec!["x", "y", "x"])) as ArrayRef;
        let liquid = transcode_liquid_inner(&strings, &states).unwrap();
        let (new_batch, bytes) = to_liquid.squeeze(CacheEntry::memory_liquid(liquid), &states);
        let (data, _) = new_batch.into_parts();
        match (data, bytes) {
            (CachedData::MemoryHybridLiquid(_), Some(b)) => assert!(!b.is_empty()),
            other => panic!("unexpected: {other:?}"),
        }

        // MemoryHybridLiquid -> DiskLiquid, no bytes
        let strings = Arc::new(StringArray::from(vec!["m", "n"])) as ArrayRef;
        let liquid = transcode_liquid_inner(&strings, &states).unwrap();
        let hybrid = liquid.squeeze(None).unwrap().0;
        let (new_batch, bytes) =
            to_liquid.squeeze(CacheEntry::memory_hybrid_liquid(hybrid), &states);
        let (data, _) = new_batch.into_parts();
        match (data, bytes) {
            (CachedData::DiskLiquid(DataType::Utf8), None) => {}
            other => panic!("unexpected: {other:?}"),
        }

        // Disk* -> unchanged
        let (b1, w1) = to_liquid.squeeze(CacheEntry::disk_arrow(DataType::Utf8), &states);
        assert!(matches!(b1.data(), CachedData::DiskArrow(DataType::Utf8)) && w1.is_none());
        let (b2, w2) = to_liquid.squeeze(CacheEntry::disk_liquid(DataType::Utf8), &states);
        assert!(matches!(b2.data(), CachedData::DiskLiquid(DataType::Utf8)) && w2.is_none());
    }
}
