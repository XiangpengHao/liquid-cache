//! Squeeze policies for liquid cache.

use std::sync::Arc;

use arrow::array::{Array, ArrayRef, StructArray};
use arrow_schema::DataType;
use bytes::Bytes;
use parquet_variant_compute::VariantArray;

use crate::cache::{
    CacheExpression, LiquidCompressorStates,
    cached_batch::{CacheEntry, CachedData},
    transcode_liquid_inner,
    utils::arrow_to_bytes,
};
use crate::liquid_array::{HybridBacking, LiquidHybridArrayRef, VariantExtractedArray};

/// What to do when we need to squeeze an entry?
pub trait SqueezePolicy: std::fmt::Debug + Send + Sync {
    /// Squeeze the entry.
    /// Returns the squeezed entry and the bytes that were used to store the entry on disk.
    fn squeeze(
        &self,
        entry: CacheEntry,
        compressor: &LiquidCompressorStates,
        squeeze_hint: Option<&CacheExpression>,
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
        _squeeze_hint: Option<&CacheExpression>,
    ) -> (CacheEntry, Option<Bytes>) {
        let data = entry.into_data();

        match data {
            CachedData::MemoryArrow(array) => {
                let bytes = arrow_to_bytes(&array).expect("failed to convert arrow to bytes");
                (
                    CacheEntry::disk_arrow(array.data_type().clone()),
                    Some(bytes),
                )
            }
            CachedData::MemoryLiquid(liquid_array) => {
                let disk_data = liquid_array.to_bytes();
                (
                    CacheEntry::disk_liquid(liquid_array.original_arrow_data_type()),
                    Some(Bytes::from(disk_data)),
                )
            }
            CachedData::MemoryHybridLiquid(hybrid_array) => {
                let data_type = hybrid_array.original_arrow_data_type();
                let entry = match hybrid_array.disk_backing() {
                    HybridBacking::Liquid => CacheEntry::disk_liquid(data_type),
                    HybridBacking::Arrow => CacheEntry::disk_arrow(data_type),
                };
                (entry, None)
            }
            CachedData::DiskLiquid(_) | CachedData::DiskArrow(_) => {
                (CacheEntry::from_data(data), None)
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
        squeeze_hint: Option<&CacheExpression>,
    ) -> (CacheEntry, Option<Bytes>) {
        let data = entry.into_data();

        match data {
            CachedData::MemoryArrow(array) => {
                if let Some(CacheExpression::VariantGet { path, .. }) = squeeze_hint
                    && let Some((hybrid_array, bytes)) =
                        try_variant_squeeze(&array, path.as_ref(), compressor)
                {
                    return (CacheEntry::memory_hybrid_liquid(hybrid_array), Some(bytes));
                }
                match transcode_liquid_inner(&array, compressor) {
                    Ok(liquid_array) => (CacheEntry::memory_liquid(liquid_array), None),
                    Err(_) => {
                        let bytes =
                            arrow_to_bytes(&array).expect("failed to convert arrow to bytes");
                        (
                            CacheEntry::disk_arrow(array.data_type().clone()),
                            Some(bytes),
                        )
                    }
                }
            }
            CachedData::MemoryLiquid(liquid_array) => {
                let (hybrid_array, bytes) = match liquid_array.squeeze(squeeze_hint) {
                    Some(result) => result,
                    None => {
                        let bytes = Bytes::from(liquid_array.to_bytes());
                        return (
                            CacheEntry::disk_liquid(liquid_array.original_arrow_data_type()),
                            Some(bytes),
                        );
                    }
                };
                (CacheEntry::memory_hybrid_liquid(hybrid_array), Some(bytes))
            }
            CachedData::MemoryHybridLiquid(hybrid_array) => {
                let data_type = hybrid_array.original_arrow_data_type();
                let entry = match hybrid_array.disk_backing() {
                    HybridBacking::Liquid => CacheEntry::disk_liquid(data_type),
                    HybridBacking::Arrow => CacheEntry::disk_arrow(data_type),
                };
                (entry, None)
            }
            CachedData::DiskLiquid(_) | CachedData::DiskArrow(_) => {
                (CacheEntry::from_data(data), None)
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
        _squeeze_hint: Option<&CacheExpression>,
    ) -> (CacheEntry, Option<Bytes>) {
        let data = entry.into_data();

        match data {
            CachedData::MemoryArrow(array) => match transcode_liquid_inner(&array, compressor) {
                Ok(liquid_array) => (CacheEntry::memory_liquid(liquid_array), None),
                Err(_) => {
                    let bytes = arrow_to_bytes(&array).expect("failed to convert arrow to bytes");
                    (
                        CacheEntry::disk_arrow(array.data_type().clone()),
                        Some(bytes),
                    )
                }
            },
            CachedData::MemoryLiquid(liquid_array) => {
                let bytes = Bytes::from(liquid_array.to_bytes());
                (
                    CacheEntry::disk_liquid(liquid_array.original_arrow_data_type()),
                    Some(bytes),
                )
            }
            CachedData::MemoryHybridLiquid(hybrid_array) => {
                let data_type = hybrid_array.original_arrow_data_type();
                let entry = match hybrid_array.disk_backing() {
                    HybridBacking::Liquid => CacheEntry::disk_liquid(data_type),
                    HybridBacking::Arrow => CacheEntry::disk_arrow(data_type),
                };
                (entry, None)
            }
            CachedData::DiskLiquid(_) | CachedData::DiskArrow(_) => {
                (CacheEntry::from_data(data), None)
            }
        }
    }
}

fn try_variant_squeeze(
    array: &ArrayRef,
    path: &str,
    compressor: &LiquidCompressorStates,
) -> Option<(LiquidHybridArrayRef, Bytes)> {
    let struct_array = array.as_any().downcast_ref::<StructArray>()?;
    let variant_array = VariantArray::try_new(struct_array).ok()?;
    if variant_array.is_empty() {
        return None;
    }

    let metadata = Arc::new(variant_array.metadata_field().clone());
    let nulls = variant_array.inner().nulls().cloned();
    let owned_path = path.trim().to_string();
    if owned_path.is_empty() {
        return None;
    }

    let typed_root = variant_array.typed_value_field()?;
    let typed_root = typed_root.as_any().downcast_ref::<StructArray>()?;
    let path_values = typed_root.column_by_name(owned_path.as_str())?;
    let path_struct = path_values.as_any().downcast_ref::<StructArray>()?;
    let typed_values = path_struct.column_by_name("typed_value")?.clone();
    if typed_values.len() != array.len() {
        return None;
    }
    if !is_supported_leaf_type(typed_values.data_type()) {
        return None;
    }

    let liquid_values = match transcode_liquid_inner(&typed_values, compressor) {
        Ok(liquid) => liquid,
        Err(_) => return None,
    };
    let bytes = arrow_to_bytes(array).ok()?;
    let hybrid = VariantExtractedArray::new(
        owned_path,
        liquid_values,
        metadata,
        nulls,
        array.data_type().clone(),
    );

    Some((Arc::new(hybrid) as LiquidHybridArrayRef, bytes))
}

fn is_supported_leaf_type(data_type: &DataType) -> bool {
    matches!(
        data_type,
        DataType::Int64
            | DataType::Float64
            | DataType::Boolean
            | DataType::Utf8
            | DataType::Binary
            | DataType::Date32
            | DataType::Timestamp(_, _)
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cache::CacheExpression;
    use crate::cache::cached_batch::CacheEntry;
    use crate::liquid_array::HybridBacking;
    use arrow::array::{ArrayRef, Int32Array, StringArray, StructArray};
    use arrow_schema::{DataType, Field};
    use parquet_variant::VariantPath;
    use parquet_variant_compute::{GetOptions, json_to_variant, variant_get};
    use std::sync::Arc;

    fn int_array(n: i32) -> ArrayRef {
        Arc::new(Int32Array::from_iter_values(0..n))
    }

    #[test]
    fn variant_hint_creates_hybrid_array() {
        let states = LiquidCompressorStates::new();
        let string_array: ArrayRef = Arc::new(StringArray::from(vec![
            Some(r#"{"name":"Alice"}"#),
            Some(r#"{"name":"Bob"}"#),
            Some(r#"{"name":"Carol"}"#),
        ]));
        let variant = json_to_variant(&string_array).expect("variant conversion");
        let struct_array: StructArray = variant.into();
        let array: ArrayRef = Arc::new(struct_array);

        let entry = CacheEntry::memory_arrow(array.clone());
        let squeeze_hint = Some(&CacheExpression::variant_get("name", DataType::Utf8));

        let (new_entry, bytes) = TranscodeSqueezeEvict.squeeze(entry, &states, squeeze_hint);
        assert!(bytes.is_some());

        let data = new_entry.into_data();
        let CachedData::MemoryHybridLiquid(hybrid) = data else {
            panic!("expected hybrid liquid array");
        };
        assert_eq!(hybrid.disk_backing(), HybridBacking::Arrow);

        let (evicted_entry, evicted_bytes) = Evict.squeeze(
            CacheEntry::memory_hybrid_liquid(hybrid.clone()),
            &states,
            None,
        );
        assert!(evicted_bytes.is_none());
        match evicted_entry.data() {
            CachedData::DiskArrow(dt) => assert_eq!(dt, array.data_type()),
            other => panic!("expected disk arrow from hybrid eviction, got {other:?}"),
        }

        let field = Arc::new(Field::new("name", DataType::Utf8, true));
        let rebuilt = hybrid.to_arrow_array().expect("to arrow");
        let optimized = variant_get(
            &rebuilt,
            GetOptions::new_with_path(VariantPath::from("name")).with_as_type(Some(field.clone())),
        )
        .expect("variant_get optimized");
        let baseline = variant_get(
            &array,
            GetOptions::new_with_path(VariantPath::from("name")).with_as_type(Some(field.clone())),
        )
        .expect("variant_get baseline");
        assert_eq!(optimized.as_ref(), baseline.as_ref());
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

    fn struct_array() -> ArrayRef {
        let values = Arc::new(Int32Array::from(vec![Some(1), None, Some(3)])) as ArrayRef;
        let field = Arc::new(Field::new("value", DataType::Int32, true));
        Arc::new(StructArray::from(vec![(field, values)]))
    }

    #[test]
    fn test_squeeze_to_disk_policy() {
        let disk = Evict;
        let states = LiquidCompressorStates::new();

        // MemoryArrow -> DiskArrow + bytes (Arrow IPC)
        let arr = int_array(8);
        let (new_batch, bytes) = disk.squeeze(CacheEntry::memory_arrow(arr.clone()), &states, None);
        let data = new_batch.into_data();
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
        let (new_batch, bytes) =
            disk.squeeze(CacheEntry::memory_liquid(liquid.clone()), &states, None);
        let data = new_batch.into_data();
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
        let (new_batch, bytes) =
            disk.squeeze(CacheEntry::memory_hybrid_liquid(hybrid), &states, None);
        let data = new_batch.into_data();
        match (data, bytes) {
            (CachedData::DiskLiquid(_data_type), None) => {}
            other => panic!("unexpected: {other:?}"),
        }

        // Disk* -> unchanged, no bytes
        let (b1, w1) = disk.squeeze(CacheEntry::disk_arrow(DataType::Utf8), &states, None);
        assert!(matches!(b1.data(), CachedData::DiskArrow(DataType::Utf8)) && w1.is_none());
        let (b2, w2) = disk.squeeze(CacheEntry::disk_liquid(DataType::Utf8), &states, None);
        assert!(matches!(b2.data(), CachedData::DiskLiquid(DataType::Utf8)) && w2.is_none());
    }

    #[test]
    fn test_squeeze_to_liquid_policy() {
        let to_liquid = TranscodeSqueezeEvict;
        let states = LiquidCompressorStates::new();

        // MemoryArrow -> MemoryLiquid, no bytes
        let arr = int_array(8);
        let (new_batch, bytes) =
            to_liquid.squeeze(CacheEntry::memory_arrow(arr.clone()), &states, None);
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
        let (new_batch, bytes) =
            to_liquid.squeeze(CacheEntry::memory_liquid(liquid), &states, None);
        let data = new_batch.into_data();
        match (data, bytes) {
            (CachedData::MemoryHybridLiquid(_), Some(b)) => assert!(!b.is_empty()),
            other => panic!("unexpected: {other:?}"),
        }

        // MemoryHybridLiquid -> DiskLiquid, no bytes
        let strings = Arc::new(StringArray::from(vec!["m", "n"])) as ArrayRef;
        let liquid = transcode_liquid_inner(&strings, &states).unwrap();
        let hybrid = liquid.squeeze(None).unwrap().0;
        let (new_batch, bytes) =
            to_liquid.squeeze(CacheEntry::memory_hybrid_liquid(hybrid), &states, None);
        let data = new_batch.into_data();
        match (data, bytes) {
            (CachedData::DiskLiquid(DataType::Utf8), None) => {}
            other => panic!("unexpected: {other:?}"),
        }

        // Disk* -> unchanged
        let (b1, w1) = to_liquid.squeeze(CacheEntry::disk_arrow(DataType::Utf8), &states, None);
        assert!(matches!(b1.data(), CachedData::DiskArrow(DataType::Utf8)) && w1.is_none());
        let (b2, w2) = to_liquid.squeeze(CacheEntry::disk_liquid(DataType::Utf8), &states, None);
        assert!(matches!(b2.data(), CachedData::DiskLiquid(DataType::Utf8)) && w2.is_none());
    }

    #[test]
    fn transcode_squeeze_struct_falls_back_to_disk_arrow() {
        let to_liquid = TranscodeSqueezeEvict;
        let states = LiquidCompressorStates::new();
        let struct_arr = struct_array();
        let (new_batch, bytes) =
            to_liquid.squeeze(CacheEntry::memory_arrow(struct_arr.clone()), &states, None);
        match (new_batch.data(), bytes) {
            (CachedData::DiskArrow(dt), Some(b)) => {
                assert_eq!(dt, struct_arr.data_type());
                assert_eq!(decode_arrow(&b).as_ref(), struct_arr.as_ref());
            }
            other => panic!("expected disk arrow fallback, got {other:?}"),
        }
    }

    #[test]
    fn transcode_evict_struct_falls_back_to_disk_arrow() {
        let to_disk = TranscodeEvict;
        let states = LiquidCompressorStates::new();
        let struct_arr = struct_array();
        let (new_batch, bytes) =
            to_disk.squeeze(CacheEntry::memory_arrow(struct_arr.clone()), &states, None);
        match (new_batch.data(), bytes) {
            (CachedData::DiskArrow(dt), Some(b)) => {
                assert_eq!(dt, struct_arr.data_type());
                assert_eq!(decode_arrow(&b).as_ref(), struct_arr.as_ref());
            }
            other => panic!("expected disk arrow fallback, got {other:?}"),
        }
    }
}
