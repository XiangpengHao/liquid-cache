//! Squeeze policies for liquid cache.

use std::sync::Arc;

use arrow::array::{Array, ArrayRef, StructArray};
use bytes::Bytes;
use parquet_variant_compute::VariantArray;

use crate::cache::{
    CacheExpression, LiquidCompressorStates, VariantRequest,
    cached_batch::{CacheEntry, CachedData},
    transcode_liquid_inner,
    utils::arrow_to_bytes,
};
use crate::liquid_array::{HybridBacking, LiquidHybridArrayRef, VariantStructHybridArray};

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
                if let Some(requests) =
                    squeeze_hint.and_then(|expression| expression.variant_requests())
                    && let Some((hybrid_array, bytes)) =
                        try_variant_squeeze(&array, requests)
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
    requests: &[VariantRequest],
) -> Option<(LiquidHybridArrayRef, Bytes)> {
    let struct_array = array.as_any().downcast_ref::<StructArray>()?;
    let variant_array = VariantArray::try_new(struct_array).ok()?;
    if variant_array.is_empty() {
        return None;
    }

    if requests.is_empty() {
        return None;
    }

    let typed_root = variant_array.typed_value_field()?;
    let typed_root = typed_root.as_any().downcast_ref::<StructArray>()?;

    let mut collected = Vec::new();
    for request in requests {
        let path = request.path().trim();
        if path.is_empty() {
            continue;
        }
        let Some(path_struct) = extract_typed_values_for_path(typed_root, path) else {
            continue;
        };
        let path_struct = path_struct.as_any().downcast_ref::<StructArray>()?;
        let Some(typed_values) = path_struct.column_by_name("typed_value") else {
            continue;
        };
        if typed_values.len() != array.len() {
            continue;
        }
        collected.push((Arc::<str>::from(path.to_string()), typed_values.clone()));
    }

    if collected.is_empty() {
        return None;
    }

    let nulls = variant_array.inner().nulls().cloned();
    let bytes = arrow_to_bytes(array).ok()?;
    let hybrid = VariantStructHybridArray::new(collected, nulls, array.data_type().clone());
    Some((Arc::new(hybrid) as LiquidHybridArrayRef, bytes))
}

fn split_variant_path(path: &str) -> Vec<String> {
    path
        .split('.')
        .filter(|segment| !segment.is_empty())
        .map(|segment| segment.to_string())
        .collect()
}

fn extract_typed_values_for_path(typed_root: &StructArray, path: &str) -> Option<ArrayRef> {
    let segments = split_variant_path(path);
    if segments.is_empty() {
        return None;
    }

    let mut cursor = typed_root;
    for (idx, segment) in segments.iter().enumerate() {
        let field = cursor.column_by_name(segment)?;
        let struct_field = field.as_any().downcast_ref::<StructArray>()?;
        if idx == segments.len() - 1 {
            return Some(field.clone());
        }
        let typed_value = struct_field.column_by_name("typed_value")?;
        cursor = typed_value.as_any().downcast_ref::<StructArray>()?;
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cache::CacheExpression;
    use crate::cache::cached_batch::CacheEntry;
    use crate::liquid_array::{HybridBacking, LiquidHybridArray, VariantStructHybridArray};
    use arrow::array::{Array, ArrayRef, BinaryViewArray, Int32Array, StringArray, StructArray};
    use arrow_schema::Fields;
    use arrow_schema::{DataType, Field};
    use parquet::variant::VariantPath;
    use parquet_variant_compute::{GetOptions, json_to_variant, variant_get};
    use std::collections::BTreeMap;
    use std::sync::Arc;

    #[derive(Default)]
    struct VariantTestTree {
        leaf: Option<ArrayRef>,
        children: BTreeMap<String, VariantTestTree>,
    }

    impl VariantTestTree {
        fn insert(&mut self, segments: &[String], values: ArrayRef) {
            if segments.is_empty() {
                self.leaf = Some(values);
                return;
            }
            let (head, tail) = segments.split_first().unwrap();
            self.children
                .entry(head.clone())
                .or_default()
                .insert(tail, values);
        }

        fn into_typed_value_array(self) -> ArrayRef {
            if let Some(values) = self.leaf {
                return values;
            }
            let mut fields = Vec::with_capacity(self.children.len());
            let mut arrays = Vec::with_capacity(self.children.len());
            for (name, child) in self.children {
                let field_array = child.into_struct_array();
                fields.push(Arc::new(Field::new(
                    name.as_str(),
                    field_array.data_type().clone(),
                    false,
                )));
                arrays.push(field_array);
            }
            Arc::new(StructArray::new(Fields::from(fields), arrays, None)) as ArrayRef
        }

        fn into_struct_array(self) -> ArrayRef {
            let typed_value = self.into_typed_value_array();
            let len = typed_value.len();
            let value_field = Arc::new(Field::new("value", DataType::BinaryView, true));
            let typed_field = Arc::new(Field::new(
                "typed_value",
                typed_value.data_type().clone(),
                true,
            ));
            Arc::new(StructArray::new(
                Fields::from(vec![value_field, typed_field]),
                vec![
                    Arc::new(BinaryViewArray::from(vec![None::<&[u8]>; len])) as ArrayRef,
                    typed_value,
                ],
                None,
            )) as ArrayRef
        }
    }

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

    fn enriched_variant_array(path: &str, data_type: DataType) -> ArrayRef {
        enriched_variant_array_with_paths(&[(path, data_type)])
    }

    fn enriched_variant_array_with_paths(entries: &[(&str, DataType)]) -> ArrayRef {
        let values: ArrayRef = Arc::new(StringArray::from(vec![
            Some(r#"{"name": "Alice", "age": 30}"#),
            Some(r#"{"name": "Bob", "age": 25}"#),
            Some(r#"{"name": "Charlie", "age": 35}"#),
        ]));
        let base_variant = json_to_variant(&values).unwrap();
        let base_arr: ArrayRef = Arc::new(base_variant.inner().clone());

        let mut typed_trees: BTreeMap<String, VariantTestTree> = BTreeMap::new();

        for (path, data_type) in entries.iter() {
            let typed_values = variant_get(
                &base_arr,
                GetOptions::new_with_path(VariantPath::from(*path)).with_as_type(Some(Arc::new(
                    Field::new("typed_value", data_type.clone(), true),
                ))),
            )
            .unwrap();
            let segments = split_variant_path(path);
            if segments.is_empty() {
                continue;
            }
            let head = segments[0].clone();
            typed_trees
                .entry(head)
                .or_default()
                .insert(&segments[1..], typed_values.clone());
        }

        let mut typed_fields: Vec<Arc<Field>> = Vec::new();
        let mut typed_columns: Vec<ArrayRef> = Vec::new();
        for (name, tree) in typed_trees {
            let array = tree.into_struct_array();
            typed_fields.push(Arc::new(Field::new(
                name.as_str(),
                array.data_type().clone(),
                true,
            )));
            typed_columns.push(array);
        }

        let typed_struct = Arc::new(StructArray::new(
            Fields::from(typed_fields),
            typed_columns,
            base_variant.inner().nulls().cloned(),
        ));

        let inner = base_variant.inner();
        use arrow::array::BinaryViewArray;
        Arc::new(StructArray::new(
            Fields::from(vec![
                Arc::new(Field::new("metadata", DataType::BinaryView, false)),
                Arc::new(Field::new("value", DataType::BinaryView, true)),
                Arc::new(Field::new(
                    "typed_value",
                    typed_struct.data_type().clone(),
                    true,
                )),
            ]),
            vec![
                inner
                    .column_by_name("metadata")
                    .cloned()
                    .unwrap_or_else(|| Arc::new(base_variant.metadata_field().clone()) as ArrayRef),
                inner.column_by_name("value").cloned().unwrap_or_else(|| {
                    Arc::new(BinaryViewArray::from(vec![None::<&[u8]>; inner.len()])) as ArrayRef
                }),
                typed_struct as ArrayRef,
            ],
            inner.nulls().cloned(),
        )) as ArrayRef
    }

    fn assert_variant_hybrid(hybrid: &LiquidHybridArrayRef, expected_path: &str, bytes: &Bytes) {
        assert!(!bytes.is_empty());
        assert_eq!(hybrid.disk_backing(), HybridBacking::Arrow);
        let struct_hybrid = hybrid
            .as_any()
            .downcast_ref::<VariantStructHybridArray>()
            .expect("hybrid variant struct");
        let arrow_array = struct_hybrid
            .to_arrow_array()
            .expect("reconstruct arrow struct");
        let struct_array = arrow_array
            .as_any()
            .downcast_ref::<StructArray>()
            .expect("variant struct");
        let value_column = struct_array
            .column_by_name("value")
            .expect("value column present");
        assert_eq!(value_column.len(), value_column.null_count());
        let typed_struct = struct_array
            .column_by_name("typed_value")
            .expect("typed_value column")
            .as_any()
            .downcast_ref::<StructArray>()
            .expect("typed struct");
        assert!(
            extract_typed_values_for_path(typed_struct, expected_path).is_some(),
            "typed path {expected_path} missing from squeezed variant"
        );
    }

    #[test]
    fn test_variant_squeeze_with_hint() {
        let policy = TranscodeSqueezeEvict;
        let states = LiquidCompressorStates::new();
        let variant_arr = enriched_variant_array("name", DataType::Utf8);
        let hint = CacheExpression::variant_get("name", DataType::Utf8);

        let (new_batch, bytes) =
            policy.squeeze(CacheEntry::memory_arrow(variant_arr), &states, Some(&hint));

        match (new_batch.into_data(), bytes) {
            (CachedData::MemoryHybridLiquid(hybrid), Some(b)) => {
                assert_variant_hybrid(&hybrid, "name", &b);
            }
            other => panic!("expected MemoryHybridLiquid with bytes, got {other:?}"),
        }
    }

    #[test]
    fn test_variant_squeeze_with_int64_path() {
        let policy = TranscodeSqueezeEvict;
        let states = LiquidCompressorStates::new();
        let variant_arr = enriched_variant_array("age", DataType::Int64);
        let hint = CacheExpression::variant_get("age", DataType::Int64);

        let (new_batch, bytes) =
            policy.squeeze(CacheEntry::memory_arrow(variant_arr), &states, Some(&hint));

        match (new_batch.into_data(), bytes) {
            (CachedData::MemoryHybridLiquid(hybrid), Some(b)) => {
                assert_variant_hybrid(&hybrid, "age", &b);
            }
            other => panic!("expected MemoryHybridLiquid with bytes, got {other:?}"),
        }
    }

    #[test]
    fn test_variant_squeeze_with_multiple_paths_preserves_all_fields() {
        let policy = TranscodeSqueezeEvict;
        let states = LiquidCompressorStates::new();
        let variant_arr = enriched_variant_array_with_paths(&[
            ("name", DataType::Utf8),
            ("age", DataType::Int64),
        ]);
        let hint = CacheExpression::variant_get("name", DataType::Utf8);

        let (new_batch, bytes) =
            policy.squeeze(CacheEntry::memory_arrow(variant_arr), &states, Some(&hint));

        match (new_batch.into_data(), bytes) {
            (CachedData::MemoryHybridLiquid(hybrid), Some(b)) => {
                assert!(!b.is_empty());
                let struct_hybrid = hybrid
                    .as_any()
                    .downcast_ref::<VariantStructHybridArray>()
                    .expect("multi-field hybrid array");
                let arrow_array = struct_hybrid
                    .to_arrow_array()
                    .expect("arrow reconstruction");
                let struct_array = arrow_array
                    .as_any()
                    .downcast_ref::<StructArray>()
                    .expect("variant struct");
                let typed_value = struct_array
                    .column_by_name("typed_value")
                    .expect("typed value column");
                let typed_struct = typed_value
                    .as_any()
                    .downcast_ref::<StructArray>()
                    .expect("typed struct");
                assert!(typed_struct.column_by_name("name").is_some());
                assert!(typed_struct.column_by_name("age").is_none());
            }
            other => panic!("expected MemoryHybridLiquid with bytes, got {other:?}"),
        }
    }

    #[test]
    fn test_variant_squeeze_without_hint() {
        let policy = TranscodeSqueezeEvict;
        let states = LiquidCompressorStates::new();
        let variant_arr = enriched_variant_array("name", DataType::Utf8);

        let (new_batch, bytes) =
            policy.squeeze(CacheEntry::memory_arrow(variant_arr), &states, None);

        match (new_batch.into_data(), bytes) {
            (CachedData::DiskArrow(_), Some(b)) => assert!(!b.is_empty()),
            (CachedData::MemoryLiquid(_), None) => {}
            other => panic!("expected DiskArrow with bytes or MemoryLiquid, got {other:?}"),
        }
    }

    #[test]
    fn test_variant_squeeze_skips_when_path_missing() {
        let policy = TranscodeSqueezeEvict;
        let states = LiquidCompressorStates::new();
        let variant_arr = enriched_variant_array("name", DataType::Utf8);
        let hint = CacheExpression::variant_get("age", DataType::Int64);

        let (new_batch, bytes) =
            policy.squeeze(CacheEntry::memory_arrow(variant_arr.clone()), &states, Some(&hint));

        match (new_batch.into_data(), bytes) {
            (CachedData::DiskArrow(dt), Some(b)) => {
                assert_eq!(dt, variant_arr.data_type().clone());
                assert!(!b.is_empty());
            }
            other => panic!("expected DiskArrow fallback when path missing, got {other:?}"),
        }
    }
}
