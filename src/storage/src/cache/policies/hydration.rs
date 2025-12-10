//! Hydration policies decide whether and how to promote squeezed/on-disk entries back into memory.

use std::sync::Arc;

use arrow::array::ArrayRef;

use crate::{
    cache::{
        CacheExpression, LiquidCompressorStates, VariantRequest, cached_batch::CacheEntry,
        utils::EntryID,
    },
    liquid_array::{
        LiquidArrayRef, LiquidSqueezedArray, LiquidSqueezedArrayRef, VariantStructSqueezedArray,
    },
};

use super::squeeze::try_variant_squeeze;

/// The materialized representation produced by a cache read.
#[derive(Debug, Clone)]
pub enum MaterializedEntry<'a> {
    /// Arrow array in memory.
    Arrow(&'a ArrayRef),
    /// Liquid array in memory.
    Liquid(&'a LiquidArrayRef),
}

/// Request context provided to a [`HydrationPolicy`].
#[derive(Debug, Clone)]
pub struct HydrationRequest<'a> {
    /// Cache key being materialized.
    pub entry_id: EntryID,
    /// The cached entry before materialization (e.g., `DiskArrow`).
    pub cached: &'a CacheEntry,
    /// The fully materialized entry produced by the read path.
    pub materialized: MaterializedEntry<'a>,
    /// Optional expression hint associated with the read.
    pub expression: Option<&'a CacheExpression>,
    /// Compressor state used for hydrating into squeezed representations.
    pub compressor: Arc<LiquidCompressorStates>,
}

/// Decide if a materialized entry should be promoted back into memory.
pub trait HydrationPolicy: std::fmt::Debug + Send + Sync {
    /// Determine how to hydrate a cache entry that was just materialized.
    /// Return a new cache entry to insert if hydration is desired.
    fn hydrate(&self, request: &HydrationRequest<'_>) -> Option<CacheEntry>;
}

/// Default hydration policy: always keep a materialized cache miss in memory
/// by promoting along the path: disk -> squeezed -> liquid -> arrow.
#[derive(Debug, Default, Clone)]
pub struct AlwaysHydrate;

impl AlwaysHydrate {
    /// Create a new [`AlwaysHydrate`] policy.
    pub fn new() -> Self {
        Self
    }
}

fn hydrate_variant_paths(
    requests: &[VariantRequest],
    materialized: &ArrayRef,
    squeezed: &VariantStructSqueezedArray,
    compressor: &LiquidCompressorStates,
) -> Option<CacheEntry> {
    let missing_requests: Vec<VariantRequest> = requests
        .iter()
        .filter(|request| !squeezed.contains_path(request.path()))
        .cloned()
        .collect();
    if missing_requests.is_empty() {
        return None;
    }

    let (new_squeezed, _) = try_variant_squeeze(materialized, &missing_requests, compressor)?;
    let new_variant = new_squeezed
        .as_any()
        .downcast_ref::<VariantStructSqueezedArray>()?;

    let mut combined_values = squeezed.typed_values();
    combined_values.extend(new_variant.typed_values());

    let nulls = squeezed.nulls().or_else(|| new_variant.nulls());
    let merged = VariantStructSqueezedArray::new(
        combined_values,
        nulls,
        squeezed.original_arrow_data_type(),
    );
    Some(CacheEntry::memory_squeezed_liquid(
        Arc::new(merged) as LiquidSqueezedArrayRef
    ))
}

impl HydrationPolicy for AlwaysHydrate {
    fn hydrate(&self, request: &HydrationRequest<'_>) -> Option<CacheEntry> {
        match (request.cached, &request.materialized) {
            (CacheEntry::DiskArrow(_), MaterializedEntry::Arrow(arr)) => {
                if let Some(CacheExpression::VariantGet { requests }) = request.expression
                    && let Some((squeezed, _bytes)) =
                        try_variant_squeeze(arr, requests, request.compressor.as_ref())
                {
                    return Some(CacheEntry::memory_squeezed_liquid(squeezed));
                }
                Some(CacheEntry::memory_arrow((*arr).clone()))
            }
            (CacheEntry::DiskLiquid(_), MaterializedEntry::Liquid(liq)) => {
                Some(CacheEntry::memory_liquid((*liq).clone()))
            }
            (CacheEntry::MemoryLiquid(_), _) => None,
            // When already squeezed/hybrid or liquid in memory, prefer promoting to Arrow if available.
            (CacheEntry::MemorySqueezedLiquid(squeezed_entry), MaterializedEntry::Arrow(arr)) => {
                if let Some(CacheExpression::VariantGet { requests }) = request.expression
                    && let Some(squeezed) = squeezed_entry
                        .as_any()
                        .downcast_ref::<VariantStructSqueezedArray>()
                    && let Some(entry) =
                        hydrate_variant_paths(requests, arr, squeezed, request.compressor.as_ref())
                {
                    return Some(entry);
                }
                Some(CacheEntry::memory_arrow((*arr).clone()))
            }
            (CacheEntry::MemorySqueezedLiquid(_), MaterializedEntry::Liquid(liq)) => {
                Some(CacheEntry::memory_liquid((*liq).clone()))
            }
            _ => None,
        }
    }
}

/// No hydration policy: never promote a materialized entry back into memory.
#[derive(Debug, Default, Clone)]
pub struct NoHydration;

impl NoHydration {
    /// Create a new [`NoHydration`] policy.
    pub fn new() -> Self {
        Self
    }
}

impl HydrationPolicy for NoHydration {
    fn hydrate(&self, _request: &HydrationRequest<'_>) -> Option<CacheEntry> {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use parquet_variant_compute::json_to_variant;

    use crate::cache::utils::LiquidCompressorStates;
    use arrow::array::StringArray;
    use arrow_schema::DataType;

    fn variant_array_from_json(values: &[&str]) -> ArrayRef {
        let strings: ArrayRef = Arc::new(StringArray::from_iter_values(values.iter().copied()));
        let variant = json_to_variant(&strings).expect("variant array");
        let struct_array = variant.into_inner();
        Arc::new(struct_array) as ArrayRef
    }

    #[test]
    fn hydrates_disk_arrow_variant_to_squeezed() {
        let arr = variant_array_from_json(&[r#"{"name":"Ada","age":30}"#]);
        let expr = CacheExpression::variant_get("age", DataType::Int64);
        let policy = AlwaysHydrate::new();
        let compressor = Arc::new(LiquidCompressorStates::new());
        let cached_entry = CacheEntry::disk_arrow(arr.data_type().clone());

        let hydrated = policy.hydrate(&HydrationRequest {
            entry_id: EntryID::from(0),
            cached: &cached_entry,
            materialized: MaterializedEntry::Arrow(&arr),
            expression: Some(&expr),
            compressor,
        });

        let entry = hydrated.expect("should hydrate");
        match entry {
            CacheEntry::MemorySqueezedLiquid(squeezed) => {
                let variant = squeezed
                    .as_any()
                    .downcast_ref::<VariantStructSqueezedArray>()
                    .expect("variant squeezed");
                assert!(variant.contains_path("age"));
            }
            other => panic!("expected squeezed entry, got {other:?}"),
        }
    }

    #[test]
    fn hydrates_squeezed_variant_adds_missing_path() {
        let arr = variant_array_from_json(&[r#"{"name":"Ada","age":30}"#]);
        let name_expr = CacheExpression::variant_get("name", DataType::Utf8);
        let age_expr = CacheExpression::variant_get("age", DataType::Int64);
        let compressor = Arc::new(LiquidCompressorStates::new());

        // Build an initial squeezed array containing only the "name" path.
        let (squeezed, _) = try_variant_squeeze(
            &arr,
            name_expr.variant_requests().unwrap(),
            compressor.as_ref(),
        )
        .expect("squeeze name");
        let cached_entry = CacheEntry::memory_squeezed_liquid(squeezed.clone());

        let policy = AlwaysHydrate::new();
        let hydrated = policy.hydrate(&HydrationRequest {
            entry_id: EntryID::from(1),
            cached: &cached_entry,
            materialized: MaterializedEntry::Arrow(&arr),
            expression: Some(&age_expr),
            compressor,
        });

        let entry = hydrated.expect("should hydrate");
        let squeezed = match entry {
            CacheEntry::MemorySqueezedLiquid(sq) => sq,
            other => panic!("expected squeezed entry, got {other:?}"),
        };
        let variant = squeezed
            .as_any()
            .downcast_ref::<VariantStructSqueezedArray>()
            .expect("variant squeezed");
        assert!(variant.contains_path("name"));
        assert!(variant.contains_path("age"));
    }
}
