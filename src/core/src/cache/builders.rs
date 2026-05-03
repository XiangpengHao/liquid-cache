use std::future::{Future, IntoFuture};
use std::pin::Pin;

use arrow::array::{
    Array, ArrayData, ArrayRef, BinaryViewArray, BooleanArray, StringViewArray, make_array,
};
use arrow::buffer::BooleanBuffer;

use super::cached_batch::CacheEntry;
use super::core::LiquidCache;
use super::io_context::{DefaultCacheMetadata, EntryMetadata};
use super::policies::{CachePolicy, HydrationPolicy, SqueezePolicy, TranscodeSqueezeEvict};
use super::{CacheExpression, EntryID, LiquidExpr, LiquidPolicy};
use crate::sync::Arc;

/// Builder for [LiquidCache].
///
/// Example:
/// ```rust
/// use liquid_cache::cache::LiquidCacheBuilder;
/// use liquid_cache::cache_policies::LiquidPolicy;
///
/// tokio_test::block_on(async {
///     let _storage = LiquidCacheBuilder::new()
///         .with_batch_size(8192)
///         .with_max_memory_bytes(1024 * 1024 * 1024)
///         .with_cache_policy(Box::new(LiquidPolicy::new()))
///         .build()
///         .await;
/// });
/// ```
pub struct LiquidCacheBuilder {
    batch_size: usize,
    max_memory_bytes: usize,
    max_disk_bytes: usize,
    disk_watermark: f64,
    cache_policy: Box<dyn CachePolicy>,
    hydration_policy: Box<dyn HydrationPolicy>,
    squeeze_policy: Box<dyn SqueezePolicy>,
    metadata: Option<Arc<dyn EntryMetadata>>,
    store: Option<t4::Store>,
    squeeze_victims_concurrently: bool,
}

impl Default for LiquidCacheBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl LiquidCacheBuilder {
    /// Create a new instance of [LiquidCacheBuilder].
    pub fn new() -> Self {
        Self {
            batch_size: 8192,
            max_memory_bytes: 1024 * 1024 * 1024,
            max_disk_bytes: usize::MAX,
            disk_watermark: 0.9,
            cache_policy: Box::new(LiquidPolicy::new()),
            hydration_policy: Box::new(super::AlwaysHydrate::new()),
            squeeze_policy: Box::new(TranscodeSqueezeEvict),
            metadata: None,
            store: None,
            squeeze_victims_concurrently: !cfg!(test),
        }
    }

    /// Set the batch size for the cache.
    /// Default is 8192.
    pub fn with_batch_size(mut self, batch_size: usize) -> Self {
        self.batch_size = batch_size;
        self
    }

    /// Set the max memory bytes for the cache.
    /// Default is 1GB.
    pub fn with_max_memory_bytes(mut self, max_memory_bytes: usize) -> Self {
        self.max_memory_bytes = max_memory_bytes;
        self
    }

    /// Set the max disk bytes for the cache.
    /// Default is unlimited — the cache will use available disk space without a cap.
    pub fn with_max_disk_bytes(mut self, max_disk_bytes: usize) -> Self {
        self.max_disk_bytes = max_disk_bytes;
        self
    }

    /// Set the disk watermark ratio (0.0–1.0). Default is 0.9.
    /// Eviction triggers when disk usage exceeds this fraction of max_disk_bytes.
    pub fn with_disk_watermark(mut self, ratio: f64) -> Self {
        self.disk_watermark = ratio;
        self
    }

    /// Set the cache policy for the cache.
    /// Default is [LiquidPolicy].
    pub fn with_cache_policy(mut self, policy: Box<dyn CachePolicy>) -> Self {
        self.cache_policy = policy;
        self
    }

    /// Set the hydration policy for the cache.
    /// Default is [crate::cache::NoHydration].
    pub fn with_hydration_policy(mut self, policy: Box<dyn HydrationPolicy>) -> Self {
        self.hydration_policy = policy;
        self
    }

    /// Set the squeeze policy for the cache.
    /// Default is [TranscodeSqueezeEvict].
    pub fn with_squeeze_policy(mut self, policy: Box<dyn SqueezePolicy>) -> Self {
        self.squeeze_policy = policy;
        self
    }

    /// Set the [EntryMetadata] for the cache.
    /// Default is [DefaultCacheMetadata].
    pub fn with_metadata(mut self, metadata: Arc<dyn EntryMetadata>) -> Self {
        self.metadata = Some(metadata);
        self
    }

    /// Set the [`t4::Store`] used for on-disk IO.
    /// If not provided, the builder mounts a fresh store at a temporary directory.
    pub fn with_store(mut self, store: t4::Store) -> Self {
        self.store = Some(store);
        self
    }

    /// Set whether cache victims are squeezed concurrently.
    pub fn with_squeeze_victims_concurrently(mut self, enabled: bool) -> Self {
        self.squeeze_victims_concurrently = enabled;
        self
    }

    /// Build the cache storage.
    ///
    /// The cache storage is wrapped in an [Arc] to allow for concurrent access.
    /// When no [`t4::Store`] is provided, one is mounted at a temporary directory.
    pub async fn build(self) -> Arc<LiquidCache> {
        let store = match self.store {
            Some(store) => store,
            None => {
                let cache_dir = tempfile::tempdir().unwrap().keep();
                let store_path = cache_dir.join("liquid_cache.t4");
                t4::mount(&store_path)
                    .await
                    .expect("failed to mount t4 store")
            }
        };
        let metadata = self
            .metadata
            .unwrap_or_else(|| Arc::new(DefaultCacheMetadata::new()));
        Arc::new(LiquidCache::new(
            self.batch_size,
            self.max_memory_bytes,
            self.max_disk_bytes,
            self.disk_watermark,
            self.squeeze_policy,
            self.cache_policy,
            self.hydration_policy,
            metadata,
            store,
            self.squeeze_victims_concurrently,
        ))
    }
}

/// Builder returned by [`LiquidCache::insert`] for configuring cache writes.
#[derive(Debug)]
pub struct Insert<'a> {
    pub(super) storage: &'a Arc<LiquidCache>,
    pub(super) entry_id: EntryID,
    pub(super) batch: ArrayRef,
    pub(super) skip_gc: bool,
    pub(super) squeeze_hint: Option<Arc<CacheExpression>>,
}

impl<'a> Insert<'a> {
    pub(super) fn new(storage: &'a Arc<LiquidCache>, entry_id: EntryID, batch: ArrayRef) -> Self {
        Self {
            storage,
            entry_id,
            batch,
            skip_gc: false,
            squeeze_hint: None,
        }
    }

    /// Skip garbage collection of view arrays.
    pub fn with_skip_gc(mut self) -> Self {
        self.skip_gc = true;
        self
    }

    /// Set a squeeze hint for the entry.
    pub fn with_squeeze_hint(mut self, expression: Arc<CacheExpression>) -> Self {
        self.squeeze_hint = Some(expression);
        self
    }

    async fn run(self) {
        let batch = if self.skip_gc {
            self.batch.clone()
        } else {
            maybe_gc_view_arrays(&self.batch).unwrap_or_else(|| self.batch.clone())
        };
        if let Some(squeeze_hint) = self.squeeze_hint {
            self.storage.add_squeeze_hint(&self.entry_id, squeeze_hint);
        }
        let batch = CacheEntry::memory_arrow(batch);
        self.storage.insert_inner(self.entry_id, batch).await;
    }
}

impl<'a> IntoFuture for Insert<'a> {
    type Output = ();
    type IntoFuture = Pin<Box<dyn Future<Output = ()> + Send + 'a>>;

    fn into_future(self) -> Self::IntoFuture {
        Box::pin(async move { self.run().await })
    }
}

/// Builder returned by [`LiquidCache::get`] for configuring cache reads.
#[derive(Debug)]
pub struct Get<'a> {
    pub(super) storage: &'a LiquidCache,
    pub(super) entry_id: &'a EntryID,
    pub(super) selection: Option<&'a BooleanBuffer>,
    pub(super) expression_hint: Option<Arc<CacheExpression>>,
}

impl<'a> Get<'a> {
    pub(super) fn new(storage: &'a LiquidCache, entry_id: &'a EntryID) -> Self {
        Self {
            storage,
            entry_id,
            selection: None,
            expression_hint: None,
        }
    }

    /// Attach a selection bitmap used to filter rows prior to materialization.
    pub fn with_selection(mut self, selection: &'a BooleanBuffer) -> Self {
        self.selection = Some(selection);
        self
    }

    /// Attach an expression hint that may help optimize cache materialization.
    pub fn with_expression_hint(mut self, expression: Arc<CacheExpression>) -> Self {
        self.expression_hint = Some(expression);
        self
    }

    /// Attach an optional expression hint.
    pub fn with_optional_expression_hint(
        mut self,
        expression: Option<Arc<CacheExpression>>,
    ) -> Self {
        self.expression_hint = expression;
        self
    }

    /// Materialize the cached array as [`ArrayRef`].
    pub async fn read(self) -> Option<ArrayRef> {
        self.storage.observer().on_get(self.selection.is_some());
        self.storage
            .read_arrow_array(
                self.entry_id,
                self.selection,
                self.expression_hint.as_deref(),
            )
            .await
    }
}

impl<'a> IntoFuture for Get<'a> {
    type Output = Option<ArrayRef>;
    type IntoFuture = Pin<Box<dyn std::future::Future<Output = Option<ArrayRef>> + Send + 'a>>;

    fn into_future(self) -> Self::IntoFuture {
        Box::pin(async move { self.read().await })
    }
}

/// Recursively garbage collects view arrays (BinaryView/StringView) within an array tree.
fn maybe_gc_view_arrays(array: &ArrayRef) -> Option<ArrayRef> {
    if let Some(binary_view) = array.as_any().downcast_ref::<BinaryViewArray>() {
        return Some(Arc::new(binary_view.gc()));
    }
    if let Some(utf8_view) = array.as_any().downcast_ref::<StringViewArray>() {
        return Some(Arc::new(utf8_view.gc()));
    }

    let data = array.to_data();
    if data.child_data().is_empty() {
        return None;
    }

    let mut changed = false;
    let mut children: Vec<ArrayData> = Vec::with_capacity(data.child_data().len());
    for child in data.child_data() {
        let child_array = make_array(child.clone());
        if let Some(gc_child) = maybe_gc_view_arrays(&child_array) {
            changed = true;
            children.push(gc_child.to_data());
        } else {
            children.push(child.clone());
        }
    }

    if !changed {
        return None;
    }

    let new_data = data.into_builder().child_data(children).build().ok()?;
    Some(make_array(new_data))
}

/// Builder for predicate evaluation on cached data.
#[derive(Debug)]
pub struct EvaluatePredicate<'a> {
    pub(super) storage: &'a LiquidCache,
    pub(super) entry_id: &'a EntryID,
    pub(super) predicate: &'a LiquidExpr,
    pub(super) selection: Option<&'a BooleanBuffer>,
}

impl<'a> EvaluatePredicate<'a> {
    pub(super) fn new(
        storage: &'a LiquidCache,
        entry_id: &'a EntryID,
        predicate: &'a LiquidExpr,
    ) -> Self {
        Self {
            storage,
            entry_id,
            predicate,
            selection: None,
        }
    }

    /// Attach a selection bitmap used to pre-filter rows before predicate evaluation.
    pub fn with_selection(mut self, selection: &'a BooleanBuffer) -> Self {
        self.selection = Some(selection);
        self
    }

    /// Evaluate the predicate against the cached data.
    pub async fn read(self) -> Option<BooleanArray> {
        self.storage
            .eval_predicate_internal(self.entry_id, self.selection, self.predicate)
            .await
    }
}

impl<'a> IntoFuture for EvaluatePredicate<'a> {
    type Output = Option<BooleanArray>;
    type IntoFuture = Pin<Box<dyn std::future::Future<Output = Option<BooleanArray>> + Send + 'a>>;

    fn into_future(self) -> Self::IntoFuture {
        Box::pin(async move { self.read().await })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::{AsArray, StructArray};
    use arrow_schema::{DataType, Field, Fields};

    #[tokio::test]
    async fn insert_gcs_view_arrays_recursively() {
        // Build view arrays then slice to create non-zero offsets (and larger backing buffers).
        let bin = Arc::new(BinaryViewArray::from(vec![
            Some(b"long_prefix_m0" as &[u8]),
            Some(b"m1"),
        ])) as ArrayRef;
        let str_view = Arc::new(StringViewArray::from(vec![
            Some("long_prefix_s0"),
            Some("s1"),
        ])) as ArrayRef;
        let variant_metadata = Arc::new(BinaryViewArray::from(vec![
            Some(b"meta0" as &[u8]),
            Some(b"meta1"),
        ])) as ArrayRef;
        let variant_value = Arc::new(BinaryViewArray::from(vec![
            Some(b"value0" as &[u8]),
            Some(b"value1"),
        ])) as ArrayRef;

        // Slice to keep only the second element so buffers still reference unused bytes.
        let bin_slice = bin.slice(1, 1);
        let str_slice = str_view.slice(1, 1);
        let variant_metadata_slice = variant_metadata.slice(1, 1);
        let variant_value_slice = variant_value.slice(1, 1);

        // Variant-like struct: metadata (BinaryView), value (BinaryView), and a typed string view.
        let variant_typed_fields = Fields::from(vec![Arc::new(Field::new(
            "typed_str",
            DataType::Utf8View,
            true,
        ))]);
        let variant_struct_fields = Fields::from(vec![
            Arc::new(Field::new("metadata", DataType::BinaryView, true)),
            Arc::new(Field::new("value", DataType::BinaryView, true)),
            Arc::new(Field::new(
                "typed_value",
                DataType::Struct(variant_typed_fields.clone()),
                true,
            )),
        ]);
        let variant_struct = Arc::new(StructArray::new(
            variant_struct_fields.clone(),
            vec![
                variant_metadata_slice.clone(),
                variant_value_slice.clone(),
                Arc::new(StructArray::new(
                    variant_typed_fields.clone(),
                    vec![str_slice.clone()],
                    None,
                )) as ArrayRef,
            ],
            None,
        ));

        let root_fields = Fields::from(vec![
            Arc::new(Field::new("bin_view", DataType::BinaryView, true)),
            Arc::new(Field::new("str_view", DataType::Utf8View, true)),
            Arc::new(Field::new(
                "variant",
                DataType::Struct(variant_struct_fields.clone()),
                true,
            )),
        ]);
        let root = Arc::new(StructArray::new(
            root_fields,
            vec![
                bin_slice.clone(),
                str_slice.clone(),
                variant_struct.clone() as ArrayRef,
            ],
            None,
        )) as ArrayRef;

        let pre_size = root.get_array_memory_size();

        let cache = LiquidCacheBuilder::new().build().await;
        let entry_id = EntryID::from(123usize);
        cache.insert(entry_id, root.clone()).await;

        let stored = cache.get(&entry_id).await.expect("array present");
        let post_size = stored.get_array_memory_size();

        // GC should have compacted the view arrays, reducing memory footprint.
        assert!(post_size < pre_size, "expected gc to reduce memory usage");

        // Validate values are preserved.
        let struct_out = stored
            .as_any()
            .downcast_ref::<StructArray>()
            .expect("struct array");

        assert_eq!(struct_out.len(), 1);

        let bin_out = struct_out
            .column_by_name("bin_view")
            .unwrap()
            .as_binary_view();
        assert_eq!(bin_out.value(0), b"m1");

        let str_out = struct_out
            .column_by_name("str_view")
            .unwrap()
            .as_string_view();
        assert_eq!(str_out.value(0), "s1");

        let variant_out = struct_out.column_by_name("variant").unwrap().as_struct();
        let meta_out = variant_out
            .column_by_name("metadata")
            .unwrap()
            .as_binary_view();
        assert_eq!(meta_out.value(0), b"meta1");

        let val_out = variant_out
            .column_by_name("value")
            .unwrap()
            .as_binary_view();
        assert_eq!(val_out.value(0), b"value1");

        let typed_out = variant_out
            .column_by_name("typed_value")
            .unwrap()
            .as_struct();
        let typed_str_out = typed_out
            .column_by_name("typed_str")
            .unwrap()
            .as_string_view();
        assert_eq!(typed_str_out.value(0), "s1");
    }
}
