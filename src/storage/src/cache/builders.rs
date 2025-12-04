use std::future::{Future, IntoFuture};
use std::path::PathBuf;
use std::pin::Pin;

use arrow::array::{ArrayRef, BooleanArray};
use arrow::buffer::BooleanBuffer;
use datafusion::physical_plan::PhysicalExpr;

use super::cached_batch::CacheEntry;
use super::core::LiquidCache;
use super::io_context::{DefaultIoContext, IoContext};
use super::policies::{CachePolicy, HydrationPolicy, SqueezePolicy, TranscodeSqueezeEvict};
use super::{CacheExpression, EntryID, LiquidPolicy};
use crate::sync::Arc;

/// Builder for [LiquidCache].
///
/// Example:
/// ```rust
/// use liquid_cache_storage::cache::LiquidCacheBuilder;
/// use liquid_cache_storage::cache_policies::LiquidPolicy;
///
///
/// let _storage = LiquidCacheBuilder::new()
///     .with_batch_size(8192)
///     .with_max_cache_bytes(1024 * 1024 * 1024)
///     .with_cache_policy(Box::new(LiquidPolicy::new()))
///     .build();
/// ```
pub struct LiquidCacheBuilder {
    batch_size: usize,
    max_cache_bytes: usize,
    cache_dir: Option<PathBuf>,
    cache_policy: Box<dyn CachePolicy>,
    hydration_policy: Box<dyn HydrationPolicy>,
    squeeze_policy: Box<dyn SqueezePolicy>,
    io_worker: Option<Arc<dyn IoContext>>,
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
            max_cache_bytes: 1024 * 1024 * 1024,
            cache_dir: None,
            cache_policy: Box::new(LiquidPolicy::new()),
            hydration_policy: Box::new(super::NoHydration::new()),
            squeeze_policy: Box::new(TranscodeSqueezeEvict),
            io_worker: None,
        }
    }

    /// Set the cache directory for the cache.
    /// Default is a temporary directory.
    pub fn with_cache_dir(mut self, cache_dir: PathBuf) -> Self {
        self.cache_dir = Some(cache_dir);
        self
    }

    /// Set the batch size for the cache.
    /// Default is 8192.
    pub fn with_batch_size(mut self, batch_size: usize) -> Self {
        self.batch_size = batch_size;
        self
    }

    /// Set the max cache bytes for the cache.
    /// Default is 1GB.
    pub fn with_max_cache_bytes(mut self, max_cache_bytes: usize) -> Self {
        self.max_cache_bytes = max_cache_bytes;
        self
    }

    /// Set the cache policy for the cache.
    /// Default is [LiquidPolicy].
    pub fn with_cache_policy(mut self, policy: Box<dyn CachePolicy>) -> Self {
        self.cache_policy = policy;
        self
    }

    /// Set the hydration policy for the cache.
    /// Default is [NoHydration].
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

    /// Set the io worker for the cache.
    /// Default is [DefaultIoContext].
    pub fn with_io_worker(mut self, io_worker: Arc<dyn IoContext>) -> Self {
        self.io_worker = Some(io_worker);
        self
    }

    /// Build the cache storage.
    ///
    /// The cache storage is wrapped in an [Arc] to allow for concurrent access.
    pub fn build(self) -> Arc<LiquidCache> {
        let cache_dir = self
            .cache_dir
            .unwrap_or_else(|| tempfile::tempdir().unwrap().keep());
        let io_worker = self
            .io_worker
            .unwrap_or_else(|| Arc::new(DefaultIoContext::new(cache_dir.clone())));
        Arc::new(LiquidCache::new(
            self.batch_size,
            self.max_cache_bytes,
            cache_dir,
            self.squeeze_policy,
            self.cache_policy,
            self.hydration_policy,
            io_worker,
        ))
    }
}

/// Builder returned by [`LiquidCache::insert`] for configuring cache writes.
#[derive(Debug)]
pub struct Insert<'a> {
    pub(super) storage: &'a Arc<LiquidCache>,
    pub(super) entry_id: EntryID,
    pub(super) batch: ArrayRef,
}

impl<'a> Insert<'a> {
    pub(super) fn new(storage: &'a Arc<LiquidCache>, entry_id: EntryID, batch: ArrayRef) -> Self {
        Self {
            storage,
            entry_id,
            batch,
        }
    }

    async fn run(self) {
        let batch = CacheEntry::memory_arrow(self.batch);
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
        if self.selection.is_some() {
            self.storage.runtime_stats.incr_get_with_selection();
        } else {
            self.storage.runtime_stats.incr_get_arrow_array();
        }
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

/// Builder for predicate evaluation on cached data.
#[derive(Debug)]
pub struct EvaluatePredicate<'a> {
    pub(super) storage: &'a LiquidCache,
    pub(super) entry_id: &'a EntryID,
    pub(super) predicate: &'a Arc<dyn PhysicalExpr>,
    pub(super) selection: Option<&'a BooleanBuffer>,
}

impl<'a> EvaluatePredicate<'a> {
    pub(super) fn new(
        storage: &'a LiquidCache,
        entry_id: &'a EntryID,
        predicate: &'a Arc<dyn PhysicalExpr>,
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
    pub async fn read(self) -> Option<Result<BooleanArray, ArrayRef>> {
        self.storage
            .eval_predicate_internal(self.entry_id, self.selection, self.predicate)
            .await
    }
}

impl<'a> IntoFuture for EvaluatePredicate<'a> {
    type Output = Option<Result<BooleanArray, ArrayRef>>;
    type IntoFuture = Pin<
        Box<dyn std::future::Future<Output = Option<Result<BooleanArray, ArrayRef>>> + Send + 'a>,
    >;

    fn into_future(self) -> Self::IntoFuture {
        Box::pin(async move { self.read().await })
    }
}
