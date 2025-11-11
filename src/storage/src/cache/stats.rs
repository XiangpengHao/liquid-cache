use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

/// Atomic runtime counters for cache API calls.
#[derive(Debug, Default)]
pub struct RuntimeStats {
    /// Number of `get_arrow_array` calls issued via `CachedData`.
    pub(crate) get_arrow_array_calls: AtomicU64,
    /// Number of `get_with_selection` calls issued via `CachedData`.
    pub(crate) get_with_selection_calls: AtomicU64,
    /// Number of `get_with_predicate` calls issued via `CachedData`.
    pub(crate) get_with_predicate_calls: AtomicU64,
    /// Number of Hybrid-Liquid predicate evaluations finished without IO.
    pub(crate) get_predicate_hybrid_success: AtomicU64,
    /// Number of Hybrid-Liquid predicate paths that required IO.
    pub(crate) get_predicate_hybrid_needs_io: AtomicU64,
    /// Number of Hybrid-Liquid predicate paths that were unsupported and fell back.
    pub(crate) get_predicate_hybrid_unsupported: AtomicU64,
    /// Number of `try_read_liquid` calls issued via `CachedData`.
    pub(crate) try_read_liquid_calls: AtomicU64,
    /// Number of `hit_date32_expression` calls.
    pub(crate) hit_date32_expression_calls: AtomicU64,
}

/// Immutable snapshot of [`RuntimeStats`].
#[derive(Debug, Clone, Copy)]
pub struct RuntimeStatsSnapshot {
    /// Total `get_arrow_array` calls.
    pub get_arrow_array_calls: u64,
    /// Total `get_with_selection` calls.
    pub get_with_selection_calls: u64,
    /// Total `get_with_predicate` calls.
    pub get_with_predicate_calls: u64,
    /// Total Hybrid-Liquid predicate successes (no IO).
    pub get_predicate_hybrid_success: u64,
    /// Total Hybrid-Liquid predicate paths requiring IO.
    pub get_predicate_hybrid_needs_io: u64,
    /// Total Hybrid-Liquid predicate paths that were unsupported.
    pub get_predicate_hybrid_unsupported: u64,
    /// Total `try_read_liquid` calls.
    pub try_read_liquid_calls: u64,
    /// Total `hit_date32_expression` calls.
    pub hit_date32_expression_calls: u64,
}

impl RuntimeStats {
    /// Return an immutable snapshot of the current runtime counters and reset the stats to 0.
    pub fn consume_snapshot(&self) -> RuntimeStatsSnapshot {
        let v = RuntimeStatsSnapshot {
            get_arrow_array_calls: self.get_arrow_array_calls.load(Ordering::Relaxed),
            get_with_selection_calls: self.get_with_selection_calls.load(Ordering::Relaxed),
            get_with_predicate_calls: self.get_with_predicate_calls.load(Ordering::Relaxed),
            get_predicate_hybrid_success: self.get_predicate_hybrid_success.load(Ordering::Relaxed),
            get_predicate_hybrid_needs_io: self
                .get_predicate_hybrid_needs_io
                .load(Ordering::Relaxed),
            get_predicate_hybrid_unsupported: self
                .get_predicate_hybrid_unsupported
                .load(Ordering::Relaxed),
            try_read_liquid_calls: self.try_read_liquid_calls.load(Ordering::Relaxed),
            hit_date32_expression_calls: self.hit_date32_expression_calls.load(Ordering::Relaxed),
        };
        self.reset();
        v
    }

    /// Increment `get_arrow_array` counter.
    #[inline]
    pub fn incr_get_arrow_array(&self) {
        self.get_arrow_array_calls.fetch_add(1, Ordering::Relaxed);
    }

    /// Increment `get_with_selection` counter.
    #[inline]
    pub fn incr_get_with_selection(&self) {
        self.get_with_selection_calls
            .fetch_add(1, Ordering::Relaxed);
    }

    /// Increment `get_with_predicate` counter.
    #[inline]
    pub fn incr_get_with_predicate(&self) {
        self.get_with_predicate_calls
            .fetch_add(1, Ordering::Relaxed);
    }

    /// Increment `try_read_liquid` counter.
    #[inline]
    pub fn incr_try_read_liquid(&self) {
        self.try_read_liquid_calls.fetch_add(1, Ordering::Relaxed);
    }

    /// Increment Hybrid-Liquid predicate success counter.
    #[inline]
    pub fn incr_get_predicate_hybrid_success(&self) {
        self.get_predicate_hybrid_success
            .fetch_add(1, Ordering::Relaxed);
    }

    /// Increment Hybrid-Liquid predicate needs-IO counter.
    #[inline]
    pub fn incr_get_predicate_hybrid_needs_io(&self) {
        self.get_predicate_hybrid_needs_io
            .fetch_add(1, Ordering::Relaxed);
    }

    /// Increment Hybrid-Liquid predicate unsupported counter.
    #[inline]
    pub fn incr_get_predicate_hybrid_unsupported(&self) {
        self.get_predicate_hybrid_unsupported
            .fetch_add(1, Ordering::Relaxed);
    }

    /// Increment `hit_date32_expression` counter.
    #[inline]
    pub fn incr_hit_date32_expression(&self) {
        self.hit_date32_expression_calls
            .fetch_add(1, Ordering::Relaxed);
    }

    /// Reset the runtime stats to 0.
    pub fn reset(&self) {
        self.get_arrow_array_calls.store(0, Ordering::Relaxed);
        self.get_with_selection_calls.store(0, Ordering::Relaxed);
        self.get_with_predicate_calls.store(0, Ordering::Relaxed);
        self.get_predicate_hybrid_success
            .store(0, Ordering::Relaxed);
        self.get_predicate_hybrid_needs_io
            .store(0, Ordering::Relaxed);
        self.get_predicate_hybrid_unsupported
            .store(0, Ordering::Relaxed);
        self.try_read_liquid_calls.store(0, Ordering::Relaxed);
        self.hit_date32_expression_calls.store(0, Ordering::Relaxed);
    }
}

/// Snapshot of cache statistics.
#[derive(Debug, Clone)]
pub struct CacheStats {
    /// Total number of entries in the cache.
    pub total_entries: usize,
    /// Number of in-memory Arrow entries.
    pub memory_arrow_entries: usize,
    /// Number of in-memory Liquid entries.
    pub memory_liquid_entries: usize,
    /// Number of in-memory Hybrid-Liquid entries.
    pub memory_hybrid_liquid_entries: usize,
    /// Number of on-disk Liquid entries.
    pub disk_liquid_entries: usize,
    /// Number of on-disk Arrow entries.
    pub disk_arrow_entries: usize,
    /// Total memory usage of the cache.
    pub memory_usage_bytes: usize,
    /// Total disk usage of the cache.
    pub disk_usage_bytes: usize,
    /// Maximum cache size.
    pub max_cache_bytes: usize,
    /// Cache root directory.
    pub cache_root_dir: PathBuf,
    /// Runtime counters snapshot.
    pub runtime: RuntimeStatsSnapshot,
}
