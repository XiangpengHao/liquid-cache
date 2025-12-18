mod internal_tracing;
mod stats;
mod tracer;

pub use internal_tracing::EventTrace;
pub use stats::{CacheStats, RuntimeStats, RuntimeStatsSnapshot};
pub use tracer::CacheTracer;

use std::path::Path;

use internal_tracing::EventTracer;
use stats::{RuntimeStats as RuntimeStatsInner, RuntimeStatsSnapshot as RuntimeStatsSnapshotInner};

#[derive(Debug)]
/// Cache-side observer for runtime stats, debug traces, and optional cache tracing.
pub struct Observer {
    runtime: RuntimeStatsInner,
    event_tracer: EventTracer,
    cache_tracer: CacheTracer,
}

impl Default for Observer {
    fn default() -> Self {
        Self::new()
    }
}

impl Observer {
    /// Create a new observer with all counters and traces reset.
    pub fn new() -> Self {
        Self {
            runtime: RuntimeStatsInner::default(),
            event_tracer: EventTracer::new(),
            cache_tracer: CacheTracer::new(),
        }
    }

    /// Snapshot runtime counters and reset them to zero.
    pub fn runtime_snapshot(&self) -> RuntimeStatsSnapshotInner {
        self.runtime.consume_snapshot()
    }

    /// Consume and clear the in-memory debug event trace.
    pub fn consume_event_trace(&self) -> EventTrace {
        self.event_tracer.drain()
    }

    /// Enable recording cache trace events (for offline analysis).
    pub fn enable_cache_trace(&self) {
        self.cache_tracer.enable();
    }

    /// Disable recording cache trace events.
    pub fn disable_cache_trace(&self) {
        self.cache_tracer.disable();
    }

    /// Flush recorded cache trace events to a Parquet file.
    pub fn flush_cache_trace(&self, to_file: impl AsRef<Path>) {
        self.cache_tracer.flush(to_file);
    }

    /// Access the underlying cache tracer.
    pub fn cache_tracer(&self) -> &CacheTracer {
        &self.cache_tracer
    }

    #[inline]
    pub(crate) fn on_get(&self, selection: bool) {
        self.runtime.incr_get();
        if selection {
            self.runtime.incr_get_with_selection();
        }
    }

    #[inline]
    pub(crate) fn on_try_read_liquid(&self) {
        self.runtime.incr_try_read_liquid();
    }

    #[inline]
    pub(crate) fn on_eval_predicate(&self) {
        self.runtime.incr_eval_predicate();
    }

    #[inline]
    pub(crate) fn on_eval_predicate_squeezed_success(&self) {
        self.runtime.incr_eval_predicate_squeezed_success();
    }

    #[inline]
    pub(crate) fn on_eval_predicate_squeezed_needs_io(&self) {
        self.runtime.incr_eval_predicate_squeezed_needs_io();
    }

    #[inline]
    pub(crate) fn on_get_squeezed_success(&self) {
        self.runtime.incr_get_squeezed_success();
    }

    #[inline]
    pub(crate) fn on_get_squeezed_needs_io(&self) {
        self.runtime.incr_get_squeezed_needs_io();
    }

    #[inline]
    pub(crate) fn on_hit_date32_expression(&self) {
        self.runtime.incr_hit_date32_expression();
    }

    pub(crate) fn record_internal(&self, event: InternalEvent) {
        match event {
            InternalEvent::IoWrite { .. } => self.runtime.incr_write_io_count(),
            InternalEvent::IoReadArrow { .. }
            | InternalEvent::IoReadLiquid { .. }
            | InternalEvent::IoReadSqueezedBacking { .. } => self.runtime.incr_read_io_count(),
            _ => {}
        }

        if cfg!(debug_assertions) {
            self.event_tracer.record(event);
        }
    }
}

pub(crate) use internal_tracing::InternalEvent;
