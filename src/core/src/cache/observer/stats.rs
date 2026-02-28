use std::fmt;
use std::sync::atomic::{AtomicU64, Ordering};

/// Macro to define runtime statistics metrics.
///
/// Usage:
/// ```ignore
/// define_runtime_stats! {
///     (field_name, "doc comment", method_name),
///     ...
/// }
/// ```
///
/// This generates:
/// - Fields in `RuntimeStats` struct
/// - Fields in `RuntimeStatsSnapshot` struct
/// - Increment methods (`incr_*`)
/// - `consume_snapshot` implementation
/// - `reset` implementation
macro_rules! define_runtime_stats {
    (
        $(
            ($field:ident, $doc:literal, $method:ident)
        ),* $(,)?
    ) => {
        /// Atomic runtime counters for cache API calls.
        #[derive(Debug, Default)]
        pub struct RuntimeStats {
            $(
                #[doc = $doc]
                pub(crate) $field: AtomicU64,
            )*
        }

        /// Immutable snapshot of [`RuntimeStats`].
        #[derive(Debug, Clone, Copy, serde::Serialize, serde::Deserialize)]
        pub struct RuntimeStatsSnapshot {
            $(
                #[doc = concat!("Total ", stringify!($field), ".")]
                pub $field: u64,
            )*
        }

        impl RuntimeStats {
            /// Return an immutable snapshot of the current runtime counters and reset the stats to 0.
            pub fn consume_snapshot(&self) -> RuntimeStatsSnapshot {
                let v = RuntimeStatsSnapshot {
                    $(
                        $field: self.$field.load(Ordering::Relaxed),
                    )*
                };
                self.reset();
                v
            }

            $(
                /// Increment counter.
                #[inline]
                pub fn $method(&self) {
                    self.$field.fetch_add(1, Ordering::Relaxed);
                }
            )*

            /// Reset the runtime stats to 0.
            pub fn reset(&self) {
                $(
                    self.$field.store(0, Ordering::Relaxed);
                )*
            }
        }

        impl fmt::Display for RuntimeStats {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                writeln!(f, "RuntimeStats:")?;
                $(
                    writeln!(f, "  {}: {}", stringify!($field), self.$field.load(Ordering::Relaxed))?;
                )*
                Ok(())
            }
        }

        impl fmt::Display for RuntimeStatsSnapshot {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                writeln!(f, "RuntimeStatsSnapshot:")?;
                $(
                    writeln!(f, "  {}: {}", stringify!($field), self.$field)?;
                )*
                Ok(())
            }
        }
    };
}

// Define all runtime statistics metrics here.
// To add a new metric, add a line: (field_name, "doc comment", method_name)
define_runtime_stats! {
    (get, "Number of `get` calls issued via `CachedData`.", incr_get),
    (get_with_selection, "Number of `get_with_selection` calls issued via `CachedData`.", incr_get_with_selection),
    (eval_predicate, "Number of `eval_predicate` calls issued via `CachedData`.", incr_eval_predicate),
    (get_squeezed_success, "Number of Squeezed-Liquid full evaluations finished without IO.", incr_get_squeezed_success),
    (get_squeezed_needs_io, "Number of Squeezed-Liquid full paths that required IO.", incr_get_squeezed_needs_io),
    (try_read_liquid_calls, "Number of `try_read_liquid` calls issued via `CachedData`.", incr_try_read_liquid),
    (hit_date32_expression_calls, "Number of `hit_date32_expression` calls.", incr_hit_date32_expression),
    (read_io_count, "Number of read IO operations.", incr_read_io_count),
    (write_io_count, "Number of write IO operations.", incr_write_io_count),
    (eval_predicate_on_liquid_failed, "Number of `eval_predicate` calls that failed on Liquid array.", incr_eval_predicate_on_liquid_failed),
    (squeezed_decompressed_count, "Number of decompressed Squeezed-Liquid entries.", __incr_squeezed_decompressed_count),
    (squeezed_total_count, "Total number of Squeezed-Liquid entries.", __incr_squeezed_total_count),
    (squeeze_io_saved, "Number of io saved by squeezing.", incr_squeeze_io_saved),
}

impl RuntimeStats {
    /// Track the number of decompressed Squeezed-Liquid entries.
    pub fn track_decompress_squeezed_count(&self, decompressed: usize, total: usize) {
        self.squeezed_decompressed_count
            .fetch_add(decompressed as u64, Ordering::Relaxed);
        self.squeezed_total_count
            .fetch_add(total as u64, Ordering::Relaxed);
    }
}

/// Snapshot of cache statistics.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct CacheStats {
    /// Total number of entries in the cache.
    pub total_entries: usize,
    /// Number of in-memory Arrow entries.
    pub memory_arrow_entries: usize,
    /// Number of in-memory Liquid entries.
    pub memory_liquid_entries: usize,
    /// Number of in-memory Squeezed-Liquid entries.
    pub memory_squeezed_liquid_entries: usize,
    /// Number of on-disk Liquid entries.
    pub disk_liquid_entries: usize,
    /// Number of on-disk Arrow entries.
    pub disk_arrow_entries: usize,
    /// Total size of in-memory Arrow entries in bytes.
    pub memory_arrow_bytes: usize,
    /// Total size of in-memory Liquid entries in bytes.
    pub memory_liquid_bytes: usize,
    /// Total size of in-memory Squeezed-Liquid entries in bytes.
    pub memory_squeezed_liquid_bytes: usize,
    /// Total memory usage of the cache.
    pub memory_usage_bytes: usize,
    /// Total disk usage of the cache.
    pub disk_usage_bytes: usize,
    /// Maximum cache size.
    pub max_cache_bytes: usize,
    /// Runtime counters snapshot.
    pub runtime: RuntimeStatsSnapshot,
}
