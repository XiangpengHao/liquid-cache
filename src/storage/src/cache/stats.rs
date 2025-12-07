use std::fmt;
use std::path::PathBuf;
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
    (get_arrow_array_calls, "Number of `get_arrow_array` calls issued via `CachedData`.", incr_get_arrow_array),
    (get_with_selection_calls, "Number of `get_with_selection` calls issued via `CachedData`.", incr_get_with_selection),
    (get_with_predicate_calls, "Number of `get_with_predicate` calls issued via `CachedData`.", incr_get_with_predicate),
    (get_predicate_squeezed_success, "Number of Squeezed-Liquid predicate evaluations finished without IO.", incr_get_predicate_squeezed_success),
    (get_predicate_squeezed_needs_io, "Number of Squeezed-Liquid predicate paths that required IO.", incr_get_predicate_squeezed_needs_io),
    (get_predicate_squeezed_unsupported, "Number of Squeezed-Liquid predicate paths that were unsupported and fell back.", incr_get_predicate_squeezed_unsupported),
    (get_squeezed_success, "Number of Squeezed-Liquid full evaluations finished without IO.", incr_get_squeezed_success),
    (get_squeezed_needs_io, "Number of Squeezed-Liquid full paths that required IO.", incr_get_squeezed_needs_io),
    (try_read_liquid_calls, "Number of `try_read_liquid` calls issued via `CachedData`.", incr_try_read_liquid),
    (hit_date32_expression_calls, "Number of `hit_date32_expression` calls.", incr_hit_date32_expression),
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
