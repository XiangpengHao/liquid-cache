use crate::trace::simulator::EntryOperation;
use crate::trace::{CacheKind, CacheSimulator, VictimStatus};
use dioxus::prelude::*;

/// Format bytes into a human-readable string
fn format_bytes(bytes: u64) -> String {
    if bytes < 1024 {
        format!("{} B", bytes)
    } else if bytes < 1024 * 1024 {
        format!("{:.1} KB", bytes as f64 / 1024.0)
    } else if bytes < 1024 * 1024 * 1024 {
        format!("{:.1} MB", bytes as f64 / (1024.0 * 1024.0))
    } else {
        format!("{:.2} GB", bytes as f64 / (1024.0 * 1024.0 * 1024.0))
    }
}

/// Get the CSS class for an operation badge
fn get_operation_badge_class(op: &EntryOperation) -> &'static str {
    match op {
        EntryOperation::Reading { .. } => "bg-blue-100 text-blue-700 border border-blue-300",
        EntryOperation::ReadingSqueezed { .. } => {
            "bg-cyan-100 text-cyan-700 border border-cyan-300"
        }
        EntryOperation::Hydrating { .. } => {
            "bg-purple-100 text-purple-700 border border-purple-300"
        }
        EntryOperation::IoRead | EntryOperation::IoWrite => {
            "bg-amber-100 text-amber-700 border border-amber-300"
        }
    }
}

/// Get the label for an operation
fn get_operation_label(op: &EntryOperation) -> String {
    match op {
        EntryOperation::Reading { expr: Some(e) } => format!("read [{}]", e),
        EntryOperation::Reading { expr: None } => "read".to_string(),
        EntryOperation::ReadingSqueezed { expr } => format!("read-sq [{}]", expr),
        EntryOperation::Hydrating { to } => format!("hydrate → {}", to.display_name()),
        EntryOperation::IoRead => "io_read".to_string(),
        EntryOperation::IoWrite => "io_write".to_string(),
    }
}

#[component]
pub fn CacheStateView(simulator: Signal<CacheSimulator>) -> Element {
    let sim = simulator.read();
    let state = sim.current_state();
    let entries = state.get_entries();

    // Count entries by kind
    let memory_arrow_count = state.count_by_kind(&CacheKind::MemoryArrow);
    let memory_liquid_count = state.count_by_kind(&CacheKind::MemoryLiquid);
    let memory_squeezed_count = state.count_by_kind(&CacheKind::MemorySqueezedLiquid);
    let disk_liquid_count = state.count_by_kind(&CacheKind::DiskLiquid);
    let disk_arrow_count = state.count_by_kind(&CacheKind::DiskArrow);

    rsx! {
        div {
            class: "cache-state-view h-full flex flex-col bg-white",

            // Header with I/O stats on the same row
            div {
                class: "cache-header p-4 border-b border-gray-200",
                div {
                    class: "flex items-start gap-4",
                    
                    // Left half: Title and count
                    div {
                        class: "flex-1",
                        h2 {
                            class: "text-lg font-semibold text-gray-900",
                            "Cache State"
                        }
                        div {
                            class: "text-sm text-gray-500",
                            "Total: {state.total_entries()}"
                        }
                    }
                    
                    // Right half: I/O stats in compact label format
                    div {
                        class: "flex-1",
                        div {
                            class: "grid grid-cols-2 gap-x-3 gap-y-1 text-sm",
                            
                            // Read stats
                            div {
                                class: "flex items-center gap-1.5",
                                span { class: "text-gray-500", "Read:" }
                                span { class: "font-semibold text-gray-900", "{state.io_stats.read_requests}" }
                                span { class: "text-gray-400", "ops" }
                                if state.io_stats_delta.read_requests_delta > 0 {
                                    span {
                                        class: "text-[10px] px-1 py-0.5 bg-amber-100 text-amber-700 rounded font-medium",
                                        "+{state.io_stats_delta.read_requests_delta}"
                                    }
                                }
                            }
                            
                            div {
                                class: "flex items-center gap-1.5",
                                span { class: "font-semibold text-gray-900", "{format_bytes(state.io_stats.bytes_read)}" }
                                if state.io_stats_delta.bytes_read_delta > 0 {
                                    span {
                                        class: "text-[10px] px-1 py-0.5 bg-amber-100 text-amber-700 rounded font-medium",
                                        "+{format_bytes(state.io_stats_delta.bytes_read_delta)}"
                                    }
                                }
                            }
                            
                            // Write stats
                            div {
                                class: "flex items-center gap-1.5",
                                span { class: "text-gray-500", "Write:" }
                                span { class: "font-semibold text-gray-900", "{state.io_stats.write_requests}" }
                                span { class: "text-gray-400", "ops" }
                                if state.io_stats_delta.write_requests_delta > 0 {
                                    span {
                                        class: "text-[10px] px-1 py-0.5 bg-amber-100 text-amber-700 rounded font-medium",
                                        "+{state.io_stats_delta.write_requests_delta}"
                                    }
                                }
                            }
                            
                            div {
                                class: "flex items-center gap-1.5",
                                span { class: "font-semibold text-gray-900", "{format_bytes(state.io_stats.bytes_written)}" }
                                if state.io_stats_delta.bytes_written_delta > 0 {
                                    span {
                                        class: "text-[10px] px-1 py-0.5 bg-amber-100 text-amber-700 rounded font-medium",
                                        "+{format_bytes(state.io_stats_delta.bytes_written_delta)}"
                                    }
                                }
                            }
                        }
                    }
                }
            }

            // Statistics - Cache Entries by Type
            div {
                class: "stats p-4 border-b border-gray-200",

                div {
                    h3 {
                        class: "text-xs font-medium text-gray-500 uppercase tracking-wide mb-2",
                        "Cache Entries by Type"
                    }
                    div {
                        class: "grid grid-cols-5 gap-2 text-center",

                        div {
                            class: "p-1 bg-gray-50 rounded border border-gray-200",
                            div {
                                class: "text-lg font-semibold text-gray-900",
                                "{memory_arrow_count}"
                            }
                            div {
                                class: "text-xs text-gray-500",
                                "MemoryArrow"
                            }
                        }

                        div {
                            class: "p-1 bg-gray-50 rounded border border-gray-200",
                            div {
                                class: "text-lg font-semibold text-gray-900",
                                "{memory_liquid_count}"
                            }
                            div {
                                class: "text-xs text-gray-500",
                                "MemoryLiquid"
                            }
                        }

                        div {
                            class: "p-1 bg-gray-50 rounded border border-gray-200",
                            div {
                                class: "text-lg font-semibold text-gray-900",
                                "{memory_squeezed_count}"
                            }
                            div {
                                class: "text-xs text-gray-500",
                                "SqueezedLiquid"
                            }
                        }

                        div {
                            class: "p-1 bg-gray-50 rounded border border-gray-200",
                            div {
                                class: "text-lg font-semibold text-gray-900",
                                "{disk_liquid_count}"
                            }
                            div {
                                class: "text-xs text-gray-500",
                                "DiskLiquid"
                            }
                        }

                        div {
                            class: "p-1 bg-gray-50 rounded border border-gray-200",
                            div {
                                class: "text-lg font-semibold text-gray-900",
                                "{disk_arrow_count}"
                            }
                            div {
                                class: "text-xs text-gray-500",
                                "DiskArrow"
                            }
                        }
                    }
                }
            }

            // Entries list
            div {
                class: "entries flex-1 overflow-y-auto p-4",
                h3 {
                    class: "text-sm font-medium text-gray-700 mb-3",
                    "Cache Entries"
                }

                if entries.is_empty() && state.failed_inserts.is_empty() {
                    div {
                        class: "text-center text-gray-400 py-8",
                        "No entries in cache"
                    }
                } else {
                    div {
                        class: "grid grid-cols-1 gap-2",

                        // Show failed inserts as ghost entries (only if entry doesn't already exist)
                        for (entry_id, kind) in state.failed_inserts.iter() {
                            {
                                let entry_exists = entries.iter().any(|e| e.entry_id == *entry_id);

                                // Only render ghost entry if it doesn't exist in actual entries
                                if !entry_exists {
                                    let current_op = state.current_operations.get(entry_id);

                                    rsx! {
                                        div {
                                            key: "failed-{entry_id}",
                                            class: "entry-item p-2.5 border-2 border-dashed border-amber-300 rounded-md",

                                            // Entry header with ID and badges
                                            div {
                                                class: "flex items-center gap-2 mb-1.5 flex-wrap",
                                                span {
                                                    class: "text-sm font-mono text-amber-700",
                                                    "Entry {entry_id}"
                                                }
                                                span {
                                                    class: "text-xs px-2 py-0.5 bg-amber-100 text-amber-700 border border-amber-400 rounded",
                                                    "not inserted"
                                                }
                                                if let Some(op) = current_op {
                                                    span {
                                                        class: "text-xs px-2 py-0.5 rounded {get_operation_badge_class(op)}",
                                                        "{get_operation_label(op)}"
                                                    }
                                                }
                                            }

                                            // Show attempted kind
                                            div {
                                                class: "text-xs flex items-center gap-1 flex-wrap text-amber-600",
                                                span {
                                                    class: "text-amber-600 italic",
                                                    "attempted: {kind.display_name()}"
                                                }
                                            }
                                        }
                                    }
                                } else {
                                    rsx! {}
                                }
                            }
                        }

                        // Show actual entries
                        for entry in entries {
                            {
                                let victim_status = state.victim_status.get(&entry.entry_id);
                                let current_op = state.current_operations.get(&entry.entry_id);
                                let has_failed_insert = state.failed_inserts.contains_key(&entry.entry_id);

                                // Choose border color based on whether there's a failed insert attempt
                                let border_class = if has_failed_insert {
                                    "border-2 border-amber-400"
                                } else {
                                    "border border-gray-200"
                                };

                                rsx! {
                                    div {
                                        key: "{entry.entry_id}",
                                        class: "entry-item p-2.5 bg-white {border_class} rounded-md",

                                        // Entry header with ID and badges
                                        div {
                                            class: "flex items-center gap-2 mb-1.5 flex-wrap",
                                            span {
                                                class: "text-sm font-mono text-gray-700",
                                                "Entry {entry.entry_id}"
                                            }
                                            if let Some(status) = victim_status {
                                                span {
                                                    class: if status == &VictimStatus::Selected {
                                                        "text-xs px-2 py-0.5 bg-amber-100 text-amber-700 border border-amber-300 rounded"
                                                    } else {
                                                        "text-xs px-2 py-0.5 bg-gray-200 text-gray-600 border border-gray-400 rounded"
                                                    },
                                                    if status == &VictimStatus::Selected {
                                                        "victim"
                                                    } else {
                                                        "squeezed"
                                                    }
                                                }
                                            }
                                            if let Some(op) = current_op {
                                                span {
                                                    class: "text-xs px-2 py-0.5 rounded {get_operation_badge_class(op)}",
                                                    "{get_operation_label(op)}"
                                                }
                                            }
                                            if has_failed_insert {
                                                if let Some(kind) = state.failed_inserts.get(&entry.entry_id) {
                                                    span {
                                                        class: "text-xs px-2 py-0.5 bg-amber-100 text-amber-700 border border-amber-300 rounded",
                                                        "insert failed → {kind.display_name()}"
                                                    }
                                                }
                                            }
                                        }

                                        // Always show history
                                        div {
                                            class: "text-xs flex items-center gap-1 flex-wrap",
                                            for (idx, kind) in entry.history.iter().enumerate() {
                                                {
                                                    let is_current = idx == entry.history.len() - 1;
                                                    rsx! {
                                                        span {
                                                            key: "{idx}",
                                                            class: if is_current {
                                                                "text-gray-900 font-semibold"
                                                            } else {
                                                                "text-gray-500"
                                                            },
                                                            "{kind.display_name()}"
                                                        }
                                                        if idx < entry.history.len() - 1 {
                                                            span { class: "text-gray-300", "→" }
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}
