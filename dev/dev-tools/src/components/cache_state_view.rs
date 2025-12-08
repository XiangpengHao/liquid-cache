use dioxus::prelude::*;
use crate::trace::{CacheSimulator, CacheKind};
use crate::trace::simulator::EntryOperation;

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
        EntryOperation::Writing => "bg-amber-100 text-amber-700 border border-amber-300",
        EntryOperation::Hydrating => "bg-purple-100 text-purple-700 border border-purple-300",
        EntryOperation::ReadingSqueezed { .. } => "bg-cyan-100 text-cyan-700 border border-cyan-300",
    }
}

/// Get the label for an operation
fn get_operation_label(op: &EntryOperation) -> String {
    match op {
        EntryOperation::Reading { expr: Some(e) } => format!("read [{}]", e),
        EntryOperation::Reading { expr: None } => "reading".to_string(),
        EntryOperation::Writing => "writing".to_string(),
        EntryOperation::Hydrating => "hydrating".to_string(),
        EntryOperation::ReadingSqueezed { expr } => format!("read-sq [{}]", expr),
    }
}

#[component]
pub fn CacheStateView(simulator: Signal<CacheSimulator>) -> Element {
    let sim = simulator.read();
    let state = sim.current_state();
    let entries = state.get_entries();

    // Count entries by kind
    let memory_arrow_count = state.count_by_kind(&CacheKind::MemoryArrow);
    let memory_squeezed_count = state.count_by_kind(&CacheKind::MemorySqueezedLiquid);
    let disk_arrow_count = state.count_by_kind(&CacheKind::DiskArrow);

    rsx! {
        div {
            class: "cache-state-view h-full flex flex-col bg-white",
            
            // Header
            div {
                class: "cache-header p-4 border-b border-gray-200",
                h2 {
                    class: "text-lg font-semibold text-gray-900 mb-1",
                    "Cache State"
                }
                div {
                    class: "text-sm text-gray-500",
                    "Total Entries: {state.total_entries()}"
                }
            }

            // Statistics
            div {
                class: "stats p-4 border-b border-gray-200",
                
                // Entry type statistics - compact
                div {
                    class: "mb-4",
                    h3 {
                        class: "text-xs font-medium text-gray-500 uppercase tracking-wide mb-2",
                        "Cache Entries"
                    }
                    div {
                        class: "grid grid-cols-3 gap-2 text-center",
                        
                        div {
                            class: "p-2 bg-gray-50 rounded border border-gray-200",
                            div {
                                class: "text-lg font-semibold text-gray-900",
                                "{memory_arrow_count}"
                            }
                            div {
                                class: "text-xs text-gray-500",
                                "Memory"
                            }
                        }
                        
                        div {
                            class: "p-2 bg-gray-50 rounded border border-gray-200",
                            div {
                                class: "text-lg font-semibold text-gray-900",
                                "{memory_squeezed_count}"
                            }
                            div {
                                class: "text-xs text-gray-500",
                                "Squeezed"
                            }
                        }
                        
                        div {
                            class: "p-2 bg-gray-50 rounded border border-gray-200",
                            div {
                                class: "text-lg font-semibold text-gray-900",
                                "{disk_arrow_count}"
                            }
                            div {
                                class: "text-xs text-gray-500",
                                "Disk"
                            }
                        }
                    }
                }

                // Disk I/O statistics - single row
                div {
                    h3 {
                        class: "text-xs font-medium text-gray-500 uppercase tracking-wide mb-2",
                        "Disk I/O"
                    }
                    div {
                        class: "px-2 py-1.5 bg-gray-50 rounded border border-gray-200",
                        div {
                            class: "flex items-center justify-between text-xs gap-3",
                            
                            div {
                                class: "flex items-center gap-1",
                                span { class: "text-gray-500", "R:" }
                                span { class: "font-semibold text-gray-900", "{state.io_stats.read_requests}" }
                            }
                            
                            div {
                                class: "flex items-center gap-1",
                                span { class: "font-semibold text-gray-900", "{format_bytes(state.io_stats.bytes_read)}" }
                            }
                            
                            div {
                                class: "flex items-center gap-1",
                                span { class: "text-gray-500", "W:" }
                                span { class: "font-semibold text-gray-900", "{state.io_stats.write_requests}" }
                            }
                            
                            div {
                                class: "flex items-center gap-1",
                                span { class: "font-semibold text-gray-900", "{format_bytes(state.io_stats.bytes_written)}" }
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
                
                if entries.is_empty() {
                    div {
                        class: "text-center text-gray-400 py-8",
                        "No entries in cache"
                    }
                } else {
                    div {
                        class: "grid grid-cols-1 gap-2",
                        for entry in entries {
                            {
                                let is_victim = state.squeeze_victims.contains(&entry.entry_id);
                                let current_op = state.current_operations.get(&entry.entry_id);

                                rsx! {
                                    div {
                                        key: "{entry.entry_id}",
                                        class: "entry-item p-2.5 bg-white border border-gray-200 rounded-md",
                                        
                                        // Entry header with ID and badges
                                        div {
                                            class: "flex items-center gap-2 mb-1.5 flex-wrap",
                                            span {
                                                class: "text-sm font-mono text-gray-700",
                                                "Entry {entry.entry_id}"
                                            }
                                            if is_victim {
                                                span {
                                                    class: "text-xs px-2 py-0.5 bg-gray-900 text-white rounded",
                                                    "victim"
                                                }
                                            }
                                            if let Some(op) = current_op {
                                                span {
                                                    class: "text-xs px-2 py-0.5 rounded {get_operation_badge_class(op)}",
                                                    "{get_operation_label(op)}"
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

