use crate::components::{Badge, Card, List, ListItem, Stat};
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

    // I/O deltas formatted
    let read_ops_delta = if state.io_stats_delta.read_requests_delta > 0 {
        Some(format!("+{}", state.io_stats_delta.read_requests_delta))
    } else {
        None
    };
    let read_bytes_delta = if state.io_stats_delta.bytes_read_delta > 0 {
        Some(format!(
            "+{}",
            format_bytes(state.io_stats_delta.bytes_read_delta)
        ))
    } else {
        None
    };
    let write_ops_delta = if state.io_stats_delta.write_requests_delta > 0 {
        Some(format!("+{}", state.io_stats_delta.write_requests_delta))
    } else {
        None
    };
    let write_bytes_delta = if state.io_stats_delta.bytes_written_delta > 0 {
        Some(format!(
            "+{}",
            format_bytes(state.io_stats_delta.bytes_written_delta)
        ))
    } else {
        None
    };

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

                    // Right half: I/O stats (compact)
                    div {
                        class: "flex-1",
                        div {
                            class: "grid grid-cols-2 gap-x-3 gap-y-1 text-sm",

                            // Read stats
                            div { class: "flex items-center gap-1.5",
                                span { class: "text-gray-500", "Read:" }
                                span { class: "font-semibold text-gray-900", "{state.io_stats.read_requests}" }
                                span { class: "text-gray-400", "ops" }
                                if let Some(delta) = &read_ops_delta {
                                    span { class: "text-[10px] px-1 py-0.5 bg-amber-100 text-amber-700 rounded font-medium", "{delta}" }
                                }
                            }

                            div { class: "flex items-center gap-1.5",
                                span { class: "font-semibold text-gray-900", "{format_bytes(state.io_stats.bytes_read)}" }
                                if let Some(delta) = &read_bytes_delta {
                                    span { class: "text-[10px] px-1 py-0.5 bg-amber-100 text-amber-700 rounded font-medium", "{delta}" }
                                }
                            }

                            // Write stats
                            div { class: "flex items-center gap-1.5",
                                span { class: "text-gray-500", "Write:" }
                                span { class: "font-semibold text-gray-900", "{state.io_stats.write_requests}" }
                                span { class: "text-gray-400", "ops" }
                                if let Some(delta) = &write_ops_delta {
                                    span { class: "text-[10px] px-1 py-0.5 bg-amber-100 text-amber-700 rounded font-medium", "{delta}" }
                                }
                            }

                            div { class: "flex items-center gap-1.5",
                                span { class: "font-semibold text-gray-900", "{format_bytes(state.io_stats.bytes_written)}" }
                                if let Some(delta) = &write_bytes_delta {
                                    span { class: "text-[10px] px-1 py-0.5 bg-amber-100 text-amber-700 rounded font-medium", "{delta}" }
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

                        Stat { class: "bg-gray-50 rounded border border-gray-200 p-1".to_string(), label: "MemoryArrow".to_string(), value: memory_arrow_count.to_string(), delta: None, tone: "neutral" }
                        Stat { class: "bg-gray-50 rounded border border-gray-200 p-1".to_string(), label: "MemoryLiquid".to_string(), value: memory_liquid_count.to_string(), delta: None, tone: "neutral" }
                        Stat { class: "bg-gray-50 rounded border border-gray-200 p-1".to_string(), label: "SqueezedLiquid".to_string(), value: memory_squeezed_count.to_string(), delta: None, tone: "neutral" }
                        Stat { class: "bg-gray-50 rounded border border-gray-200 p-1".to_string(), label: "DiskLiquid".to_string(), value: disk_liquid_count.to_string(), delta: None, tone: "neutral" }
                        Stat { class: "bg-gray-50 rounded border border-gray-200 p-1".to_string(), label: "DiskArrow".to_string(), value: disk_arrow_count.to_string(), delta: None, tone: "neutral" }
                    }
                }
            }

            // Entries list
            Card {
                class: "entries flex-1 overflow-y-auto p-4".to_string(),
                children: rsx! {
                    h3 { class: "text-sm font-medium text-gray-700 mb-3", "Cache Entries" }

                    if entries.is_empty() && state.failed_inserts.is_empty() {
                        div { class: "text-center text-gray-400 py-8", "No entries in cache" }
                    } else {
                        List {
                            class: "grid grid-cols-1 gap-2".to_string(),
                            children: rsx! {
                                // Show failed inserts as ghost entries (only if entry doesn't already exist)
                                for (entry_id, kind) in state
                                    .failed_inserts
                                    .iter()
                                    .filter(|(id, _)| !entries.iter().any(|e| e.entry_id == **id))
                                {
                                    ListItem {
                                        class: "border-2 border-dashed border-amber-300 bg-amber-50".to_string(),
                                        title: format!("Entry {}", entry_id),
                                        meta: Some("not inserted".to_string()),
                                        children: rsx! {
                                            div { class: "flex items-center gap-1.5 flex-wrap mb-1",
                                                if let Some(op) = state.current_operations.get(entry_id) {
                                                    Badge { label: get_operation_label(op), tone: "warn", class: get_operation_badge_class(op).to_string() }
                                                }
                                            }
                                            div { class: "text-xs text-amber-700 italic",
                                                "attempted: {kind.display_name()}"
                                            }
                                        }
                                    }
                                }

                                // Show actual entries
                                for entry in entries {

                                    ListItem {
                                        class: format!("entry-item bg-white {}",
                                            if state.failed_inserts.contains_key(&entry.entry_id) {
                                                "border-2 border-amber-400"
                                            } else {
                                                "border border-gray-200"
                                            }
                                        ),
                                        title: format!("Entry {}", entry.entry_id),
                                        meta: None,
                                        children: rsx! {
                                            div { class: "flex items-center gap-2 mb-1.5 flex-wrap",
                                                match state.victim_status.get(&entry.entry_id) {
                                                    Some(VictimStatus::Selected) => rsx!( Badge { label: "victim".to_string(), tone: "warn", class: "".to_string() } ),
                                                    Some(VictimStatus::Squeezed) => rsx!( Badge { label: "squeezed".to_string(), tone: "neutral", class: "".to_string() } ),
                                                    None => rsx! {},
                                                }
                                                if let Some(op) = state.current_operations.get(&entry.entry_id) {
                                                    Badge { label: get_operation_label(op), tone: "warn", class: get_operation_badge_class(op).to_string() }
                                                }
                                                if state.failed_inserts.contains_key(&entry.entry_id) {
                                                    if let Some(kind) = state.failed_inserts.get(&entry.entry_id) {
                                                        Badge { label: format!("insert failed → {}", kind.display_name()), tone: "warn", class: "".to_string() }
                                                    }
                                                }
                                            }
                                            div {
                                                class: "text-xs flex items-center gap-1 flex-wrap",
                                                for (idx, kind) in entry.history.iter().enumerate() {
                                                    span {
                                                        class: if idx == entry.history.len() - 1 { "text-gray-900 font-semibold" } else { "text-gray-500" },
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
