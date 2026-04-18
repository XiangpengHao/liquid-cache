use crate::components::{Badge, List, Stat};
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

/// Get the label for an operation
fn get_operation_label(op: &EntryOperation) -> String {
    match op {
        EntryOperation::Reading { expr: Some(e) } => format!("read [{}]", e),
        EntryOperation::Reading { expr: None } => "read".to_string(),
        EntryOperation::ReadingSqueezed { expr } => format!("read-sq [{}]", expr),
        EntryOperation::Hydrating { to } => format!("hydrate → {}", to.display_name()),
        EntryOperation::IoRead => "io_read".to_string(),
        EntryOperation::IoWrite => "io_write".to_string(),
        EntryOperation::EvalPredicate => "eval_predicate".to_string(),
        EntryOperation::DecompressSqueezed {
            decompressed,
            total,
        } => format!("decompress_squeezed {} / {}", decompressed, total),
    }
}

/// Format entry ID as u16_u16_u16_u16
fn format_entry_id_parts(entry_id: u64) -> String {
    let part0 = (entry_id & 0xFFFF) as u16;
    let part1 = ((entry_id >> 16) & 0xFFFF) as u16;
    let part2 = ((entry_id >> 32) & 0xFFFF) as u16;
    let part3 = ((entry_id >> 48) & 0xFFFF) as u16;
    format!("{}_{}_{}_{}", part3, part2, part1, part0)
}

#[component]
pub fn CacheStateView(simulator: Signal<CacheSimulator>) -> Element {
    let sim = simulator.read();
    let state = sim.current_state();
    let entries = state.get_entries();

    // Create a unified sorted list of all entry IDs (both actual and failed)
    let mut all_entry_ids: Vec<u64> = entries.iter().map(|e| e.entry_id).collect();
    for (entry_id, _) in state.failed_inserts.iter() {
        if !all_entry_ids.contains(entry_id) {
            all_entry_ids.push(*entry_id);
        }
    }
    all_entry_ids.sort();

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
            class: "cache-state-view h-full flex flex-col",

            // Header with I/O stats on the same row
            div {
                class: "p-4 border-b border-base-300",
                div {
                    class: "flex items-start gap-4",

                    // Left half: Title and count
                    div {
                        class: "flex-1",
                        h2 {
                            class: "text-lg font-bold",
                            "Cache State"
                        }
                        div {
                            class: "text-sm opacity-60",
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
                                span { class: "opacity-60", "Read:" }
                                span { class: "font-semibold", "{state.io_stats.read_requests}" }
                                span { class: "opacity-40", "ops" }
                                if let Some(delta) = &read_ops_delta {
                                    span { class: "badge badge-warning badge-xs", "{delta}" }
                                }
                            }

                            div { class: "flex items-center gap-1.5",
                                span { class: "font-semibold", "{format_bytes(state.io_stats.bytes_read)}" }
                                if let Some(delta) = &read_bytes_delta {
                                    span { class: "badge badge-warning badge-xs", "{delta}" }
                                }
                            }

                            // Write stats
                            div { class: "flex items-center gap-1.5",
                                span { class: "opacity-60", "Write:" }
                                span { class: "font-semibold", "{state.io_stats.write_requests}" }
                                span { class: "opacity-40", "ops" }
                                if let Some(delta) = &write_ops_delta {
                                    span { class: "badge badge-warning badge-xs", "{delta}" }
                                }
                            }

                            div { class: "flex items-center gap-1.5",
                                span { class: "font-semibold", "{format_bytes(state.io_stats.bytes_written)}" }
                                if let Some(delta) = &write_bytes_delta {
                                    span { class: "badge badge-warning badge-xs", "{delta}" }
                                }
                            }
                        }
                    }
                }
            }

            // Statistics - Cache Entries by Type using daisyUI stats
            div {
                class: "p-4 border-b border-base-300",
                h3 {
                    class: "text-xs font-medium opacity-60 uppercase tracking-wide mb-2",
                    "Cache Entries by Type"
                }
                div {
                    class: "stats stats-horizontal shadow w-full",

                    Stat { class: "".to_string(), label: "MemoryArrow".to_string(), value: memory_arrow_count.to_string(), delta: None, tone: "neutral" }
                    Stat { class: "".to_string(), label: "MemoryLiquid".to_string(), value: memory_liquid_count.to_string(), delta: None, tone: "neutral" }
                    Stat { class: "".to_string(), label: "SqueezedLiquid".to_string(), value: memory_squeezed_count.to_string(), delta: None, tone: "neutral" }
                    Stat { class: "".to_string(), label: "DiskLiquid".to_string(), value: disk_liquid_count.to_string(), delta: None, tone: "neutral" }
                    Stat { class: "".to_string(), label: "DiskArrow".to_string(), value: disk_arrow_count.to_string(), delta: None, tone: "neutral" }
                }
            }

            // Entries list
            div {
                class: "flex-1 overflow-y-auto p-4",
                h3 { class: "text-sm font-medium mb-3", "Cache Entries" }

                if all_entry_ids.is_empty() {
                    div { class: "text-center opacity-40 py-8", "No entries in cache" }
                } else {
                    List {
                        class: "entries-container grid grid-cols-1 gap-2".to_string(),
                        children: rsx! {
                            // Show all entries in sorted order
                            for entry_id in &all_entry_ids {
                                // Find if this is an actual entry or just a failed insert
                                if let Some(entry) = entries.iter().find(|e| e.entry_id == *entry_id) {
                                    // Actual entry exists
                                    div {
                                        key: "{entry_id}",
                                        class: format!("card card-compact bg-base-100 border border-base-300 p-2 {}",
                                            if state.failed_inserts.contains_key(&entry.entry_id) {
                                                "border-2 border-warning"
                                            } else {
                                                ""
                                            }
                                        ),

                                        div { class: "flex items-center justify-between gap-2 mb-1",
                                            div { class: "flex items-center gap-2",
                                                span { class: "text-xs font-mono font-semibold", "Entry {entry.entry_id}" }
                                                span { class: "text-xs font-mono opacity-50", "({format_entry_id_parts(entry.entry_id)})" }
                                            }
                                            div { class: "flex items-center gap-1 flex-wrap min-h-5",
                                                match state.victim_status.get(&entry.entry_id) {
                                                    Some(VictimStatus::Selected) => rsx!( Badge { label: "victim".to_string(), tone: "warn", class: "".to_string() } ),
                                                    Some(VictimStatus::Squeezed) => rsx!( Badge { label: "squeezed".to_string(), tone: "neutral", class: "".to_string() } ),
                                                    None => rsx! {},
                                                }
                                                if let Some(op) = state.current_operations.get(&entry.entry_id) {
                                                    Badge { label: get_operation_label(op), tone: "info", class: "".to_string() }
                                                }
                                                if state.failed_inserts.contains_key(&entry.entry_id) {
                                                    if let Some(kind) = state.failed_inserts.get(&entry.entry_id) {
                                                        Badge { label: format!("insert failed → {}", kind.display_name()), tone: "warn", class: "".to_string() }
                                                    }
                                                }
                                            }
                                        }

                                        div {
                                            class: "text-xs flex items-center gap-1 flex-wrap",
                                            for (idx, kind) in entry.history.iter().enumerate() {
                                                span {
                                                    class: if idx == entry.history.len() - 1 { "font-semibold" } else { "opacity-60" },
                                                    "{kind.display_name()}"
                                                }
                                                if idx < entry.history.len() - 1 {
                                                    span { class: "opacity-30", "→" }
                                                }
                                            }
                                        }
                                    }
                                } else if let Some(kind) = state.failed_inserts.get(entry_id) {
                                    // Failed insert only (no actual entry yet)
                                    div {
                                        key: "{entry_id}",
                                        class: "card card-compact bg-base-100 border-2 border-dashed border-warning p-2",
                                        div { class: "flex items-center justify-between gap-2 mb-1",
                                            div { class: "flex items-center gap-2",
                                                span { class: "text-xs font-mono font-semibold", "Entry {entry_id}" }
                                                span { class: "text-xs font-mono opacity-50", "({format_entry_id_parts(*entry_id)})" }
                                            }
                                            span { class: "text-xs opacity-60", "not inserted" }
                                        }
                                        div { class: "flex items-center gap-1 flex-wrap min-h-5",
                                            if let Some(op) = state.current_operations.get(entry_id) {
                                                Badge { label: get_operation_label(op), tone: "warn", class: "".to_string() }
                                            }
                                            span { class: "text-xs opacity-70 italic", "attempted: {kind.display_name()}" }
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
