use dioxus::prelude::*;
use crate::trace::{TraceEvent, CacheSimulator};

#[component]
pub fn TraceViewer(simulator: Signal<CacheSimulator>) -> Element {
    let sim = simulator.read();
    let events = sim.events();
    let current_idx = sim.current_index();

    rsx! {
        div {
            class: "trace-viewer h-full flex flex-col bg-white",
            
            // Header
            div {
                class: "trace-header p-4 border-b border-gray-200",
                h2 {
                    class: "text-lg font-semibold text-gray-900",
                    "Event Trace"
                }
                div {
                    class: "text-sm text-gray-500",
                    "Event {current_idx} of {events.len()}"
                }
            }

            // Event list
            div {
                class: "trace-events flex-1 overflow-y-auto",
                for (idx, event) in events.iter().enumerate() {
                    {
                        // Only highlight the last executed event (current state)
                        let is_current = idx + 1 == current_idx;
                        let is_executed = idx < current_idx;
                        
                        let bg_class = if is_current {
                            "bg-gray-100 border-gray-900"
                        } else if is_executed {
                            "bg-white border-gray-300"
                        } else {
                            "bg-white border-gray-200"
                        };

                        rsx! {
                            div {
                                key: "{idx}",
                                class: "event-item p-3 border-l-2 border-b border-gray-100 {bg_class} hover:bg-gray-50 transition-colors cursor-pointer",
                                onclick: move |_| {
                                    let target_idx = idx + 1;
                                    simulator.write().jump_to(target_idx);
                                },
                                
                                div {
                                    class: "flex items-start gap-3",
                                    
                                    // Event index
                                    div {
                                        class: "text-xs font-mono text-gray-400 w-12 flex-shrink-0",
                                        "{idx + 1}"
                                    }
                                    
                                    // Event type badge
                                    div {
                                        class: "event-type-badge px-2 py-0.5 rounded text-xs font-medium flex-shrink-0 {get_event_badge_class(event)}",
                                        "{event.event_type()}"
                                    }
                                    
                                    // Event description
                                    div {
                                        class: "text-sm text-gray-700 flex-1",
                                        "{event.description()}"
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

fn get_event_badge_class(event: &TraceEvent) -> &'static str {
    match event {
        TraceEvent::InsertSuccess { .. } => "bg-gray-100 text-gray-700 border border-gray-300",
        TraceEvent::InsertFailed { .. } => "bg-red-100 text-red-700 border border-red-300",
        TraceEvent::SqueezeBegin { .. } => "bg-gray-900 text-white",
        TraceEvent::SqueezeVictim { .. } => "bg-gray-800 text-white",
        TraceEvent::IoWrite { .. } => "bg-gray-100 text-gray-600 border border-gray-300",
        TraceEvent::IoReadArrow { .. } => "bg-gray-100 text-gray-600 border border-gray-300",
        TraceEvent::Hydrate { .. } => "bg-gray-700 text-white",
        TraceEvent::Read { .. } => "bg-gray-100 text-gray-600 border border-gray-300",
        TraceEvent::ReadSqueezedDate { .. } => "bg-gray-100 text-gray-600 border border-gray-300",
        TraceEvent::Unknown { .. } => "bg-gray-100 text-gray-500 border border-gray-200",
    }
}
