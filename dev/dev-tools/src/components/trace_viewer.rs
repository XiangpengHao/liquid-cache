use crate::trace::CacheSimulator;
use dioxus::prelude::*;

#[component]
pub fn TraceViewer(simulator: Signal<CacheSimulator>) -> Element {
    let sim = simulator.read();
    let events = sim.events();
    let current_idx = sim.current_index();

    rsx! {
        div {
            class: "trace-viewer h-full flex flex-col",

            // Header
            div {
                class: "p-4 border-b border-base-300",
                h2 {
                    class: "text-lg font-bold",
                    "Event Trace"
                }
                div {
                    class: "text-sm opacity-60",
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

                        let border_class = if is_current {
                            "border-l-4 border-primary bg-base-200"
                        } else if is_executed {
                            "border-l-2 border-base-300"
                        } else {
                            "border-l-2 border-base-200 opacity-50"
                        };

                        rsx! {
                            div {
                                key: "{idx}",
                                class: "event-item p-2 border-b border-base-200 {border_class} hover:bg-base-200 transition-colors cursor-pointer",
                                onclick: move |_| {
                                    let target_idx = idx + 1;
                                    simulator.write().jump_to(target_idx);
                                },

                                div {
                                    class: "flex items-start gap-3",

                                    // Event index
                                    div {
                                        class: "text-xs font-mono opacity-40 w-10 flex-shrink-0 text-right",
                                        "{idx + 1}"
                                    }

                                    // Event type with fixed width for alignment
                                    div {
                                        class: "text-xs font-medium opacity-70 w-32 flex-shrink-0",
                                        "{event.event_type()}"
                                    }

                                    // Event description with monospace font
                                    div {
                                        class: "text-xs font-mono flex-1",
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
