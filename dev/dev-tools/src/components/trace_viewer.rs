use crate::trace::CacheSimulator;
use dioxus::prelude::*;
use std::collections::HashMap;
use std::rc::Rc;

#[component]
pub fn TraceViewer(simulator: Signal<CacheSimulator>) -> Element {
    let sim = simulator.read();
    let events = sim.events();
    let current_idx = sim.current_index();

    // Store mounted element references for scrolling
    let mut element_refs: Signal<HashMap<usize, Rc<MountedData>>> = use_signal(HashMap::new);
    // Store reference to the scroll container
    let mut container_ref: Signal<Option<Rc<MountedData>>> = use_signal(|| None);

    // Scroll current element into view only when it's outside the visible area
    use_effect(move || {
        let idx = simulator.read().current_index();
        if idx > 0 {
            let display_idx = idx - 1; // current_index is 1-based, display is 0-based
            let container = container_ref.read().clone();
            let element = element_refs.read().get(&display_idx).cloned();

            if let (Some(container), Some(element)) = (container, element) {
                spawn(async move {
                    // Get bounding rects to check visibility
                    if let (Ok(container_rect), Ok(element_rect)) =
                        (container.get_client_rect().await, element.get_client_rect().await)
                    {
                        let element_top = element_rect.origin.y;
                        let element_bottom = element_top + element_rect.size.height;
                        let container_top = container_rect.origin.y;
                        let container_bottom = container_top + container_rect.size.height;

                        // Only scroll if element is outside visible area
                        let is_visible =
                            element_top >= container_top && element_bottom <= container_bottom;

                        if !is_visible {
                            // scroll_to will respect the CSS scroll-margin we set on the element
                            let _ = element.scroll_to(ScrollBehavior::Smooth).await;
                        }
                    }
                });
            }
        }
    });

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
                onmounted: move |evt| {
                    container_ref.set(Some(evt.data()));
                },
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
                                // scroll-my-48 adds 192px scroll-margin (top & bottom) for ~6 items of context
                                class: "event-item p-2 border-b border-base-200 scroll-my-48 {border_class} hover:bg-base-200 transition-colors cursor-pointer",
                                onmounted: move |evt| {
                                    element_refs.write().insert(idx, evt.data());
                                },
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
