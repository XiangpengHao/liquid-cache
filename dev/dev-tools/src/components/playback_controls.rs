use crate::trace::CacheSimulator;
use dioxus::prelude::*;

#[component]
pub fn PlaybackControls(simulator: Signal<CacheSimulator>) -> Element {
    let sim = simulator.read();
    let current_idx = sim.current_index();
    let total_events = sim.total_events();
    let progress_percent = if total_events > 0 {
        (current_idx as f64 / total_events as f64 * 100.0) as u32
    } else {
        0
    };

    rsx! {
        div {
            class: "playback-controls px-4 py-2 bg-white border-t border-gray-200",

            // Progress bar only
            div {
                class: "progress-bar max-w-screen-2xl mx-auto",
                div {
                    class: "flex justify-between text-xs text-gray-500 mb-1.5",
                    span { "Event {current_idx}" }
                    span { "{total_events} total" }
                }
                div {
                    class: "w-full bg-gray-200 rounded-full h-1.5 overflow-hidden",
                    div {
                        class: "bg-gray-900 h-full transition-all duration-200",
                        style: "width: {progress_percent}%",
                    }
                }
            }
        }
    }
}
