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
            class: "playback-controls px-4 py-2 bg-base-100 border-t border-base-300",

            // Progress bar using daisyUI
            div {
                class: "progress-bar max-w-screen-2xl mx-auto",
                div {
                    class: "flex justify-between text-xs opacity-60 mb-1.5",
                    span { "Event {current_idx}" }
                    span { "{total_events} total" }
                }
                progress {
                    class: "progress progress-primary w-full",
                    value: "{progress_percent}",
                    max: "100"
                }
            }
        }
    }
}
