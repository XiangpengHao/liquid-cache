use crate::components::{CacheStateView, PlaybackControls, TraceViewer};
use crate::trace::{CacheSimulator, list_snapshots, load_snapshot, parse_trace};
use dioxus::prelude::*;

/// The Home page component that will be rendered when the current route is `[Route::Home]`
#[component]
pub fn Home() -> Element {
    // State for snapshot files
    let mut available_snapshots = use_signal(Vec::<String>::new);
    let mut selected_snapshot = use_signal(String::new);
    let mut is_loading = use_signal(|| false);
    let mut error_message = use_signal(|| Option::<String>::None);

    // Parse the trace and create a simulator
    let mut simulator = use_signal(|| CacheSimulator::new(vec![]));

    let mut trace_input = use_signal(String::new);
    let mut show_input = use_signal(|| false);

    // Load available snapshots on mount
    use_effect(move || {
        spawn(async move {
            match list_snapshots().await {
                Ok(files) => {
                    if !files.is_empty() {
                        available_snapshots.set(files.clone());
                        let first_file = files[0].clone();
                        selected_snapshot.set(first_file.clone());

                        // Auto-load the first snapshot
                        if let Ok(content) = load_snapshot(first_file).await {
                            let events = parse_trace(&content);
                            simulator.set(CacheSimulator::new(events));
                        }
                    }
                }
                Err(e) => {
                    error_message.set(Some(format!("Failed to list snapshots: {}", e)));
                }
            }
        });
    });

    rsx! {
        div {
            class: "home-page h-screen flex flex-col bg-base-100",
            tabindex: 0,
            onkeydown: move |event| {
                let key = event.key();
                match key {
                    Key::ArrowDown => {
                        let current = simulator.read().current_index();
                        let total = simulator.read().total_events();
                        if current < total {
                            simulator.write().jump_to(current + 1);
                        }
                    }
                    Key::ArrowUp => {
                        let current = simulator.read().current_index();
                        if current > 0 {
                            simulator.write().jump_to(current - 1);
                        }
                    }
                    _ => {}
                }
            },

            // Header with navbar
            div {
                class: "navbar bg-base-100 border-b border-base-300 px-4",
                div {
                    class: "max-w-screen-2xl mx-auto w-full flex justify-between items-center gap-4",
                    h1 {
                        class: "text-2xl font-bold",
                        "LiquidCache Trace Visualizer"
                    }

                    // Snapshot selector
                    div {
                        class: "flex items-center gap-2 flex-1 max-w-md",
                        label {
                            class: "label-text font-medium whitespace-nowrap",
                            "Snapshot:"
                        }
                        select {
                            class: "select select-bordered select-sm flex-1",
                            disabled: is_loading(),
                            value: "{selected_snapshot}",
                            onchange: move |evt| {
                                let filename = evt.value();
                                selected_snapshot.set(filename.clone());
                                is_loading.set(true);
                                error_message.set(None);

                                spawn(async move {
                                    match load_snapshot(filename).await {
                                        Ok(content) => {
                                            let events = parse_trace(&content);
                                            simulator.set(CacheSimulator::new(events));
                                            error_message.set(None);
                                        }
                                        Err(e) => {
                                            error_message.set(Some(format!("Failed to load snapshot: {}", e)));
                                        }
                                    }
                                    is_loading.set(false);
                                });
                            },

                            if available_snapshots().is_empty() {
                                option { value: "", "Loading..." }
                            } else {
                                for snapshot in available_snapshots() {
                                    option {
                                        key: "{snapshot}",
                                        value: "{snapshot}",
                                        "{snapshot}"
                                    }
                                }
                            }
                        }

                        if is_loading() {
                            span {
                                class: "loading loading-spinner loading-sm",
                            }
                        }
                    }

                    button {
                        class: "btn btn-sm btn-outline",
                        onclick: move |_| {
                            show_input.set(!show_input());
                        },
                        if show_input() {
                            "Hide Input"
                        } else {
                            "Load Custom"
                        }
                    }
                }
            }

            // Error message with alert
            if let Some(error) = error_message() {
                div {
                    class: "px-4 py-2",
                    div {
                        class: "alert alert-error max-w-screen-2xl mx-auto",
                        svg {
                            class: "stroke-current shrink-0 h-6 w-6",
                            xmlns: "http://www.w3.org/2000/svg",
                            fill: "none",
                            view_box: "0 0 24 24",
                            path {
                                stroke_linecap: "round",
                                stroke_linejoin: "round",
                                stroke_width: "2",
                                d: "M10 14l2-2m0 0l2-2m-2 2l-2-2m2 2l2 2m7-2a9 9 0 11-18 0 9 9 0 0118 0z"
                            }
                        }
                        span { "{error}" }
                    }
                }
            }

            // Trace input area (collapsible)
            if show_input() {
                div {
                    class: "p-4 bg-base-200 border-b border-base-300",
                    div {
                        class: "max-w-screen-2xl mx-auto",
                        label {
                            class: "label",
                            span { class: "label-text font-medium", "Paste your trace here (logfmt format):" }
                        }
                        textarea {
                            class: "textarea textarea-bordered w-full h-32 font-mono text-sm",
                            placeholder: "event=insert_success entry=0 kind=MemoryArrow\nevent=insert_success entry=1 kind=DiskArrow\n...",
                            value: "{trace_input}",
                            oninput: move |evt| {
                                trace_input.set(evt.value());
                            }
                        }
                        div {
                            class: "mt-2 flex gap-2",
                            button {
                                class: "btn btn-primary btn-sm",
                                onclick: move |_| {
                                    let input = trace_input();
                                    if !input.is_empty() {
                                        let events = parse_trace(&input);
                                        simulator.set(CacheSimulator::new(events));
                                        show_input.set(false);
                                    }
                                },
                                "Load"
                            }
                            button {
                                class: "btn btn-ghost btn-sm",
                                onclick: move |_| {
                                    trace_input.set(String::new());
                                },
                                "Clear"
                            }
                        }
                    }
                }
            }

            // Main content area with two panels
            div {
                class: "main-content flex-1 flex overflow-hidden max-w-screen-2xl mx-auto w-full",

                // Left panel - Trace viewer
                div {
                    class: "left-panel w-1/2 border-r border-base-300 flex flex-col",
                    TraceViewer { simulator }
                }

                // Right panel - Cache state
                div {
                    class: "right-panel w-1/2 flex flex-col",
                    CacheStateView { simulator }
                }
            }

            // Bottom panel - Playback controls
            PlaybackControls { simulator }
        }
    }
}
