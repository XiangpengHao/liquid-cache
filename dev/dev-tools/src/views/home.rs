use dioxus::prelude::*;
use crate::trace::{parse_trace, CacheSimulator};
use crate::components::{TraceViewer, CacheStateView, PlaybackControls};

/// Sample trace data for demonstration
const SAMPLE_TRACE: &str = r#"
event=insert_success entry=0 kind=MemoryArrow
event=insert_success entry=1 kind=MemoryArrow
event=insert_failed entry=2 kind=MemoryArrow
event=squeeze_begin victims=[0,1]
event=squeeze_victim entry=0
event=io_write entry=0 kind=MemorySqueezedLiquid bytes=7816
event=insert_success entry=0 kind=MemorySqueezedLiquid
event=squeeze_victim entry=1
event=io_write entry=1 kind=MemorySqueezedLiquid bytes=7816
event=insert_success entry=1 kind=MemorySqueezedLiquid
event=insert_success entry=2 kind=MemoryArrow
event=read entry=0 selection=false expr=VariantGet[name:Utf8] cached=MemorySqueezedLiquid
event=read_squeezed_date entry=0 expression=VariantGet[name:Utf8]
event=read entry=0 selection=false expr=VariantGet[age:Int64] cached=MemorySqueezedLiquid
event=io_read_arrow entry=0 bytes=7816
event=hydrate entry=0 cached=DiskArrow new=MemoryArrow
event=insert_failed entry=0 kind=MemoryArrow
event=squeeze_begin victims=[2,0,1]
event=squeeze_victim entry=2
event=io_write entry=2 kind=MemorySqueezedLiquid bytes=7816
event=insert_success entry=2 kind=MemorySqueezedLiquid
event=squeeze_victim entry=0
event=insert_success entry=0 kind=DiskArrow
event=squeeze_victim entry=1
event=insert_success entry=1 kind=DiskArrow
event=insert_failed entry=0 kind=MemoryArrow
event=squeeze_begin victims=[2]
event=squeeze_victim entry=2
event=insert_success entry=2 kind=DiskArrow
event=insert_failed entry=0 kind=MemoryArrow
event=io_write entry=0 kind=DiskArrow bytes=7816
event=insert_success entry=0 kind=DiskArrow
event=read_squeezed_date entry=0 expression=VariantGet[age:Int64]
event=read entry=1 selection=false expr=VariantGet[address.zipcode:Int64] cached=DiskArrow
event=io_read_arrow entry=1 bytes=7816
event=hydrate entry=1 cached=DiskArrow new=MemoryArrow
event=insert_failed entry=1 kind=MemoryArrow
event=io_write entry=1 kind=DiskArrow bytes=7816
event=insert_success entry=1 kind=DiskArrow
"#;

/// The Home page component that will be rendered when the current route is `[Route::Home]`
#[component]
pub fn Home() -> Element {
    // Parse the trace and create a simulator
    let mut simulator = use_signal(|| {
        let events = parse_trace(SAMPLE_TRACE);
        CacheSimulator::new(events)
    });

    let mut trace_input = use_signal(|| String::new());
    let mut show_input = use_signal(|| false);

    rsx! {
        div {
            class: "home-page h-screen flex flex-col bg-white",
            
            // Header
            div {
                class: "header p-4 border-b border-gray-200 bg-white",
                div {
                    class: "max-w-screen-2xl mx-auto flex justify-between items-center",
                    h1 {
                        class: "text-2xl font-semibold text-gray-900",
                        "Cache Trace Visualizer"
                    }
                    button {
                        class: "px-4 py-2 text-sm font-medium bg-white border border-gray-300 rounded-md hover:bg-gray-50 transition-colors",
                        onclick: move |_| {
                            show_input.set(!show_input());
                        },
                        if show_input() {
                            "Hide Input"
                        } else {
                            "Load Trace"
                        }
                    }
                }
            }

            // Trace input area (collapsible)
            if show_input() {
                div {
                    class: "trace-input-area p-4 bg-gray-50 border-b border-gray-200",
                    div {
                        class: "max-w-screen-2xl mx-auto",
                        label {
                            class: "block text-sm font-medium text-gray-700 mb-2",
                            "Paste your trace here (logfmt format):"
                        }
                        textarea {
                            class: "w-full h-32 p-3 bg-white border border-gray-300 rounded-md text-sm font-mono resize-y focus:outline-none focus:ring-2 focus:ring-gray-400 focus:border-transparent",
                            placeholder: "event=insert_success entry=0 kind=MemoryArrow\nevent=insert_success entry=1 kind=DiskArrow\n...",
                            value: "{trace_input}",
                            oninput: move |evt| {
                                trace_input.set(evt.value());
                            }
                        }
                        div {
                            class: "mt-2 flex gap-2",
                            button {
                                class: "px-4 py-2 text-sm font-medium text-white bg-gray-900 rounded-md hover:bg-gray-800 transition-colors",
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
                                class: "px-4 py-2 text-sm font-medium text-gray-700 bg-white border border-gray-300 rounded-md hover:bg-gray-50 transition-colors",
                                onclick: move |_| {
                                    let events = parse_trace(SAMPLE_TRACE);
                                    simulator.set(CacheSimulator::new(events));
                                    show_input.set(false);
                                },
                                "Load Sample"
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
                    class: "left-panel w-1/2 border-r border-gray-200 flex flex-col bg-white",
                    TraceViewer { simulator }
                }
                
                // Right panel - Cache state
                div {
                    class: "right-panel w-1/2 flex flex-col bg-white",
                    CacheStateView { simulator }
                }
            }

            // Bottom panel - Playback controls
            PlaybackControls { simulator }
        }
    }
}
