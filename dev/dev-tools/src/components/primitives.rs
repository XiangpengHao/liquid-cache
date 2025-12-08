use dioxus::prelude::*;

/// Simple card container with consistent padding/borders.
#[component]
pub fn Card(
    #[props(default = "bg-white border border-gray-200 rounded-md p-3".to_string())] class: String,
    children: Element,
) -> Element {
    rsx! {
        div { class: "{class}", {children} }
    }
}

/// Badge with small tonal variants.
#[component]
pub fn Badge(
    label: String,
    #[props(default = "neutral")] tone: &'static str,
    #[props(default = "".to_string())] class: String,
) -> Element {
    let tone_class = match tone {
        "info" => "bg-blue-100 text-blue-700 border border-blue-300",
        "success" => "bg-green-100 text-green-700 border border-green-300",
        "warn" => "bg-amber-100 text-amber-700 border border-amber-300",
        "danger" => "bg-red-100 text-red-700 border border-red-300",
        _ => "bg-gray-100 text-gray-700 border border-gray-300",
    };

    rsx! {
        span { class: "text-xs px-2 py-0.5 rounded {tone_class} {class}", {label} }
    }
}

/// Compact stat with optional delta badge.
#[component]
pub fn Stat(
    label: String,
    value: String,
    #[props(optional)] delta: Option<String>,
    #[props(default = "neutral")] tone: &'static str,
    #[props(default = "".to_string())] class: String,
) -> Element {
    let delta_tone = match tone {
        "warn" => "bg-amber-100 text-amber-700 border border-amber-300",
        "success" => "bg-green-100 text-green-700 border border-green-300",
        _ => "bg-gray-100 text-gray-700 border border-gray-300",
    };

    rsx! {
        div { class: "text-center {class}",
            div { class: "text-lg font-semibold text-gray-900", "{value}" }
            div { class: "text-xs text-gray-500", "{label}" }
            if let Some(delta) = delta {
                span { class: "inline-block mt-1 text-[10px] px-1.5 py-0.5 rounded {delta_tone}", "{delta}" }
            }
        }
    }
}

/// List container for vertical stacking.
#[component]
pub fn List(#[props(default = "".to_string())] class: String, children: Element) -> Element {
    rsx! {
        div { class: "flex flex-col gap-2 {class}", {children} }
    }
}

/// List item with optional meta and body content.
#[derive(Props, Clone, PartialEq)]
pub struct ListItemProps {
    pub title: String,
    #[props(optional)]
    pub meta: Option<String>,
    #[props(default = "".to_string())]
    pub class: String,
    pub children: Element,
}

#[component]
pub fn ListItem(props: ListItemProps) -> Element {
    rsx! {
        div { class: "p-2.5 bg-white border border-gray-200 rounded-md {props.class}",
            div { class: "flex items-center gap-2 mb-1 flex-wrap",
                span { class: "text-sm font-mono text-gray-800", "{props.title}" }
                if let Some(meta) = props.meta.clone() {
                    span { class: "text-xs text-gray-500", "{meta}" }
                }
            }
            {props.children}
        }
    }
}
