use dioxus::prelude::*;

/// Simple card container using daisyUI card component.
#[component]
pub fn Card(#[props(default = "".to_string())] class: String, children: Element) -> Element {
    rsx! {
        div { class: "card bg-base-100 {class}", {children} }
    }
}

/// Badge with tonal variants using daisyUI badge component.
#[component]
pub fn Badge(
    label: String,
    #[props(default = "neutral")] tone: &'static str,
    #[props(default = "".to_string())] class: String,
) -> Element {
    let tone_class = match tone {
        "info" => "badge-info",
        "success" => "badge-success",
        "warn" => "badge-warning",
        "danger" => "badge-error",
        _ => "badge-ghost",
    };

    rsx! {
        span { class: "badge badge-sm {tone_class} {class}", {label} }
    }
}

/// Compact stat with optional delta badge using daisyUI stat component.
#[component]
pub fn Stat(
    label: String,
    value: String,
    #[props(optional)] delta: Option<String>,
    #[props(default = "neutral")] tone: &'static str,
    #[props(default = "".to_string())] class: String,
) -> Element {
    let delta_tone = match tone {
        "warn" => "badge-warning",
        "success" => "badge-success",
        _ => "badge-ghost",
    };

    rsx! {
        div { class: "stat {class}",
            div { class: "stat-value text-lg", "{value}" }
            div { class: "stat-title text-xs", "{label}" }
            if let Some(delta) = delta {
                div { class: "stat-desc",
                    span { class: "badge badge-xs {delta_tone}", "{delta}" }
                }
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

/// List item with optional meta and body content using daisyUI card.
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
        div { class: "card card-compact bg-base-100 border border-base-300 {props.class}",
            div { class: "card-body",
                div { class: "flex items-center gap-2 mb-1 flex-wrap",
                    span { class: "text-sm font-mono font-semibold", "{props.title}" }
                    if let Some(meta) = props.meta.clone() {
                        span { class: "text-xs opacity-60", "{meta}" }
                    }
                }
                {props.children}
            }
        }
    }
}
