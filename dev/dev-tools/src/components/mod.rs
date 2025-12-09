//! The components module contains all shared components for our app. Components are the building blocks of dioxus apps.
//! They can be used to defined common UI elements like buttons, forms, and modals. In this template, we define a Hero
//! component for fullstack apps to be used in our app.

mod trace_viewer;
pub use trace_viewer::TraceViewer;

mod cache_state_view;
pub use cache_state_view::CacheStateView;

mod playback_controls;
pub use playback_controls::PlaybackControls;

mod primitives;
pub use primitives::{Badge, List, Stat};
