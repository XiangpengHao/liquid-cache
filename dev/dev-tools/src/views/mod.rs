//! The views module contains the components for all Layouts and Routes for our app. Each layout and route in our [`Route`]
//! enum will render one of these components.
//!
//!
//! The [`Home`] component will be rendered when the current route is [`Route::Home`].

mod home;
pub use home::Home;
