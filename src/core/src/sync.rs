#[cfg(all(feature = "shuttle", test))]
#[allow(unused_imports)]
pub use shuttle::{sync::*, thread};
#[cfg(not(all(feature = "shuttle", test)))]
#[allow(unused_imports)]
pub use std::{sync::*, thread};
