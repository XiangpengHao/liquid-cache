mod loader;
mod parser;
pub mod simulator;

pub use loader::{list_snapshots, load_snapshot};
pub use parser::{CacheKind, TraceEvent, parse_trace};
pub use simulator::{CacheSimulator, VictimStatus};
