mod parser;
pub mod simulator;
mod loader;

pub use parser::{parse_trace, TraceEvent, CacheKind};
pub use simulator::{CacheSimulator, VictimStatus};
pub use loader::{list_snapshots, load_snapshot};
