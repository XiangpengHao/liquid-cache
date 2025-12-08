mod parser;
pub mod simulator;

pub use parser::{parse_trace, TraceEvent, CacheKind};
pub use simulator::CacheSimulator;
