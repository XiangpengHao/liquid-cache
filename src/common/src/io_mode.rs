use std::{fmt::Display, str::FromStr};

use serde::Serialize;

/// Mode in which Disk IO is done (buffered or direct)
#[derive(Debug, Clone, PartialEq, Default, Serialize)]
pub enum IoMode {
    /// Bypasses the page cache, uses direct IO
    Direct,
    /// Uses the page cache
    #[default]
    Buffered,
}

impl Display for IoMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}",
            match self {
                IoMode::Buffered => "buffered",
                IoMode::Direct => "direct",
            }
        )
    }
}

impl FromStr for IoMode {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Ok(match s {
            "direct" => IoMode::Direct,
            "buffered" => IoMode::Buffered,
            _ => return Err(format!("Invalid IO mode: {s}")),
        })
    }
}
