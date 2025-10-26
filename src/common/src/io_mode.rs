use std::{fmt::Display, str::FromStr};

use serde::Serialize;

/// Mode in which Disk IO is done (direct IO or page cache)
#[derive(Debug, Clone, PartialEq, Default, Serialize)]
pub enum IoMode {
    /// Bypasses the page cache, uses direct IO
    DirectIO,
    /// Uses OS's page cache
    #[default]
    PageCache,
}

impl Display for IoMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}",
            match self {
                IoMode::PageCache => "page-cache",
                IoMode::DirectIO => "direct",
            }
        )
    }
}

impl FromStr for IoMode {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Ok(match s {
            "direct" => IoMode::DirectIO,
            "page-cache" => IoMode::PageCache,
            _ => return Err(format!("Invalid IO mode: {s}")),
        })
    }
}
