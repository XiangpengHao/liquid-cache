use std::{fmt::Display, str::FromStr};

use serde::Serialize;

/// Mode in which Disk IO is done (direct IO or page cache)
#[derive(Debug, Clone, Copy, PartialEq, Default, Serialize)]
pub enum IoMode {
    /// Uses io_uring and bypass the page cache (uses direct IO), only available on Linux
    #[serde(rename = "uring-direct")]
    UringDirectIO,

    /// Uses io_uring and uses OS's page cache, only available on Linux
    #[serde(rename = "uring")]
    Uring,

    /// Uses rust's std::fs::File, this is blocking IO.
    /// On Linux, this is essentially `pread/pwrite`
    /// This is the default until we optimized the performance of uring.
    #[default]
    #[serde(rename = "std-blocking")]
    StdBlockingIO,

    /// Uses tokio's async IO, this is non-blocking IO, but quite slow: https://github.com/tokio-rs/tokio/issues/3664
    #[serde(rename = "tokio")]
    TokioIO,

    /// Use rust's std::fs::File, but will try to `spawn_blocking`, just like `object_store` does:
    /// https://github.com/apache/arrow-rs-object-store/blob/28b2fc563feb44bb3d15718cf58036772334a704/src/local.rs#L440-L448
    #[serde(rename = "std-spawn-blocking")]
    StdSpawnBlockingIO,
}

impl Display for IoMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}",
            match self {
                IoMode::Uring => "uring",
                IoMode::UringDirectIO => "uring-direct",
                IoMode::StdBlockingIO => "std-blocking",
                IoMode::TokioIO => "tokio",
                IoMode::StdSpawnBlockingIO => "std-spawn-blocking",
            }
        )
    }
}

impl FromStr for IoMode {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Ok(match s {
            "uring-direct" => IoMode::UringDirectIO,
            "uring" => IoMode::Uring,
            "std-blocking" => IoMode::StdBlockingIO,
            "tokio" => IoMode::TokioIO,
            "std-spawn-blocking" => IoMode::StdSpawnBlockingIO,
            _ => return Err(format!("Invalid IO mode: {s}")),
        })
    }
}
