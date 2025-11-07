use std::{fmt::Display, str::FromStr};

use serde::Serialize;

/// Mode in which Disk IO is done (direct IO or page cache)
#[derive(Debug, Clone, Copy, PartialEq, Default, Serialize)]
pub enum IoMode {
    /// Uses io_uring and bypass the page cache (uses direct IO), only available on Linux
    #[serde(rename = "uring-direct")]
    UringDirect,

    /// Uses io_uring and uses OS's page cache, only available on Linux
    #[serde(rename = "uring")]
    Uring,

    /// Uses multiple async io_uring instances leased per future, only available on Linux
    #[serde(rename = "uring-multi-async")]
    #[cfg_attr(target_os = "linux", default)]
    UringMultiAsync,

    /// Uses io_uring with a single shared ring on the runtime thread, only available on Linux
    #[serde(rename = "uring-shared")]
    UringShared,

    /// Uses io_uring on the calling thread and blocks until completion.
    #[serde(rename = "uring-blocking")]
    UringBlocking,

    /// Uses rust's std::fs::File, this is blocking IO.
    /// On Linux, this is essentially `pread/pwrite`
    /// This is the default on non-Linux platforms.
    #[cfg_attr(not(target_os = "linux"), default)]
    #[serde(rename = "std-blocking")]
    StdBlocking,

    /// Uses tokio's async IO, this is non-blocking IO, but quite slow: <https://github.com/tokio-rs/tokio/issues/3664>
    #[serde(rename = "tokio")]
    TokioIO,

    /// Use rust's std::fs::File, but will try to `spawn_blocking`, just like `object_store` does:
    /// <https://github.com/apache/arrow-rs-object-store/blob/28b2fc563feb44bb3d15718cf58036772334a704/src/local.rs#L440-L448>
    #[serde(rename = "std-spawn-blocking")]
    StdSpawnBlocking,
}

impl Display for IoMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}",
            match self {
                IoMode::Uring => "uring",
                IoMode::UringDirect => "uring-direct",
                IoMode::UringMultiAsync => "uring-multi-async",
                IoMode::UringShared => "uring-shared",
                IoMode::UringBlocking => "uring-blocking",
                IoMode::StdBlocking => "std-blocking",
                IoMode::TokioIO => "tokio",
                IoMode::StdSpawnBlocking => "std-spawn-blocking",
            }
        )
    }
}

impl FromStr for IoMode {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Ok(match s {
            "uring-direct" => IoMode::UringDirect,
            "uring" => IoMode::Uring,
            "uring-multi-async" => IoMode::UringMultiAsync,
            "uring-shared" => IoMode::UringShared,
            "uring-blocking" => IoMode::UringBlocking,
            "std-blocking" => IoMode::StdBlocking,
            "tokio" => IoMode::TokioIO,
            "std-spawn-blocking" => IoMode::StdSpawnBlocking,
            _ => return Err(format!("Invalid IO mode: {s}")),
        })
    }
}
