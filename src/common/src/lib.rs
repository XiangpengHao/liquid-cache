use std::fmt::Display;
use std::str::FromStr;
pub mod rpc;
pub mod utils;

/// Specify how LiquidCache should cache the data
#[derive(Clone, Debug, Default, Copy, PartialEq, Eq)]
pub enum CacheMode {
    /// Cache parquet files
    Parquet,
    /// Cache LiquidArray, transcode happens in background
    #[default]
    Liquid,
    /// Transcode blocks query execution
    LiquidEagerTranscode,
    /// Cache Arrow, transcode happens in background
    Arrow,
}

impl Display for CacheMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}",
            match self {
                CacheMode::Parquet => "parquet",
                CacheMode::Liquid => "liquid",
                CacheMode::LiquidEagerTranscode => "liquid_eager_transcode",
                CacheMode::Arrow => "arrow",
            }
        )
    }
}

impl FromStr for CacheMode {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Ok(match s {
            "parquet" => CacheMode::Parquet,
            "liquid" => CacheMode::Liquid,
            "liquid_eager_transcode" => CacheMode::LiquidEagerTranscode,
            "arrow" => CacheMode::Arrow,
            _ => {
                return Err(format!(
                    "Invalid cache mode: {}, must be one of: parquet, liquid, liquid_eager_transcode, arrow",
                    s
                ));
            }
        })
    }
}
