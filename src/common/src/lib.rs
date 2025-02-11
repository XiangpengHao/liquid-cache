use std::fmt::Display;
use std::str::FromStr;
pub mod rpc;

#[derive(Clone, Debug, Default, Copy, PartialEq, Eq)]
pub enum ParquetMode {
    #[default]
    Original,
    Liquid,
    Arrow,
}

impl Display for ParquetMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", match self {
            ParquetMode::Original => "original",
            ParquetMode::Liquid => "liquid",
            ParquetMode::Arrow => "arrow",
        })
    }
}

impl FromStr for ParquetMode {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Ok(match s {
            "original" => ParquetMode::Original,
            "liquid" => ParquetMode::Liquid,
            "arrow" => ParquetMode::Arrow,
            _ => {
                return Err(format!(
                    "Invalid parquet mode: {}, must be one of: original, liquid, arrow",
                    s
                ));
            }
        })
    }
}
