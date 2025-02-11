use std::{fmt::Display, str::FromStr};

use arrow_flight::sql::{Any, ProstMessageExt};

pub enum LiquidCacheActions {
    RegisterTable,
    ExecutionMetrics,
    ResetCache,
}

impl FromStr for LiquidCacheActions {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "RegisterTable" => Ok(LiquidCacheActions::RegisterTable),
            "ExecutionMetrics" => Ok(LiquidCacheActions::ExecutionMetrics),
            "ResetCache" => Ok(LiquidCacheActions::ResetCache),
            _ => Err(format!("Invalid action: {}", s)),
        }
    }
}

impl Display for LiquidCacheActions {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", match self {
            LiquidCacheActions::RegisterTable => "RegisterTable",
            LiquidCacheActions::ExecutionMetrics => "ExecutionMetrics",
            LiquidCacheActions::ResetCache => "ResetCache",
        })
    }
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct RegisterTableRequest {
    #[prost(string, tag = "1")]
    pub url: ::prost::alloc::string::String,

    #[prost(string, tag = "2")]
    pub table_name: ::prost::alloc::string::String,

    #[prost(string, tag = "3")]
    pub table_provider: ::prost::alloc::string::String,
}

impl ProstMessageExt for RegisterTableRequest {
    fn type_url() -> &'static str {
        "type.googleapis.com/datafusion.example.com.sql.ActionRegisterTableRequest"
    }

    fn as_any(&self) -> Any {
        Any {
            type_url: RegisterTableRequest::type_url().to_string(),
            value: ::prost::Message::encode_to_vec(self).into(),
        }
    }
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct FetchResults {
    #[prost(uint64, tag = "1")]
    pub handle: u64,

    #[prost(uint32, tag = "2")]
    pub partition: u32,
}

impl ProstMessageExt for FetchResults {
    fn type_url() -> &'static str {
        "type.googleapis.com/datafusion.example.com.sql.FetchResults"
    }

    fn as_any(&self) -> Any {
        Any {
            type_url: FetchResults::type_url().to_string(),
            value: ::prost::Message::encode_to_vec(self).into(),
        }
    }
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct ExecutionMetricsResponse {
    #[prost(uint64, tag = "1")]
    pub pushdown_eval_time: u64,
    #[prost(uint64, tag = "2")]
    pub cache_memory_usage: u64,
}

impl ProstMessageExt for ExecutionMetricsResponse {
    fn type_url() -> &'static str {
        "type.googleapis.com/datafusion.example.com.sql.ActionGetMostRecentExecutionMetricsResponse"
    }

    fn as_any(&self) -> Any {
        Any {
            type_url: Self::type_url().to_string(),
            value: ::prost::Message::encode_to_vec(self).into(),
        }
    }
}
