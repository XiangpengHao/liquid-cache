//! RPC definitions for the LiquidCache service.
//! You should not use this module directly.
//! Instead, use the `liquid_cache_server` and `liquid_cache_client` crates to interact with the LiquidCache service.
use std::collections::HashMap;

use arrow_flight::{
    Action, Ticket,
    sql::{Any, ProstMessageExt},
};
use bytes::Bytes;
use prost::Message;

/// The actions that can be performed on the LiquidCache service.
pub enum LiquidCacheActions {
    /// Get the most recent execution metrics from the LiquidCache service.
    ExecutionMetrics(ExecutionMetricsRequest),
    /// Reset the cache.
    ResetCache,
    /// Register an object store with the LiquidCache service.
    RegisterObjectStore(RegisterObjectStoreRequest),
    RegisterPlan(RegisterPlanRequest),
}

impl From<LiquidCacheActions> for Action {
    fn from(action: LiquidCacheActions) -> Self {
        match action {
            LiquidCacheActions::ExecutionMetrics(request) => Action {
                r#type: "ExecutionMetrics".to_string(),
                body: request.as_any().encode_to_vec().into(),
            },
            LiquidCacheActions::ResetCache => Action {
                r#type: "ResetCache".to_string(),
                body: Bytes::new(),
            },
            LiquidCacheActions::RegisterObjectStore(request) => Action {
                r#type: "RegisterObjectStore".to_string(),
                body: request.as_any().encode_to_vec().into(),
            },
            LiquidCacheActions::RegisterPlan(request) => Action {
                r#type: "RegisterPlan".to_string(),
                body: request.as_any().encode_to_vec().into(),
            },
        }
    }
}

impl From<Action> for LiquidCacheActions {
    fn from(action: Action) -> Self {
        match action.r#type.as_str() {
            "ExecutionMetrics" => {
                let any = Any::decode(action.body).unwrap();
                let request = any.unpack::<ExecutionMetricsRequest>().unwrap().unwrap();
                LiquidCacheActions::ExecutionMetrics(request)
            }
            "ResetCache" => LiquidCacheActions::ResetCache,
            "RegisterObjectStore" => {
                let any = Any::decode(action.body).unwrap();
                let request = any.unpack::<RegisterObjectStoreRequest>().unwrap().unwrap();
                LiquidCacheActions::RegisterObjectStore(request)
            }
            "RegisterPlan" => {
                let any = Any::decode(action.body).unwrap();
                let request = any.unpack::<RegisterPlanRequest>().unwrap().unwrap();
                LiquidCacheActions::RegisterPlan(request)
            }
            _ => panic!("Invalid action: {}", action.r#type),
        }
    }
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct RegisterPlanRequest {
    #[prost(bytes, tag = "1")]
    pub plan: ::prost::alloc::vec::Vec<u8>,

    #[prost(bytes, tag = "2")]
    pub handle: Bytes,

    #[prost(string, tag = "3")]
    pub cache_mode: String,
}

impl ProstMessageExt for RegisterPlanRequest {
    fn type_url() -> &'static str {
        "type.googleapis.com/datafusion.example.com.sql.ActionRegisterPlanRequest"
    }

    fn as_any(&self) -> Any {
        Any {
            type_url: RegisterPlanRequest::type_url().to_string(),
            value: ::prost::Message::encode_to_vec(self).into(),
        }
    }
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct ExecutionMetricsRequest {
    #[prost(string, tag = "1")]
    pub handle: String,
}

impl ProstMessageExt for ExecutionMetricsRequest {
    fn type_url() -> &'static str {
        "type.googleapis.com/datafusion.example.com.sql.ExecutionMetricsRequest"
    }

    fn as_any(&self) -> Any {
        Any {
            type_url: ExecutionMetricsRequest::type_url().to_string(),
            value: ::prost::Message::encode_to_vec(self).into(),
        }
    }
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct RegisterTableRequest {
    #[prost(string, tag = "1")]
    pub url: ::prost::alloc::string::String,

    #[prost(string, tag = "2")]
    pub table_name: ::prost::alloc::string::String,

    #[prost(string, tag = "3")]
    pub cache_mode: ::prost::alloc::string::String,
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
pub struct RegisterObjectStoreRequest {
    #[prost(string, tag = "1")]
    pub url: ::prost::alloc::string::String,

    #[prost(map = "string, string", tag = "2")]
    pub options: HashMap<String, String>,
}

impl ProstMessageExt for RegisterObjectStoreRequest {
    fn type_url() -> &'static str {
        "type.googleapis.com/datafusion.example.com.sql.ActionRegisterObjectStoreRequest"
    }

    fn as_any(&self) -> Any {
        Any {
            type_url: RegisterObjectStoreRequest::type_url().to_string(),
            value: ::prost::Message::encode_to_vec(self).into(),
        }
    }
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct FetchResults {
    #[prost(bytes, tag = "1")]
    pub handle: Bytes,

    #[prost(uint32, tag = "2")]
    pub partition: u32,
}

impl FetchResults {
    pub fn into_ticket(self) -> Ticket {
        Ticket {
            ticket: self.as_any().encode_to_vec().into(),
        }
    }
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
    #[prost(uint64, tag = "3")]
    pub liquid_cache_usage: u64,
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
