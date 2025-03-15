use std::collections::HashMap;

use arrow_flight::{
    Action,
    sql::{Any, ProstMessageExt},
};
use bytes::Bytes;
use prost::Message;

pub enum LiquidCacheActions {
    RegisterTable(RegisterTableRequest),
    ExecutionMetrics,
    ResetCache,
    RegisterObjectStore(RegisterObjectStoreRequest),
}

impl From<LiquidCacheActions> for Action {
    fn from(action: LiquidCacheActions) -> Self {
        match action {
            LiquidCacheActions::RegisterTable(request) => Action {
                r#type: "RegisterTable".to_string(),
                body: request.as_any().encode_to_vec().into(),
            },
            LiquidCacheActions::ExecutionMetrics => Action {
                r#type: "ExecutionMetrics".to_string(),
                body: Bytes::new(),
            },
            LiquidCacheActions::ResetCache => Action {
                r#type: "ResetCache".to_string(),
                body: Bytes::new(),
            },
            LiquidCacheActions::RegisterObjectStore(request) => Action {
                r#type: "RegisterObjectStore".to_string(),
                body: request.as_any().encode_to_vec().into(),
            },
        }
    }
}

impl From<Action> for LiquidCacheActions {
    fn from(action: Action) -> Self {
        match action.r#type.as_str() {
            "RegisterTable" => {
                let any = Any::decode(action.body).unwrap();
                let request = any.unpack::<RegisterTableRequest>().unwrap().unwrap();
                LiquidCacheActions::RegisterTable(request)
            }
            "ExecutionMetrics" => LiquidCacheActions::ExecutionMetrics,
            "ResetCache" => LiquidCacheActions::ResetCache,
            "RegisterObjectStore" => {
                let any = Any::decode(action.body).unwrap();
                let request = any.unpack::<RegisterObjectStoreRequest>().unwrap().unwrap();
                LiquidCacheActions::RegisterObjectStore(request)
            }
            _ => panic!("Invalid action: {}", action.r#type),
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
