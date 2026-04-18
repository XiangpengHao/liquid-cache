//! RPC definitions for the LiquidCache service.
//! You should not use this module directly.
//! Instead, use the `liquid_cache_datafusion_server` and `liquid_cache_datafusion_client` crates to interact with the LiquidCache service.
use std::collections::HashMap;

use arrow_flight::{
    Action, Ticket,
    sql::{Any, ProstMessageExt},
};
use bytes::Bytes;
use prost::Message;

/// The actions that can be performed on the LiquidCache service.
pub enum LiquidCacheActions {
    /// Register an object store with the LiquidCache service.
    RegisterObjectStore(RegisterObjectStoreRequest),
    /// Register a plan with the LiquidCache service.
    RegisterPlan(RegisterPlanRequest),
    /// Prefetch parquet files from the object store.
    PrefetchFromObjectStore(PrefetchFromObjectStoreRequest),
}

impl From<LiquidCacheActions> for Action {
    fn from(action: LiquidCacheActions) -> Self {
        match action {
            LiquidCacheActions::RegisterObjectStore(request) => Action {
                r#type: "RegisterObjectStore".to_string(),
                body: request.as_any().encode_to_vec().into(),
            },
            LiquidCacheActions::RegisterPlan(request) => Action {
                r#type: "RegisterPlan".to_string(),
                body: request.as_any().encode_to_vec().into(),
            },
            LiquidCacheActions::PrefetchFromObjectStore(request) => Action {
                r#type: "PrefetchFromObjectStore".to_string(),
                body: request.as_any().encode_to_vec().into(),
            },
        }
    }
}

impl From<Action> for LiquidCacheActions {
    fn from(action: Action) -> Self {
        match action.r#type.as_str() {
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
            "PrefetchFromObjectStore" => {
                let any = Any::decode(action.body).unwrap();
                let request = any
                    .unpack::<PrefetchFromObjectStoreRequest>()
                    .unwrap()
                    .unwrap();
                LiquidCacheActions::PrefetchFromObjectStore(request)
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
}

impl ProstMessageExt for RegisterPlanRequest {
    fn type_url() -> &'static str {
        ""
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
        ""
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
        ""
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
        ""
    }

    fn as_any(&self) -> Any {
        Any {
            type_url: RegisterObjectStoreRequest::type_url().to_string(),
            value: ::prost::Message::encode_to_vec(self).into(),
        }
    }
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct PrefetchFromObjectStoreRequest {
    /// Url of the object store. eg. s3://bucket
    #[prost(string, tag = "1")]
    pub url: ::prost::alloc::string::String,

    /// Config options for the object store
    /// For S3, this might include "access_key_id", "secret_access_key", etc.
    #[prost(map = "string, string", tag = "2")]
    pub store_options: HashMap<String, String>,

    /// Location of the file within the object store. eg. path/to/file.parquet
    #[prost(string, tag = "3")]
    pub location: ::prost::alloc::string::String,

    /// The start byte offset of the range to prefetch (inclusive)
    /// If not specified, prefetch from the beginning of the file
    #[prost(uint64, optional, tag = "4")]
    pub range_start: Option<u64>,

    /// The end byte offset of the range to prefetch (exclusive)
    /// If not specified, prefetch until the end of the file
    #[prost(uint64, optional, tag = "5")]
    pub range_end: Option<u64>,
}

impl ProstMessageExt for PrefetchFromObjectStoreRequest {
    fn type_url() -> &'static str {
        ""
    }
    fn as_any(&self) -> Any {
        Any {
            type_url: PrefetchFromObjectStoreRequest::type_url().to_string(),
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

    #[prost(string, tag = "3")]
    pub traceparent: String,
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
        ""
    }

    fn as_any(&self) -> Any {
        Any {
            type_url: FetchResults::type_url().to_string(),
            value: ::prost::Message::encode_to_vec(self).into(),
        }
    }
}

#[derive(Clone, PartialEq, serde::Serialize, serde::Deserialize, Debug)]
pub struct ExecutionMetricsResponse {
    pub pushdown_eval_time: u64,
    pub cache_memory_usage: u64,
    pub liquid_cache_usage: u64,
}

impl ExecutionMetricsResponse {
    pub fn zero() -> Self {
        Self {
            pushdown_eval_time: 0,
            cache_memory_usage: 0,
            liquid_cache_usage: 0,
        }
    }
}
