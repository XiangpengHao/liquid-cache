// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

use arrow::ipc::writer::IpcWriteOptions;
use arrow_flight::{
    Action, FlightDescriptor, FlightEndpoint, FlightInfo, HandshakeRequest, HandshakeResponse,
    IpcMessage, SchemaAsIpc, Ticket,
    encode::{DictionaryHandling, FlightDataEncoderBuilder},
    flight_descriptor::DescriptorType,
    flight_service_server::FlightService,
    sql::{
        ActionClosePreparedStatementRequest, Any, CommandGetDbSchemas,
        CommandPreparedStatementUpdate, CommandStatementQuery, ProstMessageExt, SqlInfo,
        server::{FlightSqlService, PeekableFlightDataStream},
    },
};
use datafusion::{
    error::DataFusionError,
    execution::{SessionStateBuilder, object_store::ObjectStoreUrl},
    physical_plan::ExecutionPlanProperties,
    prelude::{SessionConfig, SessionContext},
};
use futures::{Stream, TryStreamExt};
use log::info;
use prost::Message;
use prost::bytes::Bytes;
use service::LiquidCacheServiceInner;
use std::pin::Pin;
use std::sync::Arc;
use tonic::{Request, Response, Status, Streaming};
use uuid::Uuid;
mod service;
mod utils;
use utils::FinalStream;

use liquid_parquet::{LiquidCacheMode, LiquidCacheRef};

pub static ACTION_REGISTER_TABLE: &str = "RegisterTable";

pub struct LiquidCacheConfig {
    pub liquid_cache_mode: LiquidCacheMode,
}

impl LiquidCacheConfig {
    pub fn with_liquid_cache_mode(mut self, liquid_cache_mode: LiquidCacheMode) -> Self {
        self.liquid_cache_mode = liquid_cache_mode;
        self
    }
}

impl Default for LiquidCacheConfig {
    fn default() -> Self {
        Self {
            liquid_cache_mode: LiquidCacheMode::InMemoryLiquid,
        }
    }
}

/// A trait to collect stats for the execution plan.
/// The server calls `start` right before polling the stream,
/// and calls `stop` right after exhausting the stream.
pub trait StatsCollector: Send + Sync {
    fn start(&self);
    fn stop(&self);
}

pub struct LiquidCacheService {
    inner: LiquidCacheServiceInner,
    stats_collector: Vec<Arc<dyn StatsCollector>>,
}

impl LiquidCacheService {
    pub fn try_new() -> Result<Self, DataFusionError> {
        let ctx = Self::context(None)?;
        let config = LiquidCacheConfig::default();
        Ok(Self::new_with_context_and_config(ctx, config))
    }

    pub fn new_with_context_and_config(
        default_ctx: SessionContext,
        config: LiquidCacheConfig,
    ) -> Self {
        Self {
            inner: LiquidCacheServiceInner::new(Arc::new(default_ctx), config),
            stats_collector: vec![],
        }
    }

    pub fn new_with_ctx_and_cache(ctx: Arc<SessionContext>, cache: LiquidCacheRef) -> Self {
        Self {
            inner: LiquidCacheServiceInner::new_ctx_and_cache(ctx, cache),
            stats_collector: vec![],
        }
    }

    pub fn add_stats_collector(&mut self, collector: Arc<dyn StatsCollector>) {
        self.stats_collector.push(collector);
    }

    /// Create a new SessionContext with good defaults
    pub fn context(partitions: Option<usize>) -> Result<SessionContext, DataFusionError> {
        let mut session_config = SessionConfig::from_env()?;
        let options_mut = session_config.options_mut();
        options_mut.execution.parquet.pushdown_filters = true;
        options_mut.execution.parquet.binary_as_string = true;

        {
            // View types only provide benefits for parquet decoding and filtering.
            // but liquid cache has its own encodings, and don't need to use view types.
            options_mut.execution.parquet.schema_force_view_types = false;
        }

        if let Some(partitions) = partitions {
            options_mut.execution.target_partitions = partitions;
        }

        let object_store_url = ObjectStoreUrl::parse("file://").unwrap();
        let object_store = object_store::local::LocalFileSystem::new();

        let state = SessionStateBuilder::new()
            .with_config(session_config)
            .with_default_features()
            .with_object_store(object_store_url.as_ref(), Arc::new(object_store))
            .build();

        let ctx = SessionContext::new_with_state(state);
        Ok(ctx)
    }
}

#[tonic::async_trait]
impl FlightSqlService for LiquidCacheService {
    type FlightService = LiquidCacheService;

    async fn do_handshake(
        &self,
        _request: Request<Streaming<HandshakeRequest>>,
    ) -> Result<
        Response<Pin<Box<dyn Stream<Item = Result<HandshakeResponse, Status>> + Send>>>,
        Status,
    > {
        unimplemented!("We don't do handshake")
    }

    async fn get_flight_info_schemas(
        &self,
        query: CommandGetDbSchemas,
        _request: Request<FlightDescriptor>,
    ) -> Result<Response<FlightInfo>, Status> {
        let table_name = query
            .db_schema_filter_pattern
            .ok_or(Status::invalid_argument(
                "db_schema_filter_pattern is required",
            ))?;
        let schema = self.inner.get_table_schema(&table_name).await?;

        let mut info = FlightInfo::new();
        info.schema = encode_schema_to_ipc_bytes(&schema);

        Ok(Response::new(info))
    }

    async fn do_get_fallback(
        &self,
        _request: Request<Ticket>,
        message: Any,
    ) -> Result<Response<<Self as FlightService>::DoGetStream>, Status> {
        if !message.is::<FetchResults>() {
            Err(Status::unimplemented(format!(
                "do_get: The defined request is invalid: {}",
                message.type_url
            )))?
        }

        let fetch_results: FetchResults = message
            .unpack()
            .map_err(|e| Status::internal(format!("{e:?}")))?
            .ok_or_else(|| Status::internal("Expected FetchResults but got None!"))?;

        let handle = fetch_results.handle;
        let partition = fetch_results.partition as usize;
        let stream = self.inner.execute_plan(&handle, partition).await;
        let stream = FinalStream::new(
            stream,
            self.stats_collector.clone(),
            self.inner.batch_size(),
        )
        .map_err(|e| {
            panic!("Error executing plan: {:?}", e);
        });

        let ipc_options = IpcWriteOptions::default();
        let stream = FlightDataEncoderBuilder::new()
            .with_options(ipc_options)
            .with_dictionary_handling(DictionaryHandling::Resend)
            .build(stream)
            .map_err(Status::from);

        Ok(Response::new(Box::pin(stream)))
    }

    async fn get_flight_info_statement(
        &self,
        cmd: CommandStatementQuery,
        _request: Request<FlightDescriptor>,
    ) -> Result<Response<FlightInfo>, Status> {
        let user_query = cmd.query.as_str();
        let handle = Uuid::new_v4().hyphenated().to_string();
        let physical_plan = self
            .inner
            .prepare_and_register_plan(user_query, &handle)
            .await?;
        let partition_count = physical_plan.output_partitioning().partition_count();

        let schema = physical_plan.schema();

        let flight_desc = FlightDescriptor {
            r#type: DescriptorType::Cmd.into(),
            cmd: Default::default(),
            path: vec![],
        };

        let mut info = FlightInfo::new().with_descriptor(flight_desc);
        info.schema = encode_schema_to_ipc_bytes(&schema);

        for partition in 0..partition_count {
            let fetch = FetchResults {
                handle: handle.clone(),
                partition: partition as u32,
            };
            let buf = fetch.as_any().encode_to_vec().into();
            let ticket = Ticket { ticket: buf };
            let endpoint = FlightEndpoint::new().with_ticket(ticket.clone());
            info = info.with_endpoint(endpoint);
        }

        let resp = Response::new(info);
        Ok(resp)
    }

    async fn do_put_prepared_statement_update(
        &self,
        _handle: CommandPreparedStatementUpdate,
        _request: Request<PeekableFlightDataStream>,
    ) -> Result<i64, Status> {
        info!("do_put_prepared_statement_update");
        // statements like "CREATE TABLE.." or "SET datafusion.nnn.." call this function
        // and we are required to return some row count here
        Ok(-1)
    }

    async fn do_action_close_prepared_statement(
        &self,
        handle: ActionClosePreparedStatementRequest,
        _request: Request<Action>,
    ) -> Result<(), Status> {
        let handle = std::str::from_utf8(&handle.prepared_statement_handle);
        if let Ok(handle) = handle {
            self.inner.remove_plan(handle);
        }
        Ok(())
    }

    async fn do_action_fallback(
        &self,
        request: Request<Action>,
    ) -> Result<Response<<Self as FlightService>::DoActionStream>, Status> {
        if request.get_ref().r#type == ACTION_REGISTER_TABLE {
            let any = Any::decode(&*request.get_ref().body).map_err(decode_error_to_status)?;
            let cmd: ActionRegisterTableRequest = any
                .unpack()
                .map_err(arrow_error_to_status)?
                .ok_or_else(|| {
                    Status::invalid_argument("Unable to unpack ActionRegisterTableRequest.")
                })?;
            self.inner
                .register_table(&cmd.url, &cmd.table_name)
                .await
                .map_err(df_error_to_status)?;
            let output = futures::stream::iter(vec![Ok(arrow_flight::Result {
                body: Bytes::default(),
            })]);
            return Ok(Response::new(Box::pin(output)));
        }

        Err(Status::invalid_argument(format!(
            "do_action: The defined request is invalid: {:?}",
            request.get_ref().r#type
        )))
    }

    async fn register_sql_info(&self, _id: i32, _result: &SqlInfo) {}
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct FetchResults {
    #[prost(string, tag = "1")]
    pub handle: ::prost::alloc::string::String,

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
pub struct ActionRegisterTableRequest {
    #[prost(string, tag = "1")]
    pub url: ::prost::alloc::string::String,

    #[prost(string, tag = "2")]
    pub table_name: ::prost::alloc::string::String,
}

impl ProstMessageExt for ActionRegisterTableRequest {
    fn type_url() -> &'static str {
        "type.googleapis.com/datafusion.example.com.sql.ActionRegisterTableRequest"
    }

    fn as_any(&self) -> Any {
        Any {
            type_url: ActionRegisterTableRequest::type_url().to_string(),
            value: ::prost::Message::encode_to_vec(self).into(),
        }
    }
}

fn decode_error_to_status(err: prost::DecodeError) -> Status {
    Status::invalid_argument(format!("{err:?}"))
}

fn arrow_error_to_status(err: arrow_schema::ArrowError) -> Status {
    Status::internal(format!("{err:?}"))
}

fn df_error_to_status(err: datafusion::error::DataFusionError) -> Status {
    Status::internal(format!("{err:?}"))
}

// TODO: we need to workaround a arrow-flight bug here:
// https://github.com/apache/arrow-rs/issues/7058
fn encode_schema_to_ipc_bytes(schema: &arrow_schema::Schema) -> Bytes {
    let options = IpcWriteOptions::default();
    let schema_as_ipc = SchemaAsIpc::new(schema, &options);
    let IpcMessage(schema) = schema_as_ipc.try_into().unwrap();
    schema
}
