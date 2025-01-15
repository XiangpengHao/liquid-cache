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
    encode::FlightDataEncoderBuilder,
    flight_descriptor::DescriptorType,
    flight_service_server::FlightService,
    sql::{
        server::{FlightSqlService, PeekableFlightDataStream},
        ActionClosePreparedStatementRequest, Any, CommandGetDbSchemas,
        CommandPreparedStatementUpdate, CommandStatementQuery, ProstMessageExt, SqlInfo,
    },
    Action, FlightDescriptor, FlightEndpoint, FlightInfo, HandshakeRequest, HandshakeResponse,
    Ticket,
};
use dashmap::DashMap;
use datafusion::{
    error::DataFusionError,
    execution::object_store::ObjectStoreUrl,
    physical_plan::{ExecutionPlan, ExecutionPlanProperties},
    prelude::{ParquetReadOptions, SessionConfig, SessionContext},
};
use futures::{Stream, TryStreamExt};
use log::{debug, info};
use prost::bytes::Bytes;
use prost::Message;
use std::collections::HashSet;
use std::pin::Pin;
use std::sync::Arc;
use tokio::sync::Mutex;
use tonic::{Request, Response, Status, Streaming};
use url::Url;
use uuid::Uuid;

mod utils;
use utils::GcStream;

pub(crate) static ACTION_REGISTER_TABLE: &str = "RegisterTable";

pub struct SplitSqlService {
    execution_plans: Arc<DashMap<String, Arc<dyn ExecutionPlan>>>,
    registered_tables: Mutex<HashSet<String>>,
    default_ctx: Arc<SessionContext>,
}

impl SplitSqlService {
    pub fn try_new() -> Result<Self, DataFusionError> {
        let ctx = Self::context()?;
        Ok(Self::new_with_context(ctx))
    }

    pub fn new_with_context(default_ctx: SessionContext) -> Self {
        Self {
            execution_plans: Default::default(),
            registered_tables: Default::default(),
            default_ctx: Arc::new(default_ctx),
        }
    }

    pub fn context() -> Result<SessionContext, DataFusionError> {
        let mut session_config = SessionConfig::from_env()?;
        let options_mut = session_config.options_mut();
        options_mut.execution.parquet.pushdown_filters = true;
        options_mut.execution.parquet.binary_as_string = true;
        let ctx = SessionContext::new_with_config(session_config);
        let object_store_url = ObjectStoreUrl::parse("file://").unwrap();
        let object_store = object_store::local::LocalFileSystem::new();
        ctx.register_object_store(object_store_url.as_ref(), Arc::new(object_store));
        Ok(ctx)
    }

    async fn register_table(&self, url: String, table_name: String) -> Result<(), Status> {
        let url = Url::parse(&url).map_err(|e| Status::invalid_argument(format!("{e:?}")))?;

        let mut registered_tables = self.registered_tables.lock().await;
        if registered_tables.contains(&table_name) {
            // already registered
            info!("table {table_name} already registered");
            return Ok(());
        }

        self.default_ctx
            .register_parquet(&table_name, &url, ParquetReadOptions::default())
            .await
            .map_err(df_error_to_status)?;
        info!("registered table {table_name} from {url}");
        registered_tables.insert(table_name);
        Ok(())
    }
}

impl SplitSqlService {
    fn get_ctx<T>(&self, _req: &Request<T>) -> Result<Arc<SessionContext>, Status> {
        Ok(self.default_ctx.clone())
    }

    fn get_result(&self, handle: &str) -> Result<Arc<dyn ExecutionPlan>, Status> {
        if let Some(result) = self.execution_plans.get(handle) {
            Ok(result.clone())
        } else {
            Err(Status::internal(format!(
                "Request handle not found: {handle}"
            )))?
        }
    }

    fn remove_result(&self, handle: &str) -> Result<(), Status> {
        self.execution_plans.remove(&handle.to_string());
        Ok(())
    }
}

#[tonic::async_trait]
impl FlightSqlService for SplitSqlService {
    type FlightService = SplitSqlService;

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
        let schema = self
            .default_ctx
            .table_provider(&table_name)
            .await
            .unwrap()
            .schema();

        let info = FlightInfo::new()
            .try_with_schema(&schema)
            .expect("encoding failed");
        Ok(Response::new(info))
    }

    async fn do_get_fallback(
        &self,
        request: Request<Ticket>,
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

        debug!("getting results for {handle}");
        let execution_plan = self.get_result(&handle)?;

        let displayable = datafusion::physical_plan::display::DisplayableExecutionPlan::new(
            execution_plan.as_ref(),
        );
        debug!("physical plan:\n{}", displayable.indent(false));

        let ctx = self.get_ctx(&request)?;

        let schema = execution_plan.schema();
        debug!("execution plan schema: {:?}", schema);
        let stream = execution_plan
            .execute(fetch_results.partition as usize, ctx.task_ctx())
            .unwrap();
        let stream = GcStream::new(stream).map_err(|e| {
            panic!("Error executing plan: {:?}", e);
        });

        let ipc_options = IpcWriteOptions::default().with_preserve_dict_id(false);
        let stream = FlightDataEncoderBuilder::new()
            .with_options(ipc_options)
            .build(stream)
            .map_err(Status::from);

        Ok(Response::new(Box::pin(stream)))
    }

    async fn get_flight_info_statement(
        &self,
        cmd: CommandStatementQuery,
        request: Request<FlightDescriptor>,
    ) -> Result<Response<FlightInfo>, Status> {
        let user_query = cmd.query.as_str();
        info!("running query: {user_query}");

        let ctx = self.get_ctx(&request)?;

        let plan = ctx.sql(user_query).await.expect("Error generating plan");
        let (state, plan) = plan.into_parts();
        let plan = state.optimize(&plan).expect("Error optimizing plan");
        let physical_plan = state
            .create_physical_plan(&plan)
            .await
            .expect("Error creating physical plan");

        let partition_count = physical_plan.output_partitioning().partition_count();

        let schema = physical_plan.schema();

        let handle = Uuid::new_v4().hyphenated().to_string();
        self.execution_plans.insert(handle.clone(), physical_plan);

        let flight_desc = FlightDescriptor {
            r#type: DescriptorType::Cmd.into(),
            cmd: Default::default(),
            path: vec![],
        };

        let mut info = FlightInfo::new()
            .try_with_schema(&schema)
            .expect("encoding failed")
            .with_descriptor(flight_desc);

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
            info!("do_action_close_prepared_statement: removing plan and results for {handle}");
            let _ = self.remove_result(handle);
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
            self.register_table(cmd.url, cmd.table_name).await?;
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
