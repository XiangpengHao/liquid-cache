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

#![warn(missing_docs)]
#![cfg_attr(not(doctest), doc = include_str!(concat!("../", std::env!("CARGO_PKG_README"))))]

use arrow::ipc::writer::IpcWriteOptions;
use arrow_flight::{
    Action, HandshakeRequest, HandshakeResponse, Ticket,
    encode::{DictionaryHandling, FlightDataEncoderBuilder},
    flight_service_server::FlightService,
    sql::{
        Any, CommandPreparedStatementUpdate, SqlInfo,
        server::{FlightSqlService, PeekableFlightDataStream},
    },
};
use datafusion::{
    error::DataFusionError,
    execution::{SessionStateBuilder, object_store::ObjectStoreUrl},
    prelude::{SessionConfig, SessionContext},
};
use datafusion_proto::bytes::physical_plan_from_bytes;
use fastrace::prelude::SpanContext;
use futures::{Stream, TryStreamExt};
use liquid_cache_common::{
    CacheMode,
    rpc::{FetchResults, LiquidCacheActions},
};
use liquid_cache_parquet::LiquidCacheRef;
use log::info;
use prost::bytes::Bytes;
use service::LiquidCacheServiceInner;
use std::{collections::HashMap, path::PathBuf, sync::Arc};
use std::{pin::Pin, str::FromStr};
use tonic::{Request, Response, Status, Streaming};
use url::Url;
use uuid::Uuid;
mod service;
mod utils;
use utils::FinalStream;
pub mod admin_server;
mod local_cache;

#[cfg(test)]
mod tests;

/// A trait to collect stats for the execution plan.
/// The server calls `start` right before polling the stream,
/// and calls `stop` right after exhausting the stream.
pub trait StatsCollector: Send + Sync {
    /// Start the stats collector.
    fn start(&self);
    /// Stop the stats collector.
    fn stop(&self);
}

/// The LiquidCache server.
///
/// # Example
///
/// ```rust
/// use arrow_flight::flight_service_server::FlightServiceServer;
/// use datafusion::prelude::SessionContext;
/// use liquid_cache_server::LiquidCacheService;
/// use tonic::transport::Server;
/// let liquid_cache = LiquidCacheService::new(SessionContext::new(), None, None);
/// let flight = FlightServiceServer::new(liquid_cache);
/// Server::builder()
///     .add_service(flight)
///     .serve("0.0.0.0:50051".parse().unwrap());
/// ```
pub struct LiquidCacheService {
    inner: LiquidCacheServiceInner,
    stats_collector: Vec<Arc<dyn StatsCollector>>,
}

impl Default for LiquidCacheService {
    fn default() -> Self {
        Self::try_new().unwrap()
    }
}

impl LiquidCacheService {
    /// Create a new [LiquidCacheService] with a default [SessionContext]
    /// With no disk cache and unbounded memory usage.
    pub fn try_new() -> Result<Self, DataFusionError> {
        let ctx = Self::context()?;
        Ok(Self::new(ctx, None, None))
    }

    /// Create a new [LiquidCacheService] with a custom [SessionContext]
    ///
    /// # Arguments
    ///
    /// * `ctx` - The [SessionContext] to use
    /// * `max_cache_bytes` - The maximum number of bytes to cache in memory
    /// * `disk_cache_dir` - The directory to store the disk cache
    pub fn new(
        ctx: SessionContext,
        max_cache_bytes: Option<usize>,
        disk_cache_dir: Option<PathBuf>,
    ) -> Self {
        Self {
            inner: LiquidCacheServiceInner::new(Arc::new(ctx), max_cache_bytes, disk_cache_dir),
            stats_collector: vec![],
        }
    }

    /// Get a reference to the cache
    pub fn cache(&self) -> &LiquidCacheRef {
        self.inner.cache()
    }

    /// Add a stats collector to the service
    pub fn add_stats_collector(&mut self, collector: Arc<dyn StatsCollector>) {
        self.stats_collector.push(collector);
    }

    /// Create a new [SessionContext] with good defaults
    /// This is the recommended way to create a [SessionContext] for LiquidCache
    pub fn context() -> Result<SessionContext, DataFusionError> {
        let mut session_config = SessionConfig::from_env()?;
        let options_mut = session_config.options_mut();
        options_mut.execution.parquet.pushdown_filters = true;
        options_mut.execution.parquet.binary_as_string = true;
        options_mut.execution.batch_size = 8192 * 2;

        {
            // view types cause excessive memory usage because they are not gced.
            // For Arrow memory mode, we need to read as UTF-8
            // For Liquid cache, we have our own way of handling string columns
            options_mut.execution.parquet.schema_force_view_types = false;
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

    /// Get all registered tables and their cache modes
    pub async fn get_registered_tables(&self) -> HashMap<String, (String, CacheMode)> {
        self.inner.get_registered_tables().await
    }

    /// Get the parquet cache directory
    pub fn get_parquet_cache_dir(&self) -> &PathBuf {
        self.inner.get_parquet_cache_dir()
    }

    pub(crate) fn inner(&self) -> &LiquidCacheServiceInner {
        &self.inner
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

    #[fastrace::trace]
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

        let span_context = SpanContext::decode_w3c_traceparent(&fetch_results.traceparent).unwrap();
        let span = fastrace::Span::root("poll_stream", span_context);

        let handle = Uuid::from_bytes_ref(fetch_results.handle.as_ref().try_into().unwrap());
        let partition = fetch_results.partition as usize;
        let stream = self.inner.execute_plan(handle, partition).await;
        let stream = FinalStream::new(
            stream,
            self.stats_collector.clone(),
            self.inner.batch_size(),
            span,
        )
        .map_err(|e| {
            panic!("Error executing plan: {e:?}");
        });

        let ipc_options = IpcWriteOptions::default();
        let stream = FlightDataEncoderBuilder::new()
            .with_options(ipc_options)
            .with_dictionary_handling(DictionaryHandling::Resend)
            .build(stream)
            .map_err(Status::from);

        Ok(Response::new(Box::pin(stream)))
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

    #[fastrace::trace]
    async fn do_action_fallback(
        &self,
        request: Request<Action>,
    ) -> Result<Response<<Self as FlightService>::DoActionStream>, Status> {
        let action = LiquidCacheActions::from(request.into_inner());
        match action {
            LiquidCacheActions::RegisterObjectStore(cmd) => {
                self.inner
                    .register_object_store(&Url::parse(&cmd.url).unwrap(), cmd.options)
                    .await
                    .map_err(df_error_to_status)?;

                let output = futures::stream::iter(vec![Ok(arrow_flight::Result {
                    body: Bytes::default(),
                })]);
                return Ok(Response::new(Box::pin(output)));
            }
            LiquidCacheActions::RegisterPlan(cmd) => {
                let plan = cmd.plan;
                let plan = physical_plan_from_bytes(&plan, self.inner.get_ctx()).unwrap();
                let handle = Uuid::from_bytes_ref(cmd.handle.as_ref().try_into().unwrap());
                let cache_mode = CacheMode::from_str(&cmd.cache_mode).unwrap();
                self.inner.register_plan(*handle, plan, cache_mode);
                let output = futures::stream::iter(vec![Ok(arrow_flight::Result {
                    body: Bytes::default(),
                })]);
                return Ok(Response::new(Box::pin(output)));
            }
        }
    }

    async fn register_sql_info(&self, _id: i32, _result: &SqlInfo) {}
}

fn df_error_to_status(err: datafusion::error::DataFusionError) -> Status {
    Status::internal(format!("{err:?}"))
}
