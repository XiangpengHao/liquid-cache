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
    CacheEvictionStrategy, CacheMode,
    rpc::{FetchResults, LiquidCacheActions},
};
use liquid_cache_parquet::LiquidCacheRef;
use log::info;
use prost::bytes::Bytes;
use service::LiquidCacheServiceInner;
use std::pin::Pin;
use std::{path::PathBuf, sync::Arc};
use tonic::{Request, Response, Status, Streaming};
use url::Url;
use uuid::Uuid;
mod service;
mod utils;
use utils::FinalStream;
mod admin_server;
mod errors;
mod local_cache;
pub use admin_server::{models::*, run_admin_server};
pub use errors::{
    LiquidCacheErrorExt, LiquidCacheResult, anyhow_to_status, df_error_to_status_with_trace,
};
use object_store::path::Path;
use object_store::{GetOptions, GetRange};

#[cfg(test)]
mod tests;

/// The LiquidCache server.
///
/// # Example
///
/// ```rust
/// use arrow_flight::flight_service_server::FlightServiceServer;
/// use datafusion::prelude::SessionContext;
/// use liquid_cache_server::LiquidCacheService;
/// use tonic::transport::Server;
/// use liquid_cache_common::CacheEvictionStrategy::Discard;
/// let liquid_cache = LiquidCacheService::new(SessionContext::new(), None, None, Default::default(), Discard).unwrap();
/// let flight = FlightServiceServer::new(liquid_cache);
/// Server::builder()
///     .add_service(flight)
///     .serve("0.0.0.0:15214".parse().unwrap());
/// ```
pub struct LiquidCacheService {
    inner: LiquidCacheServiceInner,
}

impl Default for LiquidCacheService {
    fn default() -> Self {
        Self::try_new().unwrap()
    }
}

impl LiquidCacheService {
    /// Create a new [LiquidCacheService] with a default [SessionContext]
    /// With no disk cache and unbounded memory usage.
    pub fn try_new() -> anyhow::Result<Self> {
        let ctx = Self::context()?;
        Self::new(
            ctx,
            None,
            None,
            CacheMode::LiquidEagerTranscode,
            CacheEvictionStrategy::Discard,
        )
    }

    /// Create a new [LiquidCacheService] with a custom [SessionContext]
    ///
    /// # Arguments
    ///
    /// * `ctx` - The [SessionContext] to use
    /// * `max_cache_bytes` - The maximum number of bytes to cache in memory
    /// * `disk_cache_dir` - The directory to store the disk cache
    /// * `cache_mode` - The [CacheMode] to use
    pub fn new(
        ctx: SessionContext,
        max_cache_bytes: Option<usize>,
        disk_cache_dir: Option<PathBuf>,
        cache_mode: CacheMode,
        cache_policy: CacheEvictionStrategy,
    ) -> anyhow::Result<Self> {
        let disk_cache_dir = match disk_cache_dir {
            Some(dir) => dir,
            None => tempfile::tempdir()?.keep(),
        };
        Ok(Self {
            inner: LiquidCacheServiceInner::new(
                Arc::new(ctx),
                max_cache_bytes,
                disk_cache_dir,
                cache_mode,
                cache_policy,
            ),
        })
    }

    /// Get a reference to the cache
    pub fn cache(&self) -> &Option<LiquidCacheRef> {
        self.inner.cache()
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

        let object_store_url = ObjectStoreUrl::parse("file://")?;
        let object_store = object_store::local::LocalFileSystem::new();

        let state = SessionStateBuilder::new()
            .with_config(session_config)
            .with_default_features()
            .with_object_store(object_store_url.as_ref(), Arc::new(object_store))
            .build();

        let ctx = SessionContext::new_with_state(state);
        Ok(ctx)
    }

    /// Get the parquet cache directory
    pub fn get_parquet_cache_dir(&self) -> &PathBuf {
        self.inner.get_parquet_cache_dir()
    }

    pub(crate) fn inner(&self) -> &LiquidCacheServiceInner {
        &self.inner
    }

    async fn do_get_fallback_inner(
        &self,
        message: Any,
    ) -> anyhow::Result<Response<<Self as FlightService>::DoGetStream>> {
        if !message.is::<FetchResults>() {
            return Err(anyhow::anyhow!(
                "do_get: The defined request is invalid: {}",
                message.type_url
            ));
        }

        let fetch_results: FetchResults = message
            .unpack()?
            .ok_or_else(|| anyhow::anyhow!("Expected FetchResults but got None!"))?;

        let span_context = SpanContext::decode_w3c_traceparent(&fetch_results.traceparent)
            .ok_or_else(|| anyhow::anyhow!("Failed to decode traceparent"))?;
        let span = fastrace::Span::root("poll_stream", span_context);

        let handle = Uuid::from_bytes_ref(fetch_results.handle.as_ref().try_into()?);
        let partition = fetch_results.partition as usize;
        let stream = self.inner.execute_plan(handle, partition).await?;
        let stream = FinalStream::new(stream, self.inner.batch_size(), span).map_err(|e| {
            let status = anyhow_to_status(anyhow::Error::from(e).context("Error executing plan"));
            arrow_flight::error::FlightError::Tonic(Box::new(status))
        });

        let ipc_options = IpcWriteOptions::default();
        let stream = FlightDataEncoderBuilder::new()
            .with_options(ipc_options)
            .with_dictionary_handling(DictionaryHandling::Resend)
            .build(stream)
            .map_err(Status::from);

        Ok(Response::new(Box::pin(stream)))
    }

    async fn do_action_inner(
        &self,
        action: LiquidCacheActions,
    ) -> anyhow::Result<Response<<Self as FlightService>::DoActionStream>> {
        match action {
            LiquidCacheActions::RegisterObjectStore(cmd) => {
                let url = Url::parse(&cmd.url)?;
                self.inner.register_object_store(&url, cmd.options).await?;

                let output = futures::stream::iter(vec![Ok(arrow_flight::Result {
                    body: Bytes::default(),
                })]);
                Ok(Response::new(Box::pin(output)))
            }
            LiquidCacheActions::RegisterPlan(cmd) => {
                let plan = cmd.plan;
                let plan = physical_plan_from_bytes(&plan, self.inner.get_ctx())?;
                let handle = Uuid::from_bytes_ref(cmd.handle.as_ref().try_into()?);
                self.inner.register_plan(*handle, plan);
                let output = futures::stream::iter(vec![Ok(arrow_flight::Result {
                    body: Bytes::default(),
                })]);
                Ok(Response::new(Box::pin(output)))
            }
            LiquidCacheActions::PrefetchFromObjectStore(cmd) => {
                // Parse the object store URL (e.g., s3://bucket)
                let url = Url::parse(&cmd.url)?;

                // Register the object store if not already registered
                self.inner
                    .register_object_store(&url, cmd.store_options)
                    .await?;

                // Get the local cache wrapper for this object store
                let local_cache = self.inner.get_object_store(&url)?;

                // Parse the path to the object within the store (e.g., path/to/file.parquet)
                let path = Path::from(cmd.location);

                // Create a range for the specific chunk we want to prefetch, if specified
                let get_options = if cmd.range_start.is_some() && cmd.range_end.is_some() {
                    let chunk_range =
                        GetRange::Bounded(cmd.range_start.unwrap()..cmd.range_end.unwrap());
                    GetOptions {
                        range: Some(chunk_range),
                        ..GetOptions::default()
                    }
                } else {
                    GetOptions::default()
                };

                // Call the underlying object store to get the data and cache it
                // The LocalCache wrapper will handle caching the fetched data
                local_cache.get_opts(&path, get_options).await?;

                // Return empty response to indicate successful prefetch
                let output = futures::stream::iter(vec![Ok(arrow_flight::Result {
                    body: Bytes::default(),
                })]);
                Ok(Response::new(Box::pin(output)))
            }
        }
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
        self.do_get_fallback_inner(message)
            .await
            .map_err(anyhow_to_status)
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
        self.do_action_inner(action).await.map_err(anyhow_to_status)
    }

    async fn register_sql_info(&self, _id: i32, _result: &SqlInfo) {}
}

#[cfg(test)]
mod server_actions_tests {
    use super::*;
    use LiquidCacheService;
    use liquid_cache_common::rpc::PrefetchFromObjectStoreRequest;
    use std::collections::HashMap;
    use tokio::fs::File;
    use tokio::io::AsyncWriteExt;

    async fn create_test_file(file_path: &str, size_mb: usize) -> anyhow::Result<()> {
        let mut file = File::create(file_path).await?;
        let size_bytes = size_mb * 1024 * 1024;

        let chunk_size = 64 * 1024;
        let chunk_data = vec![0u8; chunk_size];
        let mut written = 0;

        while written < size_bytes {
            let remaining = size_bytes - written;
            let write_size = std::cmp::min(chunk_size, remaining);
            file.write_all(&chunk_data[..write_size]).await?;
            written += write_size;
        }

        file.flush().await?;
        Ok(())
    }

    #[tokio::test]
    async fn test_prefetch_from_object_store() {
        let service = LiquidCacheService::default();

        // Create temporary test file
        let temp_dir = tempfile::tempdir().unwrap();
        let file_path = temp_dir.path().join("test_prefetch_data.bin");
        let file_path_str = file_path.to_string_lossy().to_string();

        // Generate 16MB test file
        create_test_file(&file_path_str, 16).await.unwrap();

        // Test with local file system
        let url = Url::parse("file:///").unwrap();

        let request = PrefetchFromObjectStoreRequest {
            url: url.to_string(),
            store_options: HashMap::new(),
            location: file_path_str.clone(),
            range_start: None,
            range_end: None,
        };

        let action = LiquidCacheActions::PrefetchFromObjectStore(request);
        let result = service.do_action_inner(action).await.unwrap();

        let mut stream = result.into_inner();
        let result = stream.try_next().await.unwrap().unwrap();
        assert!(result.body.is_empty());

        // Cleanup is handled by tempdir drop
    }

    #[tokio::test]
    async fn test_prefetch_from_object_store_with_range() {
        let service = LiquidCacheService::default();

        // Create temporary test file
        let temp_dir = tempfile::tempdir().unwrap();
        let file_path = temp_dir.path().join("test_prefetch_data_range.bin");
        let file_path_str = file_path.to_string_lossy().to_string();

        // Generate 16MB test file
        create_test_file(&file_path_str, 16).await.unwrap();

        // Test with local file system
        let url = Url::parse("file:///").unwrap();

        // range from 1mb to 10mb
        let range_start = 1024 * 1024;
        let range_end = 10 * 1024 * 1024;

        let request = PrefetchFromObjectStoreRequest {
            url: url.to_string(),
            store_options: HashMap::new(),
            location: file_path_str.clone(),
            range_start: Some(range_start),
            range_end: Some(range_end),
        };

        let action = LiquidCacheActions::PrefetchFromObjectStore(request);
        let result = service.do_action_inner(action).await.unwrap();

        let mut stream = result.into_inner();
        let result = stream.try_next().await.unwrap().unwrap();
        assert!(result.body.is_empty());

        // Cleanup is handled by tempdir drop
    }

    #[tokio::test]
    async fn test_prefetch_invalid_object_store() {
        let service = LiquidCacheService::default();

        let request = PrefetchFromObjectStoreRequest {
            url: "invalid://url".to_string(),
            store_options: HashMap::new(),
            location: "test.parquet".to_string(),
            range_start: Some(0),
            range_end: Some(1024),
        };

        let action = LiquidCacheActions::PrefetchFromObjectStore(request);
        let result = service.do_action_inner(action).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_prefetch_invalid_location() {
        let service = LiquidCacheService::default();

        let url = Url::parse("file:///").unwrap();
        let request = PrefetchFromObjectStoreRequest {
            url: url.to_string(),
            store_options: HashMap::new(),
            location: "non_existent_file.parquet".to_string(),
            range_start: Some(0),
            range_end: Some(1024),
        };

        let action = LiquidCacheActions::PrefetchFromObjectStore(request);
        let result = service.do_action_inner(action).await;
        assert!(result.is_err());
    }
}
