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

use arrow_flight::flight_service_server::FlightServiceServer;
use datafusion::prelude::{ParquetReadOptions, SessionConfig, SessionContext};
use datafusion_cache::server::SplitSqlService;
use log::info;
use std::sync::Arc;
use tonic::transport::Server;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::builder()
        .format_timestamp(None)
        .filter_level(log::LevelFilter::Info)
        .init();

    let addr = "0.0.0.0:50051".parse()?;

    let mut session_config = SessionConfig::from_env()?;
    let options_mut = session_config.options_mut();
    options_mut.execution.parquet.pushdown_filters = true;
    options_mut.execution.parquet.binary_as_string = true;
    let ctx = Arc::new(SessionContext::new_with_config(session_config));

    ctx.register_parquet(
        "small_hits",
        "examples/small_hits.parquet",
        ParquetReadOptions::default(),
    )
    .await?;

    let service = SplitSqlService::new("small_hits".to_string(), ctx);
    info!("SplitSQL server listening on {addr:?}");

    let svc = FlightServiceServer::new(service);

    Server::builder().add_service(svc).serve(addr).await?;

    Ok(())
}
