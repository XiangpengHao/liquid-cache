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
use datafusion::prelude::SessionContext;
use liquid_cache_common::CacheMode;
use liquid_cache_parquet::cache::policies::LruPolicy;
use liquid_cache_server::LiquidCacheService;
use tonic::transport::Server;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let liquid_cache = LiquidCacheService::new(
        SessionContext::new(),
        Some(1024 * 1024 * 1024),          // max memory cache size 1GB
        Some(tempfile::tempdir()?.keep()), // disk cache dir
        CacheMode::LiquidEagerTranscode,
        Box::new(LruPolicy::new()),
    )?;

    let flight = FlightServiceServer::new(liquid_cache);

    Server::builder()
        .add_service(flight)
        .serve("0.0.0.0:15214".parse()?)
        .await?;

    Ok(())
}
