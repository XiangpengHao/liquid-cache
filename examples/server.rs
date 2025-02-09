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
use liquid_cache_server::LiquidCacheService;
use log::info;
use tonic::transport::Server;
use std::env;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {

    env_logger::builder().format_timestamp(None).init();

    let port: u16 = env::var("PORT")
        .unwrap_or_else(|_| "50051".to_string()) // Default to "50051" if PORT is not set
        .parse()
        .expect("Failed to parse PORT as a number");

    let addr = format!("0.0.0.0:{}", port).parse()?;

    let split_sql = LiquidCacheService::try_new()?;
    let flight = FlightServiceServer::new(split_sql);

    info!("SplitSQL server listening on {addr:?}");

    Server::builder().add_service(flight).serve(addr).await?;

    Ok(())
}
