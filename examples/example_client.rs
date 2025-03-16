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

use clap::{Parser, command};
use datafusion::{
    error::Result,
    prelude::{SessionConfig, SessionContext},
};
use liquid_cache_client::LiquidCacheTableBuilder;
use std::sync::Arc;
use url::Url;

#[derive(Parser, Clone)]
#[command(name = "Example Client")]
struct CliArgs {
    /// SQL query to execute
    #[arg(
        long,
        default_value = "SELECT COUNT(*) FROM aws_locations WHERE \"countryCode\" = 'US';"
    )]
    query: String,

    /// URL of the table to query
    #[arg(
        long,
        default_value = "https://raw.githubusercontent.com/tobilg/aws-edge-locations/main/data/aws-edge-locations.parquet"
    )]
    file: String,

    /// Server URL
    #[arg(long, default_value = "http://localhost:50051")]
    cache_server: String,
}

#[tokio::main]
pub async fn main() -> Result<()> {
    let mut session_config = SessionConfig::from_env()?;
    session_config
        .options_mut()
        .execution
        .parquet
        .pushdown_filters = true;
    let ctx = Arc::new(SessionContext::new_with_config(session_config));

    let args = CliArgs::parse();

    let cache_server = args.cache_server;
    let url = Url::parse(&args.file).unwrap();
    let table_name = url.path().split('/').next_back().unwrap_or("default");
    let sql = args.query;

    let table = LiquidCacheTableBuilder::new(cache_server, table_name, url.as_ref())
        .with_object_store(
            format!("{}://{}", url.scheme(), url.host_str().unwrap_or_default()),
            None,
        )
        .build()
        .await?;
    ctx.register_table(table_name, Arc::new(table))?;

    ctx.sql(&sql).await?.show().await?;

    Ok(())
}
