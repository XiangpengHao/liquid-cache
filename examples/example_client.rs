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
use datafusion::{error::Result, execution::object_store::ObjectStoreUrl, prelude::*};
use liquid_cache_client::LiquidCacheBuilder;
use liquid_cache_client::common::CacheMode;
use std::path::Path;
use std::sync::Arc;
use url::Url;

#[derive(Parser, Clone)]
#[command(name = "Example Client")]
struct CliArgs {
    /// SQL query to execute
    #[arg(
        long,
        default_value = "SELECT COUNT(*) FROM \"aws-edge-locations\" WHERE \"countryCode\" = 'US';"
    )]
    query: String,

    /// URL of the table to query
    #[arg(
        long,
        default_value = "https://raw.githubusercontent.com/tobilg/aws-edge-locations/main/data/aws-edge-locations.parquet"
    )]
    file: String,

    /// Server URL
    #[arg(long, default_value = "http://localhost:15214")]
    cache_server: String,
}

#[tokio::main]
pub async fn main() -> Result<()> {
    let args = CliArgs::parse();
    let url = Url::parse(&args.file).unwrap();
    let object_store_url = format!("{}://{}", url.scheme(), url.host_str().unwrap_or_default());

    let ctx = LiquidCacheBuilder::new(args.cache_server.clone())
        .with_object_store(ObjectStoreUrl::parse(object_store_url.as_str())?, None)
        .with_cache_mode(CacheMode::Liquid)
        .build(SessionConfig::from_env()?)?;
    let ctx = Arc::new(ctx);

    let table_name = Path::new(url.path())
        .file_stem()
        .unwrap_or_default()
        .to_str()
        .unwrap_or("default");
    let sql = args.query;
    let object_store = object_store::http::HttpBuilder::new()
        .with_url(object_store_url.as_str())
        .build()
        .unwrap();
    let object_store_url = ObjectStoreUrl::parse(object_store_url.as_str()).unwrap();
    ctx.register_object_store(object_store_url.as_ref(), Arc::new(object_store));
    ctx.register_parquet(table_name, url.as_ref(), Default::default())
        .await?;

    ctx.sql(&sql).await?.show().await?;

    Ok(())
}
