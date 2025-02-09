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

use std::sync::Arc;

use datafusion::{
    error::Result,
    prelude::{SessionConfig, SessionContext},
};
use liquid_cache_client::SplitSqlTableFactory;
use log::info;
use url::Url;
use std::env;

#[tokio::main]
pub async fn main() -> Result<()> {
    env_logger::builder()
        .format_timestamp(None)
        .filter_level(log::LevelFilter::Info)
        .init();

    let mut session_config = SessionConfig::from_env()?;
    session_config
        .options_mut()
        .execution
        .parquet
        .pushdown_filters = true;
    let ctx = Arc::new(SessionContext::new_with_config(session_config));

    let port: u16 = env::var("PORT")
        .unwrap_or_else(|_| "50051".to_string()) // Default to "50051" if PORT is not set
        .parse()
        .expect("Failed to parse PORT as a number");

    let entry_point = format!("http://localhost:{}", port);

    let sql = "SELECT COUNT(*) FROM small_hits WHERE \"URL\" <> '';";

    info!("SQL to be executed: {}", sql);

    let table_name = "small_hits";

    let current_dir = std::env::current_dir()?.to_string_lossy().to_string();
    let table_url = Url::parse(&format!(
        "file://{}/examples/small_hits.parquet",
        current_dir
    ))
    .unwrap();

    let table = SplitSqlTableFactory::open_table(entry_point, table_name, table_url).await?;
    ctx.register_table(table_name, Arc::new(table))?;

    ctx.sql(sql).await?.show().await?;

    Ok(())
}
