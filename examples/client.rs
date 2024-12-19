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

use std::{collections::HashMap, sync::Arc};

use datafusion::{
    error::Result,
    prelude::{SessionConfig, SessionContext},
};
use datafusion_cache::{sql::USERNAME, SplitSqlTableFactory};

#[tokio::main]
pub async fn main() -> Result<()> {
    env_logger::builder().format_timestamp(None).init();

    let mut session_config = SessionConfig::from_env()?;
    session_config
        .options_mut()
        .execution
        .parquet
        .pushdown_filters = true;
    let ctx = Arc::new(SessionContext::new_with_config(session_config));

    let entry_point = "http://localhost:50051";
    let sql = r#"
    SELECT DISTINCT "URL" FROM small_hits WHERE "URL" <> '';
    "#;

    let table = SplitSqlTableFactory::open_table(
        entry_point,
        HashMap::from([(USERNAME.into(), "whatever".into())]),
        "small_hits",
    )
    .await?;
    ctx.register_table("small_hits", Arc::new(table))?;

    let df = ctx.sql(sql).await?;
    df.show().await?;

    Ok(())
}
