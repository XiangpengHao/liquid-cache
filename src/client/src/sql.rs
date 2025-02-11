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

use crate::{FlightMetadata, FlightProperties};
use arrow_flight::Action;
use arrow_flight::sql::ProstMessageExt;
use arrow_flight::sql::client::FlightSqlServiceClient;
use arrow_flight::{error::Result, sql::CommandGetDbSchemas};
use liquid_common::ParquetMode;
use liquid_common::rpc::{RegisterTableRequest, LiquidCacheActions};
use prost::Message;
use std::collections::HashMap;
use tonic::Request;
use tonic::transport::Channel;

/// Default Flight SQL driver.
#[derive(Clone, Debug, Default)]
pub(crate) struct FlightSqlDriver {}

impl FlightSqlDriver {
    /// Returns a [FlightMetadata] from the specified channel,
    /// according to the provided table options.
    pub async fn metadata(
        &self,
        channel: Channel,
        table_name: &str,
        table_url: &str,
        parquet_mode: ParquetMode,
    ) -> Result<FlightMetadata> {
        let mut client = FlightSqlServiceClient::new(channel);

        {
            let register_table_request = RegisterTableRequest {
                url: table_url.to_string(),
                table_name: table_name.to_string(),
                table_provider: parquet_mode.to_string(),
            };
            let action = Action {
                r#type: LiquidCacheActions::RegisterTable.to_string(),
                body: register_table_request.as_any().encode_to_vec().into(),
            };
            client.do_action(Request::new(action)).await?;
        }

        let request = CommandGetDbSchemas {
            db_schema_filter_pattern: Some(table_name.to_string()),
            ..Default::default()
        };

        let info = client.get_db_schemas(request).await?;
        let mut grpc_headers = HashMap::default();
        if let Some(token) = client.token() {
            grpc_headers.insert("authorization".into(), format!("Bearer {}", token));
        }
        FlightMetadata::try_new(info, FlightProperties::default().grpc_headers(grpc_headers))
    }

    pub async fn run_sql(&self, channel: Channel, sql: &str) -> Result<FlightMetadata> {
        let mut client = FlightSqlServiceClient::new(channel);

        let info = client.execute(sql.to_string(), None).await?;

        FlightMetadata::try_new(info, FlightProperties::default())
    }
}
