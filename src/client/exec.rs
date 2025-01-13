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

//! Execution plan for reading flights from Arrow Flight services

use std::any::Any;
use std::fmt::{Debug, Formatter};
use std::pin::Pin;
use std::str::FromStr;
use std::sync::Arc;
use std::task::{ready, Context, Poll};

use crate::client::metrics::{FlightStreamMetrics, FlightTableMetrics};
use crate::client::{flight_channel, to_df_err, FlightMetadata, FlightProperties};
use arrow_array::RecordBatch;
use arrow_flight::flight_service_client::FlightServiceClient;
use arrow_flight::{FlightClient, FlightEndpoint, Ticket};
use arrow_schema::SchemaRef;
use datafusion::config::ConfigOptions;
use datafusion::datasource::schema_adapter::{DefaultSchemaAdapterFactory, SchemaMapper};
use datafusion_common::arrow::datatypes::ToByteSlice;
use datafusion_common::project_schema;
use datafusion_common::Result;
use datafusion_execution::{RecordBatchStream, SendableRecordBatchStream, TaskContext};
use datafusion_physical_expr::{EquivalenceProperties, Partitioning};
use datafusion_physical_plan::execution_plan::{Boundedness, EmissionType};
use datafusion_physical_plan::metrics::{ExecutionPlanMetricsSet, MetricsSet};
use datafusion_physical_plan::stream::RecordBatchStreamAdapter;
use datafusion_physical_plan::{DisplayAs, DisplayFormatType, ExecutionPlan, PlanProperties};
use futures::future::BoxFuture;
use futures::{Stream, TryStreamExt};
use serde::{Deserialize, Serialize};
use tonic::metadata::{AsciiMetadataKey, MetadataMap};

/// Arrow Flight physical plan that maps flight endpoints to partitions
#[derive(Clone, Debug)]
pub struct FlightExec {
    config: FlightConfig,
    plan_properties: PlanProperties,
    metadata_map: Arc<MetadataMap>,
    metrics: ExecutionPlanMetricsSet,
    flight_schema: SchemaRef,
}

impl FlightExec {
    /// Creates a FlightExec with the provided [FlightMetadata]
    /// and origin URL (used as fallback location as per the protocol spec).
    pub(crate) fn try_new(
        flight_schema: SchemaRef,
        output_schema: SchemaRef,
        metadata: FlightMetadata,
        projection: Option<&Vec<usize>>,
        origin: &str,
        limit: Option<usize>,
    ) -> Result<Self> {
        let partitions = metadata
            .info
            .endpoint
            .iter()
            .map(|endpoint| FlightPartition::new(endpoint, origin.to_string()))
            .collect();
        let schema = project_schema(&output_schema, projection).expect("Error projecting schema");
        let config = FlightConfig {
            origin: origin.into(),
            schema,
            partitions,
            properties: metadata.props,
            limit,
        };
        let boundedness = if config.properties.unbounded_stream {
            Boundedness::Unbounded {
                requires_infinite_memory: false,
            }
        } else {
            Boundedness::Bounded
        };

        let plan_properties = PlanProperties::new(
            EquivalenceProperties::new(config.schema.clone()),
            Partitioning::UnknownPartitioning(config.partitions.len()),
            EmissionType::Incremental,
            boundedness,
        );
        let mut mm = MetadataMap::new();
        for (k, v) in config.properties.grpc_headers.iter() {
            let key = AsciiMetadataKey::from_str(k.as_str()).expect("invalid header name");
            let value = v.parse().expect("invalid header value");
            mm.insert(key, value);
        }

        let metrics = ExecutionPlanMetricsSet::new();

        Ok(Self {
            config,
            plan_properties,
            flight_schema,
            metadata_map: Arc::from(mm),
            metrics,
        })
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub(crate) struct FlightConfig {
    origin: String,
    schema: SchemaRef,
    partitions: Arc<[FlightPartition]>,
    properties: FlightProperties,
    limit: Option<usize>,
}

/// The minimum information required for fetching a flight stream.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
struct FlightPartition {
    locations: String,
    ticket: FlightTicket,
}

#[derive(Clone, Deserialize, Eq, PartialEq, Serialize)]
struct FlightTicket(Arc<[u8]>);

impl From<Option<&Ticket>> for FlightTicket {
    fn from(ticket: Option<&Ticket>) -> Self {
        let bytes = match ticket {
            Some(t) => t.ticket.to_byte_slice().into(),
            None => [].into(),
        };
        Self(bytes)
    }
}

impl Debug for FlightTicket {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "[..{} bytes..]", self.0.len())
    }
}

impl FlightPartition {
    fn new(endpoint: &FlightEndpoint, fallback_location: String) -> Self {
        let locations = if endpoint.location.is_empty() {
            fallback_location
        } else {
            endpoint
                .location
                .iter()
                .map(|loc| {
                    if loc.uri.starts_with("arrow-flight-reuse-connection://") {
                        fallback_location.clone()
                    } else {
                        loc.uri.clone()
                    }
                })
                .collect()
        };

        Self {
            locations,
            ticket: endpoint.ticket.as_ref().into(),
        }
    }
}

async fn flight_client(
    source: impl Into<String>,
    grpc_headers: &MetadataMap,
) -> Result<FlightClient> {
    let channel = flight_channel(source).await?;
    let inner_client = FlightServiceClient::new(channel);
    let mut client = FlightClient::new_from_inner(inner_client);
    client.metadata_mut().clone_from(grpc_headers);
    Ok(client)
}

async fn flight_stream(
    partition: FlightPartition,
    schema: SchemaRef,
    grpc_headers: Arc<MetadataMap>,
    mut flight_metrics: FlightTableMetrics,
) -> Result<SendableRecordBatchStream> {
    flight_metrics.time_creating_client.start();
    let mut client = flight_client(partition.locations, grpc_headers.as_ref()).await?;
    flight_metrics.time_creating_client.stop();

    let ticket = Ticket::new(partition.ticket.0.to_vec());
    flight_metrics.time_getting_stream.start();
    let stream = client.do_get(ticket).await.unwrap().map_err(to_df_err);
    flight_metrics.time_getting_stream.stop();

    Ok(Box::pin(RecordBatchStreamAdapter::new(
        schema.clone(),
        stream,
    )))
}

impl DisplayAs for FlightExec {
    fn fmt_as(&self, t: DisplayFormatType, f: &mut Formatter) -> std::fmt::Result {
        match t {
            DisplayFormatType::Default => write!(
                f,
                "FlightExec: origin={}, streams={}",
                self.config.origin,
                self.config.partitions.len()
            ),
            DisplayFormatType::Verbose => write!(
                f,
                "FlightExec: origin={}, streams={}, properties={:?}",
                self.config.origin,
                self.config.partitions.len(),
                self.config.properties,
            ),
        }
    }
}

impl ExecutionPlan for FlightExec {
    fn name(&self) -> &str {
        "FlightExec"
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn properties(&self) -> &PlanProperties {
        &self.plan_properties
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        vec![]
    }

    fn with_new_children(
        self: Arc<Self>,
        _children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        Ok(self)
    }

    fn execute(
        &self,
        partition: usize,
        _context: Arc<TaskContext>,
    ) -> Result<SendableRecordBatchStream> {
        let flight_metrics = FlightTableMetrics::new(&self.metrics, partition);
        let future_stream = flight_stream(
            self.config.partitions[partition].clone(),
            self.schema(),
            self.metadata_map.clone(),
            flight_metrics,
        );
        let stream_metrics = FlightStreamMetrics::new(&self.metrics, partition);
        let (schema_adapter, _) = DefaultSchemaAdapterFactory::from_schema(self.schema())
            .map_schema(self.flight_schema.as_ref())
            .unwrap();
        Ok(Box::pin(FlightStream {
            metrics: stream_metrics,
            _partition: partition,
            state: FlightStreamState::Init,
            future_stream: Some(Box::pin(future_stream)),
            schema: self.schema(),
            schema_mapper: schema_adapter,
        }))
    }

    fn fetch(&self) -> Option<usize> {
        self.config.limit
    }

    fn repartitioned(
        &self,
        _target_partitions: usize,
        _config: &ConfigOptions,
    ) -> Result<Option<Arc<dyn ExecutionPlan>>> {
        let mut new_plan = self.clone();
        new_plan.plan_properties.partitioning =
            Partitioning::UnknownPartitioning(self.config.partitions.len());
        Ok(Some(Arc::new(new_plan)))
    }

    fn metrics(&self) -> Option<MetricsSet> {
        Some(self.metrics.clone_inner())
    }
}

enum FlightStreamState {
    Init,
    GetStream(BoxFuture<'static, Result<SendableRecordBatchStream>>),
    Processing(SendableRecordBatchStream),
}

struct FlightStream {
    metrics: FlightStreamMetrics,
    _partition: usize,
    state: FlightStreamState,
    future_stream: Option<BoxFuture<'static, Result<SendableRecordBatchStream>>>,
    schema: SchemaRef,
    schema_mapper: Arc<dyn SchemaMapper>,
}

impl FlightStream {
    fn poll_inner(&mut self, cx: &mut Context<'_>) -> Poll<Option<Result<RecordBatch>>> {
        loop {
            match &mut self.state {
                FlightStreamState::Init => {
                    self.metrics.time_reading_total.start();
                    self.state = FlightStreamState::GetStream(self.future_stream.take().unwrap());
                    continue;
                }
                FlightStreamState::GetStream(fut) => {
                    let stream = ready!(fut.as_mut().poll(cx)).unwrap();
                    self.state = FlightStreamState::Processing(stream);
                    continue;
                }
                FlightStreamState::Processing(stream) => {
                    let result = stream.as_mut().poll_next(cx);
                    self.metrics.poll_count.add(1);
                    return result;
                }
            }
        }
    }
}

impl Stream for FlightStream {
    type Item = Result<RecordBatch>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        self.metrics.time_processing.start();
        let result = self.poll_inner(cx);
        self.metrics.time_processing.stop();
        match result {
            Poll::Ready(Some(Ok(batch))) => {
                self.metrics.output_rows.add(batch.num_rows());
                self.metrics
                    .bytes_transferred
                    .add(batch.get_array_memory_size());
                let new_batch = self.schema_mapper.map_batch(batch).unwrap();
                Poll::Ready(Some(Ok(new_batch)))
            }
            Poll::Ready(None) | Poll::Ready(Some(Err(_))) => {
                self.metrics.time_reading_total.stop();
                Poll::Ready(None)
            }
            _ => Poll::Pending,
        }
    }
}

impl RecordBatchStream for FlightStream {
    fn schema(&self) -> SchemaRef {
        self.schema.clone()
    }
}
