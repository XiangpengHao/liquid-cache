use std::collections::HashMap;
use std::task::{Context, Poll};
use std::{any::Any, fmt::Formatter, sync::Arc};

use arrow::array::RecordBatch;
use arrow_flight::sql::client::FlightSqlServiceClient;
use arrow_schema::SchemaRef;
use datafusion::common::Statistics;
use datafusion::config::ConfigOptions;
use datafusion::datasource::schema_adapter::{DefaultSchemaAdapterFactory, SchemaMapper};
use datafusion::execution::object_store::ObjectStoreUrl;
use datafusion::physical_plan::Distribution;
use datafusion::physical_plan::execution_plan::CardinalityEffect;
use datafusion::physical_plan::metrics::{ExecutionPlanMetricsSet, MetricsSet};
use datafusion::physical_plan::projection::ProjectionExec;
use datafusion::{
    error::Result,
    execution::{RecordBatchStream, SendableRecordBatchStream},
    physical_plan::{
        DisplayAs, DisplayFormatType, ExecutionPlan, stream::RecordBatchStreamAdapter,
    },
};
use datafusion_proto::bytes::physical_plan_to_bytes;
use futures::{Stream, TryStreamExt, future::BoxFuture, lock::Mutex, ready};
use liquid_cache_common::CacheMode;
use liquid_cache_common::rpc::{
    FetchResults, LiquidCacheActions, RegisterObjectStoreRequest, RegisterPlanRequest,
};
use tonic::Request;
use uuid::Uuid;

use crate::metrics::FlightStreamMetrics;
use crate::{flight_channel, to_df_err};

/// The execution plan for the LiquidCache client.
#[derive(Debug)]
pub struct LiquidCacheClientExec {
    remote_plan: Arc<dyn ExecutionPlan>,
    cache_server: String,
    plan_register_lock: Arc<Mutex<Option<Uuid>>>,
    cache_mode: CacheMode,
    object_stores: Vec<(ObjectStoreUrl, HashMap<String, String>)>,
    metrics: ExecutionPlanMetricsSet,
}

impl LiquidCacheClientExec {
    pub(crate) fn new(
        remote_plan: Arc<dyn ExecutionPlan>,
        cache_server: String,
        cache_mode: CacheMode,
        object_stores: Vec<(ObjectStoreUrl, HashMap<String, String>)>,
    ) -> Self {
        Self {
            remote_plan,
            cache_server,
            plan_register_lock: Arc::new(Mutex::new(None)),
            cache_mode,
            object_stores,
            metrics: ExecutionPlanMetricsSet::new(),
        }
    }

    /// Get the UUID of the plan.
    pub async fn get_plan_uuid(&self) -> Option<Uuid> {
        *self.plan_register_lock.lock().await
    }
}

impl DisplayAs for LiquidCacheClientExec {
    fn fmt_as(&self, _t: DisplayFormatType, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "LiquidCacheClientExec")
    }
}

impl ExecutionPlan for LiquidCacheClientExec {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn name(&self) -> &str {
        "LiquidCacheClientExec"
    }

    fn properties(&self) -> &datafusion::physical_plan::PlanProperties {
        self.remote_plan.properties()
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        vec![&self.remote_plan]
    }

    fn with_new_children(
        self: Arc<Self>,
        children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> datafusion::error::Result<Arc<dyn ExecutionPlan>> {
        Ok(Arc::new(Self {
            remote_plan: children.first().unwrap().clone(),
            cache_server: self.cache_server.clone(),
            plan_register_lock: self.plan_register_lock.clone(),
            cache_mode: self.cache_mode,
            object_stores: self.object_stores.clone(),
            metrics: self.metrics.clone(),
        }))
    }

    fn execute(
        &self,
        partition: usize,
        _context: Arc<datafusion::execution::TaskContext>,
    ) -> datafusion::error::Result<datafusion::execution::SendableRecordBatchStream> {
        let cache_server = self.cache_server.clone();
        let plan = self.remote_plan.clone();
        let lock = self.plan_register_lock.clone();
        let stream_metrics = FlightStreamMetrics::new(&self.metrics, partition);
        Ok(Box::pin(FlightStream {
            future_stream: Some(Box::pin(flight_stream(
                cache_server,
                plan,
                lock,
                partition,
                self.object_stores.clone(),
                self.cache_mode,
            ))),
            state: FlightStreamState::Init,
            schema: self.remote_plan.schema().clone(),
            schema_mapper: None,
            metrics: stream_metrics,
        }))
    }

    fn required_input_distribution(&self) -> Vec<Distribution> {
        self.remote_plan.required_input_distribution()
    }

    fn benefits_from_input_partitioning(&self) -> Vec<bool> {
        self.remote_plan.benefits_from_input_partitioning()
    }

    fn repartitioned(
        &self,
        target_partitions: usize,
        config: &ConfigOptions,
    ) -> Result<Option<Arc<dyn ExecutionPlan>>> {
        self.remote_plan.repartitioned(target_partitions, config)
    }

    fn statistics(&self) -> Result<Statistics> {
        self.remote_plan.statistics()
    }

    fn supports_limit_pushdown(&self) -> bool {
        self.remote_plan.supports_limit_pushdown()
    }

    fn with_fetch(&self, limit: Option<usize>) -> Option<Arc<dyn ExecutionPlan>> {
        self.remote_plan.with_fetch(limit)
    }

    fn fetch(&self) -> Option<usize> {
        self.remote_plan.fetch()
    }

    fn cardinality_effect(&self) -> CardinalityEffect {
        self.remote_plan.cardinality_effect()
    }

    fn try_swapping_with_projection(
        &self,
        projection: &ProjectionExec,
    ) -> Result<Option<Arc<dyn ExecutionPlan>>> {
        self.remote_plan.try_swapping_with_projection(projection)
    }

    fn metrics(&self) -> Option<MetricsSet> {
        Some(self.metrics.clone_inner())
    }
}

async fn flight_stream(
    server: String,
    plan: Arc<dyn ExecutionPlan>,
    plan_register_lock: Arc<Mutex<Option<Uuid>>>,
    partition: usize,
    object_stores: Vec<(ObjectStoreUrl, HashMap<String, String>)>,
    cache_mode: CacheMode,
) -> Result<SendableRecordBatchStream> {
    let channel = flight_channel(server).await?;

    let mut client = FlightSqlServiceClient::new(channel);
    let schema = plan.schema().clone();

    // Only one partition needs to register the plan
    let handle = {
        let mut maybe_uuid = plan_register_lock.lock().await;
        match maybe_uuid.as_ref() {
            Some(uuid) => *uuid,
            None => {
                // Register object stores
                for (url, options) in &object_stores {
                    let action =
                        LiquidCacheActions::RegisterObjectStore(RegisterObjectStoreRequest {
                            url: url.to_string(),
                            options: options.clone(),
                        })
                        .into();
                    client.do_action(Request::new(action)).await?;
                }
                // Register plan
                let plan_bytes = physical_plan_to_bytes(plan)?;
                let handle = Uuid::new_v4();
                let action = LiquidCacheActions::RegisterPlan(RegisterPlanRequest {
                    plan: plan_bytes.to_vec(),
                    handle: handle.into_bytes().to_vec().into(),
                    cache_mode: cache_mode.to_string(),
                })
                .into();
                client.do_action(Request::new(action)).await?;
                *maybe_uuid = Some(handle);
                handle
            }
        }
    };

    let fetch_results = FetchResults {
        handle: handle.into_bytes().to_vec().into(),
        partition: partition as u32,
    };
    let ticket = fetch_results.into_ticket();
    let stream = client.do_get(ticket).await?.map_err(to_df_err);
    Ok(Box::pin(RecordBatchStreamAdapter::new(schema, stream)))
}

enum FlightStreamState {
    Init,
    GetStream(BoxFuture<'static, Result<SendableRecordBatchStream>>),
    Processing(SendableRecordBatchStream),
}

struct FlightStream {
    future_stream: Option<BoxFuture<'static, Result<SendableRecordBatchStream>>>,
    state: FlightStreamState,
    schema: SchemaRef,
    schema_mapper: Option<Arc<dyn SchemaMapper>>,
    metrics: FlightStreamMetrics,
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

    fn poll_next(
        mut self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Option<Self::Item>> {
        self.metrics.time_processing.start();
        let result = self.poll_inner(cx);
        match result {
            Poll::Ready(Some(Ok(batch))) => {
                let coerced_batch = if let Some(schema_mapper) = &self.schema_mapper {
                    schema_mapper.map_batch(batch).unwrap()
                } else {
                    let (schema_mapper, _) =
                        DefaultSchemaAdapterFactory::from_schema(self.schema.clone())
                            .map_schema(&batch.schema())
                            .unwrap();
                    let batch = schema_mapper.map_batch(batch).unwrap();

                    self.schema_mapper = Some(schema_mapper);
                    batch
                };
                self.metrics.output_rows.add(coerced_batch.num_rows());
                self.metrics
                    .bytes_decoded
                    .add(coerced_batch.get_array_memory_size());
                self.metrics.time_processing.stop();
                Poll::Ready(Some(Ok(coerced_batch)))
            }
            Poll::Ready(None) => {
                self.metrics.time_processing.stop();
                self.metrics.time_reading_total.stop();
                Poll::Ready(None)
            }
            Poll::Ready(Some(Err(e))) => {
                panic!("Error in flight stream: {:?}", e);
            }
            Poll::Pending => {
                self.metrics.time_processing.stop();
                Poll::Pending
            }
        }
    }
}

impl RecordBatchStream for FlightStream {
    fn schema(&self) -> SchemaRef {
        self.schema.clone()
    }
}
