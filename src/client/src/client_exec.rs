use std::collections::HashMap;
use std::task::Poll;
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

use crate::{FlightSqlDriver, flight_channel, to_df_err};

/// The execution plan for the LiquidCache client.
#[derive(Debug)]
pub struct LiquidCacheClientExec {
    remote_plan: Arc<dyn ExecutionPlan>,
    cache_server: String,
    driver: Arc<FlightSqlDriver>,
    plan_register_lock: Arc<Mutex<Option<Uuid>>>,
    cache_mode: CacheMode,
    object_stores: Vec<(ObjectStoreUrl, HashMap<String, String>)>,
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
            driver: Arc::new(FlightSqlDriver {}),
            plan_register_lock: Arc::new(Mutex::new(None)),
            cache_mode,
            object_stores,
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
            driver: self.driver.clone(),
            plan_register_lock: self.plan_register_lock.clone(),
            cache_mode: self.cache_mode,
            object_stores: self.object_stores.clone(),
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
}

impl Stream for FlightStream {
    type Item = Result<RecordBatch>;

    fn poll_next(
        mut self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Option<Self::Item>> {
        loop {
            match &mut self.state {
                FlightStreamState::Init => {
                    self.state = FlightStreamState::GetStream(self.future_stream.take().unwrap());
                    continue;
                }
                FlightStreamState::GetStream(fut) => {
                    let stream = ready!(fut.as_mut().poll(cx)).unwrap();
                    self.state = FlightStreamState::Processing(stream);
                    continue;
                }
                FlightStreamState::Processing(stream) => {
                    let result = ready!(stream.as_mut().poll_next(cx));
                    match result {
                        Some(Ok(batch)) => {
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
                            return Poll::Ready(Some(Ok(coerced_batch)));
                        }
                        None => return Poll::Ready(None),
                        Some(Err(e)) => {
                            panic!("Error in flight stream: {:?}", e);
                        }
                    }
                }
            }
        }
    }
}

impl RecordBatchStream for FlightStream {
    fn schema(&self) -> SchemaRef {
        self.schema.clone()
    }
}
