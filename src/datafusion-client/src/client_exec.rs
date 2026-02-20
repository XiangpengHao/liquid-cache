use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::task::{Context, Poll};
use std::time::Duration;
use std::{any::Any, fmt::Formatter, sync::Arc};

use arrow::array::RecordBatch;
use arrow_flight::decode::FlightRecordBatchStream;
use arrow_flight::flight_service_client::FlightServiceClient;
use arrow_schema::SchemaRef;
use datafusion::common::Statistics;
use datafusion::config::ConfigOptions;
use datafusion::execution::object_store::ObjectStoreUrl;
use datafusion::physical_expr_adapter::{BatchAdapter, BatchAdapterFactory};
use datafusion::physical_plan::execution_plan::CardinalityEffect;
use datafusion::physical_plan::filter_pushdown::{
    ChildPushdownResult, FilterDescription, FilterPushdownPhase, FilterPushdownPropagation,
};
use datafusion::physical_plan::metrics::{ExecutionPlanMetricsSet, MetricsSet};
use datafusion::physical_plan::{ExecutionPlanProperties, PhysicalExpr, PlanProperties};
use datafusion::{
    error::Result,
    execution::{RecordBatchStream, SendableRecordBatchStream},
    physical_plan::{
        DisplayAs, DisplayFormatType, ExecutionPlan, stream::RecordBatchStreamAdapter,
    },
};
use datafusion_proto::bytes::physical_plan_to_bytes;
use fastrace::Span;
use fastrace::future::FutureExt;
use fastrace::prelude::*;
use futures::{Stream, TryStreamExt, future::BoxFuture, ready};
use liquid_cache_common::rpc::{
    FetchResults, LiquidCacheActions, RegisterObjectStoreRequest, RegisterPlanRequest,
};
use tonic::Request;
use uuid::Uuid;

use crate::metrics::FlightStreamMetrics;
use crate::{flight_channel, to_df_err};

#[repr(usize)]
enum PlanRegisterState {
    NotRegistered = 0,
    InProgress = 1,
    Registered = 2,
}

/// The execution plan for the LiquidCache client.
pub struct LiquidCacheClientExec {
    remote_plan: Arc<dyn ExecutionPlan>,
    cache_server: String,
    object_stores: Vec<(ObjectStoreUrl, HashMap<String, String>)>,
    metrics: ExecutionPlanMetricsSet,
    uuid: Uuid,
    plan_registered: Arc<AtomicUsize>,
    properties: PlanProperties,
}

impl std::fmt::Debug for LiquidCacheClientExec {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "LiquidCacheClientExec")
    }
}

impl LiquidCacheClientExec {
    pub(crate) fn new(
        remote_plan: Arc<dyn ExecutionPlan>,
        cache_server: String,
        object_stores: Vec<(ObjectStoreUrl, HashMap<String, String>)>,
    ) -> Self {
        let properties = PlanProperties::new(
            remote_plan.equivalence_properties().clone(), // Equivalence Properties
            remote_plan.output_partitioning().clone(),    // Output Partitioning
            remote_plan.pipeline_behavior(),
            remote_plan.boundedness(),
        );
        let uuid = Uuid::new_v4();
        Self {
            remote_plan,
            cache_server,
            plan_registered: Arc::new(AtomicUsize::new(PlanRegisterState::NotRegistered as usize)),
            object_stores,
            uuid,
            metrics: ExecutionPlanMetricsSet::new(),
            properties,
        }
    }

    /// Get the UUID of the plan.
    pub fn get_uuid(&self) -> Uuid {
        self.uuid
    }
}

impl DisplayAs for LiquidCacheClientExec {
    fn fmt_as(&self, t: DisplayFormatType, f: &mut Formatter<'_>) -> std::fmt::Result {
        match t {
            DisplayFormatType::Default | DisplayFormatType::Verbose => {
                write!(
                    f,
                    "LiquidCacheClientExec: server={}, object_stores={:?}",
                    self.cache_server, self.object_stores
                )
            }
            DisplayFormatType::TreeRender => {
                write!(
                    f,
                    "server={}, object_stores={:?}",
                    self.cache_server, self.object_stores
                )
            }
        }
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
            plan_registered: self.plan_registered.clone(),
            object_stores: self.object_stores.clone(),
            metrics: self.metrics.clone(),
            uuid: self.uuid,
            properties: self.properties.clone(),
        }))
    }

    fn execute(
        &self,
        partition: usize,
        context: Arc<datafusion::execution::TaskContext>,
    ) -> datafusion::error::Result<datafusion::execution::SendableRecordBatchStream> {
        let cache_server = self.cache_server.clone();
        let plan = self.remote_plan.clone();
        let lock = self.plan_registered.clone();
        let stream_metrics = FlightStreamMetrics::new(&self.metrics, partition);

        let span = context
            .session_config()
            .get_extension::<Span>()
            .unwrap_or_default();
        let exec_span = Span::enter_with_parent("exec_flight_stream", &span);
        let create_stream_span = Span::enter_with_parent("create_flight_stream", &exec_span);
        let stream = flight_stream(
            cache_server,
            plan,
            lock,
            self.uuid,
            partition,
            self.object_stores.clone(),
        );
        Ok(Box::pin(FlightStream::new(
            Some(Box::pin(stream)),
            self.remote_plan.schema().clone(),
            stream_metrics,
            exec_span,
            create_stream_span,
        )))
    }

    fn statistics(&self) -> Result<Statistics> {
        self.remote_plan.partition_statistics(None)
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

    fn metrics(&self) -> Option<MetricsSet> {
        Some(self.metrics.clone_inner())
    }

    fn maintains_input_order(&self) -> Vec<bool> {
        vec![true]
    }

    fn benefits_from_input_partitioning(&self) -> Vec<bool> {
        vec![false]
    }

    fn cardinality_effect(&self) -> CardinalityEffect {
        CardinalityEffect::Equal
    }

    fn gather_filters_for_pushdown(
        &self,
        _phase: FilterPushdownPhase,
        parent_filters: Vec<Arc<dyn PhysicalExpr>>,
        _config: &ConfigOptions,
    ) -> Result<FilterDescription> {
        FilterDescription::from_children(parent_filters, &self.children())
    }

    fn handle_child_pushdown_result(
        &self,
        _phase: FilterPushdownPhase,
        child_pushdown_result: ChildPushdownResult,
        _config: &ConfigOptions,
    ) -> Result<FilterPushdownPropagation<Arc<dyn ExecutionPlan>>> {
        Ok(FilterPushdownPropagation::if_all(child_pushdown_result))
    }
}

async fn flight_stream(
    server: String,
    plan: Arc<dyn ExecutionPlan>,
    plan_registered: Arc<AtomicUsize>,
    handle: Uuid,
    partition: usize,
    object_stores: Vec<(ObjectStoreUrl, HashMap<String, String>)>,
) -> Result<SendableRecordBatchStream> {
    let channel = flight_channel(server)
        .in_span(Span::enter_with_local_parent("connect_channel"))
        .await?;

    let mut client = FlightServiceClient::new(channel).max_decoding_message_size(1024 * 1024 * 8);
    let schema = plan.schema().clone();

    match plan_registered.compare_exchange(
        PlanRegisterState::NotRegistered as usize,
        PlanRegisterState::InProgress as usize,
        Ordering::AcqRel,
        Ordering::Relaxed,
    ) {
        Ok(_) => {
            LocalSpan::add_event(Event::new("register_plan"));

            for (url, options) in &object_stores {
                let action = LiquidCacheActions::RegisterObjectStore(RegisterObjectStoreRequest {
                    url: url.to_string(),
                    options: options.clone(),
                })
                .into();
                client
                    .do_action(Request::new(action))
                    .await
                    .map_err(to_df_err)?;
            }
            // Register plan
            let plan_bytes = physical_plan_to_bytes(plan)?;
            let action = LiquidCacheActions::RegisterPlan(RegisterPlanRequest {
                plan: plan_bytes.to_vec(),
                handle: handle.into_bytes().to_vec().into(),
            })
            .into();
            client
                .do_action(Request::new(action))
                .await
                .map_err(to_df_err)?;
            plan_registered.store(PlanRegisterState::Registered as usize, Ordering::Release);
            LocalSpan::add_event(Event::new("register_plan_done"));
        }
        Err(_e) => {
            LocalSpan::add_event(Event::new("getting_existing_plan"));
            while plan_registered.load(Ordering::Acquire) != PlanRegisterState::Registered as usize
            {
                tokio::time::sleep(Duration::from_micros(100)).await;
            }
            LocalSpan::add_event(Event::new("got_existing_plan"));
        }
    };

    let current = SpanContext::current_local_parent().unwrap_or_else(SpanContext::random);

    let fetch_results = FetchResults {
        handle: handle.into_bytes().to_vec().into(),
        partition: partition as u32,
        traceparent: current.encode_w3c_traceparent(),
    };
    let ticket = fetch_results.into_ticket();
    let (md, response_stream, _ext) = client.do_get(ticket).await.map_err(to_df_err)?.into_parts();
    LocalSpan::add_event(Event::new("get_flight_stream"));
    let stream =
        FlightRecordBatchStream::new_from_flight_data(response_stream.map_err(|e| e.into()))
            .with_headers(md)
            .map_err(to_df_err);
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
    batch_adapter: Option<BatchAdapter>,
    metrics: FlightStreamMetrics,
    poll_stream_span: fastrace::Span,
    create_stream_span: Option<fastrace::Span>,
}

impl FlightStream {
    fn new(
        future_stream: Option<BoxFuture<'static, Result<SendableRecordBatchStream>>>,
        schema: SchemaRef,
        metrics: FlightStreamMetrics,
        poll_stream_span: fastrace::Span,
        create_stream_span: fastrace::Span,
    ) -> Self {
        Self {
            future_stream,
            state: FlightStreamState::Init,
            schema,
            batch_adapter: None,
            metrics,
            poll_stream_span,
            create_stream_span: Some(create_stream_span),
        }
    }
}

use futures::StreamExt;
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
                    let _guard = self.create_stream_span.as_ref().unwrap().set_local_parent();
                    let stream = ready!(fut.as_mut().poll(cx)).unwrap();
                    self.create_stream_span.take();
                    self.state = FlightStreamState::Processing(stream);
                    continue;
                }
                FlightStreamState::Processing(stream) => {
                    let result = stream.poll_next_unpin(cx);
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
        let _guard = self.poll_stream_span.set_local_parent();
        self.metrics.time_processing.start();
        let result = self.poll_inner(cx);
        match result {
            Poll::Ready(Some(Ok(batch))) => {
                let coerced_batch = if let Some(adapter) = &self.batch_adapter {
                    match adapter.adapt_batch(&batch) {
                        Ok(batch) => batch,
                        Err(e) => return Poll::Ready(Some(Err(e))),
                    }
                } else {
                    let adapter = match BatchAdapterFactory::new(self.schema.clone())
                        .make_adapter(batch.schema())
                    {
                        Ok(adapter) => adapter,
                        Err(e) => return Poll::Ready(Some(Err(e))),
                    };
                    let batch = match adapter.adapt_batch(&batch) {
                        Ok(batch) => batch,
                        Err(e) => return Poll::Ready(Some(Err(e))),
                    };
                    self.batch_adapter = Some(adapter);
                    batch
                };
                self.metrics.output_rows.add(coerced_batch.num_rows());
                self.metrics
                    .bytes_decoded
                    .add(coerced_batch.get_array_memory_size());
                self.metrics.time_processing.stop();
                LocalSpan::add_event(Event::new("emit_batch"));
                Poll::Ready(Some(Ok(coerced_batch)))
            }
            Poll::Ready(None) => {
                self.metrics.time_processing.stop();
                self.metrics.time_reading_total.stop();
                Poll::Ready(None)
            }
            Poll::Ready(Some(Err(e))) => {
                panic!("Error in flight stream: {e:?}");
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
