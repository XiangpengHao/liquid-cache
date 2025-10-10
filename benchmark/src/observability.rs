use datafusion::arrow::array::RecordBatch;
use datafusion::arrow::datatypes::SchemaRef;
use datafusion::error::Result;
use datafusion::execution::RecordBatchStream;
use datafusion::execution::SendableRecordBatchStream;
use datafusion::physical_plan::DisplayAs;
use datafusion::physical_plan::DisplayFormatType;
use datafusion::physical_plan::ExecutionPlan;
use fastrace::Span;
use fastrace_futures::InSpan;
use fastrace_futures::StreamExt as _;
use fastrace_opentelemetry::OpenTelemetryReporter;
use futures::Stream;
use futures::StreamExt;
use logforth::filter::EnvFilter;
use opentelemetry::InstrumentationScope;
use opentelemetry::KeyValue;
use opentelemetry_otlp::SpanExporter;
use opentelemetry_otlp::WithExportConfig;
use opentelemetry_sdk::Resource;
use std::any::Any;
use std::borrow::Cow;
use std::sync::Arc;

pub(crate) struct TracedExecutionPlan {
    plan: Arc<dyn ExecutionPlan>,
    span: Arc<fastrace::Span>,
}

impl std::fmt::Debug for TracedExecutionPlan {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "TracedExecutionPlan {{ plan: {:?} }}", self.plan)
    }
}

impl TracedExecutionPlan {
    pub fn new(plan: Arc<dyn ExecutionPlan>, span: fastrace::Span) -> Self {
        Self {
            plan,
            span: Arc::new(span),
        }
    }
}

impl DisplayAs for TracedExecutionPlan {
    fn fmt_as(&self, t: DisplayFormatType, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.plan.fmt_as(t, f)
    }
}

impl ExecutionPlan for TracedExecutionPlan {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn name(&self) -> &str {
        "traced_execution_plan"
    }

    fn properties(&self) -> &datafusion::physical_plan::PlanProperties {
        self.plan.properties()
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        vec![&self.plan]
    }

    fn with_new_children(
        self: Arc<Self>,
        children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> datafusion::error::Result<Arc<dyn ExecutionPlan>> {
        Ok(Arc::new(Self {
            plan: children.first().unwrap().clone(),
            span: self.span.clone(),
        }))
    }

    fn execute(
        &self,
        partition: usize,
        context: Arc<datafusion::execution::TaskContext>,
    ) -> datafusion::error::Result<datafusion::execution::SendableRecordBatchStream> {
        let span = Span::enter_with_parent("execute", &self.span);
        let stream = self.plan.execute(partition, context)?;
        Ok(Box::pin(TracedSendableRecordBatchStream::new(stream, span)))
    }
}

struct TracedSendableRecordBatchStream {
    stream: InSpan<SendableRecordBatchStream>,
    schema: SchemaRef,
}

impl TracedSendableRecordBatchStream {
    pub fn new(stream: SendableRecordBatchStream, span: fastrace::Span) -> Self {
        let schema = stream.schema();
        let stream = stream.in_span(span);
        Self { stream, schema }
    }
}

impl RecordBatchStream for TracedSendableRecordBatchStream {
    fn schema(&self) -> SchemaRef {
        self.schema.clone()
    }
}

impl Stream for TracedSendableRecordBatchStream {
    type Item = Result<RecordBatch>;

    fn poll_next(
        mut self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Option<Self::Item>> {
        self.stream.poll_next_unpin(cx)
    }
}

pub fn setup_observability(service_name: &str, jaeger_endpoint: Option<&str>) {
    let mut builder = logforth::builder();
    builder = builder.dispatch(|d| {
        d.filter(EnvFilter::from_default_env())
            .append(logforth::append::Stdout::default())
    });
    builder.apply();

    let endpoint = jaeger_endpoint
        .map(|s| s.to_string())
        .or_else(|| std::env::var("LIQUIDCACHE_JAEGER_ENDPOINT").ok())
        .or_else(|| std::env::var("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT").ok());

    let Some(endpoint) = endpoint else {
        return;
    };

    let scope = InstrumentationScope::builder(env!("CARGO_PKG_NAME"))
        .with_version(env!("CARGO_PKG_VERSION"))
        .build();
    let trace_exporter = OpenTelemetryReporter::new(
        SpanExporter::builder()
            .with_tonic()
            .with_endpoint(endpoint)
            .with_protocol(opentelemetry_otlp::Protocol::Grpc)
            .build()
            .expect("initialize otlp exporter"),
        Cow::Owned(
            Resource::builder()
                .with_attributes([KeyValue::new("service.name", service_name.to_string())])
                .build(),
        ),
        scope,
    );
    fastrace::set_reporter(trace_exporter, fastrace::collector::Config::default());
}
