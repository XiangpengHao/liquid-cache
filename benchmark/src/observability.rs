use datafusion::catalog::memory::DataSourceExec;
use datafusion::common::tree_node::Transformed;
use datafusion::common::tree_node::TreeNode;
use datafusion::common::tree_node::TreeNodeRecursion;
use datafusion::datasource::physical_plan::FileScanConfig;
use datafusion::datasource::source::DataSource;
use datafusion::physical_plan::ExecutionPlan;
use fastrace_opentelemetry::OpenTelemetryReporter;
use liquid_cache_parquet::LiquidParquetSource;
use logforth::filter::env_filter::EnvFilterBuilder;
use opentelemetry::InstrumentationScope;
use opentelemetry::KeyValue;
use opentelemetry_otlp::SpanExporter;
use opentelemetry_otlp::WithExportConfig;
use opentelemetry_sdk::Resource;
use std::borrow::Cow;
use std::sync::Arc;

pub fn instrument_liquid_source_with_span(
    plan: Arc<dyn ExecutionPlan>,
    span: fastrace::Span,
) -> Arc<dyn ExecutionPlan> {
    let rewritten = plan
        .transform_up(|node| {
            let Some(data_source) = node.as_any().downcast_ref::<DataSourceExec>() else {
                return Ok(Transformed::no(node));
            };
            let file_scan_config = data_source
                .data_source()
                .as_any()
                .downcast_ref::<FileScanConfig>()
                .expect("FileScanConfig not found");
            let mut new_config = file_scan_config.clone();
            let Some(liquid_source) = file_scan_config
                .file_source()
                .as_any()
                .downcast_ref::<LiquidParquetSource>()
            else {
                return Ok(Transformed::no(node));
            };
            let sub_span = fastrace::Span::enter_with_parent("execute", &span);
            let new_source = Arc::new(liquid_source.with_span(sub_span));
            new_config.file_source = new_source;
            let new_file_source: Arc<dyn DataSource> = Arc::new(new_config);
            let new_plan = Arc::new(DataSourceExec::new(new_file_source));
            Ok(Transformed::new(
                new_plan,
                true,
                TreeNodeRecursion::Continue,
            ))
        })
        .unwrap();
    rewritten.data
}

pub fn setup_observability(service_name: &str, jaeger_endpoint: Option<&str>) {
    logforth::starter_log::builder()
        .dispatch(|d| {
            d.filter(EnvFilterBuilder::from_default_env().build())
                .append(logforth::append::Stdout::default())
        })
        .apply();

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
