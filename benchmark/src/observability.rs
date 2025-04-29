use fastrace_opentelemetry::OpenTelemetryReporter;
use logforth::append::opentelemetry::OpentelemetryLogBuilder;
use logforth::filter::EnvFilter;
use opentelemetry::InstrumentationScope;
use opentelemetry::KeyValue;
use opentelemetry::trace::SpanKind;
use opentelemetry_otlp::LogExporter;
use opentelemetry_otlp::WithExportConfig;
use opentelemetry_otlp::{SpanExporter, WithTonicConfig};
use opentelemetry_sdk::Resource;
use std::borrow::Cow;
use tonic::metadata::MetadataMap;

fn otl_metadata(auth: &str) -> MetadataMap {
    let mut map = MetadataMap::with_capacity(3);
    map.insert("authorization", format!("Basic {auth}").parse().unwrap());
    map.insert("organization", "default".parse().unwrap());
    map.insert("stream-name", "default".parse().unwrap());
    map
}

pub fn setup_observability(service_name: &str, kind: SpanKind, auth: Option<&str>) {
    let Some(auth) = auth else {
        logforth::builder()
            .dispatch(|d| {
                d.filter(EnvFilter::from_default_env())
                    .append(logforth::append::Stdout::default())
            })
            .apply();
        return;
    };

    // Setup logging with logforth
    let log_exporter = LogExporter::builder()
        .with_tonic()
        .with_endpoint("http://localhost:5081/api/development".to_string())
        .with_protocol(opentelemetry_otlp::Protocol::Grpc)
        .with_metadata(otl_metadata(auth))
        .build()
        .unwrap();
    logforth::builder()
        .dispatch(|d| {
            d.filter(EnvFilter::from_default_env())
                .append(logforth::append::Stdout::default())
        })
        .dispatch(|d| {
            let otl_appender = OpentelemetryLogBuilder::new(service_name, log_exporter)
                .build()
                .unwrap();
            d.filter(EnvFilter::from_default_env()).append(otl_appender)
        })
        .apply();

    let trace_exporter = OpenTelemetryReporter::new(
        SpanExporter::builder()
            .with_tonic()
            .with_endpoint("http://localhost:5081/api/development".to_string())
            .with_metadata(otl_metadata(auth))
            .with_protocol(opentelemetry_otlp::Protocol::Grpc)
            .build()
            .expect("initialize oltp exporter"),
        kind,
        Cow::Owned(
            Resource::builder()
                .with_attributes([KeyValue::new("service.name", service_name.to_string())])
                .build(),
        ),
        InstrumentationScope::builder(env!("CARGO_PKG_NAME"))
            .with_version(env!("CARGO_PKG_VERSION"))
            .build(),
    );
    fastrace::set_reporter(trace_exporter, fastrace::collector::Config::default());
}
