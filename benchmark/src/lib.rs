use arrow_flight::sql::Any;
use arrow_flight::{FlightClient, flight_service_client::FlightServiceClient};
use datafusion::arrow::array::RecordBatch;
use datafusion::common::tree_node::TreeNode;
use datafusion::execution::TaskContext;
use datafusion::physical_plan::collect;
use datafusion::{error::Result, physical_plan::ExecutionPlan};
use datafusion::{
    physical_plan::metrics::MetricValue,
    prelude::{SessionConfig, SessionContext},
};
use fastrace::Span;
use fastrace::future::FutureExt as _;
use futures::StreamExt;
use liquid_cache_client::{LiquidCacheBuilder, LiquidCacheClientExec};
use liquid_cache_common::CacheMode;
use liquid_cache_common::rpc::{
    ExecutionMetricsRequest, ExecutionMetricsResponse, LiquidCacheActions,
};
use object_store::ClientConfigKey;
use prost::Message;
use serde::Serialize;
use std::time::Duration;
use std::{fmt::Display, path::Path, str::FromStr, sync::Arc};
use tonic::metadata::MetadataMap;
use tonic::transport::Channel;
use url::Url;

mod reports;
pub mod utils;

pub use reports::*;

#[derive(Clone, Debug, Default, Copy, PartialEq, Eq, Serialize)]
pub enum BenchmarkMode {
    ParquetFileserver,
    ParquetPushdown,
    ArrowPushdown,
    LiquidCache,
    #[default]
    LiquidEagerTranscode,
}

impl BenchmarkMode {
    #[fastrace::trace]
    pub async fn setup_tpch_ctx(
        &self,
        server_url: &str,
        data_dir: &Path,
        partitions: Option<usize>,
    ) -> Result<Arc<SessionContext>> {
        let mut session_config = SessionConfig::from_env()?;
        let current_dir = std::env::current_dir()?.to_string_lossy().to_string();

        let tables = [
            "customer", "lineitem", "nation", "orders", "part", "partsupp", "region", "supplier",
        ];

        let mode = match self {
            BenchmarkMode::ParquetFileserver => {
                let ctx = Arc::new(SessionContext::new_with_config(session_config));
                let base_url = Url::parse(server_url).unwrap();

                let object_store = object_store::http::HttpBuilder::new()
                    .with_url(base_url.clone())
                    .with_config(ClientConfigKey::AllowHttp, "true")
                    .build()
                    .unwrap();
                ctx.register_object_store(&base_url, Arc::new(object_store));

                for table_name in tables.iter() {
                    let table_path = Url::parse(&format!(
                        "file://{}/{}/{}.parquet",
                        current_dir,
                        data_dir.display(),
                        table_name
                    ))
                    .unwrap();
                    ctx.register_parquet(*table_name, table_path, Default::default())
                        .await?;
                }
                return Ok(ctx);
            }
            BenchmarkMode::ParquetPushdown => CacheMode::Parquet,
            BenchmarkMode::ArrowPushdown => CacheMode::Arrow,
            BenchmarkMode::LiquidCache => CacheMode::Liquid,
            BenchmarkMode::LiquidEagerTranscode => CacheMode::LiquidEagerTranscode,
        };
        session_config
            .options_mut()
            .execution
            .parquet
            .pushdown_filters = true;
        let mut session_config = SessionConfig::from_env()?;
        if let Some(partitions) = partitions {
            session_config.options_mut().execution.target_partitions = partitions;
        }
        let ctx = LiquidCacheBuilder::new(server_url)
            .with_cache_mode(mode)
            .build(session_config)?;

        for table_name in tables.iter() {
            let table_url = Url::parse(&format!(
                "file://{}/{}/{}.parquet",
                current_dir,
                data_dir.display(),
                table_name
            ))
            .unwrap();
            ctx.register_parquet(*table_name, table_url, Default::default())
                .await?;
        }

        Ok(Arc::new(ctx))
    }

    #[fastrace::trace]
    pub async fn setup_clickbench_ctx(
        &self,
        server_url: &str,
        data_url: &Path,
        partitions: Option<usize>,
    ) -> Result<Arc<SessionContext>> {
        let table_name = "hits";
        let current_dir = std::env::current_dir()?.to_string_lossy().to_string();
        let table_url =
            Url::parse(&format!("file://{}/{}", current_dir, data_url.display())).unwrap();

        let mode = match self {
            BenchmarkMode::ParquetFileserver => {
                let mut session_config = SessionConfig::from_env()?;
                if let Some(partitions) = partitions {
                    session_config.options_mut().execution.target_partitions = partitions;
                }
                let ctx = Arc::new(SessionContext::new_with_config(session_config));
                let base_url = Url::parse(server_url).unwrap();

                let object_store = object_store::http::HttpBuilder::new()
                    .with_url(base_url.clone())
                    .with_config(ClientConfigKey::AllowHttp, "true")
                    .build()
                    .unwrap();
                ctx.register_object_store(&base_url, Arc::new(object_store));

                ctx.register_parquet(
                    "hits",
                    format!("{}/hits.parquet", server_url),
                    Default::default(),
                )
                .await?;
                return Ok(ctx);
            }
            BenchmarkMode::ParquetPushdown => CacheMode::Parquet,
            BenchmarkMode::ArrowPushdown => CacheMode::Arrow,
            BenchmarkMode::LiquidCache => CacheMode::Liquid,
            BenchmarkMode::LiquidEagerTranscode => CacheMode::LiquidEagerTranscode,
        };
        let mut session_config = SessionConfig::from_env()?;
        if let Some(partitions) = partitions {
            session_config.options_mut().execution.target_partitions = partitions;
        }
        let ctx = LiquidCacheBuilder::new(server_url)
            .with_cache_mode(mode)
            .build(session_config)?;

        ctx.register_parquet(table_name, table_url, Default::default())
            .await?;
        Ok(Arc::new(ctx))
    }

    #[fastrace::trace]
    pub async fn get_execution_metrics(
        &self,
        server_url: &str,
        execution_plan: &Arc<dyn ExecutionPlan>,
    ) -> ExecutionMetricsResponse {
        match self {
            BenchmarkMode::ParquetFileserver => {
                // for parquet fileserver, the memory usage is the bytes scanned.
                // It's not easy to get the memory usage as it is cached in the kernel's page cache.
                // So the bytes scanned is the minimum cache memory usage, actual usage is slightly higher.
                let mut plan = execution_plan;
                while let Some(child) = plan.children().first() {
                    plan = child;
                }
                if plan.name() != "ParquetExec" {
                    // the scan is completely pruned, so the memory usage is 0
                    return ExecutionMetricsResponse {
                        pushdown_eval_time: 0,
                        cache_memory_usage: 0,
                        liquid_cache_usage: 0,
                    };
                }
                let metrics = plan
                    .metrics()
                    .unwrap()
                    .aggregate_by_name()
                    .sorted_for_display()
                    .timestamps_removed();

                let mut bytes_scanned = 0;

                for metric in metrics.iter() {
                    if let MetricValue::Count { name, count } = metric.value() {
                        if name == "bytes_scanned" {
                            bytes_scanned = count.value();
                        }
                    }
                }

                ExecutionMetricsResponse {
                    pushdown_eval_time: 0,
                    cache_memory_usage: bytes_scanned as u64,
                    liquid_cache_usage: 0,
                }
            }
            BenchmarkMode::ParquetPushdown
            | BenchmarkMode::ArrowPushdown
            | BenchmarkMode::LiquidCache
            | BenchmarkMode::LiquidEagerTranscode => {
                let mut handles = Vec::new();
                execution_plan
                    .apply(|plan| {
                        let any_plan = plan.as_any();
                        if let Some(flight_exec) = any_plan.downcast_ref::<LiquidCacheClientExec>()
                        {
                            handles.push(flight_exec);
                        }
                        Ok(datafusion::common::tree_node::TreeNodeRecursion::Continue)
                    })
                    .unwrap();
                let mut flight_client = get_flight_client(server_url).await;
                let mut metrics = Vec::new();
                for handle in handles {
                    let action = LiquidCacheActions::ExecutionMetrics(ExecutionMetricsRequest {
                        handle: handle.get_plan_uuid().await.unwrap().to_string(),
                    })
                    .into();
                    let mut result_stream = flight_client.do_action(action).await.unwrap();
                    let result = result_stream.next().await.unwrap().unwrap();
                    let any = Any::decode(&*result).unwrap();
                    metrics.push(any.unpack::<ExecutionMetricsResponse>().unwrap().unwrap());
                }
                let metric =
                    metrics
                        .iter()
                        .fold(None, |acc: Option<ExecutionMetricsResponse>, m| {
                            if let Some(acc) = acc {
                                Some(ExecutionMetricsResponse {
                                    pushdown_eval_time: acc.pushdown_eval_time
                                        + m.pushdown_eval_time,
                                    cache_memory_usage: acc.cache_memory_usage,
                                    liquid_cache_usage: acc.liquid_cache_usage,
                                })
                            } else {
                                Some(m.clone())
                            }
                        });
                metric.unwrap_or_default()
            }
        }
    }

    pub async fn reset_cache(&self, server_url: &str) -> Result<()> {
        if self == &BenchmarkMode::ParquetFileserver {
            // File server relies on OS page cache, so we don't need to reset it
            return Ok(());
        }
        let mut flight_client = get_flight_client(server_url).await;
        let action = LiquidCacheActions::ResetCache.into();
        let mut result_stream = flight_client.do_action(action).await.unwrap();
        let _result = result_stream.next().await.unwrap().unwrap();
        Ok(())
    }
}

impl Display for BenchmarkMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}",
            match self {
                BenchmarkMode::ParquetFileserver => "parquet-fileserver",
                BenchmarkMode::ParquetPushdown => "parquet-pushdown",
                BenchmarkMode::LiquidCache => "liquid-cache",
                BenchmarkMode::ArrowPushdown => "arrow-pushdown",
                BenchmarkMode::LiquidEagerTranscode => "liquid-eager-transcode",
            }
        )
    }
}

impl FromStr for BenchmarkMode {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Ok(match s {
            "parquet-fileserver" => BenchmarkMode::ParquetFileserver,
            "parquet-pushdown" => BenchmarkMode::ParquetPushdown,
            "arrow-pushdown" => BenchmarkMode::ArrowPushdown,
            "liquid-cache" => BenchmarkMode::LiquidCache,
            "liquid-eager-transcode" => BenchmarkMode::LiquidEagerTranscode,
            _ => return Err(format!("Invalid benchmark mode: {}", s)),
        })
    }
}

async fn get_flight_client(server_url: &str) -> FlightClient {
    let endpoint = Channel::from_shared(server_url.to_string()).unwrap();
    let channel = endpoint.connect().await.unwrap();
    let inner_client = FlightServiceClient::new(channel);
    FlightClient::new_from_inner(inner_client)
}

#[fastrace::trace]
pub async fn run_query(
    ctx: &Arc<SessionContext>,
    query: &str,
) -> Result<(Vec<RecordBatch>, Arc<dyn ExecutionPlan>)> {
    let df = ctx
        .sql(query)
        .in_span(Span::enter_with_local_parent("logical_plan"))
        .await?;
    let (state, logical_plan) = df.into_parts();
    let physical_plan = state
        .create_physical_plan(&logical_plan)
        .in_span(Span::enter_with_local_parent("physical_plan"))
        .await?;

    let ctx = TaskContext::from(&state);
    let cfg = ctx
        .session_config()
        .clone()
        .with_extension(Arc::new(Span::enter_with_local_parent(
            "poll_physical_plan",
        )));
    let ctx = ctx.with_session_config(cfg);
    let results = collect(physical_plan.clone(), Arc::new(ctx)).await?;
    Ok((results, physical_plan))
}

#[derive(Serialize)]
pub struct BenchmarkResult<T: Serialize> {
    pub args: T,
    pub results: Vec<QueryResult>,
}

#[derive(Serialize)]
pub struct QueryResult {
    id: u32,
    query: String,
    iteration_results: Vec<IterationResult>,
}

impl QueryResult {
    pub fn new(id: u32, query: String) -> Self {
        Self {
            id,
            query,
            iteration_results: Vec::new(),
        }
    }

    pub fn add(&mut self, iteration_result: IterationResult) {
        self.iteration_results.push(iteration_result);
    }
}
#[derive(Serialize)]
pub struct IterationResult {
    pub network_traffic: u64,
    pub time_millis: u64,
    pub cache_cpu_time: u64,
    pub cache_memory_usage: u64,
    pub starting_timestamp: Duration,
}

use fastrace_opentelemetry::OpenTelemetryReporter;
use opentelemetry::InstrumentationScope;
use opentelemetry::KeyValue;
use opentelemetry::trace::SpanKind;
use opentelemetry_otlp::WithExportConfig;
use opentelemetry_otlp::{SpanExporter, WithTonicConfig};
use opentelemetry_sdk::Resource;
use std::borrow::Cow;

fn otl_metadata() -> MetadataMap {
    let mut map = MetadataMap::with_capacity(3);

    map.insert(
        "authorization",
        format!("Basic cm9vdEBleGFtcGxlLmNvbTpFeUIycDFuSXNicXJLekNI") // This is picked from the Ingestion tab openobserve
            .parse()
            .unwrap(),
    );
    map.insert("organization", "default".parse().unwrap());
    map.insert("stream-name", "default".parse().unwrap());
    map
}

pub fn setup_observability(service_name: &str, kind: SpanKind) {
    // Setup logging with logforth
    logforth::builder()
        .dispatch(|d| {
            d.filter(log::LevelFilter::Info)
                .append(logforth::append::Stdout::default())
        })
        // enable after: https://github.com/fast/logforth/issues/125
        // .dispatch(|d| {
        //     let otl_appender =
        //         OpentelemetryLogBuilder::new(service_name, "http://localhost:5081/api/development")
        //             .protocol(OpentelemetryWireProtocol::Grpc)
        //             .build()
        //             .unwrap();
        //     d.append(otl_appender)
        // })
        .apply();

    let reporter = OpenTelemetryReporter::new(
        SpanExporter::builder()
            .with_tonic()
            .with_endpoint("http://localhost:5081/api/development".to_string())
            .with_metadata(otl_metadata())
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
    fastrace::set_reporter(reporter, fastrace::collector::Config::default());
}
