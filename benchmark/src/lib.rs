use clap::Parser;
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
use liquid_cache_client::{LiquidCacheBuilder, LiquidCacheClientExec};
use liquid_cache_common::CacheMode;
use liquid_cache_common::rpc::ExecutionMetricsResponse;
use log::info;
use object_store::ClientConfigKey;
use serde::Serialize;
use std::path::PathBuf;
use std::time::Duration;
use std::{fmt::Display, path::Path, str::FromStr, sync::Arc};
use url::Url;

mod observability;
mod reports;
pub mod utils;

pub use observability::*;
pub use reports::*;

#[derive(Parser, Serialize, Clone)]
pub struct CommonBenchmarkArgs {
    /// Server URL
    #[arg(long, default_value = "http://localhost:50051")]
    pub server: String,

    /// Admin server URL
    #[arg(long, default_value = "http://localhost:50052")]
    pub admin_server: String,

    /// Number of times to run each query
    #[arg(long, default_value = "3")]
    pub iteration: u32,

    /// Path to the output JSON file
    #[arg(long)]
    pub output: Option<PathBuf>,

    /// Path to the baseline directory
    #[arg(long = "answer-dir")]
    pub answer_dir: Option<PathBuf>,

    /// Benchmark mode to use
    #[arg(long = "bench-mode", default_value = "liquid-eager-transcode")]
    pub bench_mode: BenchmarkMode,

    /// Reset the cache before running a new query
    #[arg(long = "reset-cache", default_value = "false")]
    pub reset_cache: bool,

    /// Number of partitions to use
    #[arg(long)]
    pub partitions: Option<usize>,

    /// Query number to run, if not provided, all queries will be run
    #[arg(long)]
    pub query: Option<u32>,

    /// Openobserve auth token
    #[arg(long)]
    pub openobserve_auth: Option<String>,

    /// Path to save the cache trace
    #[arg(long = "cache-trace-dir")]
    pub cache_trace_dir: Option<PathBuf>,
}

impl CommonBenchmarkArgs {
    pub async fn start_trace(&self) {
        if self.cache_trace_dir.is_some() {
            let client = reqwest::Client::new();
            let response = client
                .get(format!("{}/start_trace", self.admin_server))
                .send()
                .await
                .unwrap();
            if response.status().is_success() {
                info!("Cache trace collection started");
            }
        }
    }

    pub async fn stop_trace(&self) {
        if let Some(cache_trace_dir) = &self.cache_trace_dir {
            let response = reqwest::Client::new()
                .get(format!(
                    "{}/stop_trace?path={}",
                    self.admin_server,
                    cache_trace_dir.display()
                ))
                .send()
                .await
                .unwrap();
            let response_body = response.text().await.unwrap();
            info!("Cache trace collection stopped: {response_body}");
        }
    }
}

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
                    format!("{server_url}/hits.parquet"),
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
        admin_url: &str,
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
                let mut metrics = Vec::new();
                for handle in handles {
                    let plan_id = handle.get_plan_uuid().await.unwrap();
                    let response = reqwest::Client::new()
                        .get(format!("{admin_url}/execution_metrics?plan_id={plan_id}"))
                        .send()
                        .await
                        .unwrap();
                    let v = response.json::<ExecutionMetricsResponse>().await.unwrap();
                    metrics.push(v);
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
                metric.expect("No metrics found")
            }
        }
    }

    pub async fn reset_cache(&self, admin_url: &str) -> Result<()> {
        if self == &BenchmarkMode::ParquetFileserver {
            // File server relies on OS page cache, so we don't need to reset it
            return Ok(());
        }
        let client = reqwest::Client::new();
        client
            .post(format!("{admin_url}/reset_cache"))
            .send()
            .await
            .unwrap()
            .text()
            .await
            .unwrap();
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
            _ => return Err(format!("Invalid benchmark mode: {s}")),
        })
    }
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
