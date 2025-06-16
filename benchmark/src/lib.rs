use clap::Parser;
use datafusion::arrow::array::RecordBatch;
use datafusion::catalog::memory::DataSourceExec;
use datafusion::common::tree_node::{TreeNode, TreeNodeRecursion};
use datafusion::execution::TaskContext;
use datafusion::physical_plan::collect;
use datafusion::{error::Result, physical_plan::ExecutionPlan};
use datafusion::{physical_plan::metrics::MetricValue, prelude::SessionContext};
use fastrace::Span;
use fastrace::future::FutureExt as _;
use liquid_cache_common::rpc::ExecutionMetricsResponse;
use liquid_cache_server::{ApiResponse, ExecutionStats};
use log::info;
use serde::Serialize;
use std::path::PathBuf;
use std::time::Duration;
use std::{fmt::Display, str::FromStr, sync::Arc};
use uuid::Uuid;

pub mod inprocess;
mod observability;
pub mod runner;
pub mod tpch;
pub mod utils;

pub use inprocess::*;
pub use observability::*;
pub use runner::*;

pub struct Query {
    pub id: u32,
    pub sql: String,
}

#[derive(Parser, Serialize, Clone)]
pub struct CommonBenchmarkArgs {
    /// LiquidCache server URL
    #[arg(long, default_value = "http://localhost:15214")]
    pub server: String,

    /// LiquidCache admin server URL
    #[arg(long, default_value = "http://localhost:53703")]
    pub admin_server: String,

    /// Number of times to run each query
    #[arg(long, default_value = "3")]
    pub iteration: u32,

    /// Path to the output JSON file to save the benchmark results
    #[arg(long)]
    pub output: Option<PathBuf>,

    /// Path to the answer directory
    #[arg(long = "answer-dir")]
    pub answer_dir: Option<PathBuf>,

    /// Benchmark mode to use
    #[arg(long = "bench-mode", default_value = "liquid-eager-transcode")]
    pub bench_mode: BenchmarkMode,

    /// Reset the cache before running a new query
    #[arg(long = "reset-cache", default_value = "false")]
    pub reset_cache: bool,

    /// Number of partitions to use,
    /// impacts LiquidCache **server's** number of threads to use
    /// Checkout datafusion partition docs for more details:
    /// <https://datafusion.apache.org/user-guide/configs.html#:~:text=datafusion.execution.target_partitions>
    #[arg(long)]
    pub partitions: Option<usize>,

    /// Query number to run, if not provided, all queries will be run
    #[arg(long)]
    pub query: Option<u32>,

    /// Openobserve auth token
    #[arg(long)]
    pub openobserve_auth: Option<String>,

    /// Path to save the cache trace
    /// It tells the **server** to collect the cache trace.
    #[arg(long = "cache-trace-dir")]
    pub cache_trace_dir: Option<PathBuf>,

    /// Path to save the cache stats
    /// It tells the **server** to collect the cache stats.
    #[arg(long = "cache-stats-dir")]
    pub cache_stats_dir: Option<PathBuf>,

    /// Profile the execution with flamegraph
    /// It tells the **server** to collect the flamegraph execution.
    /// It saves the flamegraph to the admin dashboard, usually:
    /// <https://liquid-cache-admin.xiangpeng.systems/?host=http://localhost:53703>
    #[arg(long)]
    pub flamegraph: bool,
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

    pub async fn start_flamegraph(&self) {
        if self.flamegraph {
            let response = reqwest::Client::new()
                .get(format!("{}/start_flamegraph", self.admin_server))
                .send()
                .await
                .unwrap();
            let response_body = response.text().await.unwrap();
            info!("Flamegraph collection started: {response_body}");
        }
    }

    pub async fn stop_flamegraph(&self) -> Option<String> {
        if self.flamegraph {
            let response = reqwest::Client::new()
                .get(format!("{}/stop_flamegraph", self.admin_server))
                .send()
                .await
                .unwrap();
            let response_body = response.json::<ApiResponse>().await.unwrap();
            if response_body.status == "success" {
                info!("Flamegraph saved to admin dashboard");
                Some(response_body.message)
            } else {
                info!("Failed to save flamegraph to admin dashboard");
                None
            }
        } else {
            None
        }
    }

    pub async fn get_cache_stats(&self) {
        if let Some(cache_stats_dir) = &self.cache_stats_dir {
            let response = reqwest::Client::new()
                .get(format!(
                    "{}/cache_stats?path={}",
                    self.admin_server,
                    cache_stats_dir.display()
                ))
                .send()
                .await
                .unwrap();
            let response_body = response.text().await.unwrap();
            info!("Cache stats: {response_body}");
        }
    }

    #[fastrace::trace]
    pub async fn get_execution_metrics(
        &self,
        execution_plan: &Arc<dyn ExecutionPlan>,
    ) -> ExecutionMetricsResponse {
        match self.bench_mode {
            BenchmarkMode::ParquetFileserver => {
                // for parquet fileserver, the memory usage is the bytes scanned.
                // It's not easy to get the memory usage as it is cached in the kernel's page cache.
                // So the bytes scanned is the minimum cache memory usage, actual usage is slightly higher.

                // Collect metrics from all DataSourceExec nodes using TreeNode traversal
                let mut total_bytes_scanned = 0;
                let _ = execution_plan.apply(|plan| {
                    if plan.as_any().downcast_ref::<DataSourceExec>().is_some() {
                        let metrics = plan
                            .metrics()
                            .unwrap()
                            .aggregate_by_name()
                            .sorted_for_display()
                            .timestamps_removed();

                        for metric in metrics.iter() {
                            if let MetricValue::Count { name, count } = metric.value()
                                && name == "bytes_scanned"
                            {
                                total_bytes_scanned += count.value();
                            }
                        }
                    }
                    Ok(TreeNodeRecursion::Continue)
                });

                if total_bytes_scanned == 0 {
                    // the scan is completely pruned, so the memory usage is 0
                    return ExecutionMetricsResponse {
                        pushdown_eval_time: 0,
                        cache_memory_usage: 0,
                        liquid_cache_usage: 0,
                    };
                }

                ExecutionMetricsResponse {
                    pushdown_eval_time: 0,
                    cache_memory_usage: total_bytes_scanned as u64,
                    liquid_cache_usage: 0,
                }
            }
            BenchmarkMode::ParquetPushdown
            | BenchmarkMode::ArrowPushdown
            | BenchmarkMode::LiquidCache
            | BenchmarkMode::LiquidEagerTranscode => {
                let uuids = utils::get_plan_uuids(execution_plan);
                let mut metrics = Vec::new();
                for uuid in uuids {
                    let response = reqwest::Client::new()
                        .get(format!(
                            "{}/execution_metrics?plan_id={uuid}",
                            self.admin_server
                        ))
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
                                    cache_memory_usage: acc.cache_memory_usage
                                        + m.cache_memory_usage,
                                    liquid_cache_usage: acc.liquid_cache_usage
                                        + m.liquid_cache_usage,
                                })
                            } else {
                                Some(m.clone())
                            }
                        });
                // If the query plan does not scan any data, the metrics will be empty
                metric.unwrap_or_else(ExecutionMetricsResponse::zero)
            }
        }
    }

    pub async fn reset_cache(&self) -> Result<()> {
        if self.bench_mode == BenchmarkMode::ParquetFileserver {
            // File server relies on OS page cache, so we don't need to reset it
            return Ok(());
        }
        let client = reqwest::Client::new();
        client
            .get(format!("{}/reset_cache", self.admin_server))
            .send()
            .await
            .unwrap()
            .text()
            .await
            .unwrap();
        Ok(())
    }

    pub async fn set_execution_stats(
        &self,
        plan_uuid: Vec<Uuid>,
        flamegraph: Option<String>,
        display_name: String,
        network_traffic_bytes: u64,
        execution_time_ms: u64,
        user_sql: String,
    ) {
        let params = ExecutionStats {
            plan_ids: plan_uuid.iter().map(|uuid| uuid.to_string()).collect(),
            display_name,
            flamegraph_svg: flamegraph,
            network_traffic_bytes,
            execution_time_ms,
            user_sql,
        };
        let client = reqwest::Client::new();
        client
            .post(format!("{}/set_execution_stats", self.admin_server))
            .json(&params)
            .send()
            .await
            .unwrap();
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
) -> Result<(Vec<RecordBatch>, Arc<dyn ExecutionPlan>, Vec<Uuid>)> {
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
    let plan_uuids = utils::get_plan_uuids(&physical_plan);
    Ok((results, physical_plan, plan_uuids))
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
    pub liquid_cache_usage: u64,
    pub starting_timestamp: Duration,
    pub disk_bytes_read: u64,
    pub disk_bytes_written: u64,
}

impl Display for IterationResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Query time: {} ms\n network: {} bytes\n cache cpu time: {} ms\n cache memory: {} bytes, liquid cache memory: {} bytes\n disk read: {} bytes, disk written: {} bytes",
            self.time_millis,
            self.network_traffic,
            self.cache_cpu_time,
            self.cache_memory_usage,
            self.liquid_cache_usage,
            self.disk_bytes_read,
            self.disk_bytes_written,
        )
    }
}
