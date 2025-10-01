use clap::Parser;
use datafusion::arrow::array::RecordBatch;
use datafusion::execution::TaskContext;
use datafusion::physical_plan::collect;
use datafusion::prelude::SessionContext;
use datafusion::{error::Result, physical_plan::ExecutionPlan};
use fastrace::Span;
use fastrace::future::FutureExt as _;
use liquid_cache_common::rpc::ExecutionMetricsResponse;
use liquid_cache_server::{ApiResponse, ExecutionStats};
use liquid_cache_storage::cache::CacheStats;
use liquid_cache_storage::cache::squeeze_policies::{
    Evict, SqueezePolicy, TranscodeEvict, TranscodeSqueezeEvict,
};
use log::info;
use serde::Serialize;
use std::path::PathBuf;
use std::time::Duration;
use std::{fmt::Display, str::FromStr, sync::Arc};
use uuid::Uuid;

pub mod client_runner;
pub mod inprocess_runner;
mod manifest;
mod observability;
pub mod tpch;
pub mod utils;

pub use client_runner::*;
pub use inprocess_runner::*;
pub use manifest::BenchmarkManifest;
pub use observability::*;

#[derive(Serialize, Clone)]
pub struct Query {
    id: u32,
    statement: Vec<String>,
}

impl Query {
    pub fn new(id: u32, statement: Vec<String>) -> Self {
        Self { id, statement }
    }

    pub fn id(&self) -> u32 {
        self.id
    }

    pub fn statement(&self) -> &Vec<String> {
        &self.statement
    }
}

#[derive(Parser, Serialize, Clone)]
pub struct ClientBenchmarkArgs {
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

impl ClientBenchmarkArgs {
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
        let metric = metrics
            .iter()
            .fold(None, |acc: Option<ExecutionMetricsResponse>, m| {
                if let Some(acc) = acc {
                    Some(ExecutionMetricsResponse {
                        pushdown_eval_time: acc.pushdown_eval_time + m.pushdown_eval_time,
                        cache_memory_usage: acc.cache_memory_usage + m.cache_memory_usage,
                        liquid_cache_usage: acc.liquid_cache_usage + m.liquid_cache_usage,
                    })
                } else {
                    Some(m.clone())
                }
            });
        // If the query plan does not scan any data, the metrics will be empty
        metric.unwrap_or_else(ExecutionMetricsResponse::zero)
    }

    pub async fn reset_cache(&self) -> Result<()> {
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
        user_sql: Vec<String>,
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
    Arrow,
    #[default]
    Liquid,
    LiquidNoSqueeze,
}

impl BenchmarkMode {
    pub fn to_squeeze_policy(&self) -> Box<dyn SqueezePolicy> {
        match self {
            BenchmarkMode::Arrow => Box::new(Evict),
            BenchmarkMode::Liquid => Box::new(TranscodeSqueezeEvict),
            BenchmarkMode::LiquidNoSqueeze => Box::new(TranscodeEvict),
        }
    }
}

impl Display for BenchmarkMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}",
            match self {
                BenchmarkMode::Arrow => "arrow",
                BenchmarkMode::Liquid => "liquid",
                BenchmarkMode::LiquidNoSqueeze => "liquid-no-squeeze",
            }
        )
    }
}

impl FromStr for BenchmarkMode {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Ok(match s {
            "arrow" => BenchmarkMode::Arrow,
            "liquid" => BenchmarkMode::Liquid,
            "liquid-no-squeeze" => BenchmarkMode::LiquidNoSqueeze,
            _ => return Err(format!("Invalid benchmark mode: {s}")),
        })
    }
}

#[fastrace::trace]
pub async fn run_query(
    ctx: &Arc<SessionContext>,
    query: &str,
) -> (Vec<RecordBatch>, Arc<dyn ExecutionPlan>, Vec<Uuid>) {
    let df = ctx
        .sql(query)
        .in_span(Span::enter_with_local_parent("logical_plan"))
        .await
        .unwrap();
    let (state, logical_plan) = df.into_parts();
    let physical_plan = state
        .create_physical_plan(&logical_plan)
        .in_span(Span::enter_with_local_parent("physical_plan"))
        .await
        .unwrap();

    let ctx = TaskContext::from(&state);
    let cfg = ctx
        .session_config()
        .clone()
        .with_extension(Arc::new(Span::enter_with_local_parent(
            "poll_physical_plan",
        )));
    let ctx = ctx.with_session_config(cfg);
    let results = collect(physical_plan.clone(), Arc::new(ctx)).await.unwrap();
    let plan_uuids = utils::get_plan_uuids(&physical_plan);
    (results, physical_plan, plan_uuids)
}

#[derive(Serialize)]
pub struct BenchmarkResult<T: Serialize> {
    pub args: T,
    pub results: Vec<QueryResult>,
}

#[derive(Serialize)]
pub struct QueryResult {
    query: Query,
    iteration_results: Vec<IterationResult>,
}

impl QueryResult {
    pub fn new(query: Query) -> Self {
        Self {
            query,
            iteration_results: Vec::new(),
        }
    }

    pub fn add(&mut self, iteration_result: IterationResult) {
        self.iteration_results.push(iteration_result);
    }
}

#[derive(Serialize, Debug)]
pub struct SerializableCacheStats {
    pub total_entries: usize,
    pub memory_arrow_entries: usize,
    pub memory_liquid_entries: usize,
    pub memory_hybrid_liquid_entries: usize,
    pub disk_liquid_entries: usize,
}

impl SerializableCacheStats {
    pub fn from(cache_stats: CacheStats) -> Self {
        Self {
            total_entries: cache_stats.total_entries,
            memory_arrow_entries: cache_stats.memory_arrow_entries,
            memory_liquid_entries: cache_stats.memory_liquid_entries,
            memory_hybrid_liquid_entries: cache_stats.memory_hybrid_liquid_entries,
            disk_liquid_entries: cache_stats.disk_liquid_entries,
        }
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
    pub cache_stats: Option<SerializableCacheStats>,
}

impl Display for IterationResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Query time: {} ms\n network: {} bytes\n cache cpu time: {} ms\n cache memory: {} bytes, liquid cache memory: {} bytes\n disk read: {} bytes, disk written: {} bytes\n",
            self.time_millis,
            self.network_traffic,
            self.cache_cpu_time,
            self.cache_memory_usage,
            self.liquid_cache_usage,
            self.disk_bytes_read,
            self.disk_bytes_written,
        )?;
        if let Some(cache_stats) = &self.cache_stats {
            write!(
                f,
                " cache stats: total_entries: {}, memory_arrow_entries: {}, memory_liquid_entries: {}, memory_hybrid_liquid_entries: {}, disk_liquid_entries: {}",
                cache_stats.total_entries,
                cache_stats.memory_arrow_entries,
                cache_stats.memory_liquid_entries,
                cache_stats.memory_hybrid_liquid_entries,
                cache_stats.disk_liquid_entries,
            )
        } else {
            Ok(())
        }
    }
}
