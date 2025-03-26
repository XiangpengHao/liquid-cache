use arrow_flight::sql::Any;
use arrow_flight::{FlightClient, flight_service_client::FlightServiceClient};
use datafusion::common::tree_node::TreeNode;
use datafusion::{error::Result, physical_plan::ExecutionPlan};
use datafusion::{
    physical_plan::metrics::MetricValue,
    prelude::{SessionConfig, SessionContext},
};
use futures::StreamExt;
use liquid_cache_client::{FlightExec, LiquidCacheTableBuilder};
use liquid_cache_common::CacheMode;
use liquid_cache_common::rpc::{
    ExecutionMetricsRequest, ExecutionMetricsResponse, LiquidCacheActions,
};
use liquid_cache_parquet::LiquidCacheRef;
use liquid_cache_server::StatsCollector;
use object_store::ClientConfigKey;
use pprof::ProfilerGuard;
use prost::Message;
use serde::Serialize;
use std::time::Duration;
use std::{
    fmt::Display,
    path::{Path, PathBuf},
    str::FromStr,
    sync::{
        Arc, Mutex,
        atomic::{AtomicU32, AtomicUsize},
    },
};
use tonic::transport::Channel;
use url::Url;

pub mod utils;

pub struct FlameGraphReport {
    output_dir: PathBuf,
    guard: Mutex<Option<ProfilerGuard<'static>>>,
    running_count: AtomicUsize,
    flame_graph_id: AtomicU32,
}

impl FlameGraphReport {
    pub fn new(output: PathBuf) -> Self {
        Self {
            output_dir: output,
            guard: Mutex::new(None),
            running_count: AtomicUsize::new(0),
            flame_graph_id: AtomicU32::new(0),
        }
    }
}

impl StatsCollector for FlameGraphReport {
    fn start(&self, _partition: usize, _plan: &Arc<dyn ExecutionPlan>) {
        self.running_count
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let mut guard = self.guard.lock().unwrap();
        if guard.is_some() {
            return;
        }
        let old = guard.take();
        assert!(old.is_none(), "FlameGraphReport is already started");
        *guard = Some(
            pprof::ProfilerGuardBuilder::default()
                .frequency(500)
                .blocklist(&["libpthread.so.0", "libm.so.6", "libgcc_s.so.1"])
                .build()
                .unwrap(),
        );
    }

    fn stop(&self, _partition: usize, _plan: &Arc<dyn ExecutionPlan>) {
        let previous = self
            .running_count
            .fetch_sub(1, std::sync::atomic::Ordering::Relaxed);
        if previous != 1 {
            return;
        }

        let mut lock_guard = self.guard.lock().unwrap();
        if lock_guard.is_none() {
            log::warn!("FlameGraphReport is not started");
            return;
        }
        let profiler = lock_guard.take().unwrap();
        drop(lock_guard);
        let report = profiler.report().build().unwrap();
        drop(profiler);

        let now = std::time::SystemTime::now();
        let datetime = now.duration_since(std::time::UNIX_EPOCH).unwrap();
        let minute = (datetime.as_secs() / 60) % 60;
        let second = datetime.as_secs() % 60;
        let flame_graph_id = self
            .flame_graph_id
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let filename = format!(
            "flamegraph-id{:02}-{:02}-{:03}.svg",
            flame_graph_id, minute, second
        );
        let filepath = self.output_dir.join(filename);
        let file = std::fs::File::create(&filepath).unwrap();
        report.flamegraph(file).unwrap();
        log::info!("Flamegraph saved to {}", filepath.display());
    }
}

pub struct StatsReport {
    output_dir: PathBuf,
    cache: LiquidCacheRef,
    running_count: AtomicUsize,
    stats_id: AtomicU32,
}

impl StatsReport {
    pub fn new(output: PathBuf, cache: LiquidCacheRef) -> Self {
        Self {
            output_dir: output,
            cache,
            running_count: AtomicUsize::new(0),
            stats_id: AtomicU32::new(0),
        }
    }
}

impl StatsCollector for StatsReport {
    fn start(&self, _partition: usize, _plan: &Arc<dyn ExecutionPlan>) {
        self.running_count
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }

    fn stop(&self, _partition: usize, _plan: &Arc<dyn ExecutionPlan>) {
        let previous = self
            .running_count
            .fetch_sub(1, std::sync::atomic::Ordering::Relaxed);
        // We only write stats once, after every thread finishes.
        if previous != 1 {
            return;
        }
        let now = std::time::SystemTime::now();
        let datetime = now.duration_since(std::time::UNIX_EPOCH).unwrap();
        let minute = (datetime.as_secs() / 60) % 60;
        let second = datetime.as_secs() % 60;
        let stats_id = self
            .stats_id
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let filename = format!(
            "stats-id{:02}-{:02}-{:03}.parquet",
            stats_id, minute, second
        );
        let parquet_file_path = self.output_dir.join(filename);
        self.cache.write_stats(&parquet_file_path).unwrap();
        log::info!("Stats saved to {}", parquet_file_path.display());
    }
}

#[derive(Clone, Debug, Default, Copy, PartialEq, Eq, Serialize)]
pub enum BenchmarkMode {
    ParquetFileserver,
    ParquetPushdown,
    ArrowPushdown,
    #[default]
    LiquidCache,
    LiquidEagerTranscode,
}

impl BenchmarkMode {
    pub async fn setup_tpch_ctx(
        &self,
        server_url: &str,
        data_dir: &Path,
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
        let ctx = Arc::new(SessionContext::new_with_config(session_config));

        for table_name in tables.iter() {
            let table_url = Url::parse(&format!(
                "file://{}/{}/{}.parquet",
                current_dir,
                data_dir.display(),
                table_name
            ))
            .unwrap();
            let table = LiquidCacheTableBuilder::new(server_url, table_name, table_url)
                .with_cache_mode(mode)
                .build()
                .await?;
            ctx.register_table(*table_name, Arc::new(table))?;
        }

        Ok(ctx)
    }

    pub async fn setup_clickbench_ctx(
        &self,
        server_url: &str,
        data_url: &Path,
    ) -> Result<Arc<SessionContext>> {
        let mut session_config = SessionConfig::from_env()?;
        let table_name = "hits";
        let current_dir = std::env::current_dir()?.to_string_lossy().to_string();
        let table_url =
            Url::parse(&format!("file://{}/{}", current_dir, data_url.display())).unwrap();

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
        session_config
            .options_mut()
            .execution
            .parquet
            .pushdown_filters = true;
        let ctx = Arc::new(SessionContext::new_with_config(session_config));

        let table = LiquidCacheTableBuilder::new(server_url, table_name, table_url)
            .with_cache_mode(mode)
            .build()
            .await?;
        ctx.register_table(table_name, Arc::new(table))?;
        Ok(ctx)
    }

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
                        if let Some(flight_exec) = any_plan.downcast_ref::<FlightExec>() {
                            let plan_handle = flight_exec.plan_handle();
                            handles.push(plan_handle);
                        }
                        Ok(datafusion::common::tree_node::TreeNodeRecursion::Continue)
                    })
                    .unwrap();
                let mut flight_client = get_flight_client(server_url).await;
                let mut metrics = Vec::new();
                for handle in handles {
                    let action = LiquidCacheActions::ExecutionMetrics(ExecutionMetricsRequest {
                        handle: handle.to_string(),
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
                metric.unwrap()
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
