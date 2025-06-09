use clap::Parser;
use datafusion::catalog::memory::DataSourceExec;
use datafusion::common::tree_node::{TreeNode, TreeNodeRecursion};
use datafusion::physical_plan::ExecutionPlan;
use datafusion::physical_plan::metrics::MetricValue;
use datafusion::prelude::SessionConfig;
use datafusion::{arrow::array::RecordBatch, error::Result, prelude::SessionContext};
use liquid_cache_benchmarks::{Query, run_query, setup_observability, tpch};
use liquid_cache_common::{CacheEvictionStrategy, LiquidCacheMode};
use liquid_cache_parquet::{LiquidCacheInProcessBuilder, LiquidCacheRef};
use log::info;
use mimalloc::MiMalloc;
use serde::Serialize;
use std::{
    fs::{File, create_dir_all},
    path::PathBuf,
    sync::Arc,
    time::{Duration, Instant},
};
use sysinfo::Disks;
use url::Url;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

#[derive(Clone, Debug, Default, Copy, PartialEq, Eq, Serialize)]
enum BenchmarkMode {
    Parquet,
    Arrow,
    #[default]
    Liquid,
    LiquidEagerTranscode,
}

impl std::str::FromStr for BenchmarkMode {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Ok(match s {
            "parquet" => BenchmarkMode::Parquet,
            "arrow" => BenchmarkMode::Arrow,
            "liquid" => BenchmarkMode::Liquid,
            "liquid-eager-transcode" => BenchmarkMode::LiquidEagerTranscode,
            _ => {
                return Err(format!(
                    "Invalid in-process benchmark mode: {s}, must be one of: parquet, arrow, liquid, liquid-eager-transcode"
                ));
            }
        })
    }
}

#[derive(Parser, Serialize, Clone)]
#[command(name = "TPCH In-Process Benchmark")]
struct TpchInProcessBenchmark {
    /// Path to the query directory
    #[arg(long = "query-dir")]
    pub query_dir: PathBuf,

    /// Path to the data directory with TPCH data
    #[arg(long = "data-dir")]
    pub data_dir: PathBuf,

    /// Benchmark mode to use
    #[arg(long = "bench-mode", default_value = "liquid-eager-transcode")]
    pub bench_mode: BenchmarkMode,

    /// Number of times to run each query
    #[arg(long, default_value = "3")]
    pub iteration: u32,

    /// Path to the output JSON file to save the benchmark results
    #[arg(long)]
    pub output: Option<PathBuf>,

    /// Reset the cache before running a new query
    #[arg(long = "reset-cache", default_value = "false")]
    pub reset_cache: bool,

    /// Number of partitions to use
    #[arg(long)]
    pub partitions: Option<usize>,

    /// Query number to run, if not provided, all queries will be run
    #[arg(long)]
    pub query: Option<u32>,

    /// Maximum cache size in bytes
    #[arg(long = "max-cache-mb")]
    pub max_cache_mb: Option<usize>,

    /// Directory to write flamegraph SVG files to
    #[arg(long)]
    pub flamegraph_dir: Option<PathBuf>,
}

#[derive(Serialize)]
pub struct BenchmarkResult {
    args: TpchInProcessBenchmark,
    results: Vec<QueryResult>,
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
    pub time_millis: u64,
    pub cache_memory_usage: u64,
    pub starting_timestamp: Duration,
    pub bytes_read: u64,
    pub bytes_written: u64,
}

impl IterationResult {
    pub fn log(&self) {
        info!(
            "Query: {} ms, cache memory: {} bytes, disk I/O: read {} bytes, written {} bytes",
            self.time_millis, self.cache_memory_usage, self.bytes_read, self.bytes_written,
        );
    }
}

impl TpchInProcessBenchmark {
    async fn setup_context(&self) -> Result<(Arc<SessionContext>, Option<LiquidCacheRef>)> {
        let mut session_config = SessionConfig::from_env()?;
        let current_dir = std::env::current_dir()?.to_string_lossy().to_string();

        let tables = [
            "customer", "lineitem", "nation", "orders", "part", "partsupp", "region", "supplier",
        ];

        session_config
            .options_mut()
            .execution
            .parquet
            .pushdown_filters = true;
        if let Some(partitions) = self.partitions {
            session_config.options_mut().execution.target_partitions = partitions;
        }
        let cache_size = self
            .max_cache_mb
            .map(|size| size * 1024 * 1024)
            .unwrap_or(usize::MAX);

        let cache_dir = std::env::current_dir()?.join("benchmark/tpch/data/cache");
        create_dir_all(&cache_dir)?;
        let (ctx, cache): (SessionContext, Option<LiquidCacheRef>) = match self.bench_mode {
            BenchmarkMode::Parquet => (SessionContext::new_with_config(session_config), None),
            BenchmarkMode::Arrow => {
                let v = LiquidCacheInProcessBuilder::new()
                    .with_max_cache_bytes(cache_size)
                    .with_cache_mode(LiquidCacheMode::Arrow)
                    .with_cache_dir(cache_dir)
                    .with_cache_strategy(CacheEvictionStrategy::ToDisk)
                    .build(session_config)?;
                (v.0, Some(v.1))
            }
            BenchmarkMode::Liquid => {
                let v = LiquidCacheInProcessBuilder::new()
                    .with_max_cache_bytes(cache_size)
                    .with_cache_mode(LiquidCacheMode::Liquid {
                        transcode_in_background: true,
                    })
                    .with_cache_dir(cache_dir)
                    .with_cache_strategy(CacheEvictionStrategy::ToDisk)
                    .build(session_config)?;
                (v.0, Some(v.1))
            }
            BenchmarkMode::LiquidEagerTranscode => {
                let v = LiquidCacheInProcessBuilder::new()
                    .with_max_cache_bytes(cache_size)
                    .with_cache_mode(LiquidCacheMode::Liquid {
                        transcode_in_background: false,
                    })
                    .with_cache_dir(cache_dir)
                    .with_cache_strategy(CacheEvictionStrategy::ToDisk)
                    .build(session_config)?;
                (v.0, Some(v.1))
            }
        };
        for table_name in tables.iter() {
            let table_path = Url::parse(&format!(
                "file://{}/{}/{}.parquet",
                current_dir,
                self.data_dir.display(),
                table_name
            ))
            .unwrap();
            ctx.register_parquet(*table_name, table_path, Default::default())
                .await?;
        }

        Ok((Arc::new(ctx), cache))
    }

    async fn get_queries(&self) -> Result<Vec<Query>> {
        tpch::get_all_queries(&self.query_dir)
    }

    async fn execute_query(
        &self,
        ctx: &Arc<SessionContext>,
        query: &Query,
    ) -> Result<(Vec<RecordBatch>, Arc<dyn ExecutionPlan>)> {
        if query.id == 15 {
            // Q15 has three queries, the second one is the one we want to test
            let queries: Vec<&str> = query.sql.split(';').collect();
            let mut results = Vec::new();
            let mut plan = None;
            for (i, q) in queries.iter().enumerate() {
                let (result, p, _) = run_query(ctx, q).await?;
                if i == 1 {
                    results = result;
                    plan = Some(p);
                }
            }
            Ok((results, plan.unwrap()))
        } else {
            let (results, plan, _) = run_query(ctx, &query.sql).await?;
            Ok((results, plan))
        }
    }

    async fn run_single_iteration(
        &self,
        ctx: &Arc<SessionContext>,
        query: &Query,
        bench_start_time: Instant,
        cache: Option<LiquidCacheRef>,
        iteration: u32,
    ) -> Result<IterationResult> {
        info!("Running query {}: \n{}", query.id, query.sql);

        // Start flamegraph profiling if enabled
        let profiler_guard = if self.flamegraph_dir.is_some() {
            Some(
                pprof::ProfilerGuardBuilder::default()
                    .frequency(500)
                    .blocklist(&["libpthread.so.0", "libm.so.6", "libgcc_s.so.1"])
                    .build()
                    .unwrap(),
            )
        } else {
            None
        };

        // Capture disk I/O metrics before query execution
        let mut disk_info = Disks::new_with_refreshed_list();

        let now = Instant::now();
        let starting_timestamp = bench_start_time.elapsed();

        let (_results, execution_plan) = self.execute_query(ctx, query).await?;
        let elapsed = now.elapsed();

        disk_info.refresh(true);
        let disk_read: u64 = disk_info.iter().map(|disk| disk.usage().read_bytes).sum();
        let total_read: u64 = disk_info
            .iter()
            .map(|disk| disk.usage().total_read_bytes)
            .sum();
        let total_written: u64 = disk_info
            .iter()
            .map(|disk| disk.usage().total_written_bytes)
            .sum();
        let disk_written: u64 = disk_info
            .iter()
            .map(|disk| disk.usage().written_bytes)
            .sum();
        println!(
            "read: {}, written: {}, total_read: {}, total_written: {}",
            disk_read, disk_written, total_read, total_written
        );

        // Stop flamegraph profiling and write to file if enabled
        if let Some(profiler) = profiler_guard
            && let Some(flamegraph_dir) = &self.flamegraph_dir
        {
            let report = profiler.report().build().unwrap();
            let mut svg_data = Vec::new();
            report.flamegraph(&mut svg_data).unwrap();
            create_dir_all(flamegraph_dir)?;

            // Get current time for filename prefix
            let now = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap();
            let secs = now.as_secs();
            let hour = (secs / 3600) % 24;
            let minute = (secs / 60) % 60;
            let second = secs % 60;

            let filename = format!(
                "{:02}h{:02}m{:02}s_q{}_i{}.svg",
                hour, minute, second, query.id, iteration
            );
            let filepath = flamegraph_dir.join(filename);
            std::fs::write(&filepath, svg_data).unwrap();
            info!("Flamegraph written to: {}", filepath.display());
        }

        let cache_memory_usage = if let Some(cache) = cache {
            cache.memory_usage_bytes()
        } else {
            let mut total_bytes_scanned = 0;
            let _ = execution_plan
                .apply(|plan| {
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
                })
                .unwrap();
            total_bytes_scanned
        };

        let result = IterationResult {
            time_millis: elapsed.as_millis() as u64,
            cache_memory_usage: cache_memory_usage as u64,
            starting_timestamp,
            bytes_read: disk_read,
            bytes_written: disk_written,
        };

        result.log();
        Ok(result)
    }

    pub async fn run(self) -> Result<BenchmarkResult> {
        let (ctx, cache) = self.setup_context().await?;
        let queries = self.get_queries().await?;
        let queries = if let Some(query) = self.query {
            vec![queries.into_iter().find(|q| q.id == query).unwrap()]
        } else {
            queries
        };

        let mut benchmark_result = BenchmarkResult {
            args: self.clone(),
            results: Vec::new(),
        };

        std::fs::create_dir_all("benchmark/data/results")?;

        let bench_start_time = Instant::now();

        for query in queries {
            let mut query_result = QueryResult::new(query.id, query.sql.clone());

            for it in 0..self.iteration {
                let iteration_result = self
                    .run_single_iteration(&ctx, &query, bench_start_time, cache.clone(), it + 1)
                    .await?;

                query_result.add(iteration_result);
            }

            if self.reset_cache
                && let Some(cache) = &cache
            {
                cache.reset();
            }

            benchmark_result.results.push(query_result);
        }

        if let Some(output_path) = &self.output {
            let output_file = File::create(output_path)?;
            serde_json::to_writer_pretty(output_file, &benchmark_result).unwrap();
        }

        Ok(benchmark_result)
    }
}

#[tokio::main]
pub async fn main() -> Result<()> {
    setup_observability(
        "tpch-inprocess",
        opentelemetry::trace::SpanKind::Client,
        None,
    );
    let benchmark = TpchInProcessBenchmark::parse();
    benchmark.run().await?;
    Ok(())
}
