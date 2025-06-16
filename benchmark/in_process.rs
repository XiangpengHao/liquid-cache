use anyhow::Result;
use clap::Parser;
use datafusion::execution::object_store::ObjectStoreUrl;
use datafusion::prelude::SessionConfig;
use datafusion::prelude::SessionContext;
use liquid_cache_benchmarks::{
    BenchmarkResult, IterationResult, Query, QueryResult, run_query, setup_observability,
};
use liquid_cache_common::{CacheEvictionStrategy, LiquidCacheMode};
use liquid_cache_parquet::{LiquidCacheInProcessBuilder, LiquidCacheRef};
use log::info;
use mimalloc::MiMalloc;
use serde::{Deserialize, Serialize};
use std::{
    collections::HashMap,
    fs::{File, create_dir_all},
    path::PathBuf,
    sync::Arc,
    time::Instant,
};
use sysinfo::Disks;

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
                    "Invalid benchmark mode: {s}, must be one of: parquet, arrow, liquid, liquid-eager-transcode"
                ));
            }
        })
    }
}

#[derive(Deserialize, Debug, Clone)]
pub struct ObjectStoreConfig {
    /// Object store URL (e.g., "s3://bucket-name", "gs://bucket-name", "http://localhost:9000")
    pub url: String,
    /// Optional configuration options for the object store
    pub options: Option<HashMap<String, String>>,
}

#[derive(Deserialize, Debug, Clone)]
pub struct BenchmarkManifest {
    /// Name of the benchmark suite
    pub name: String,
    /// Description of the benchmark
    pub description: Option<String>,
    /// Data tables configuration: table name -> parquet file path
    pub tables: HashMap<String, String>,
    /// Array of SQL queries - each element can be a file path or inline SQL
    pub queries: Vec<String>,
    /// Optional object store configurations
    pub object_stores: Option<Vec<ObjectStoreConfig>>,
}

#[derive(Parser, Serialize, Clone)]
#[command(name = "In-Process Benchmark")]
struct InProcessBenchmark {
    /// Path to the benchmark manifest file (JSON)
    #[arg(long = "manifest")]
    pub manifest: PathBuf,

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

    /// Maximum cache size in bytes
    #[arg(long = "max-cache-mb")]
    pub max_cache_mb: Option<usize>,

    /// Directory to write flamegraph SVG files to
    #[arg(long = "flamegraph-dir")]
    pub flamegraph_dir: Option<PathBuf>,

    /// Query index to run (0-based), if not provided, all queries will be run
    #[arg(long)]
    pub query_index: Option<usize>,
}

fn write_flamegraph(
    profiler: &pprof::ProfilerGuard,
    flamegraph_dir: &PathBuf,
    query_index: usize,
    iteration: u32,
) -> Result<()> {
    let report = profiler.report().build()?;
    let mut svg_data = Vec::new();
    report.flamegraph(&mut svg_data)?;
    create_dir_all(flamegraph_dir)?;

    let now = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH)?;
    let secs = now.as_secs();
    let hour = (secs / 3600) % 24;
    let minute = (secs / 60) % 60;
    let second = secs % 60;

    let filename = format!(
        "{:02}h{:02}m{:02}s_q{}_i{}.svg",
        hour, minute, second, query_index, iteration
    );
    let filepath = flamegraph_dir.join(filename);
    std::fs::write(&filepath, svg_data)?;
    info!("Flamegraph written to: {}", filepath.display());

    Ok(())
}

impl InProcessBenchmark {
    fn load_manifest(&self) -> Result<BenchmarkManifest> {
        let manifest_content = std::fs::read_to_string(&self.manifest)?;
        let manifest: BenchmarkManifest = serde_json::from_str(&manifest_content)?;
        Ok(manifest)
    }

    /// Determine if a string is a file path or inline SQL
    /// If it ends with .sql or contains path separators, treat as file path
    /// Otherwise, treat as inline SQL
    fn is_file_path(query_str: &str) -> bool {
        query_str.ends_with(".sql")
            || query_str.contains('/')
            || query_str.contains('\\')
            || (!query_str.contains(' ') && !query_str.contains('\n'))
    }

    async fn setup_context(
        &self,
        manifest: &BenchmarkManifest,
    ) -> Result<(Arc<SessionContext>, Option<LiquidCacheRef>)> {
        let mut session_config = SessionConfig::from_env()?;

        session_config
            .options_mut()
            .execution
            .parquet
            .pushdown_filters = true;
        if let Some(partitions) = self.partitions {
            session_config.options_mut().execution.target_partitions = partitions;
        }
        session_config.options_mut().execution.batch_size = 8192 * 2;

        let cache_size = self
            .max_cache_mb
            .map(|size| size * 1024 * 1024)
            .unwrap_or(usize::MAX);

        let cache_dir = std::env::current_dir()?.join("benchmark/data/cache");
        if cache_dir.exists() {
            std::fs::remove_dir_all(&cache_dir)?;
        }
        std::fs::create_dir_all(&cache_dir)?;

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

        // Register object stores if configured
        if let Some(object_stores) = &manifest.object_stores {
            for store_config in object_stores {
                let url = ObjectStoreUrl::parse(&store_config.url)?;
                let options = store_config.options.clone().unwrap_or_default();

                let (object_store, _) = object_store::parse_url_opts(url.as_ref(), options)?;
                ctx.register_object_store(url.as_ref(), Arc::new(object_store));
                info!("Registered object store: {}", store_config.url);
            }
        }

        // Register tables from manifest
        for (table_name, table_path_str) in &manifest.tables {
            ctx.register_parquet(table_name, table_path_str, Default::default())
                .await?;
            info!("Registered table '{}' from {}", table_name, table_path_str);
        }

        Ok((Arc::new(ctx), cache))
    }

    async fn load_queries(&self, manifest: &BenchmarkManifest) -> Result<Vec<Query>> {
        let mut queries = Vec::new();

        for (index, query_str) in manifest.queries.iter().enumerate() {
            let sql = if Self::is_file_path(query_str) {
                // Treat as file path
                let sql = std::fs::read_to_string(query_str)?;
                info!("Loaded query {} from file: {}", index, query_str);
                sql
            } else {
                // Treat as inline SQL
                info!("Loaded query {} from inline SQL", index);
                query_str.clone()
            };

            queries.push(Query {
                id: index as u32,
                sql,
            });
        }

        Ok(queries)
    }

    async fn run_single_iteration(
        &self,
        ctx: &Arc<SessionContext>,
        query: &Query,
        bench_start_time: Instant,
        cache: Option<LiquidCacheRef>,
        iteration: u32,
    ) -> Result<IterationResult> {
        info!(
            "Running query {} iteration {}: \n{}",
            query.id, iteration, query.sql
        );

        // Start flamegraph profiling if enabled
        let profiler_guard = if self.flamegraph_dir.is_some() {
            Some(
                pprof::ProfilerGuardBuilder::default()
                    .frequency(500)
                    .blocklist(&["libpthread.so.0", "libm.so.6", "libgcc_s.so.1"])
                    .build()?,
            )
        } else {
            None
        };

        // Capture disk I/O metrics before query execution
        let mut disk_info = Disks::new_with_refreshed_list();

        let now = Instant::now();
        let starting_timestamp = bench_start_time.elapsed();

        let (_results, _execution_plan, _) = run_query(ctx, &query.sql).await?;
        let elapsed = now.elapsed();

        disk_info.refresh(true);
        let disk_read: u64 = disk_info.iter().map(|disk| disk.usage().read_bytes).sum();
        let disk_written: u64 = disk_info
            .iter()
            .map(|disk| disk.usage().written_bytes)
            .sum();

        // Stop flamegraph profiling and write to file if enabled
        if let Some(profiler) = profiler_guard
            && let Some(flamegraph_dir) = &self.flamegraph_dir
        {
            write_flamegraph(&profiler, flamegraph_dir, query.id as usize, iteration)?;
        }

        let cache_memory_usage = if let Some(cache) = cache {
            cache.memory_usage_bytes()
        } else {
            0
        };

        let result = IterationResult {
            network_traffic: 0,
            time_millis: elapsed.as_millis() as u64,
            cache_cpu_time: 0, // Not easily available for in-process
            cache_memory_usage: cache_memory_usage as u64,
            liquid_cache_usage: cache_memory_usage as u64,
            disk_bytes_read: disk_read,
            disk_bytes_written: disk_written,
            starting_timestamp,
        };

        info!("\n{result}");
        Ok(result)
    }

    pub async fn run(self) -> Result<BenchmarkResult<InProcessBenchmark>> {
        let manifest = self.load_manifest()?;
        info!("Loaded benchmark manifest: {}", manifest.name);

        let (ctx, cache) = self.setup_context(&manifest).await?;
        let queries = self.load_queries(&manifest).await?;

        // Filter queries if specific query index is requested
        let query_indices: Vec<usize> = if let Some(query_index) = self.query_index {
            if query_index >= queries.len() {
                return Err(anyhow::anyhow!(
                    "Query index {} out of range (max: {})",
                    query_index,
                    queries.len() - 1
                ));
            }
            vec![query_index]
        } else {
            (0..queries.len()).collect()
        };

        let mut benchmark_result = BenchmarkResult {
            args: self.clone(),
            results: Vec::new(),
        };

        let bench_start_time = Instant::now();

        for query_index in query_indices {
            let query = &queries[query_index];
            let mut query_result = QueryResult::new(query.id, query.sql.clone());

            for it in 0..self.iteration {
                let iteration_result = self
                    .run_single_iteration(&ctx, query, bench_start_time, cache.clone(), it + 1)
                    .await?;

                query_result.add(iteration_result);

                if self.reset_cache
                    && let Some(cache) = &cache
                {
                    cache.reset();
                }
            }

            benchmark_result.results.push(query_result);
        }

        if let Some(output_path) = &self.output {
            let output_file = File::create(output_path)?;
            serde_json::to_writer_pretty(output_file, &benchmark_result)?;
        }

        Ok(benchmark_result)
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    setup_observability("inprocess", opentelemetry::trace::SpanKind::Client, None);
    let benchmark = InProcessBenchmark::parse();
    benchmark.run().await?;
    Ok(())
}
