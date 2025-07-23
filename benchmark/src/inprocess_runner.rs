use crate::manifest::BenchmarkManifest;
use crate::{BenchmarkResult, IterationResult, Query, QueryResult, run_query};
use anyhow::Result;
use datafusion::arrow::util::pretty::pretty_format_batches;
use datafusion::prelude::{SessionConfig, SessionContext};
use liquid_cache_common::LiquidCacheMode;
use liquid_cache_parquet::cache::policies::ToDiskPolicy;
use liquid_cache_parquet::{
    LiquidCacheInProcessBuilder, LiquidCacheRef, extract_execution_metrics,
};
use log::info;
use serde::Serialize;
use std::{
    fs::{File, create_dir_all},
    path::PathBuf,
    sync::Arc,
    time::Instant,
};
use sysinfo::Disks;

#[derive(Clone, Debug, Default, Copy, PartialEq, Eq, Serialize)]
pub enum InProcessBenchmarkMode {
    Parquet,
    Arrow,
    #[default]
    Liquid,
    LiquidEagerTranscode,
}

impl std::str::FromStr for InProcessBenchmarkMode {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Ok(match s {
            "parquet" => InProcessBenchmarkMode::Parquet,
            "arrow" => InProcessBenchmarkMode::Arrow,
            "liquid" => InProcessBenchmarkMode::Liquid,
            "liquid-eager-transcode" => InProcessBenchmarkMode::LiquidEagerTranscode,
            _ => {
                return Err(format!(
                    "Invalid in-process benchmark mode: {s}, must be one of: parquet, arrow, liquid, liquid-eager-transcode"
                ));
            }
        })
    }
}

pub struct InProcessBenchmarkRunner {
    pub bench_mode: InProcessBenchmarkMode,
    pub iteration: u32,
    pub reset_cache: bool,
    pub partitions: Option<usize>,
    pub max_cache_mb: Option<usize>,
    pub flamegraph_dir: Option<PathBuf>,
    pub query_filter: Option<usize>,
}

impl Default for InProcessBenchmarkRunner {
    fn default() -> Self {
        Self::new()
    }
}

impl InProcessBenchmarkRunner {
    pub fn new() -> Self {
        Self {
            bench_mode: InProcessBenchmarkMode::default(),
            iteration: 3,
            reset_cache: false,
            partitions: None,
            max_cache_mb: None,
            flamegraph_dir: None,
            query_filter: None,
        }
    }

    pub fn with_bench_mode(mut self, mode: InProcessBenchmarkMode) -> Self {
        self.bench_mode = mode;
        self
    }

    pub fn with_iteration(mut self, iteration: u32) -> Self {
        self.iteration = iteration;
        self
    }

    pub fn with_reset_cache(mut self, reset_cache: bool) -> Self {
        self.reset_cache = reset_cache;
        self
    }

    pub fn with_partitions(mut self, partitions: Option<usize>) -> Self {
        self.partitions = partitions;
        self
    }

    pub fn with_max_cache_mb(mut self, max_cache_mb: Option<usize>) -> Self {
        self.max_cache_mb = max_cache_mb;
        self
    }

    pub fn with_flamegraph_dir(mut self, flamegraph_dir: Option<PathBuf>) -> Self {
        self.flamegraph_dir = flamegraph_dir;
        self
    }

    pub fn with_query_filter(mut self, query_filter: Option<usize>) -> Self {
        self.query_filter = query_filter;
        self
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
            InProcessBenchmarkMode::Parquet => {
                let mut session_config = session_config.clone();
                session_config
                    .options_mut()
                    .execution
                    .parquet
                    .pushdown_filters = true;
                (SessionContext::new_with_config(session_config), None)
            }
            InProcessBenchmarkMode::Arrow => {
                let v = LiquidCacheInProcessBuilder::new()
                    .with_max_cache_bytes(cache_size)
                    .with_cache_mode(LiquidCacheMode::Arrow)
                    .with_cache_dir(cache_dir)
                    .with_cache_strategy(Box::new(ToDiskPolicy::new()))
                    .build(session_config)?;
                (v.0, Some(v.1))
            }
            InProcessBenchmarkMode::Liquid => {
                let v = LiquidCacheInProcessBuilder::new()
                    .with_max_cache_bytes(cache_size)
                    .with_cache_mode(LiquidCacheMode::Liquid {
                        transcode_in_background: true,
                    })
                    .with_cache_dir(cache_dir)
                    .with_cache_strategy(Box::new(ToDiskPolicy::new()))
                    .build(session_config)?;
                (v.0, Some(v.1))
            }
            InProcessBenchmarkMode::LiquidEagerTranscode => {
                let v = LiquidCacheInProcessBuilder::new()
                    .with_max_cache_bytes(cache_size)
                    .with_cache_mode(LiquidCacheMode::Liquid {
                        transcode_in_background: false,
                    })
                    .with_cache_dir(cache_dir)
                    .with_cache_strategy(Box::new(ToDiskPolicy::new()))
                    .build(session_config)?;
                (v.0, Some(v.1))
            }
        };

        if let Some(object_stores) = manifest.get_object_store() {
            for (url, object_store) in object_stores {
                ctx.register_object_store(url.as_ref(), Arc::new(object_store));
            }
        }

        // Register tables from manifest
        for (table_name, table_path_str) in &manifest.tables {
            ctx.register_parquet(table_name, table_path_str, Default::default())
                .await?;
            info!("Registered table '{table_name}' from {table_path_str}");
        }

        Ok((Arc::new(ctx), cache))
    }

    async fn execute_query(
        &self,
        ctx: &Arc<SessionContext>,
        query: &Query,
    ) -> Vec<(
        Vec<datafusion::arrow::array::RecordBatch>,
        Arc<dyn datafusion::physical_plan::ExecutionPlan>,
    )> {
        let mut results = Vec::new();

        for q in query.statement() {
            let (r, plan, _) = run_query(ctx, &q).await;
            results.push((r, plan));
        }
        results
    }

    fn write_flamegraph(
        &self,
        profiler: &pprof::ProfilerGuard,
        query_id: u32,
        iteration: u32,
    ) -> Result<()> {
        if let Some(flamegraph_dir) = &self.flamegraph_dir {
            let report = profiler.report().build()?;
            let mut svg_data = Vec::new();
            report.flamegraph(&mut svg_data)?;
            create_dir_all(flamegraph_dir)?;

            let now = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH)?;
            let secs = now.as_secs();
            let hour = (secs / 3600) % 24;
            let minute = (secs / 60) % 60;
            let second = secs % 60;

            let filename =
                format!("{hour:02}h{minute:02}m{second:02}s_q{query_id:02}_i{iteration:02}.svg");
            let filepath = flamegraph_dir.join(filename);
            std::fs::write(&filepath, svg_data)?;
            info!("Flamegraph written to: {}", filepath.display());
        }
        Ok(())
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
            "Running query {} iteration {}: \n{:?}",
            query.id(),
            iteration,
            query.statement()
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

        let (results, execution_plan) = {
            let r = self.execute_query(ctx, query).await;
            if query.id() == 15 && query.statement().len() == 3 {
                // special handle for tpch q15, this is very ugly
                r[1].clone()
            } else {
                r[0].clone()
            }
        };
        let elapsed = now.elapsed();

        disk_info.refresh(true);
        let disk_read: u64 = disk_info.iter().map(|disk| disk.usage().read_bytes).sum();
        let disk_written: u64 = disk_info
            .iter()
            .map(|disk| disk.usage().written_bytes)
            .sum();

        // Stop flamegraph profiling and write to file if enabled
        if let Some(profiler) = profiler_guard {
            self.write_flamegraph(&profiler, query.id, iteration)?;
        }

        // Extract execution metrics using the shared function
        let metrics = extract_execution_metrics(&execution_plan, cache.as_ref());

        let result = IterationResult {
            network_traffic: 0,
            time_millis: elapsed.as_millis() as u64,
            cache_cpu_time: metrics.pushdown_eval_time,
            cache_memory_usage: metrics.cache_memory_usage,
            liquid_cache_usage: metrics.liquid_cache_usage,
            disk_bytes_read: disk_read,
            disk_bytes_written: disk_written,
            starting_timestamp,
        };

        println!("{}", pretty_format_batches(&results).unwrap());

        info!("\n{result}");
        Ok(result)
    }

    pub async fn run<T: Serialize + Clone>(
        self,
        manifest: BenchmarkManifest,
        benchmark_args: T,
        output_path: Option<PathBuf>,
    ) -> Result<BenchmarkResult<T>> {
        info!("Running benchmark: {}", manifest.name);

        let (ctx, cache) = self.setup_context(&manifest).await?;
        let queries = manifest.load_queries(0);

        // Filter queries if specific query index is requested
        let query_indices: Vec<usize> = if let Some(query_index) = self.query_filter {
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
            args: benchmark_args,
            results: Vec::new(),
        };

        let bench_start_time = Instant::now();

        for query_index in query_indices {
            let query = &queries[query_index];
            let mut query_result = QueryResult::new(query.clone());

            for it in 0..self.iteration {
                let iteration_result = self
                    .run_single_iteration(&ctx, query, bench_start_time, cache.clone(), it + 1)
                    .await?;

                query_result.add(iteration_result);

                if self.reset_cache
                    && let Some(cache) = &cache
                {
                    unsafe {
                        cache.reset();
                    }
                }
            }

            benchmark_result.results.push(query_result);
        }

        if let Some(output_path) = &output_path {
            let output_file = File::create(output_path)?;
            serde_json::to_writer_pretty(output_file, &benchmark_result)?;
        }

        Ok(benchmark_result)
    }
}
