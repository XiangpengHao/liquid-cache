use crate::manifest::BenchmarkManifest;
use crate::{BenchmarkResult, IterationResult, PerfEventStats, Query, QueryResult, run_query};
use anyhow::Result;
use datafusion::arrow::array::RecordBatch;
use datafusion::arrow::datatypes::SchemaRef;
use datafusion::arrow::util::pretty::pretty_format_batches;
use datafusion::parquet::{
    arrow::ArrowWriter, basic::Compression, file::properties::WriterProperties,
};
use datafusion::prelude::{SessionConfig, SessionContext};
use liquid_cache_common::IoMode;
use liquid_cache_datafusion_local::LiquidCacheLocalBuilder;
use liquid_cache_datafusion::{LiquidCacheParquetRef, extract_execution_metrics};
use liquid_cache::cache::NoHydration;
use liquid_cache::cache::squeeze_policies::{Evict, TranscodeEvict, TranscodeSqueezeEvict};
use liquid_cache::cache_policies::LiquidPolicy;
use log::{info, warn};
use perf_event::{
    Builder as PerfBuilder, Counter, Group,
    events::{Hardware, Software},
};
use serde::Serialize;
use std::{
    fs::{File, create_dir_all},
    io,
    path::{Path, PathBuf},
    sync::Arc,
    time::Instant,
};

struct DiskIoGuard {
    system: sysinfo::System,
    pid: sysinfo::Pid,
    start_read_total: u64,
    start_written_total: u64,
}

impl DiskIoGuard {
    fn new() -> Self {
        let mut system = sysinfo::System::new();
        let pid = sysinfo::get_current_pid().unwrap();
        system.refresh_processes_specifics(
            sysinfo::ProcessesToUpdate::Some(&[pid]),
            true,
            sysinfo::ProcessRefreshKind::nothing().with_disk_usage(),
        );
        let p = system.process(pid).unwrap();
        let du = p.disk_usage();
        Self {
            system,
            pid,
            start_read_total: du.total_read_bytes,
            start_written_total: du.total_written_bytes,
        }
    }

    fn stop(mut self) -> (u64, u64) {
        self.system.refresh_processes_specifics(
            sysinfo::ProcessesToUpdate::Some(&[self.pid]),
            true,
            sysinfo::ProcessRefreshKind::nothing().with_disk_usage(),
        );
        if let Some(p) = self.system.process(self.pid) {
            let du = p.disk_usage();
            (
                du.total_read_bytes.saturating_sub(self.start_read_total),
                du.total_written_bytes
                    .saturating_sub(self.start_written_total),
            )
        } else {
            (0, 0)
        }
    }
}

struct PerfEventCollector {
    group: Group,
    cycles: Counter,
    instructions: Counter,
    cache_references: Counter,
    cache_misses: Counter,
    context_switches: Counter,
    page_faults: Counter,
}

impl PerfEventCollector {
    fn new() -> io::Result<Self> {
        let mut group = Group::new()?;
        // Target the whole benchmark process (all threads), not just the current one.
        let pid = std::process::id() as i32;
        let mut cycles_builder = PerfBuilder::new(Hardware::CPU_CYCLES);
        cycles_builder.observe_pid(pid);

        let mut instructions_builder = PerfBuilder::new(Hardware::INSTRUCTIONS);
        instructions_builder.observe_pid(pid);

        let mut cache_refs_builder = PerfBuilder::new(Hardware::CACHE_REFERENCES);
        cache_refs_builder.observe_pid(pid);

        let mut cache_misses_builder = PerfBuilder::new(Hardware::CACHE_MISSES);
        cache_misses_builder.observe_pid(pid);

        let mut ctx_switch_builder = PerfBuilder::new(Software::CONTEXT_SWITCHES);
        ctx_switch_builder.observe_pid(pid);

        let mut page_faults_builder = PerfBuilder::new(Software::PAGE_FAULTS);
        page_faults_builder.observe_pid(pid);

        let cycles = group.add(&cycles_builder)?;
        let instructions = group.add(&instructions_builder)?;
        let cache_references = group.add(&cache_refs_builder)?;
        let cache_misses = group.add(&cache_misses_builder)?;
        let context_switches = group.add(&ctx_switch_builder)?;
        let page_faults = group.add(&page_faults_builder)?;

        Ok(Self {
            group,
            cycles,
            instructions,
            cache_references,
            cache_misses,
            context_switches,
            page_faults,
        })
    }

    fn start(&mut self) -> io::Result<()> {
        self.group.enable()
    }

    fn stop(mut self) -> io::Result<PerfEventStats> {
        self.group.disable()?;
        let counts = self.group.read()?;
        Ok(PerfEventStats {
            cpu_cycles: counts
                .get(&self.cycles)
                .map(|entry| entry.value())
                .unwrap_or(0),
            instructions: counts
                .get(&self.instructions)
                .map(|entry| entry.value())
                .unwrap_or(0),
            cache_references: counts
                .get(&self.cache_references)
                .map(|entry| entry.value())
                .unwrap_or(0),
            cache_misses: counts
                .get(&self.cache_misses)
                .map(|entry| entry.value())
                .unwrap_or(0),
            context_switches: counts
                .get(&self.context_switches)
                .map(|entry| entry.value())
                .unwrap_or(0),
            page_faults: counts
                .get(&self.page_faults)
                .map(|entry| entry.value())
                .unwrap_or(0),
        })
    }
}

#[derive(Clone, Debug, Default, Copy, PartialEq, Eq, Serialize)]
pub enum InProcessBenchmarkMode {
    Parquet,
    /// Plain DataFusion with default SessionConfig (no tweaks)
    DataFusionDefault,
    Arrow,
    #[default]
    Liquid,
    LiquidNoSqueeze,
}

impl std::str::FromStr for InProcessBenchmarkMode {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Ok(match s {
            "parquet" => InProcessBenchmarkMode::Parquet,
            "datafusion-default" | "datafusion" => InProcessBenchmarkMode::DataFusionDefault,
            "arrow" => InProcessBenchmarkMode::Arrow,
            "liquid" => InProcessBenchmarkMode::Liquid,
            "liquid-no-squeeze" => InProcessBenchmarkMode::LiquidNoSqueeze,
            _ => {
                return Err(format!(
                    "Invalid in-process benchmark mode: {s}, must be one of: parquet, datafusion-default, arrow, liquid, liquid-no-squeeze"
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
    pub cache_dir: Option<PathBuf>,
    pub io_mode: IoMode,
    pub output_dir: Option<PathBuf>,
    pub collect_perf_events: bool,
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
            cache_dir: None,
            io_mode: IoMode::default(),
            output_dir: None,
            collect_perf_events: false,
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

    pub fn with_perf_events(mut self, collect: bool) -> Self {
        self.collect_perf_events = collect;
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

    pub fn with_cache_dir(mut self, cache_dir: Option<PathBuf>) -> Self {
        self.cache_dir = cache_dir;
        self
    }

    pub fn with_io_mode(mut self, io_mode: IoMode) -> Self {
        self.io_mode = io_mode;
        self
    }

    pub fn with_output_dir(mut self, output_dir: Option<PathBuf>) -> Self {
        self.output_dir = output_dir;
        self
    }

    #[fastrace::trace]
    async fn setup_context(
        &self,
        manifest: &BenchmarkManifest,
    ) -> Result<(Arc<SessionContext>, Option<LiquidCacheParquetRef>)> {
        let mut session_config = SessionConfig::from_env()?;
        // Apply some helpful defaults for non-DataFusionDefault modes
        if self.bench_mode != InProcessBenchmarkMode::DataFusionDefault {
            session_config
                .options_mut()
                .execution
                .parquet
                .pushdown_filters = true;
            if let Some(partitions) = self.partitions {
                session_config.options_mut().execution.target_partitions = partitions;
            }
            // session_config.options_mut().execution.batch_size = 8192 * 2;
        }

        let cache_size = self
            .max_cache_mb
            .map(|size| size * 1024 * 1024)
            .unwrap_or(usize::MAX);

        let cache_dir = self
            .cache_dir
            .clone()
            .unwrap_or(std::env::current_dir()?.join("benchmark/data/cache"));
        if cache_dir.exists() {
            std::fs::remove_dir_all(&cache_dir)?;
        }
        std::fs::create_dir_all(&cache_dir)?;

        let (ctx, cache): (SessionContext, Option<LiquidCacheParquetRef>) = match self.bench_mode {
            InProcessBenchmarkMode::Parquet => {
                let mut session_config = session_config.clone();
                session_config
                    .options_mut()
                    .execution
                    .parquet
                    .pushdown_filters = true;
                (SessionContext::new_with_config(session_config), None)
            }
            InProcessBenchmarkMode::DataFusionDefault => {
                // Use DataFusion's default SessionConfig without any custom tweaks
                (SessionContext::new_with_config(SessionConfig::new()), None)
            }
            InProcessBenchmarkMode::Arrow => {
                let v = LiquidCacheLocalBuilder::new()
                    .with_max_cache_bytes(cache_size)
                    .with_cache_dir(cache_dir)
                    .with_cache_policy(Box::new(LiquidPolicy::new()))
                    .with_hydration_policy(Box::new(NoHydration::new()))
                    .with_squeeze_policy(Box::new(Evict))
                    .build(session_config)?;
                (v.0, Some(v.1))
            }
            InProcessBenchmarkMode::Liquid => {
                let v = LiquidCacheLocalBuilder::new()
                    .with_max_cache_bytes(cache_size)
                    .with_cache_dir(cache_dir)
                    .with_cache_policy(Box::new(LiquidPolicy::new()))
                    .with_hydration_policy(Box::new(NoHydration::new()))
                    .with_squeeze_policy(Box::new(TranscodeSqueezeEvict))
                    .with_io_mode(self.io_mode)
                    .with_eager_shredding(true)
                    .build(session_config)?;
                (v.0, Some(v.1))
            }
            InProcessBenchmarkMode::LiquidNoSqueeze => {
                let v = LiquidCacheLocalBuilder::new()
                    .with_max_cache_bytes(cache_size)
                    .with_cache_dir(cache_dir)
                    .with_cache_policy(Box::new(LiquidPolicy::new()))
                    .with_hydration_policy(Box::new(NoHydration::new()))
                    .with_squeeze_policy(Box::new(TranscodeEvict))
                    .with_io_mode(self.io_mode)
                    .build(session_config)?;
                (v.0, Some(v.1))
            }
        };

        if let Some(object_stores) = manifest.get_object_store() {
            for (url, object_store) in object_stores {
                ctx.register_object_store(url.as_ref(), Arc::new(object_store));
            }
        }

        Self::register_manifest_tables(&ctx, manifest).await?;

        Ok((Arc::new(ctx), cache))
    }

    async fn register_manifest_tables(
        ctx: &SessionContext,
        manifest: &BenchmarkManifest,
    ) -> Result<()> {
        for (table_name, table_path_str) in &manifest.tables {
            ctx.register_parquet(
                format!("\"{table_name}\""),
                table_path_str,
                Default::default(),
            )
            .await?;
            info!("Registered table '{table_name}' from {table_path_str}");
        }
        Ok(())
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
            let (r, plan, _) = run_query(ctx, q).await;
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

    fn write_query_result(
        &self,
        output_dir: &Path,
        query: &Query,
        iteration: u32,
        schema: &SchemaRef,
        batches: &[RecordBatch],
    ) -> Result<()> {
        create_dir_all(output_dir)?;
        let filename = format!(
            "q{query_id:02}_i{iteration:02}.parquet",
            query_id = query.id()
        );
        let path = output_dir.join(filename);
        let file = File::create(&path)?;
        let props = WriterProperties::builder()
            .set_compression(Compression::SNAPPY)
            .build();
        let mut writer = ArrowWriter::try_new(file, schema.clone(), Some(props))?;
        for batch in batches {
            writer.write(batch)?;
        }
        writer.close()?;
        info!(
            "Query {} iteration {} result saved to {}",
            query.id(),
            iteration,
            path.display()
        );
        Ok(())
    }

    #[fastrace::trace]
    async fn run_single_iteration(
        &self,
        ctx: &Arc<SessionContext>,
        query: &Query,
        bench_start_time: Instant,
        cache: Option<LiquidCacheParquetRef>,
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

        let io_guard = DiskIoGuard::new();

        let perf_collector = if self.collect_perf_events {
            match PerfEventCollector::new() {
                Ok(mut collector) => {
                    if let Err(err) = collector.start() {
                        warn!("Failed to enable perf events; skipping counters: {err}");
                        None
                    } else {
                        Some(collector)
                    }
                }
                Err(err) => {
                    warn!("Failed to initialize perf events; skipping counters: {err}");
                    None
                }
            }
        } else {
            None
        };

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

        let perf_events = if let Some(collector) = perf_collector {
            match collector.stop() {
                Ok(stats) => Some(stats),
                Err(err) => {
                    warn!("Failed to read perf events; skipping counters: {err}");
                    None
                }
            }
        } else {
            None
        };

        let (disk_read, disk_written) = io_guard.stop();

        let cache_stats = cache.as_ref().map(|cache| cache.storage().stats());

        // Stop flamegraph profiling and write to file if enabled
        if let Some(profiler) = profiler_guard {
            self.write_flamegraph(&profiler, query.id, iteration)?;
        }

        // Extract execution metrics using the shared function
        let metrics = extract_execution_metrics(&execution_plan, cache.as_ref());

        if let Some(output_dir) = &self.output_dir {
            let schema = results
                .first()
                .map(|b| b.schema())
                .unwrap_or_else(|| execution_plan.schema());
            self.write_query_result(output_dir, query, iteration, &schema, &results)?;
        }

        let result = IterationResult {
            network_traffic: 0,
            time_millis: elapsed.as_millis() as u64,
            cache_cpu_time: metrics.pushdown_eval_time,
            cache_memory_usage: metrics.cache_memory_usage,
            liquid_cache_usage: metrics.liquid_cache_usage,
            disk_bytes_read: disk_read,
            disk_bytes_written: disk_written,
            starting_timestamp,
            cache_stats,
            perf_events,
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
                crate::tracepoints::iteration_start(query.id(), it);
                let iteration_result = self
                    .run_single_iteration(&ctx, query, bench_start_time, cache.clone(), it)
                    .await?;

                query_result.add(iteration_result);
            }

            if self.reset_cache
                && let Some(cache) = &cache
            {
                unsafe {
                    cache.reset();
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
