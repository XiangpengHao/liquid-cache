use std::{
    fmt::Display,
    fs::File,
    io::{BufRead, BufReader},
    path::{Path, PathBuf},
    str::FromStr,
    sync::Arc,
    time::{Duration, Instant},
};

use arrow_flight::{FlightClient, flight_service_client::FlightServiceClient, sql::Any};
use clap::{Parser, arg, command};
use datafusion::{
    arrow::{array::RecordBatch, util::pretty},
    error::Result,
    parquet::{
        arrow::{ArrowWriter, arrow_reader::ParquetRecordBatchReaderBuilder},
        basic::Compression,
        file::properties::WriterProperties,
    },
    physical_plan::{
        ExecutionPlan, collect, display::DisplayableExecutionPlan, metrics::MetricValue,
    },
    prelude::{SessionConfig, SessionContext},
};
use futures::StreamExt;
use liquid_cache_benchmarks::utils::assert_batch_eq;
use liquid_cache_client::LiquidCacheTableFactory;
use liquid_common::{
    ParquetMode,
    rpc::{ExecutionMetricsResponse, LiquidCacheActions},
};
use log::{debug, info};
use mimalloc::MiMalloc;
use object_store::ClientConfigKey;
use owo_colors::OwoColorize;
use prost::Message;
use serde::Serialize;
use std::fs::File as StdFile;
use sysinfo::Networks;
use tonic::transport::Channel;
use url::Url;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

#[derive(Serialize)]
struct BenchmarkResult {
    server_url: String,
    file_path: PathBuf,
    query_path: PathBuf,
    iteration: u32,
    bench_mode: BenchmarkMode,
    output_path: Option<PathBuf>,
    queries: Vec<QueryResult>,
}

#[derive(Serialize)]
struct QueryResult {
    id: u32,
    query: String,
    iteration_results: Vec<IterationResult>,
}

impl QueryResult {
    fn new(id: u32, query: String) -> Self {
        Self {
            id,
            query,
            iteration_results: Vec::new(),
        }
    }

    fn add(&mut self, iteration_result: IterationResult) {
        self.iteration_results.push(iteration_result);
    }
}
#[derive(Serialize)]
struct IterationResult {
    network_traffic: u64,
    time_millis: u64,
    cache_cpu_time: u64,
    cache_memory_usage: u64,
    starting_timestamp: Duration,
}

#[derive(Clone, Debug, Default, Copy, PartialEq, Eq, Serialize)]
enum BenchmarkMode {
    ParquetFileserver,
    ParquetPushdown,
    ArrowPushdown,
    #[default]
    LiquidCache,
    LiquidEagerTranscode,
}

impl BenchmarkMode {
    async fn setup_ctx(&self, server_url: &str, file_path: &Path) -> Result<Arc<SessionContext>> {
        let mut session_config = SessionConfig::from_env()?;
        let table_name = "hits";
        let current_dir = std::env::current_dir()?.to_string_lossy().to_string();
        let table_url =
            Url::parse(&format!("file://{}/{}", current_dir, file_path.display())).unwrap();

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
            BenchmarkMode::ParquetPushdown => ParquetMode::Original,
            BenchmarkMode::ArrowPushdown => ParquetMode::Arrow,
            BenchmarkMode::LiquidCache => ParquetMode::Liquid,
            BenchmarkMode::LiquidEagerTranscode => ParquetMode::LiquidEagerTranscode,
        };
        session_config
            .options_mut()
            .execution
            .parquet
            .pushdown_filters = true;
        let ctx = Arc::new(SessionContext::new_with_config(session_config));

        let table =
            LiquidCacheTableFactory::open_table(server_url, table_name, table_url, mode).await?;
        ctx.register_table(table_name, Arc::new(table))?;
        Ok(ctx)
    }

    async fn get_execution_metrics(
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
                let mut flight_client = get_flight_client(server_url).await;
                let action = LiquidCacheActions::ExecutionMetrics.into();
                let mut result_stream = flight_client.do_action(action).await.unwrap();
                let result = result_stream.next().await.unwrap().unwrap();
                let any = Any::decode(&*result).unwrap();
                any.unpack::<ExecutionMetricsResponse>().unwrap().unwrap()
            }
        }
    }

    async fn reset_cache(&self, server_url: &str) -> Result<()> {
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

fn get_query(
    query_path: impl AsRef<Path>,
    query_number: Option<u32>,
) -> Result<Vec<(u32, String)>> {
    let query_path = query_path.as_ref();
    let file = File::open(query_path)?;
    let reader = BufReader::new(file);
    let mut queries = Vec::new();
    for (index, line) in reader.lines().enumerate() {
        queries.push((index as u32, line?));
    }
    if let Some(query_number) = query_number {
        Ok(queries
            .into_iter()
            .filter(|(id, _)| *id == query_number)
            .collect())
    } else {
        Ok(queries)
    }
}

fn save_result(result: &[RecordBatch], query_id: u32) -> Result<()> {
    let file_path = format!("benchmark/data/results/Q{}.parquet", query_id);
    let file = File::create(&file_path)?;
    let props = WriterProperties::builder()
        .set_compression(Compression::SNAPPY)
        .build();
    let mut writer = ArrowWriter::try_new(file, result[0].schema(), Some(props)).unwrap();
    for batch in result {
        writer.write(batch).unwrap();
    }
    writer.close().unwrap();
    info!(
        "Query {} result saved to {}",
        query_id.to_string().red(),
        file_path.yellow()
    );
    Ok(())
}

fn check_result_against_answer(
    results: &Vec<RecordBatch>,
    answer_dir: &Path,
    query_id: u32,
    query: &str,
) -> Result<()> {
    // - If query returns no results, check if baseline exists
    // - If baseline does not exist, skip query
    // - If baseline exists, panic
    if results.is_empty() {
        let baseline_exists = format!("{}/Q{}.parquet", answer_dir.display(), query_id);
        match File::open(baseline_exists) {
            Err(_) => {
                info!(
                    "Query {} returned no results (matches baseline)",
                    query_id.to_string().red()
                );
                return Ok(());
            }
            Ok(_) => panic!(
                "Query {} returned no results but baseline exists",
                query_id.to_string().red()
            ),
        }
    }
    // Read answers
    let baseline_path = format!("{}/Q{}.parquet", answer_dir.display(), query_id);
    let baseline_file = File::open(baseline_path)?;
    let mut baseline_batches = Vec::new();
    let reader = ParquetRecordBatchReaderBuilder::try_new(baseline_file)?.build()?;
    for batch in reader {
        baseline_batches.push(batch?);
    }

    // Compare answers and result
    let result_batch = datafusion::arrow::compute::concat_batches(&results[0].schema(), results)?;
    let baseline_batch = datafusion::arrow::compute::concat_batches(
        &baseline_batches[0].schema(),
        &baseline_batches,
    )?;
    if query.contains("LIMIT") {
        info!(
            "Query {} contains LIMIT, only validating the shape of the result",
            query_id.to_string().red()
        );
        let (result_num_rows, result_columns) =
            (result_batch.num_rows(), result_batch.columns().len());
        let (baseline_num_rows, baseline_columns) =
            (baseline_batch.num_rows(), baseline_batch.columns().len());
        if result_num_rows != baseline_num_rows || result_columns != baseline_columns {
            save_result(results, query_id)?;
            panic!(
                "Query {} result does not match baseline. Result(num_rows: {}, num_columns: {}), Baseline(num_rows: {}, num_columns: {})",
                query_id.to_string().red(),
                result_num_rows,
                result_columns,
                baseline_num_rows,
                baseline_columns,
            );
        }
    } else if !assert_batch_eq(&result_batch, &baseline_batch) {
        save_result(results, query_id)?;
        panic!(
            "Query {} result does not match baseline. Result: {:?}, Baseline: {:?}",
            query_id.to_string().red(),
            result_batch.red(),
            baseline_batch.red()
        );
    }
    Ok(())
}

#[derive(Parser)]
#[command(name = "ClickBench Benchmark Client")]
struct CliArgs {
    /// Path to the query file
    #[arg(long)]
    query_path: PathBuf,

    /// Query number to run, if not provided, all queries will be run
    #[arg(long)]
    query: Option<u32>,

    /// Path to the ClickBench file, hit.parquet or directory to partitioned files
    #[arg(long)]
    file: PathBuf,

    /// Server URL
    #[arg(long, default_value = "http://localhost:50051")]
    server: String,

    /// Number of times to run each query
    #[arg(long, default_value = "3")]
    iteration: u32,

    /// Path to the output JSON file
    #[arg(long)]
    output: Option<PathBuf>,

    /// Path to the baseline directory
    #[arg(long = "answer-dir")]
    answer_dir: Option<PathBuf>,

    /// Benchmark mode to use
    #[arg(long = "bench-mode", default_value = "liquid-cache")]
    bench_mode: BenchmarkMode,

    /// Reset the cache before running a new query
    #[arg(long = "reset-cache", default_value = "false")]
    reset_cache: bool,
}

#[tokio::main]
pub async fn main() -> Result<()> {
    env_logger::builder().format_timestamp(None).init();

    let args = CliArgs::parse();

    let queries = get_query(&args.query_path, args.query)?;
    let bench_mode = &args.bench_mode;
    let ctx = bench_mode.setup_ctx(&args.server, &args.file).await?;

    let mut benchmark_result = BenchmarkResult {
        server_url: args.server.clone(),
        file_path: args.file.clone(),
        query_path: args.query_path.clone(),
        iteration: args.iteration,
        output_path: args.output.clone(),
        bench_mode: *bench_mode,
        queries: Vec::new(),
    };

    std::fs::create_dir_all("benchmark/data/results")?;

    let mut networks = Networks::new_with_refreshed_list();
    let bench_start_time = Instant::now();

    for (id, query) in queries {
        let mut query_result = QueryResult::new(id, query.clone());
        for _i in 0..args.iteration {
            info!("Running query {}: \n{}", id.magenta(), query.cyan());
            let now = Instant::now();
            let starting_timestamp = bench_start_time.elapsed();
            let df = ctx.sql(&query).await?;
            let (state, logical_plan) = df.into_parts();

            let physical_plan = state.create_physical_plan(&logical_plan).await?;
            let results = collect(physical_plan.clone(), state.task_ctx()).await?;
            let elapsed = now.elapsed();
            info!("Query execution time: {:?}", elapsed);

            networks.refresh(true);
            // for mac its lo0 and for linux its lo.
            let network_info = networks
                .get("lo0")
                .or_else(|| networks.get("lo"))
                .expect("No loopback interface found in networks");
            let physical_plan_with_metrics =
                DisplayableExecutionPlan::with_metrics(physical_plan.as_ref());

            debug!(
                "Physical plan: \n{}",
                physical_plan_with_metrics.indent(true).magenta()
            );
            let result_str = pretty::pretty_format_batches(&results).unwrap();
            info!("Query result: \n{}", result_str.cyan());

            // Check query answers
            if let Some(answer_dir) = &args.answer_dir {
                check_result_against_answer(&results, answer_dir, id, &query)?;
                info!("Query {} passed validation", id.to_string().red());
            }

            let metrics_response = bench_mode
                .get_execution_metrics(&args.server, &physical_plan)
                .await;
            info!(
                "Server processing time: {} ms, cache memory usage: {} bytes, liquid cache usage: {} bytes",
                metrics_response.pushdown_eval_time,
                metrics_response.cache_memory_usage,
                metrics_response.liquid_cache_usage
            );

            query_result.add(IterationResult {
                network_traffic: network_info.received(),
                time_millis: elapsed.as_millis() as u64,
                cache_cpu_time: metrics_response.pushdown_eval_time,
                cache_memory_usage: metrics_response.cache_memory_usage,
                starting_timestamp,
            });
        }
        if args.reset_cache {
            bench_mode.reset_cache(&args.server).await?;
        }
        benchmark_result.queries.push(query_result);
    }

    if let Some(output_path) = &args.output {
        let output_file = StdFile::create(output_path)?;
        serde_json::to_writer_pretty(output_file, &benchmark_result).unwrap();
    }

    Ok(())
}

async fn get_flight_client(server_url: &str) -> FlightClient {
    let endpoint = Channel::from_shared(server_url.to_string()).unwrap();
    let channel = endpoint.connect().await.unwrap();
    let inner_client = FlightServiceClient::new(channel);
    FlightClient::new_from_inner(inner_client)
}
