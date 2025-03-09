use clap::{Parser, arg};
use datafusion::{
    arrow::{array::RecordBatch, util::pretty},
    error::Result,
    parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder,
    physical_plan::{collect, display::DisplayableExecutionPlan},
};
use liquid_cache_benchmarks::{
    BenchmarkMode, BenchmarkResult, IterationResult, QueryResult, utils::assert_batch_eq,
};
use log::{debug, info};
use mimalloc::MiMalloc;
use owo_colors::OwoColorize;
use serde::Serialize;
use std::{fs::File as StdFile, path::Path};
use std::{fs::File, path::PathBuf, time::Instant};
use sysinfo::Networks;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

#[derive(Parser, Serialize, Clone)]
#[command(name = "TPCH Benchmark Client")]
struct CliArgs {
    /// Path to the query directory
    #[arg(long = "query-dir")]
    query_dir: PathBuf,

    /// Query number to run, if not provided, all queries will be run
    #[arg(long)]
    query: Option<u32>,

    /// Path to the data directory with TPCH data
    #[arg(long = "data-dir")]
    data_dir: PathBuf,

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

fn get_queries(query_dir: impl AsRef<Path>, query: Option<u32>) -> Vec<(u32, PathBuf, String)> {
    let query_dir = query_dir.as_ref();

    let query_data: Vec<(u32, PathBuf, String)> = (1..22)
        .map(|i| {
            // Convert to owned PathBuf
            let mut path = query_dir.to_owned();
            path.push(format!("q{i}.sql"));

            let query = std::fs::read_to_string(&path).unwrap();
            (i, path, query)
        })
        .filter(|(i, _, _)| query.is_none() || Some(*i) == query)
        .collect();

    query_data
}

fn check_result_against_answer(
    results: &Vec<RecordBatch>,
    answer_dir: &Path,
    query_id: u32,
) -> Result<()> {
    let baseline_path = format!("{}/q{}.parquet", answer_dir.display(), query_id);
    let baseline_file = File::open(baseline_path)?;
    let mut baseline_batches = Vec::new();
    let reader = ParquetRecordBatchReaderBuilder::try_new(baseline_file)?.build()?;
    for batch in reader {
        baseline_batches.push(batch?);
    }

    // Compare answers and result
    let result_batch = datafusion::arrow::compute::concat_batches(&results[0].schema(), results)?;
    let answer_batch = datafusion::arrow::compute::concat_batches(
        &baseline_batches[0].schema(),
        &baseline_batches,
    )?;
    assert_batch_eq(&answer_batch, &result_batch);
    Ok(())
}

#[tokio::main]
pub async fn main() -> Result<()> {
    env_logger::builder().format_timestamp(None).init();

    let args = CliArgs::parse();

    let ctx = args
        .bench_mode
        .setup_tpch_ctx(&args.server, &args.data_dir)
        .await?;

    let mut benchmark_result = BenchmarkResult {
        args: args.clone(),
        results: Vec::new(),
    };

    let queries = get_queries(&args.query_dir, args.query);

    std::fs::create_dir_all("benchmark/data/results")?;

    let mut networks = Networks::new_with_refreshed_list();
    let bench_start_time = Instant::now();

    for (id, _, query) in queries {
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
                check_result_against_answer(&results, answer_dir, id)?;
                info!("Query {} passed validation", id.to_string().red());
            }

            let metrics_response = args
                .bench_mode
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
            args.bench_mode.reset_cache(&args.server).await?;
        }
        benchmark_result.results.push(query_result);
    }

    if let Some(output_path) = &args.output {
        let output_file = StdFile::create(output_path)?;
        serde_json::to_writer_pretty(output_file, &benchmark_result).unwrap();
    }

    Ok(())
}
