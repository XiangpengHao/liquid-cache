use clap::{Parser, arg};
use datafusion::{
    arrow::{array::RecordBatch, util::pretty},
    error::Result,
    parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder,
    physical_plan::{ExecutionPlan, collect, display::DisplayableExecutionPlan},
    prelude::SessionContext,
};
use fastrace::prelude::*;
use liquid_cache_benchmarks::{
    BenchmarkMode, BenchmarkResult, IterationResult, QueryResult, setup_observability,
    utils::assert_batch_eq,
};
use log::{debug, info};
use mimalloc::MiMalloc;
use serde::Serialize;
use std::{fs::File as StdFile, path::Path, sync::Arc};
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
    #[arg(long = "bench-mode", default_value = "liquid-eager-transcode")]
    bench_mode: BenchmarkMode,

    /// Reset the cache before running a new query
    #[arg(long = "reset-cache", default_value = "false")]
    reset_cache: bool,

    /// Number of partitions to use
    #[arg(long)]
    partitions: Option<usize>,
}

/// One query file can contain multiple queries, separated by `;`
fn get_query_by_id(query_dir: impl AsRef<Path>, query_id: u32) -> Result<Vec<String>> {
    let query_dir = query_dir.as_ref();
    let mut path = query_dir.to_owned();
    path.push(format!("q{query_id}.sql"));
    let content = std::fs::read_to_string(&path)?;
    Ok(content
        .split(';')
        .map(|s| s.trim())
        .filter(|s| !s.is_empty())
        .map(|s| s.to_string())
        .collect())
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

#[fastrace::trace]
async fn run_query(
    ctx: &Arc<SessionContext>,
    query: &str,
) -> Result<(Vec<RecordBatch>, Arc<dyn ExecutionPlan>)> {
    let df = ctx
        .sql(query)
        .in_span(Span::enter_with_local_parent("plan_logical"))
        .await?;
    let (state, logical_plan) = df.into_parts();
    let physical_plan = state
        .create_physical_plan(&logical_plan)
        .in_span(Span::enter_with_local_parent("plan_physical"))
        .await?;
    let results = collect(physical_plan.clone(), state.task_ctx())
        .in_span(Span::enter_with_local_parent("collect"))
        .await?;
    Ok((results, physical_plan))
}

#[tokio::main]
pub async fn main() -> Result<()> {
    setup_observability("tpch-client", opentelemetry::trace::SpanKind::Client);

    let args = CliArgs::parse();

    let ctx = args
        .bench_mode
        .setup_tpch_ctx(&args.server, &args.data_dir, args.partitions)
        .await?;

    let mut benchmark_result = BenchmarkResult {
        args: args.clone(),
        results: Vec::new(),
    };

    let query_ids = if let Some(query) = args.query {
        vec![query]
    } else {
        (1..=22).collect()
    };

    std::fs::create_dir_all("benchmark/data/results")?;

    let mut networks = Networks::new_with_refreshed_list();
    let bench_start_time = Instant::now();

    for id in query_ids {
        let query = get_query_by_id(&args.query_dir, id)?;
        let mut query_result = QueryResult::new(id, query.join(";"));
        for it in 0..args.iteration {
            let root = Span::root(format!("tpch-client-{}-{}", id, it), SpanContext::random());
            let _g = root.set_local_parent();
            info!("Running query {}: \n{}", id, query.join(";"));
            let now = Instant::now();
            let starting_timestamp = bench_start_time.elapsed();
            let (results, physical_plan) = if id == 15 {
                // Q15 has three queries, the second one is the one we want to test
                let mut results = Vec::new();
                let mut physical_plan = None;
                for (i, q) in query.iter().enumerate() {
                    let (result, plan) = run_query(&ctx, q).await?;
                    if i == 1 {
                        physical_plan = Some(plan);
                        results = result;
                    }
                }
                (results, physical_plan.unwrap())
            } else {
                run_query(&ctx, &query[0]).await?
            };
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
                physical_plan_with_metrics.indent(true)
            );
            let result_str = pretty::pretty_format_batches(&results).unwrap();
            info!("Query result: \n{}", result_str);

            // Check query answers
            if let Some(answer_dir) = &args.answer_dir {
                check_result_against_answer(&results, answer_dir, id)?;
                info!("Query {} passed validation", id);
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

    fastrace::flush();
    Ok(())
}
