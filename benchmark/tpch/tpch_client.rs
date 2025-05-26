use clap::{Parser, arg};
use datafusion::{
    arrow::{array::RecordBatch, util::pretty},
    error::Result,
    parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder,
    physical_plan::display::DisplayableExecutionPlan,
};
use fastrace::prelude::*;
use liquid_cache_benchmarks::{
    BenchmarkResult, CommonBenchmarkArgs, IterationResult, QueryResult, run_query,
    setup_observability, utils::assert_batch_eq,
};
use log::{debug, info};
use mimalloc::MiMalloc;
use serde::Serialize;
use std::{
    fs::File,
    path::{Path, PathBuf},
    time::Instant,
};
use sysinfo::Networks;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

#[derive(Parser, Serialize, Clone)]
#[command(name = "TPCH Benchmark Client")]
struct CliArgs {
    /// Path to the query directory
    #[arg(long = "query-dir")]
    query_dir: PathBuf,

    /// Path to the data directory with TPCH data
    #[arg(long = "data-dir")]
    data_dir: PathBuf,

    #[clap(flatten)]
    common: CommonBenchmarkArgs,
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

#[tokio::main]
pub async fn main() -> Result<()> {
    let args = CliArgs::parse();
    setup_observability(
        "tpch-client",
        opentelemetry::trace::SpanKind::Client,
        args.common.openobserve_auth.as_deref(),
    );

    let ctx = args.common.setup_tpch_ctx(&args.data_dir).await?;

    let mut benchmark_result = BenchmarkResult {
        args: args.clone(),
        results: Vec::new(),
    };

    let query_ids = if let Some(query) = args.common.query {
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
        for it in 0..args.common.iteration {
            let root = Span::root(format!("tpch-client-{id}-{it}"), SpanContext::random());
            let _g = root.set_local_parent();
            args.common.start_trace().await;
            args.common.start_flamegraph().await;
            info!("Running query {}: \n{}", id, query.join(";"));
            let now = Instant::now();
            let starting_timestamp = bench_start_time.elapsed();
            let (results, physical_plan, plan_uuid) = if id == 15 {
                // Q15 has three queries, the second one is the one we want to test
                let mut results = Vec::new();
                let mut physical_plan = None;
                let mut plan_uuid = None;
                for (i, q) in query.iter().enumerate() {
                    let (result, plan, uuid) = run_query(&ctx, q).await?;
                    if i == 1 {
                        physical_plan = Some(plan);
                        results = result;
                        plan_uuid = Some(uuid);
                    }
                }
                (results, physical_plan.unwrap(), plan_uuid.unwrap())
            } else {
                run_query(&ctx, &query[0]).await?
            };
            let elapsed = now.elapsed();
            networks.refresh(true);
            // for mac its lo0 and for linux its lo.
            let network_info = networks
                .get("lo0")
                .or_else(|| networks.get("lo"))
                .expect("No loopback interface found in networks");

            let flamegraph = if let Some(plan_uuid) = plan_uuid {
                args.common.stop_flamegraph(&plan_uuid).await
            } else {
                None
            };
            args.common.stop_trace().await;

            let physical_plan_with_metrics =
                DisplayableExecutionPlan::with_metrics(physical_plan.as_ref());
            debug!(
                "Physical plan: \n{}",
                physical_plan_with_metrics.indent(true)
            );
            let result_str = pretty::pretty_format_batches(&results).unwrap();
            debug!("Query result: \n{result_str}");

            // Check query answers
            if let Some(answer_dir) = &args.common.answer_dir {
                check_result_against_answer(&results, answer_dir, id)?;
                info!("Query {id} passed validation");
            }

            args.common.get_cache_stats().await;
            let network_traffic = network_info.received();

            let metrics_response = args.common.get_execution_metrics(&physical_plan).await;

            if let Some(plan_uuid) = plan_uuid {
                args.common
                    .set_execution_stats(
                        &plan_uuid,
                        flamegraph,
                        format!("TPCH-Q{id}"),
                        network_traffic,
                    )
                    .await;
            }

            let result = IterationResult {
                network_traffic,
                time_millis: elapsed.as_millis() as u64,
                cache_cpu_time: metrics_response.pushdown_eval_time,
                cache_memory_usage: metrics_response.cache_memory_usage,
                liquid_cache_usage: metrics_response.liquid_cache_usage,
                starting_timestamp,
            };
            result.log();
            query_result.add(result);
        }
        if args.common.reset_cache {
            args.common.reset_cache().await?;
        }
        benchmark_result.results.push(query_result);
    }

    if let Some(output_path) = &args.common.output {
        let output_file = File::create(output_path)?;
        serde_json::to_writer_pretty(output_file, &benchmark_result).unwrap();
    }

    fastrace::flush();
    Ok(())
}
