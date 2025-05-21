use std::{
    fs::File,
    io::{BufRead, BufReader},
    path::{Path, PathBuf},
    time::Instant,
};

use clap::{Parser, arg, command};
use datafusion::{
    arrow::{array::RecordBatch, util::pretty},
    error::Result,
    parquet::{
        arrow::{ArrowWriter, arrow_reader::ParquetRecordBatchReaderBuilder},
        basic::Compression,
        file::properties::WriterProperties,
    },
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
use std::fs::File as StdFile;
use sysinfo::Networks;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

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
    let file_path = format!("benchmark/data/results/Q{query_id}.parquet");
    let file = File::create(&file_path)?;
    let props = WriterProperties::builder()
        .set_compression(Compression::SNAPPY)
        .build();
    let mut writer = ArrowWriter::try_new(file, result[0].schema(), Some(props)).unwrap();
    for batch in result {
        writer.write(batch).unwrap();
    }
    writer.close().unwrap();
    info!("Query {query_id} result saved to {file_path}");
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
                info!("Query {query_id} returned no results (matches baseline)");
                return Ok(());
            }
            Ok(_) => panic!("Query {query_id} returned no results but baseline exists"),
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
        info!("Query {query_id} contains LIMIT, only validating the shape of the result");
        let (result_num_rows, result_columns) =
            (result_batch.num_rows(), result_batch.columns().len());
        let (baseline_num_rows, baseline_columns) =
            (baseline_batch.num_rows(), baseline_batch.columns().len());
        if result_num_rows != baseline_num_rows || result_columns != baseline_columns {
            save_result(results, query_id)?;
            panic!(
                "Query {query_id} result does not match baseline. Result(num_rows: {result_num_rows}, num_columns: {result_columns}), Baseline(num_rows: {baseline_num_rows}, num_columns: {baseline_columns})"
            );
        }
    } else {
        assert_batch_eq(&result_batch, &baseline_batch);
    }
    Ok(())
}

#[derive(Parser, Serialize, Clone)]
#[command(name = "ClickBench Benchmark Client")]
struct CliArgs {
    /// Path to the query file
    #[arg(long)]
    query_path: PathBuf,

    /// Path to the ClickBench file, hit.parquet or directory to partitioned files
    #[arg(long)]
    file: PathBuf,

    #[clap(flatten)]
    common: CommonBenchmarkArgs,
}

#[tokio::main]
pub async fn main() -> Result<()> {
    let args = CliArgs::parse();
    setup_observability(
        "clickbench-client",
        opentelemetry::trace::SpanKind::Client,
        args.common.openobserve_auth.as_deref(),
    );

    let queries = get_query(&args.query_path, args.common.query)?;
    let ctx = args.common.setup_clickbench_ctx(&args.file).await?;

    let mut benchmark_result = BenchmarkResult {
        args: args.clone(),
        results: Vec::new(),
    };

    std::fs::create_dir_all("benchmark/data/results")?;

    let mut networks = Networks::new_with_refreshed_list();
    let bench_start_time = Instant::now();

    for (id, query) in queries {
        let mut query_result = QueryResult::new(id, query.clone());
        for it in 0..args.common.iteration {
            info!("Running query {id}: \n{query}");

            args.common.start_trace().await;
            args.common.start_flamegraph().await;

            let root = Span::root(
                format!("clickbench-client-{id}-{it}"),
                SpanContext::random(),
            );
            let _g = root.set_local_parent();
            let now = Instant::now();
            let starting_timestamp = bench_start_time.elapsed();
            let (results, physical_plan) = run_query(&ctx, &query).await?;
            let elapsed = now.elapsed();

            networks.refresh(true);
            // for mac its lo0 and for linux its lo.
            let network_info = networks
                .get("lo0")
                .or_else(|| networks.get("lo"))
                .expect("No loopback interface found in networks");

            args.common.stop_flamegraph().await;
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
                check_result_against_answer(&results, answer_dir, id, &query)?;
                info!("Query {id} passed validation");
            }

            args.common.get_cache_stats().await;

            let metrics_response = args.common.get_execution_metrics(&physical_plan).await;

            let result = IterationResult {
                network_traffic: network_info.received(),
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
        let output_file = StdFile::create(output_path)?;
        serde_json::to_writer_pretty(output_file, &benchmark_result).unwrap();
    }

    fastrace::flush();
    Ok(())
}
