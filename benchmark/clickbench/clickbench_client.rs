use clap::Parser;
use datafusion::{
    arrow::array::RecordBatch,
    parquet::{
        arrow::{ArrowWriter, arrow_reader::ParquetRecordBatchReaderBuilder},
        basic::Compression,
        file::properties::WriterProperties,
    },
    prelude::SessionContext,
};
use datafusion::{error::Result, prelude::SessionConfig};
use liquid_cache_benchmarks::{
    Benchmark, BenchmarkManifest, ClientBenchmarkArgs, utils::assert_batch_eq,
};
use liquid_cache_benchmarks::{BenchmarkRunner, Query, run_query};
use liquid_cache_client::LiquidCacheBuilder;
use log::info;
use mimalloc::MiMalloc;
use serde::Serialize;
use std::{
    fs::File,
    path::{Path, PathBuf},
    sync::Arc,
};
use uuid::Uuid;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

fn save_result(result: &[RecordBatch], query_id: u32) {
    let file_path = format!("benchmark/data/results/Q{query_id}.parquet");
    let file = File::create(&file_path).unwrap();
    let props = WriterProperties::builder()
        .set_compression(Compression::SNAPPY)
        .build();
    let mut writer = ArrowWriter::try_new(file, result[0].schema(), Some(props)).unwrap();
    for batch in result {
        writer.write(batch).unwrap();
    }
    writer.close().unwrap();
    info!("Query {query_id} result saved to {file_path}");
}

fn check_result_against_answer(results: &[RecordBatch], answer_dir: &Path, query: &Query) {
    // - If query returns no results, check if baseline exists
    // - If baseline does not exist, skip query
    // - If baseline exists, panic
    if results.is_empty() {
        let baseline_exists = format!("{}/Q{}.parquet", answer_dir.display(), query.id());
        match File::open(baseline_exists) {
            Err(_) => {
                info!(
                    "Query {} returned no results (matches baseline)",
                    query.id()
                );
                return;
            }
            Ok(_) => {
                panic!(
                    "Query {} returned no results but baseline exists",
                    query.id()
                )
            }
        }
    }
    // Read answers
    let baseline_path = format!("{}/Q{}.parquet", answer_dir.display(), query.id());
    let baseline_file = File::open(baseline_path).unwrap();
    let mut baseline_batches = Vec::new();
    let reader = ParquetRecordBatchReaderBuilder::try_new(baseline_file)
        .unwrap()
        .build()
        .unwrap();
    for batch in reader {
        baseline_batches.push(batch.unwrap());
    }

    // Compare answers and result
    let result_batch =
        datafusion::arrow::compute::concat_batches(&results[0].schema(), results).unwrap();
    let baseline_batch = datafusion::arrow::compute::concat_batches(
        &baseline_batches[0].schema(),
        &baseline_batches,
    )
    .unwrap();
    if query.statement()[0].contains("LIMIT") {
        info!(
            "Query {} contains LIMIT, only validating the shape of the result",
            query.id()
        );
        let (result_num_rows, result_columns) =
            (result_batch.num_rows(), result_batch.columns().len());
        let (baseline_num_rows, baseline_columns) =
            (baseline_batch.num_rows(), baseline_batch.columns().len());
        if result_num_rows != baseline_num_rows || result_columns != baseline_columns {
            save_result(results, query.id());
            panic!(
                "Query {} result does not match baseline. Result(num_rows: {result_num_rows}, num_columns: {result_columns}), Baseline(num_rows: {baseline_num_rows}, num_columns: {baseline_columns})",
                query.id()
            );
        }
    } else {
        assert_batch_eq(&result_batch, &baseline_batch);
    }
}

#[derive(Parser, Serialize, Clone)]
#[command(name = "ClickBench Benchmark")]
pub struct ClickBenchArgs {
    /// Path to the benchmark manifest file
    #[arg(long)]
    pub manifest: PathBuf,

    #[clap(flatten)]
    pub common: ClientBenchmarkArgs,
}

#[derive(Clone, Serialize)]
struct ClickBench {
    manifest: BenchmarkManifest,
    common_args: ClientBenchmarkArgs,
}

impl ClickBench {
    fn new(args: ClickBenchArgs) -> Self {
        let manifest = BenchmarkManifest::load_from_file(&args.manifest).unwrap();
        let common_args = args.common;
        Self {
            manifest,
            common_args,
        }
    }
}

impl Benchmark for ClickBench {
    type Args = ClickBench;

    fn common_args(&self) -> &ClientBenchmarkArgs {
        &self.common_args
    }

    fn args(&self) -> &Self::Args {
        self
    }

    #[fastrace::trace]
    async fn setup_context(&self) -> Result<Arc<SessionContext>> {
        // Load the manifest
        let mut session_config = SessionConfig::from_env()?;
        if let Some(partitions) = self.common_args.partitions {
            session_config.options_mut().execution.target_partitions = partitions;
        }

        let liquid_cache_builder = LiquidCacheBuilder::new(&self.common_args.server);
        let ctx = liquid_cache_builder.build(session_config)?;

        self.manifest.register_object_stores(&ctx).await.unwrap();
        self.manifest.register_tables(&ctx).await.unwrap();

        Ok(Arc::new(ctx))
    }

    async fn get_queries(&self) -> Result<Vec<Query>> {
        let queries = self.manifest.load_queries(0);
        Ok(queries)
    }

    async fn execute_query(
        &self,
        ctx: &Arc<SessionContext>,
        query: &Query,
    ) -> (
        Vec<RecordBatch>,
        Arc<dyn datafusion::physical_plan::ExecutionPlan>,
        Vec<Uuid>,
    ) {
        run_query(ctx, &query.statement()[0]).await
    }

    async fn validate_result(&self, query: &Query, results: &[RecordBatch]) {
        if let Some(answer_dir) = &self.common_args.answer_dir {
            check_result_against_answer(results, answer_dir, query);
            info!("Query {} passed validation", query.id());
        }
    }

    fn benchmark_name(&self) -> &'static str {
        "clickbench"
    }
}

#[tokio::main]
pub async fn main() -> Result<()> {
    let clickbench = ClickBench::new(ClickBenchArgs::parse());
    BenchmarkRunner::run(clickbench).await?;
    Ok(())
}
