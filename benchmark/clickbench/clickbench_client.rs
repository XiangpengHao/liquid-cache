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
    Benchmark, BenchmarkManifest, CommonBenchmarkArgs, utils::assert_batch_eq,
};
use liquid_cache_benchmarks::{BenchmarkMode, BenchmarkRunner, Query, run_query};
use liquid_cache_client::LiquidCacheBuilder;
use liquid_cache_common::CacheMode;
use log::info;
use mimalloc::MiMalloc;
use object_store::ClientConfigKey;
use serde::Serialize;
use std::{
    fs::File,
    path::{Path, PathBuf},
    sync::Arc,
};
use url::Url;
use uuid::Uuid;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

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
    results: &[RecordBatch],
    answer_dir: &Path,
    query: &Query,
) -> Result<()> {
    // - If query returns no results, check if baseline exists
    // - If baseline does not exist, skip query
    // - If baseline exists, panic
    if results.is_empty() {
        let baseline_exists = format!("{}/Q{}.parquet", answer_dir.display(), query.id);
        match File::open(baseline_exists) {
            Err(_) => {
                info!("Query {} returned no results (matches baseline)", query.id);
                return Ok(());
            }
            Ok(_) => {
                panic!("Query {} returned no results but baseline exists", query.id)
            }
        }
    }
    // Read answers
    let baseline_path = format!("{}/Q{}.parquet", answer_dir.display(), query.id);
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
    if query.sql.contains("LIMIT") {
        info!(
            "Query {} contains LIMIT, only validating the shape of the result",
            query.id
        );
        let (result_num_rows, result_columns) =
            (result_batch.num_rows(), result_batch.columns().len());
        let (baseline_num_rows, baseline_columns) =
            (baseline_batch.num_rows(), baseline_batch.columns().len());
        if result_num_rows != baseline_num_rows || result_columns != baseline_columns {
            save_result(results, query.id)?;
            panic!(
                "Query {} result does not match baseline. Result(num_rows: {result_num_rows}, num_columns: {result_columns}), Baseline(num_rows: {baseline_num_rows}, num_columns: {baseline_columns})",
                query.id
            );
        }
    } else {
        assert_batch_eq(&result_batch, &baseline_batch);
    }
    Ok(())
}

#[derive(Parser, Serialize, Clone)]
#[command(name = "ClickBench Benchmark")]
pub struct ClickBenchArgs {
    /// Path to the benchmark manifest file
    #[arg(long)]
    pub manifest_path: PathBuf,

    #[clap(flatten)]
    pub common: CommonBenchmarkArgs,
}

impl ClickBenchArgs {
    /// Load the benchmark manifest with caching
    fn load_manifest(&self) -> BenchmarkManifest {
        let manifest = BenchmarkManifest::load_from_file(&self.manifest_path)
            .expect("Failed to load manifest");
        manifest
    }
}

#[derive(Clone, Serialize)]
struct ClickBench {
    manifest: BenchmarkManifest,
    common_args: CommonBenchmarkArgs,
}

impl ClickBench {
    fn new(args: ClickBenchArgs) -> Self {
        let manifest = args.load_manifest();
        let common_args = args.common;
        Self {
            manifest,
            common_args,
        }
    }
}

impl Benchmark for ClickBench {
    type Args = ClickBench;

    fn common_args(&self) -> &CommonBenchmarkArgs {
        &self.common_args
    }

    fn args(&self) -> &Self::Args {
        self
    }

    #[fastrace::trace]
    async fn setup_context(&self) -> Result<Arc<SessionContext>> {
        // Load the manifest
        let mode = match self.common_args.bench_mode {
            BenchmarkMode::ParquetFileserver => {
                let mut session_config = SessionConfig::from_env()?;
                if let Some(partitions) = self.common_args.partitions {
                    session_config.options_mut().execution.target_partitions = partitions;
                }
                let ctx = Arc::new(SessionContext::new_with_config(session_config));
                let base_url = Url::parse(&self.common_args.server).unwrap();

                let object_store = object_store::http::HttpBuilder::new()
                    .with_url(base_url.clone())
                    .with_config(ClientConfigKey::AllowHttp, "true")
                    .build()
                    .unwrap();
                ctx.register_object_store(&base_url, Arc::new(object_store));

                // Register tables from manifest
                for (table_name, table_path) in &self.manifest.tables {
                    ctx.register_parquet(
                        table_name,
                        format!("{}/{}", self.common_args.server, table_path),
                        Default::default(),
                    )
                    .await?;
                }
                return Ok(ctx);
            }
            BenchmarkMode::ParquetPushdown => CacheMode::Parquet,
            BenchmarkMode::ArrowPushdown => CacheMode::Arrow,
            BenchmarkMode::LiquidCache => CacheMode::Liquid,
            BenchmarkMode::LiquidEagerTranscode => CacheMode::LiquidEagerTranscode,
        };

        let mut session_config = SessionConfig::from_env()?;
        if let Some(partitions) = self.common_args.partitions {
            session_config.options_mut().execution.target_partitions = partitions;
        }

        let liquid_cache_builder =
            LiquidCacheBuilder::new(&self.common_args.server).with_cache_mode(mode);
        let ctx = liquid_cache_builder.build(session_config)?;

        self.manifest.register_object_stores(&ctx).await.unwrap();
        self.manifest.register_tables(&ctx).await.unwrap();

        Ok(Arc::new(ctx))
    }

    async fn get_queries(&self) -> Result<Vec<Query>> {
        let queries = self.manifest.load_queries();
        Ok(queries)
    }

    async fn execute_query(
        &self,
        ctx: &Arc<SessionContext>,
        query: &Query,
    ) -> Result<(
        Vec<RecordBatch>,
        Arc<dyn datafusion::physical_plan::ExecutionPlan>,
        Vec<Uuid>,
    )> {
        run_query(ctx, &query.sql).await
    }

    async fn validate_result(&self, query: &Query, results: &[RecordBatch]) -> Result<()> {
        if let Some(answer_dir) = &self.common_args.answer_dir {
            check_result_against_answer(results, answer_dir, query)?;
            info!("Query {} passed validation", query.id);
        }
        Ok(())
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
