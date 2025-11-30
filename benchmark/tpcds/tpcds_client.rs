use clap::Parser;
use datafusion::prelude::SessionConfig;
use datafusion::{arrow::array::RecordBatch, error::Result, prelude::SessionContext};
use liquid_cache_benchmarks::BenchmarkRunner;
use liquid_cache_benchmarks::{
    Benchmark, BenchmarkManifest, ClientBenchmarkArgs, run_query, utils::check_tpcds_result,
};
use liquid_cache_client::LiquidCacheBuilder;
use log::info;
use mimalloc::MiMalloc;
use serde::Serialize;
use std::{path::PathBuf, sync::Arc};
use uuid::Uuid;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

#[derive(Parser, Clone)]
#[command(name = "TPCDS Benchmark")]
pub struct TpcdsArgs {
    /// Path to the benchmark manifest file
    #[arg(long)]
    pub manifest: PathBuf,

    #[clap(flatten)]
    pub common: ClientBenchmarkArgs,
}

#[derive(Serialize, Clone)]
struct TpcdsBenchmark {
    manifest: BenchmarkManifest,
    common_args: ClientBenchmarkArgs,
}

impl TpcdsBenchmark {
    fn new(args: TpcdsArgs) -> Self {
        let manifest = BenchmarkManifest::load_from_file(&args.manifest).unwrap();
        let common_args = args.common;
        Self {
            manifest,
            common_args,
        }
    }
}

impl Benchmark for TpcdsBenchmark {
    type Args = TpcdsBenchmark;

    fn common_args(&self) -> &ClientBenchmarkArgs {
        &self.common_args
    }

    fn args(&self) -> &Self::Args {
        self
    }

    #[fastrace::trace]
    async fn setup_context(&self) -> Result<Arc<SessionContext>> {
        let mut session_config = SessionConfig::from_env()?;

        session_config
            .options_mut()
            .execution
            .parquet
            .pushdown_filters = true;

        if let Some(partitions) = self.common_args.partitions {
            session_config.options_mut().execution.target_partitions = partitions;
        }

        let liquid_cache_builder = LiquidCacheBuilder::new(&self.common_args.server);
        let ctx = liquid_cache_builder.build(session_config)?;

        self.manifest.register_object_stores(&ctx).await.unwrap();
        self.manifest.register_tables(&ctx).await.unwrap();

        Ok(Arc::new(ctx))
    }

    async fn get_queries(&self) -> Result<Vec<liquid_cache_benchmarks::Query>> {
        let queries = self.manifest.load_queries(1);
        Ok(queries)
    }

    async fn validate_result(
        &self,
        query: &liquid_cache_benchmarks::Query,
        results: &[RecordBatch],
    ) {
        if let Some(answer_dir) = &self.common_args.answer_dir {
            check_tpcds_result(results, answer_dir, query.id());
            info!("Query {} passed validation", query.id());
        }
    }

    fn benchmark_name(&self) -> &'static str {
        "tpcds"
    }

    async fn execute_query(
        &self,
        ctx: &Arc<SessionContext>,
        query: &liquid_cache_benchmarks::Query,
    ) -> (
        Vec<RecordBatch>,
        Arc<dyn datafusion::physical_plan::ExecutionPlan>,
        Vec<Uuid>,
    ) {
        run_query(ctx, &query.statement()[0]).await
    }
}

#[tokio::main]
pub async fn main() -> Result<()> {
    let tpcds = TpcdsBenchmark::new(TpcdsArgs::parse());
    BenchmarkRunner::run(tpcds).await?;
    Ok(())
}
