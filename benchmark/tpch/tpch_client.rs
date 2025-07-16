use clap::Parser;
use datafusion::prelude::SessionConfig;
use datafusion::{arrow::array::RecordBatch, error::Result, prelude::SessionContext};
use liquid_cache_benchmarks::{Benchmark, BenchmarkManifest, CommonBenchmarkArgs, run_query, tpch};
use liquid_cache_benchmarks::{BenchmarkMode, BenchmarkRunner};
use liquid_cache_client::LiquidCacheBuilder;
use liquid_cache_common::CacheMode;
use log::info;
use mimalloc::MiMalloc;
use object_store::ClientConfigKey;
use serde::Serialize;
use std::{path::PathBuf, sync::Arc};
use url::Url;
use uuid::Uuid;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

#[derive(Parser, Serialize, Clone)]
#[command(name = "TPCH Benchmark")]
pub struct TpchArgs {
    /// Path to the benchmark manifest file
    #[arg(long)]
    pub manifest_path: PathBuf,

    #[clap(flatten)]
    pub common: CommonBenchmarkArgs,
}

impl TpchArgs {
    /// Load the benchmark manifest
    fn load_manifest(&self) -> BenchmarkManifest {
        let manifest = BenchmarkManifest::load_from_file(&self.manifest_path)
            .expect("Failed to load manifest");
        manifest
    }
}

#[derive(Clone, Serialize)]
struct TpchBenchmark {
    manifest: BenchmarkManifest,
    common_args: CommonBenchmarkArgs,
}

impl TpchBenchmark {
    fn new(args: TpchArgs) -> Self {
        let manifest = args.load_manifest();
        let common_args = args.common;
        Self {
            manifest,
            common_args,
        }
    }
}

impl Benchmark for TpchBenchmark {
    type Args = TpchBenchmark;

    fn common_args(&self) -> &CommonBenchmarkArgs {
        &self.common_args
    }

    fn args(&self) -> &Self::Args {
        self
    }

    #[fastrace::trace]
    async fn setup_context(&self) -> Result<Arc<SessionContext>> {
        let mut session_config = SessionConfig::from_env()?;

        let mode = match self.common_args.bench_mode {
            BenchmarkMode::ParquetFileserver => {
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

        session_config
            .options_mut()
            .execution
            .parquet
            .pushdown_filters = true;

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

    async fn get_queries(&self) -> Result<Vec<liquid_cache_benchmarks::Query>> {
        let queries = self.manifest.load_queries();
        Ok(queries)
    }

    async fn validate_result(
        &self,
        query: &liquid_cache_benchmarks::Query,
        results: &[RecordBatch],
    ) -> Result<()> {
        if let Some(answer_dir) = &self.common_args.answer_dir {
            tpch::check_result_against_answer(results, answer_dir, query.id)?;
            info!("Query {} passed validation", query.id);
        }
        Ok(())
    }

    fn benchmark_name(&self) -> &'static str {
        "tpch"
    }

    async fn execute_query(
        &self,
        ctx: &Arc<SessionContext>,
        query: &liquid_cache_benchmarks::Query,
    ) -> Result<(
        Vec<RecordBatch>,
        Arc<dyn datafusion::physical_plan::ExecutionPlan>,
        Vec<Uuid>,
    )> {
        run_query(ctx, &query.sql).await
    }
}

#[tokio::main]
pub async fn main() -> Result<()> {
    let tpch = TpchBenchmark::new(TpchArgs::parse());
    BenchmarkRunner::run(tpch).await?;
    Ok(())
}
