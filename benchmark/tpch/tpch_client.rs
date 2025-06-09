use clap::Parser;
use datafusion::prelude::SessionConfig;
use datafusion::{arrow::array::RecordBatch, error::Result, prelude::SessionContext};
use liquid_cache_benchmarks::{Benchmark, CommonBenchmarkArgs, run_query, tpch};
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
pub struct TpchBenchmark {
    /// Path to the query directory
    #[arg(long = "query-dir")]
    pub query_dir: PathBuf,

    /// Path to the data directory with TPCH data
    #[arg(long = "data-dir")]
    pub data_dir: PathBuf,

    #[clap(flatten)]
    pub common: CommonBenchmarkArgs,
}

impl Benchmark for TpchBenchmark {
    type Args = TpchBenchmark;

    fn common_args(&self) -> &CommonBenchmarkArgs {
        &self.common
    }

    fn args(&self) -> &Self::Args {
        self
    }

    #[fastrace::trace]
    async fn setup_context(&self) -> Result<Arc<SessionContext>> {
        let mut session_config = SessionConfig::from_env()?;
        let current_dir = std::env::current_dir()?.to_string_lossy().to_string();

        let tables = [
            "customer", "lineitem", "nation", "orders", "part", "partsupp", "region", "supplier",
        ];

        let mode = match self.common.bench_mode {
            BenchmarkMode::ParquetFileserver => {
                let ctx = Arc::new(SessionContext::new_with_config(session_config));
                let base_url = Url::parse(&self.common.server).unwrap();

                let object_store = object_store::http::HttpBuilder::new()
                    .with_url(base_url.clone())
                    .with_config(ClientConfigKey::AllowHttp, "true")
                    .build()
                    .unwrap();
                ctx.register_object_store(&base_url, Arc::new(object_store));

                for table_name in tables.iter() {
                    let table_path = Url::parse(&format!(
                        "file://{}/{}/{}.parquet",
                        current_dir,
                        self.data_dir.display(),
                        table_name
                    ))
                    .unwrap();
                    ctx.register_parquet(*table_name, table_path, Default::default())
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
        let mut session_config = SessionConfig::from_env()?;
        if let Some(partitions) = self.common.partitions {
            session_config.options_mut().execution.target_partitions = partitions;
        }
        let ctx = LiquidCacheBuilder::new(&self.common.server)
            .with_cache_mode(mode)
            .build(session_config)?;

        for table_name in tables.iter() {
            let table_url = Url::parse(&format!(
                "file://{}/{}/{}.parquet",
                current_dir,
                self.data_dir.display(),
                table_name
            ))
            .unwrap();
            ctx.register_parquet(*table_name, table_url, Default::default())
                .await?;
        }

        Ok(Arc::new(ctx))
    }

    async fn get_queries(&self) -> Result<Vec<liquid_cache_benchmarks::Query>> {
        tpch::get_all_queries(&self.query_dir)
    }

    async fn validate_result(
        &self,
        query: &liquid_cache_benchmarks::Query,
        results: &[RecordBatch],
    ) -> Result<()> {
        if let Some(answer_dir) = &self.common.answer_dir {
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
        if query.id == 15 {
            // Q15 has three queries, the second one is the one we want to test
            let queries: Vec<&str> = query.sql.split(';').collect();
            let mut results = Vec::new();
            let mut physical_plan = None;
            let mut plan_uuids = Vec::new();
            for (i, q) in queries.iter().enumerate() {
                let (result, plan, uuid) = run_query(ctx, q).await?;
                if i == 1 {
                    physical_plan = Some(plan);
                    results = result;
                    plan_uuids = uuid;
                }
            }
            Ok((results, physical_plan.unwrap(), plan_uuids))
        } else {
            run_query(ctx, &query.sql).await
        }
    }
}

#[tokio::main]
pub async fn main() -> Result<()> {
    let benchmark = TpchBenchmark::parse();
    BenchmarkRunner::run(benchmark).await?;
    Ok(())
}
