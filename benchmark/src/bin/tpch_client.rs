use std::{path::PathBuf, sync::Arc};

use datafusion::{
    arrow::array::RecordBatch,
    catalog::TableProvider,
    error::Result,
    physical_plan::{collect, displayable},
    prelude::{SessionConfig, SessionContext},
};
use liquid_cache_client::SplitSqlTableFactory;
use liquid_common::ParquetMode;
use log::{debug, info};
use owo_colors::OwoColorize;
use url::Url;

use clap::{Command, arg, value_parser};

pub const TPCH_TABLES: &[&str] = &[
    "part", "supplier", "partsupp", "customer", "orders", "lineitem", "nation", "region",
];

fn load_queries(query_dir: &PathBuf) -> Result<Vec<String>> {
    let mut queries = Vec::new();
    for entry in std::fs::read_dir(query_dir)? {
        let path = entry?.path();
        let query = std::fs::read_to_string(path)?;
        queries.push(query);
    }
    Ok(queries)
}

struct TpchRunner {
    ctx: Arc<SessionContext>,
    server_url: String,
    data_dir: PathBuf,
    iteration: u32,
    queries: Vec<String>,
    query_id: Option<u32>,
}

impl TpchRunner {
    fn new(
        server_url: String,
        data_dir: PathBuf,
        queries: Vec<String>,
        query_id: Option<u32>,
        iteration: u32,
    ) -> Result<Self> {
        let mut session_config = SessionConfig::from_env()?;
        session_config
            .options_mut()
            .execution
            .parquet
            .pushdown_filters = true;
        let ctx = Arc::new(SessionContext::new_with_config(session_config));

        Ok(Self {
            ctx,
            server_url,
            data_dir,
            queries,
            query_id,
            iteration,
        })
    }

    async fn setup(&mut self) -> Result<()> {
        for table_name in TPCH_TABLES {
            let table = self.get_table(table_name).await?;
            self.ctx.register_table(*table_name, table)?;
        }
        Ok(())
    }

    async fn run(&mut self) -> Result<()> {
        let query_id_to_run = match self.query_id {
            Some(id) => vec![id],
            None => (0..self.queries.len() as u32).collect(),
        };

        info!("Running TPC-H, queries: {:?}", query_id_to_run);

        self.setup().await?;

        for query_id in query_id_to_run.iter() {
            for _ in 0..self.iteration {
                self.benchmark_query(*query_id).await?;
            }
        }
        Ok(())
    }

    async fn benchmark_query(&mut self, query_id: u32) -> Result<Vec<RecordBatch>> {
        let query = self.queries[query_id as usize].clone();
        self.execute_query(&query).await
    }

    async fn execute_query(&mut self, query: &str) -> Result<Vec<RecordBatch>> {
        let plan = self.ctx.sql(query).await?;

        let (state, plan) = plan.into_parts();
        debug!("Logical Plan: \n{}", plan.to_string().cyan());

        let optimized_plan = state.optimize(&plan)?;
        debug!("Optimized Plan: \n{}", optimized_plan.to_string().cyan());

        let physical_plan = state.create_physical_plan(&plan).await?;
        debug!(
            "Physical Plan: \n{}",
            displayable(physical_plan.as_ref())
                .indent(true)
                .to_string()
                .cyan()
        );

        let result = collect(physical_plan, state.task_ctx()).await?;
        Ok(result)
    }

    async fn get_table(&self, table_name: &str) -> Result<Arc<dyn TableProvider>> {
        let current_dir = std::env::current_dir()?.to_string_lossy().to_string();
        let table_url = Url::parse(&format!(
            "file://{}/{}",
            current_dir,
            self.data_dir.display()
        ))
        .unwrap();

        let table = SplitSqlTableFactory::open_table(
            &self.server_url,
            table_name,
            table_url,
            ParquetMode::Liquid,
        )
        .await?;
        Ok(Arc::new(table))
    }
}

#[tokio::main]
pub async fn main() -> Result<()> {
    let matches = Command::new("SplitSQL Benchmark Client")
        .arg(
            arg!(--"query-dir" <PATH>)
                .required(true)
                .help("Path to the query directory")
                .value_parser(value_parser!(std::path::PathBuf)),
        )
        .arg(
            arg!(--query <NUMBER>)
                .required(false)
                .help("Query number to run, if not provided, all queries will be run")
                .value_parser(value_parser!(u32)),
        )
        .arg(
            arg!(--data <PATH>)
                .required(true)
                .help("Path to the TPC-H directory")
                .value_parser(value_parser!(std::path::PathBuf)),
        )
        .arg(
            arg!(--server <URL>)
                .required(false)
                .default_value("http://localhost:50051")
                .help("Server URL")
                .value_parser(value_parser!(String)),
        )
        .arg(
            arg!(--iteration <NUMBER>)
                .required(false)
                .default_value("1")
                .help("Number of iterations to run")
                .value_parser(value_parser!(u32)),
        )
        .get_matches();

    let server_url = matches.get_one::<String>("server").unwrap();
    let query_dir = matches.get_one::<PathBuf>("query-dir").unwrap();
    let data_dir = matches.get_one::<PathBuf>("data").unwrap();
    let query_id = matches.get_one::<u32>("query");
    let iteration = matches.get_one::<u32>("iteration").unwrap();
    let queries = load_queries(query_dir)?;

    let mut runner = TpchRunner::new(
        server_url.clone(),
        data_dir.clone(),
        queries,
        query_id.copied(),
        *iteration,
    )?;

    runner.run().await?;

    Ok(())
}
