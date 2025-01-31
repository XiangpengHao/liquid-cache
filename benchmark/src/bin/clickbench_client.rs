use std::{
    fs::File,
    io::{BufRead, BufReader},
    path::{Path, PathBuf},
    sync::Arc,
    time::Instant,
};

use datafusion::{
    arrow::util::pretty,
    error::Result,
    physical_plan::{collect, display::DisplayableExecutionPlan},
    prelude::{SessionConfig, SessionContext},
};
use liquid_cache_client::SplitSqlTableFactory;
use log::{debug, info};
use owo_colors::OwoColorize;
use url::Url;

use clap::{Command, arg, value_parser};
use serde::Serialize;
use std::fs::File as StdFile;

#[derive(Serialize)]
struct BenchmarkResult {
    server_url: String,
    file_path: PathBuf,
    query_path: PathBuf,
    iteration: u32,
    queries: Vec<QueryResult>,
}

#[derive(Serialize)]
struct QueryResult {
    id: u32,
    query: String,
    times_millis: Vec<u64>,
}

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

#[tokio::main]
pub async fn main() -> Result<()> {
    env_logger::builder().format_timestamp(None).init();

    let matches = Command::new("ClickBench Benchmark Client")
        .arg(
            arg!(--"query-path" <PATH>)
                .required(true)
                .help("Path to the query file")
                .value_parser(value_parser!(std::path::PathBuf)),
        )
        .arg(
            arg!(--query <NUMBER>)
                .required(false)
                .help("Query number to run, if not provided, all queries will be run")
                .value_parser(value_parser!(u32)),
        )
        .arg(
            arg!(--file <PATH>)
                .required(true)
                .help("Path to the ClickBench file")
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
                .help("Number of times to run each query")
                .value_parser(value_parser!(u32)),
        )
        .arg(
            arg!(--output <PATH>)
                .required(false)
                .help("Path to the output JSON file")
                .value_parser(value_parser!(std::path::PathBuf)),
        )
        .get_matches();

    let server_url = matches.get_one::<String>("server").unwrap();
    let file = matches.get_one::<PathBuf>("file").unwrap();
    let query_path = matches.get_one::<PathBuf>("query-path").unwrap();
    let queries = get_query(query_path, matches.get_one::<u32>("query").copied())?;
    let iteration = matches.get_one::<u32>("iteration").unwrap();
    let output_path = matches.get_one::<PathBuf>("output");

    let mut session_config = SessionConfig::from_env()?;
    session_config
        .options_mut()
        .execution
        .parquet
        .pushdown_filters = true;
    let ctx = Arc::new(SessionContext::new_with_config(session_config));

    let table_name = "hits";

    let current_dir = std::env::current_dir()?.to_string_lossy().to_string();
    let table_url = Url::parse(&format!("file://{}/{}", current_dir, file.display())).unwrap();

    let table = SplitSqlTableFactory::open_table(server_url, table_name, table_url).await?;
    ctx.register_table(table_name, Arc::new(table))?;

    let mut benchmark_result = BenchmarkResult {
        server_url: server_url.clone(),
        file_path: file.clone(),
        query_path: query_path.clone(),
        iteration: *iteration,
        queries: Vec::new(),
    };

    for (id, query) in queries {
        let mut times_millis = Vec::new();
        for _i in 0..*iteration {
            info!("SQL to be executed: \n{}", query.cyan());
            let now = Instant::now();
            let df = ctx.sql(&query).await?;
            let (state, logical_plan) = df.into_parts();

            let physical_plan = state.create_physical_plan(&logical_plan).await?;
            let results = collect(physical_plan.clone(), state.task_ctx()).await?;
            let elapsed = now.elapsed();
            times_millis.push(elapsed.as_millis() as u64);
            info!("Query execution time: {:?}", elapsed);

            let physical_plan_with_metrics =
                DisplayableExecutionPlan::with_metrics(physical_plan.as_ref());

            debug!(
                "Physical plan: \n{}",
                physical_plan_with_metrics.indent(true).magenta()
            );
            let result_str = pretty::pretty_format_batches(&results).unwrap();
            info!("Query result: \n{}", result_str.cyan());
        }
        benchmark_result.queries.push(QueryResult {
            id,
            query: query.clone(),
            times_millis,
        });
    }

    if let Some(output_path) = output_path {
        let output_file = StdFile::create(output_path)?;
        serde_json::to_writer_pretty(output_file, &benchmark_result).unwrap();
    }

    Ok(())
}
