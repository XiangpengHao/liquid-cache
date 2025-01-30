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

fn get_query(query_path: impl AsRef<Path>, query_number: Option<u32>) -> Result<Vec<String>> {
    let query_path = query_path.as_ref();
    let file = File::open(query_path)?;
    let reader = BufReader::new(file);
    let mut queries = Vec::new();
    for line in reader.lines() {
        queries.push(line?);
    }
    if let Some(query_number) = query_number {
        Ok(vec![queries[query_number as usize].clone()])
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
        .get_matches();

    let server_url = matches.get_one::<String>("server").unwrap();
    let file = matches.get_one::<PathBuf>("file").unwrap();
    let query_path = matches.get_one::<PathBuf>("query-path").unwrap();
    let queries = get_query(query_path, matches.get_one::<u32>("query").copied())?;

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

    for query in queries {
        info!("SQL to be executed: \n{}", query.cyan());
        let now = Instant::now();
        let df = ctx.sql(&query).await?;
        let (state, logical_plan) = df.into_parts();

        // create physical plan will invoke the optimizer
        let physical_plan = state.create_physical_plan(&logical_plan).await?;
        let results = collect(physical_plan.clone(), state.task_ctx()).await?;
        let elapsed = now.elapsed();
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

    Ok(())
}
