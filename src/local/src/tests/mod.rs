use arrow_schema::{DataType, Field, Schema};
use std::sync::Arc;
use tempfile::TempDir;

use arrow::util::pretty::pretty_format_batches;
use datafusion::{
    datasource::{
        file_format::parquet::ParquetFormat,
        listing::{ListingOptions, ListingTableUrl},
    },
    error::Result,
    physical_plan::{ExecutionPlan, collect, display::DisplayableExecutionPlan},
    prelude::{ParquetReadOptions, SessionConfig, SessionContext},
};
use liquid_cache_common::LiquidCacheMode;
use liquid_cache_storage::policies::ToDiskPolicy;

use crate::LiquidCacheLocalBuilder;

const TEST_FILE: &str = "../../examples/nano_hits.parquet";

async fn create_session_context_with_liquid_cache(
    cache_mode: LiquidCacheMode,
    cache_size_bytes: usize,
) -> Result<SessionContext> {
    let temp_dir = TempDir::new().unwrap();

    let mut config = SessionConfig::new();
    config.options_mut().execution.target_partitions = 4;
    let (ctx, _) = LiquidCacheLocalBuilder::new()
        .with_max_cache_bytes(cache_size_bytes)
        .with_cache_dir(temp_dir.path().to_path_buf())
        .with_cache_mode(cache_mode)
        .with_cache_strategy(Box::new(ToDiskPolicy))
        .build(config)?;

    // Register the test parquet file
    ctx.register_parquet("hits", TEST_FILE, ParquetReadOptions::default())
        .await
        .unwrap();

    Ok(ctx)
}

async fn get_physical_plan(sql: &str, ctx: &SessionContext) -> Arc<dyn ExecutionPlan> {
    let df = ctx.sql(sql).await.unwrap();
    let (state, plan) = df.into_parts();
    state.create_physical_plan(&plan).await.unwrap()
}

async fn run_sql_with_cache(
    sql: &str,
    cache_mode: LiquidCacheMode,
    cache_size_bytes: usize,
) -> (String, String) {
    let ctx = create_session_context_with_liquid_cache(cache_mode, cache_size_bytes)
        .await
        .unwrap();

    let plan = get_physical_plan(sql, &ctx).await;
    let displayable = DisplayableExecutionPlan::new(plan.as_ref());
    let plan_string = format!("{}", displayable.tree_render());

    async fn get_result(ctx: &SessionContext, sql: &str) -> String {
        let plan = get_physical_plan(sql, ctx).await;
        let batches = collect(plan, ctx.task_ctx()).await.unwrap();
        pretty_format_batches(&batches).unwrap().to_string()
    }

    let first_run = get_result(&ctx, sql).await;
    let second_run = get_result(&ctx, sql).await;

    assert_eq!(first_run, second_run);

    (first_run, plan_string)
}

async fn test_runner(sql: &str, reference: &str) {
    let cache_modes = [
        LiquidCacheMode::Arrow,
        LiquidCacheMode::Liquid,
        LiquidCacheMode::LiquidBlocking,
    ];

    let cache_sizes = [10 * 1024, 1024 * 1024, usize::MAX]; // 10KB, 1MB, unlimited

    for cache_mode in cache_modes {
        for cache_size in cache_sizes {
            let (result, _plan) = run_sql_with_cache(sql, cache_mode, cache_size).await;
            assert_eq!(
                result, reference,
                "Results differ for cache_mode: {cache_mode:?}, cache_size: {cache_size}"
            );
        }
    }
}

#[tokio::test]
async fn test_url_prefix_filtering() {
    let sql = r#"select COUNT(*) from hits where "URL" like 'https://%'"#;

    let (reference, plan) =
        run_sql_with_cache(sql, LiquidCacheMode::LiquidBlocking, 1024 * 1024).await;

    insta::assert_snapshot!(format!("plan: \n{}\nvalues: \n{}", plan, reference));
    test_runner(sql, &reference).await;
}

#[tokio::test]
async fn test_url_selection_and_ordering() {
    let sql = r#"select "URL" from hits where "URL" like '%tours%' order by "URL" desc"#;

    let (reference, plan) =
        run_sql_with_cache(sql, LiquidCacheMode::LiquidBlocking, 1024 * 1024).await;

    insta::assert_snapshot!(format!("plan: \n{}\nvalues: \n{}", plan, reference));
    test_runner(sql, &reference).await;
}

#[tokio::test]
async fn test_os_selection() {
    let sql = r#"select "OS" from hits where "URL" like '%tours%' order by "OS" desc"#;

    let (reference, plan) =
        run_sql_with_cache(sql, LiquidCacheMode::LiquidBlocking, 1024 * 1024).await;

    insta::assert_snapshot!(format!("plan: \n{}\nvalues: \n{}", plan, reference));

    test_runner(sql, &reference).await;
}

#[tokio::test]
async fn test_referer_filtering() {
    let sql = r#"select "Referer" from hits where "Referer" <> '' AND "URL" like '%tours%' order by "Referer" desc"#;

    let (reference, plan) =
        run_sql_with_cache(sql, LiquidCacheMode::LiquidBlocking, 1024 * 1024).await;

    insta::assert_snapshot!(format!("plan: \n{}\nvalues: \n{}", plan, reference));

    test_runner(sql, &reference).await;
}

#[tokio::test]
async fn test_single_column_filter_projection() {
    let sql = r#"select "WatchID" from hits where "WatchID" = 6978470580070504163"#;

    let (reference, plan) =
        run_sql_with_cache(sql, LiquidCacheMode::LiquidBlocking, 1024 * 1024).await;

    insta::assert_snapshot!(format!("plan: \n{}\nvalues: \n{}", plan, reference));

    test_runner(sql, &reference).await;
}

#[tokio::test]
async fn test_provide_schema_with_filter() {
    let sql = r#"select "WatchID", "OS", "EventTime" from hits where "OS" <> 2 order by "WatchID" desc limit 10"#;

    let (reference, plan) =
        run_sql_with_cache(sql, LiquidCacheMode::LiquidBlocking, 1024 * 1024).await;

    insta::assert_snapshot!(format!("plan: \n{}\nvalues: \n{}", plan, reference));

    let (ctx, _) = LiquidCacheLocalBuilder::new()
        .with_cache_mode(LiquidCacheMode::LiquidBlocking)
        .build(SessionConfig::new())
        .unwrap();

    let file_format = ParquetFormat::default().with_enable_pruning(true);
    let listing_options =
        ListingOptions::new(Arc::new(file_format)).with_file_extension(".parquet");

    let table_path = ListingTableUrl::parse("../../examples/nano_hits.parquet").unwrap();
    let schema = Schema::new(vec![
        Field::new("WatchID", DataType::Int64, true),
        Field::new("EventTime", DataType::Int64, true),
        Field::new("OS", DataType::Int16, true),
    ]);

    ctx.register_listing_table(
        "hits",
        &table_path,
        listing_options.clone(),
        Some(Arc::new(schema)),
        None,
    )
    .await
    .unwrap();

    let results = ctx.sql(sql).await.unwrap().collect().await.unwrap();

    let formatted_results = pretty_format_batches(&results).unwrap().to_string();
    if formatted_results != reference {
        println!("formatted_results: \n{formatted_results}");
        println!("reference: \n{reference}");
    }
    assert_eq!(formatted_results, reference);
}
