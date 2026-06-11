use arrow_schema::{DataType, Field, Schema};
use liquid_cache::{
    cache::{
        CacheStats,
        squeeze_policies::{Evict, SqueezePolicy, TranscodeEvict, TranscodeSqueezeEvict},
    },
    cache_policies::LiquidPolicy,
};
use liquid_cache_datafusion::LiquidCacheParquetRef;
use std::{fmt, path::Path, sync::Arc};
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

use crate::LiquidCacheLocalBuilder;
mod date_optimizer;
mod squeeze;
mod variants;

const TEST_FILE: &str = "../../examples/nano_hits.parquet";
const OPENOBSERVE_FILE: &str = "../../dev/test_parquet/openobserve.parquet";

#[derive(Debug, Clone)]
struct QueryOutcome {
    values: String,
    plan: String,
    stats: CacheStatsSummary,
}

#[derive(Debug, Clone)]
struct CacheStatsSummary {
    stats: CacheStats,
    entries_after_first_run: usize,
}

impl CacheStatsSummary {
    fn from_stats(stats: CacheStats, entries_after_first_run: usize) -> Self {
        Self {
            stats,
            entries_after_first_run,
        }
    }

    fn has_cache_hits(&self) -> bool {
        let runtime = &self.stats.runtime;
        runtime.get_with_selection > 0
            || runtime.try_read_liquid_calls > 0
            || runtime.get > 0
            || runtime.eval_predicate > 0
    }

    fn entries_reused(&self) -> bool {
        self.stats.total_entries == self.entries_after_first_run
    }
}

impl fmt::Display for CacheStatsSummary {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "entries.total: {}", self.stats.total_entries)?;
        writeln!(
            f,
            "entries.after_first_run: {}",
            self.entries_after_first_run
        )?;
        writeln!(
            f,
            "entries.memory.arrow: {}",
            self.stats.memory_arrow_entries
        )?;
        writeln!(
            f,
            "entries.memory.liquid: {}",
            self.stats.memory_liquid_entries
        )?;
        writeln!(
            f,
            "entries.memory.squeezed_liquid: {}",
            self.stats.memory_squeezed_liquid_entries
        )?;
        writeln!(f, "entries.disk.liquid: {}", self.stats.disk_liquid_entries)?;
        writeln!(f, "entries.disk.arrow: {}", self.stats.disk_arrow_entries)?;
        writeln!(f, "usage.memory_bytes: {}", self.stats.memory_usage_bytes)?;
        writeln!(f, "usage.disk_bytes: {}", self.stats.disk_usage_bytes)?;
        // Use the Display implementation for runtime stats
        write!(f, "{}", self.stats.runtime)
    }
}

async fn create_session_context_with_liquid_cache(
    squeeze_policy: Box<dyn SqueezePolicy>,
    cache_size_bytes: usize,
    cache_dir: &Path,
) -> Result<(SessionContext, LiquidCacheParquetRef)> {
    let mut config = SessionConfig::new();
    config.options_mut().execution.target_partitions = 4;
    let (ctx, cache) = LiquidCacheLocalBuilder::new()
        .with_max_memory_bytes(cache_size_bytes)
        .with_cache_dir(cache_dir.to_path_buf())
        .with_squeeze_policy(squeeze_policy)
        .with_cache_policy(Box::new(LiquidPolicy::new()))
        .build(config)
        .await?;

    // Register the test parquet file
    ctx.register_parquet("hits", TEST_FILE, ParquetReadOptions::default())
        .await
        .unwrap();

    Ok((ctx, cache))
}

async fn get_physical_plan(sql: &str, ctx: &SessionContext) -> Arc<dyn ExecutionPlan> {
    let df = ctx.sql(sql).await.unwrap();
    let (state, plan) = df.into_parts();
    state.create_physical_plan(&plan).await.unwrap()
}

async fn run_sql_with_cache(
    sql: &str,
    squeeze_policy: Box<dyn SqueezePolicy>,
    cache_size_bytes: usize,
    cache_dir: &Path,
) -> QueryOutcome {
    let (ctx, cache) =
        create_session_context_with_liquid_cache(squeeze_policy, cache_size_bytes, cache_dir)
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

    // Clear any historical runtime counters before warming the cache.
    cache.storage().stats();

    let first_run = get_result(&ctx, sql).await;
    let entries_after_first_run = cache.storage().stats().total_entries;
    let second_run = get_result(&ctx, sql).await;

    assert_eq!(first_run, second_run);

    let stats_after_second_run = cache.storage().stats();
    let stats = CacheStatsSummary::from_stats(stats_after_second_run, entries_after_first_run);

    QueryOutcome {
        values: second_run,
        plan: plan_string,
        stats,
    }
}

async fn test_runner(sql: &str, reference: &str, cache_dir: &Path) {
    let cache_sizes = [10 * 1024, 1024 * 1024, usize::MAX]; // 10KB, 1MB, unlimited

    for cache_size in cache_sizes {
        let squeeze_policies: Vec<Box<dyn SqueezePolicy>> = vec![
            Box::new(TranscodeSqueezeEvict),
            Box::new(Evict),
            Box::new(TranscodeEvict),
        ];
        for squeeze_policy in squeeze_policies {
            let QueryOutcome { values, .. } =
                run_sql_with_cache(sql, squeeze_policy, cache_size, cache_dir).await;
            assert_eq!(
                values, reference,
                "Results differ, cache_size: {cache_size}"
            );
        }
    }
}

#[tokio::test]
async fn test_url_prefix_filtering() {
    let cache_dir = TempDir::new().unwrap();
    let sql = r#"select COUNT(*) from hits where "URL" like 'https://%'"#;

    let QueryOutcome {
        values,
        plan,
        stats,
    } = run_sql_with_cache(
        sql,
        Box::new(TranscodeSqueezeEvict),
        1024 * 1024,
        cache_dir.path(),
    )
    .await;

    assert!(stats.has_cache_hits());
    assert!(stats.entries_reused());

    let reference = values.clone();

    insta::assert_snapshot!(format!(
        "plan: \n{}\nvalues: \n{}\nstats:\n{}",
        plan, values, stats
    ));
    test_runner(sql, &reference, cache_dir.path()).await;
}

#[tokio::test]
async fn test_url_selection_and_ordering() {
    let cache_dir = TempDir::new().unwrap();
    let sql = r#"select "URL" from hits where "URL" like '%tours%' order by "URL" desc"#;

    let QueryOutcome {
        values,
        plan,
        stats,
    } = run_sql_with_cache(
        sql,
        Box::new(TranscodeSqueezeEvict),
        1024 * 300,
        cache_dir.path(),
    )
    .await;

    assert!(stats.has_cache_hits());
    assert!(stats.entries_reused());

    let reference = values.clone();

    insta::assert_snapshot!(format!(
        "plan: \n{}\nvalues: \n{}\nstats:\n{}",
        plan, values, stats
    ));
    test_runner(sql, &reference, cache_dir.path()).await;
}

#[tokio::test]
async fn test_os_selection() {
    let cache_dir = TempDir::new().unwrap();
    let sql = r#"select "OS" from hits where "URL" like '%tours%' order by "OS" desc"#;

    let QueryOutcome {
        values,
        plan,
        stats,
    } = run_sql_with_cache(
        sql,
        Box::new(TranscodeSqueezeEvict),
        1024 * 1024,
        cache_dir.path(),
    )
    .await;

    assert!(stats.has_cache_hits());
    assert!(stats.entries_reused());

    let reference = values.clone();

    insta::assert_snapshot!(format!(
        "plan: \n{}\nvalues: \n{}\nstats:\n{}",
        plan, values, stats
    ));

    test_runner(sql, &reference, cache_dir.path()).await;
}

#[tokio::test]
async fn test_referer_filtering() {
    let cache_dir = TempDir::new().unwrap();
    let sql = r#"select "Referer" from hits where "Referer" <> '' AND "URL" like '%tours%' order by "Referer" desc"#;

    let QueryOutcome {
        values,
        plan,
        stats,
    } = run_sql_with_cache(
        sql,
        Box::new(TranscodeSqueezeEvict),
        1024 * 1024,
        cache_dir.path(),
    )
    .await;

    assert!(stats.has_cache_hits());
    assert!(stats.entries_reused());

    let reference = values.clone();

    insta::assert_snapshot!(format!(
        "plan: \n{}\nvalues: \n{}\nstats:\n{}",
        plan, values, stats
    ));

    test_runner(sql, &reference, cache_dir.path()).await;
}

#[tokio::test]
async fn test_single_column_filter_projection() {
    let cache_dir = TempDir::new().unwrap();
    let sql = r#"select "WatchID" from hits where "WatchID" = 6978470580070504163"#;

    let QueryOutcome {
        values,
        plan,
        stats,
    } = run_sql_with_cache(
        sql,
        Box::new(TranscodeSqueezeEvict),
        1024 * 1024,
        cache_dir.path(),
    )
    .await;

    assert!(stats.has_cache_hits());
    assert!(stats.entries_reused());

    let reference = values.clone();

    insta::assert_snapshot!(format!(
        "plan: \n{}\nvalues: \n{}\nstats:\n{}",
        plan, values, stats
    ));

    test_runner(sql, &reference, cache_dir.path()).await;
}

#[tokio::test]
async fn test_provide_schema2() {
    use std::fmt::Write as _;

    let cache_dir = TempDir::new().unwrap();
    let df_ctx = SessionContext::new();
    let mut config = SessionConfig::new();
    config.options_mut().execution.target_partitions = 4;
    let (liquid_ctx, cache) = LiquidCacheLocalBuilder::new()
        .with_cache_dir(cache_dir.path().to_path_buf())
        .with_max_memory_bytes(1024 * 1024)
        .with_squeeze_policy(Box::new(TranscodeSqueezeEvict))
        .build(config)
        .await
        .unwrap();

    let file_format = ParquetFormat::default().with_enable_pruning(true);
    let listing_options =
        ListingOptions::new(Arc::new(file_format)).with_file_extension(".parquet");
    let table_path = ListingTableUrl::parse(OPENOBSERVE_FILE).unwrap();
    let schema = Arc::new(Schema::new(vec![
        Field::new("_timestamp", DataType::Int64, false),
        Field::new("log", DataType::Utf8, true),
        Field::new("message", DataType::Utf8, true),
        Field::new("kubernetes_namespace_name", DataType::Utf8, false),
    ]));

    df_ctx
        .register_listing_table(
            "default",
            &table_path,
            listing_options.clone(),
            Some(schema.clone()),
            None,
        )
        .await
        .unwrap();
    liquid_ctx
        .register_listing_table("default", &table_path, listing_options, Some(schema), None)
        .await
        .unwrap();

    let queries = [
        "SELECT * from default where log like '%hhj%' order by _timestamp",
        "SELECT date_bin(interval '10 second', to_timestamp_micros(_timestamp), to_timestamp('2001-01-01T00:00:00')) AS zo_sql_key, count(*) AS zo_sql_num from default WHERE log like '%hhj%' or message like '%hhj%' GROUP BY zo_sql_key ORDER BY zo_sql_key",
        "SELECT _timestamp, kubernetes_namespace_name from default order by _timestamp desc limit 100",
    ];

    let mut snapshot = String::new();

    for (idx, sql) in queries.iter().enumerate() {
        let df_results = df_ctx.sql(sql).await.unwrap().collect().await.unwrap();

        let plan = get_physical_plan(sql, &liquid_ctx).await;
        let displayable = DisplayableExecutionPlan::new(plan.as_ref());
        let plan_string = format!("{}", displayable.tree_render());

        // Reset runtime counters so we measure hits from the warm run onwards.
        cache.storage().stats();

        let first_liquid_run = liquid_ctx.sql(sql).await.unwrap().collect().await.unwrap();
        assert_eq!(df_results[0].columns(), first_liquid_run[0].columns());

        let entries_after_first_run = cache.storage().stats().total_entries;
        let second_liquid_run = liquid_ctx.sql(sql).await.unwrap().collect().await.unwrap();
        assert_eq!(df_results[0].columns(), second_liquid_run[0].columns());

        let stats = CacheStatsSummary::from_stats(cache.storage().stats(), entries_after_first_run);

        assert!(stats.has_cache_hits());
        assert!(stats.entries_reused());

        writeln!(snapshot, "query[{idx}]: {sql}").unwrap();
        writeln!(snapshot, "plan: \n{}", plan_string).unwrap();
        writeln!(snapshot, "stats:\n{}", stats).unwrap();

        if idx + 1 != queries.len() {
            snapshot.push('\n');
        }
    }

    insta::assert_snapshot!(snapshot);
}

#[tokio::test]
async fn test_provide_schema_with_filter() {
    let cache_dir = TempDir::new().unwrap();
    let sql = r#"select "WatchID", "OS", "EventTime" from hits where "OS" <> 2 order by "WatchID" desc limit 10"#;

    let QueryOutcome {
        values,
        plan,
        stats,
    } = run_sql_with_cache(
        sql,
        Box::new(TranscodeSqueezeEvict),
        1024 * 1024,
        cache_dir.path(),
    )
    .await;

    assert!(stats.has_cache_hits());
    assert!(stats.entries_reused());

    let reference = values.clone();

    insta::assert_snapshot!(format!(
        "plan: \n{}\nvalues: \n{}\nstats:\n{}",
        plan, values, stats
    ));

    let (ctx, _) = LiquidCacheLocalBuilder::new()
        .with_squeeze_policy(Box::new(TranscodeSqueezeEvict))
        .build(SessionConfig::new())
        .await
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

/// Test that only predicate (WHERE clause) columns are cached.
/// Projection-only columns should NOT be cached — they read from Parquet directly.
#[tokio::test]
async fn test_predicate_only_caching() {
    let cache_dir = TempDir::new().unwrap();

    // Query: WHERE on "OS" (predicate column), SELECT "URL" (projection-only column)
    let sql = r#"SELECT "URL" FROM hits WHERE "OS" > 0 ORDER BY "URL" LIMIT 5"#;

    let mut config = SessionConfig::new();
    config.options_mut().execution.target_partitions = 2;
    let (ctx, cache) = LiquidCacheLocalBuilder::new()
        .with_max_memory_bytes(usize::MAX)
        .with_cache_dir(cache_dir.path().to_path_buf())
        .with_squeeze_policy(Box::new(TranscodeSqueezeEvict))
        .with_cache_policy(Box::new(LiquidPolicy::new()))
        .build(config)
        .await
        .unwrap();

    ctx.register_parquet("hits", TEST_FILE, ParquetReadOptions::default())
        .await
        .unwrap();

    // First run: fills cache
    let plan = get_physical_plan(sql, &ctx).await;
    let batches_1 = collect(plan, ctx.task_ctx()).await.unwrap();
    let result_1 = pretty_format_batches(&batches_1).unwrap().to_string();

    let stats_after_first = cache.storage().stats();
    let entries_after_first = stats_after_first.total_entries;

    // Should have cached ONLY OS (predicate column) batches, NOT URL
    // With nano_hits.parquet (1 row group), we expect a small number of entries
    // corresponding to OS batches only.
    assert!(
        entries_after_first > 0,
        "Cache should have entries for predicate column OS"
    );

    // Second run: should reuse cache for predicate evaluation
    let plan = get_physical_plan(sql, &ctx).await;
    let batches_2 = collect(plan, ctx.task_ctx()).await.unwrap();
    let result_2 = pretty_format_batches(&batches_2).unwrap().to_string();

    // Results must be identical
    assert_eq!(result_1, result_2, "Results should be the same on hot run");

    let stats_after_second = cache.storage().stats();

    // No new entries on second run (all predicate columns already cached)
    assert_eq!(
        stats_after_second.total_entries, entries_after_first,
        "No new cache entries on hot run"
    );

    // eval_predicate should be > 0 (OS predicate evaluated from cache on 2nd run)
    assert!(
        stats_after_second.runtime.eval_predicate > 0,
        "Predicate should be evaluated from cache on hot run, got eval_predicate={}",
        stats_after_second.runtime.eval_predicate
    );

    // get (projection read) should be 0 — URL is not cached
    assert_eq!(
        stats_after_second.runtime.get, 0,
        "Projection column URL should NOT be read from cache (get={})",
        stats_after_second.runtime.get
    );
}

/// Test that when a column is used in BOTH predicate AND projection,
/// it IS cached and served from cache for both purposes.
#[tokio::test]
async fn test_predicate_column_in_projection_is_cached() {
    let cache_dir = TempDir::new().unwrap();

    // Query: OS is in both WHERE and SELECT
    let sql = r#"SELECT "OS", COUNT(*) as cnt FROM hits WHERE "OS" > 0 GROUP BY "OS" ORDER BY cnt DESC LIMIT 5"#;

    let mut config = SessionConfig::new();
    config.options_mut().execution.target_partitions = 2;
    let (ctx, cache) = LiquidCacheLocalBuilder::new()
        .with_max_memory_bytes(usize::MAX)
        .with_cache_dir(cache_dir.path().to_path_buf())
        .with_squeeze_policy(Box::new(TranscodeSqueezeEvict))
        .with_cache_policy(Box::new(LiquidPolicy::new()))
        .build(config)
        .await
        .unwrap();

    ctx.register_parquet("hits", TEST_FILE, ParquetReadOptions::default())
        .await
        .unwrap();

    // First run: fills cache
    let plan = get_physical_plan(sql, &ctx).await;
    let batches_1 = collect(plan, ctx.task_ctx()).await.unwrap();
    let result_1 = pretty_format_batches(&batches_1).unwrap().to_string();

    // Clear counters
    cache.storage().stats();

    // Second run: should hit cache for BOTH predicate eval AND projection read
    let plan = get_physical_plan(sql, &ctx).await;
    let batches_2 = collect(plan, ctx.task_ctx()).await.unwrap();
    let result_2 = pretty_format_batches(&batches_2).unwrap().to_string();

    assert_eq!(result_1, result_2, "Results should be the same on hot run");

    let stats = cache.storage().stats();

    // OS is a predicate column → should be cached
    // eval_predicate > 0 (WHERE clause evaluated from cache)
    assert!(
        stats.runtime.eval_predicate > 0,
        "Predicate column OS should be evaluated from cache, got eval_predicate={}",
        stats.runtime.eval_predicate
    );

    // OS is also in projection → get_with_selection > 0 or get > 0
    // Since OS is a predicate column, it IS cached and should be read from cache
    let projection_reads = stats.runtime.get + stats.runtime.get_with_selection;
    assert!(
        projection_reads > 0,
        "Predicate column OS in projection should be read from cache, got get={}, get_with_selection={}",
        stats.runtime.get, stats.runtime.get_with_selection
    );
}

/// Test that string predicate columns are NOT cached (only numeric predicates are cached).
/// Query uses a string predicate (URL LIKE) and a numeric predicate (OS > 0).
/// Only the numeric predicate column should be cached.
#[tokio::test]
async fn test_only_numeric_predicate_columns_cached() {
    let cache_dir = TempDir::new().unwrap();

    // Query: WHERE on "URL" (string predicate) AND "OS" (numeric predicate), SELECT "OS"
    // Only OS should be cached; URL is a string predicate and should NOT be cached.
    let sql = r#"SELECT "OS" FROM hits WHERE "URL" LIKE '%tours%' AND "OS" > 0 ORDER BY "OS" LIMIT 5"#;

    let mut config = SessionConfig::new();
    config.options_mut().execution.target_partitions = 2;
    let (ctx, cache) = LiquidCacheLocalBuilder::new()
        .with_max_memory_bytes(usize::MAX)
        .with_cache_dir(cache_dir.path().to_path_buf())
        .with_squeeze_policy(Box::new(TranscodeSqueezeEvict))
        .with_cache_policy(Box::new(LiquidPolicy::new()))
        .build(config)
        .await
        .unwrap();

    ctx.register_parquet("hits", TEST_FILE, ParquetReadOptions::default())
        .await
        .unwrap();

    // First run: fills cache (only numeric predicate columns)
    let plan = get_physical_plan(sql, &ctx).await;
    let batches_1 = collect(plan, ctx.task_ctx()).await.unwrap();
    let result_1 = pretty_format_batches(&batches_1).unwrap().to_string();

    let stats_after_first = cache.storage().stats();
    let entries_after_first = stats_after_first.total_entries;

    // Should have cached only OS (numeric predicate), NOT URL (string predicate)
    assert!(
        entries_after_first > 0,
        "Cache should have entries for numeric predicate column OS"
    );

    // The memory should be small — OS is Int16, not a large string column.
    // If URL were cached, memory would be much larger (URL is a huge string column).
    // OS column for nano_hits: ~1000 rows × 2 bytes = ~2KB per batch.
    // URL column would be megabytes. So we check memory is modest.
    assert!(
        stats_after_first.memory_usage_bytes < 1024 * 1024, // less than 1MB
        "Cache should be small (only OS cached, not URL). Got {} bytes",
        stats_after_first.memory_usage_bytes
    );

    // Second run: should reuse cache
    let plan = get_physical_plan(sql, &ctx).await;
    let batches_2 = collect(plan, ctx.task_ctx()).await.unwrap();
    let result_2 = pretty_format_batches(&batches_2).unwrap().to_string();

    assert_eq!(result_1, result_2, "Results should be identical on hot run");

    let stats_after_second = cache.storage().stats();

    // No new entries on second run
    assert_eq!(
        stats_after_second.total_entries, entries_after_first,
        "No new cache entries on hot run"
    );

    // eval_predicate should be > 0 (OS numeric predicate evaluated from cache)
    assert!(
        stats_after_second.runtime.eval_predicate > 0,
        "Numeric predicate OS should be evaluated from cache, got eval_predicate={}",
        stats_after_second.runtime.eval_predicate
    );
}
