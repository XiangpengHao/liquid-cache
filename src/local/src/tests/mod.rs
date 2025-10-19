use arrow_schema::{DataType, Field, Schema};
use liquid_cache_parquet::LiquidCacheRef;
use liquid_cache_storage::{
    cache::{
        CacheStats,
        squeeze_policies::{Evict, SqueezePolicy, TranscodeEvict, TranscodeSqueezeEvict},
    },
    cache_policies::LiquidPolicy,
};
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
    total_entries: usize,
    entries_after_first_run: usize,
    memory_arrow_entries: usize,
    memory_liquid_entries: usize,
    memory_hybrid_liquid_entries: usize,
    disk_liquid_entries: usize,
    disk_arrow_entries: usize,
    memory_usage_bytes: usize,
    disk_usage_bytes: usize,
    runtime_get_arrow_array_calls: u64,
    runtime_get_with_selection_calls: u64,
    runtime_get_with_predicate_calls: u64,
    runtime_get_predicate_hybrid_success: u64,
    runtime_get_predicate_hybrid_needs_io: u64,
    runtime_get_predicate_hybrid_unsupported: u64,
    runtime_try_read_liquid_calls: u64,
}

impl CacheStatsSummary {
    fn from_stats(after_second_run: CacheStats, entries_after_first_run: usize) -> Self {
        let runtime = after_second_run.runtime;
        Self {
            total_entries: after_second_run.total_entries,
            entries_after_first_run,
            memory_arrow_entries: after_second_run.memory_arrow_entries,
            memory_liquid_entries: after_second_run.memory_liquid_entries,
            memory_hybrid_liquid_entries: after_second_run.memory_hybrid_liquid_entries,
            disk_liquid_entries: after_second_run.disk_liquid_entries,
            disk_arrow_entries: after_second_run.disk_arrow_entries,
            memory_usage_bytes: after_second_run.memory_usage_bytes,
            disk_usage_bytes: after_second_run.disk_usage_bytes,
            runtime_get_arrow_array_calls: runtime.get_arrow_array_calls,
            runtime_get_with_selection_calls: runtime.get_with_selection_calls,
            runtime_get_with_predicate_calls: runtime.get_with_predicate_calls,
            runtime_get_predicate_hybrid_success: runtime.get_predicate_hybrid_success,
            runtime_get_predicate_hybrid_needs_io: runtime.get_predicate_hybrid_needs_io,
            runtime_get_predicate_hybrid_unsupported: runtime.get_predicate_hybrid_unsupported,
            runtime_try_read_liquid_calls: runtime.try_read_liquid_calls,
        }
    }

    fn has_cache_hits(&self) -> bool {
        self.runtime_get_with_selection_calls > 0
            || self.runtime_try_read_liquid_calls > 0
            || self.runtime_get_arrow_array_calls > 0
            || self.runtime_get_with_predicate_calls > 0
    }

    fn entries_reused(&self) -> bool {
        self.total_entries == self.entries_after_first_run
    }
}

impl fmt::Display for CacheStatsSummary {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "entries.total: {}", self.total_entries)?;
        writeln!(
            f,
            "entries.after_first_run: {}",
            self.entries_after_first_run
        )?;
        writeln!(f, "entries.memory.arrow: {}", self.memory_arrow_entries)?;
        writeln!(f, "entries.memory.liquid: {}", self.memory_liquid_entries)?;
        writeln!(
            f,
            "entries.memory.hybrid_liquid: {}",
            self.memory_hybrid_liquid_entries
        )?;
        writeln!(f, "entries.disk.liquid: {}", self.disk_liquid_entries)?;
        writeln!(f, "entries.disk.arrow: {}", self.disk_arrow_entries)?;
        writeln!(f, "usage.memory_bytes: {}", self.memory_usage_bytes)?;
        writeln!(f, "usage.disk_bytes: {}", self.disk_usage_bytes)?;
        writeln!(
            f,
            "runtime.get_arrow_array_calls: {}",
            self.runtime_get_arrow_array_calls
        )?;
        writeln!(
            f,
            "runtime.get_with_selection_calls: {}",
            self.runtime_get_with_selection_calls
        )?;
        writeln!(
            f,
            "runtime.get_with_predicate_calls: {}",
            self.runtime_get_with_predicate_calls
        )?;
        writeln!(
            f,
            "runtime.get_predicate_hybrid_success: {}",
            self.runtime_get_predicate_hybrid_success
        )?;
        writeln!(
            f,
            "runtime.get_predicate_hybrid_needs_io: {}",
            self.runtime_get_predicate_hybrid_needs_io
        )?;
        writeln!(
            f,
            "runtime.get_predicate_hybrid_unsupported: {}",
            self.runtime_get_predicate_hybrid_unsupported
        )?;
        write!(
            f,
            "runtime.try_read_liquid_calls: {}",
            self.runtime_try_read_liquid_calls
        )
    }
}

async fn create_session_context_with_liquid_cache(
    squeeze_policy: Box<dyn SqueezePolicy>,
    cache_size_bytes: usize,
    cache_dir: &Path,
) -> Result<(SessionContext, LiquidCacheRef)> {
    let mut config = SessionConfig::new();
    config.options_mut().execution.target_partitions = 4;
    let (ctx, cache) = LiquidCacheLocalBuilder::new()
        .with_max_cache_bytes(cache_size_bytes)
        .with_cache_dir(cache_dir.to_path_buf())
        .with_squeeze_policy(squeeze_policy)
        .with_cache_policy(Box::new(LiquidPolicy::new()))
        .build(config)?;

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
    let (liquid_ctx, cache) = LiquidCacheLocalBuilder::new()
        .with_cache_dir(cache_dir.path().to_path_buf())
        .with_max_cache_bytes(1024 * 1024)
        .with_squeeze_policy(Box::new(TranscodeSqueezeEvict))
        .build(SessionConfig::new())
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
        assert_eq!(
            df_results, first_liquid_run,
            "reference mismatch on first run for query[{idx}]"
        );

        let entries_after_first_run = cache.storage().stats().total_entries;
        let second_liquid_run = liquid_ctx.sql(sql).await.unwrap().collect().await.unwrap();
        assert_eq!(
            df_results, second_liquid_run,
            "reference mismatch on warmed run for query[{idx}]"
        );

        let stats = CacheStatsSummary::from_stats(cache.storage().stats(), entries_after_first_run);

        assert!(
            stats.has_cache_hits(),
            "expected warmed cache to report hits for query[{idx}]: {sql}\nstats:\n{}",
            &stats
        );
        assert!(
            stats.entries_reused(),
            "expected warmed cache to reuse entries for query[{idx}]: {sql}\nstats:\n{}",
            &stats
        );

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
