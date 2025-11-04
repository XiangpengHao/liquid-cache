use std::sync::Arc;

use arrow::array::{Date32Array, RecordBatch};
use arrow_schema::{DataType, Field, Schema};
use datafusion::parquet::arrow::ArrowWriter;
use liquid_cache_storage::cache::squeeze_policies::TranscodeSqueezeEvict;
use tempfile::TempDir;

use crate::tests::CacheStatsSummary;

fn create_parquet_file(file_path: &str) {
    let schema = Arc::new(Schema::new(vec![Field::new(
        "date_a",
        DataType::Date32,
        false,
    )]));

    let dates = Date32Array::from_iter((0..1_000_000).map(Some));
    let batch = RecordBatch::try_new(Arc::clone(&schema), vec![Arc::new(dates)]).unwrap();
    let file = std::fs::File::create(file_path).unwrap();
    let mut writer = ArrowWriter::try_new(file, schema, None).unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();
}

#[tokio::test]
async fn test_date_extraction_and_ordering() {
    use crate::LiquidCacheLocalBuilder;
    use datafusion::physical_plan::{ExecutionPlan, collect, display::DisplayableExecutionPlan};
    use datafusion::prelude::{ParquetReadOptions, SessionConfig};

    let cache_dir = TempDir::new().unwrap();
    let temp_dir = TempDir::new().unwrap();
    let parquet_path = temp_dir.path().join("dates.parquet");

    // Create the parquet file with date32 data
    create_parquet_file(parquet_path.to_str().unwrap());

    // Set up the session context with liquid cache
    let lc_builder = LiquidCacheLocalBuilder::new()
        .with_max_cache_bytes(1024 * 1024)
        .with_cache_dir(cache_dir.path().to_path_buf())
        .with_squeeze_policy(Box::new(TranscodeSqueezeEvict))
        .with_cache_policy(Box::new(
            liquid_cache_storage::cache_policies::LiquidPolicy::new(),
        ));
    let mut config = SessionConfig::new();
    config.options_mut().execution.target_partitions = 1;
    let (ctx, cache) = lc_builder.build(config).unwrap();

    // Register the parquet file as "date_a" table
    ctx.register_parquet(
        "test_table",
        parquet_path.to_str().unwrap(),
        ParquetReadOptions::default(),
    )
    .await
    .unwrap();

    let sql = r#"select EXTRACT(YEAR from date_a) as year from test_table"#;

    // Get the physical plan
    async fn get_physical_plan(
        sql: &str,
        ctx: &datafusion::prelude::SessionContext,
    ) -> Arc<dyn ExecutionPlan> {
        let df = ctx.sql(sql).await.unwrap();
        let (state, plan) = df.into_parts();
        state.create_physical_plan(&plan).await.unwrap()
    }

    let plan = get_physical_plan(sql, &ctx).await;
    let displayable = DisplayableExecutionPlan::new(plan.as_ref());
    let plan_string = format!("{}", displayable.tree_render());

    // Clear any historical runtime counters before warming the cache
    cache.storage().stats();

    // First run - warms the cache
    let plan_first = get_physical_plan(sql, &ctx).await;
    let batches_first = collect(plan_first, ctx.task_ctx()).await.unwrap();

    let entries_after_first_run = cache.storage().stats().total_entries;

    // Second run - should hit the cache
    let plan_second = get_physical_plan(sql, &ctx).await;
    let batches_second = collect(plan_second, ctx.task_ctx()).await.unwrap();

    assert_eq!(
        batches_first, batches_second,
        "Results should be consistent between runs"
    );

    let stats_after_second_run = cache.storage().stats();
    let stats = CacheStatsSummary::from_stats(stats_after_second_run, entries_after_first_run);

    assert!(stats.has_cache_hits(), "Expected cache hits on second run");
    assert!(
        stats.entries_reused(),
        "Expected cache entries to be reused"
    );

    let snapshot = format!("plan: \n{}\nstats:\n{}", plan_string, stats);
    insta::assert_snapshot!(snapshot);
}
