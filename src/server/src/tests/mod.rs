use std::sync::Arc;

use arrow::util::pretty::pretty_format_batches;
use datafusion::{
    physical_plan::{ExecutionPlan, collect},
    prelude::SessionContext,
};
use liquid_cache_common::{CacheMode, LiquidCacheMode};
use uuid::Uuid;

use crate::{LiquidCacheService, LiquidCacheServiceInner};

const TEST_FILE: &str = "../../examples/nano_hits.parquet";

async fn get_physical_plan(sql: &str, ctx: &SessionContext) -> Arc<dyn ExecutionPlan> {
    let df = ctx.sql(sql).await.unwrap();
    let (state, plan) = df.into_parts();
    state.create_physical_plan(&plan).await.unwrap()
}

async fn run_sql(sql: &str, mode: CacheMode, cache_size: usize) -> String {
    let ctx = Arc::new(LiquidCacheService::context().unwrap());
    ctx.register_parquet("hits", TEST_FILE, Default::default())
        .await
        .unwrap();
    let service = LiquidCacheServiceInner::new(
        ctx.clone(),
        Some(cache_size),
        None,
        LiquidCacheMode::from(mode),
    );
    async fn get_result(service: &LiquidCacheServiceInner, sql: &str, mode: CacheMode) -> String {
        let handle = Uuid::new_v4();
        let ctx = service.get_ctx();
        let plan = get_physical_plan(sql, &ctx).await;
        service.register_plan(handle, plan, mode);
        let plan = service.get_plan(&handle).unwrap();
        let batches = collect(plan, ctx.task_ctx()).await.unwrap();
        pretty_format_batches(&batches).unwrap().to_string()
    }

    let first_iter = get_result(&service, sql, mode).await;
    let second_iter = get_result(&service, sql, mode).await;

    assert_eq!(first_iter, second_iter);

    first_iter
}

#[tokio::test]
async fn test_url_prefix() {
    let sql = r#"select COUNT(*) from hits where "URL" like 'https://%'"#;
    let eager = run_sql(sql, CacheMode::LiquidEagerTranscode, 573960).await;
    insta::assert_snapshot!(eager);

    let arrow = run_sql(sql, CacheMode::Arrow, usize::MAX).await;
    assert_eq!(eager, arrow);

    let lazy = run_sql(sql, CacheMode::Liquid, 573960).await;
    assert_eq!(eager, lazy);
}

#[tokio::test]
async fn test_url() {
    let sql = r#"select "URL" from hits where "URL" like '%tours%' order by "URL" desc"#;
    // 573960 is the first batch size of URL
    let eager = run_sql(sql, CacheMode::LiquidEagerTranscode, 573960).await;
    insta::assert_snapshot!(eager);

    let arrow = run_sql(sql, CacheMode::Arrow, usize::MAX).await;
    assert_eq!(eager, arrow);

    let lazy = run_sql(sql, CacheMode::Liquid, 573960).await;
    assert_eq!(eager, lazy);
}

#[tokio::test]
async fn test_url_small_cache() {
    let sql = r#"select "URL" from hits where "URL" like '%tours%' order by "URL" desc"#;
    let arrow = run_sql(sql, CacheMode::Arrow, 10).await;
    insta::assert_snapshot!(arrow);
}

#[tokio::test]
async fn test_os() {
    let sql = r#"select "OS" from hits where "URL" like '%tours%' order by "OS" desc"#;
    let eager = run_sql(sql, CacheMode::LiquidEagerTranscode, 573960).await;
    insta::assert_snapshot!(eager);

    let arrow = run_sql(sql, CacheMode::Arrow, usize::MAX).await;
    assert_eq!(eager, arrow);

    let lazy = run_sql(sql, CacheMode::Liquid, 573960).await;
    assert_eq!(eager, lazy);
}

#[tokio::test]
async fn test_referer() {
    let sql = r#"select "Referer" from hits where "Referer" <> '' AND "URL" like '%tours%' order by "Referer" desc"#;
    let eager = run_sql(sql, CacheMode::LiquidEagerTranscode, 573960).await;
    insta::assert_snapshot!(eager);

    let arrow = run_sql(sql, CacheMode::Arrow, usize::MAX).await;
    assert_eq!(eager, arrow);

    let lazy = run_sql(sql, CacheMode::Liquid, 573960).await;
    assert_eq!(eager, lazy);
}

#[tokio::test]
#[ignore = "Wait for https://github.com/apache/datafusion/pull/15827 to be merged"]
async fn test_min_max() {
    let sql = r#"select min("Referer"), max("Referer") from hits where "Referer" <> '' AND "URL" like '%tours%'"#;
    let eager = run_sql(sql, CacheMode::LiquidEagerTranscode, 573960).await;
    insta::assert_snapshot!(eager);

    let arrow = run_sql(sql, CacheMode::Arrow, usize::MAX).await;
    assert_eq!(eager, arrow);

    let lazy = run_sql(sql, CacheMode::Liquid, 573960).await;
    assert_eq!(eager, lazy);
}
