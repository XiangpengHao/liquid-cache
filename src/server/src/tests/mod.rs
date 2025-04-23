use std::sync::Arc;

use arrow::util::pretty::pretty_format_batches;
use datafusion::{physical_plan::ExecutionPlan, prelude::SessionContext};
use futures::StreamExt;
use liquid_cache_common::CacheMode;
use uuid::Uuid;

use crate::{LiquidCacheService, LiquidCacheServiceInner};

const TEST_FILE: &str = "../../examples/nano_hits.parquet";

async fn get_physical_plan(sql: &str, ctx: &SessionContext) -> Arc<dyn ExecutionPlan> {
    let df = ctx.sql(sql).await.unwrap();
    let (state, plan) = df.into_parts();
    state.create_physical_plan(&plan).await.unwrap()
}

async fn run_sql(sql: &str, mode: CacheMode, cache_size: usize) -> String {
    let ctx = Arc::new(LiquidCacheService::context(None).unwrap());
    ctx.register_parquet("hits", TEST_FILE, Default::default())
        .await
        .unwrap();
    let service = LiquidCacheServiceInner::new(ctx.clone(), Some(cache_size), None);

    let handle = Uuid::new_v4();
    let plan = get_physical_plan(sql, &ctx).await;
    service.register_plan(handle, plan, mode);

    async fn get_result(service: &LiquidCacheServiceInner, handle: Uuid) -> String {
        let mut stream = service.execute_plan(&handle, 0).await;
        let mut batches = Vec::new();
        while let Some(batch_result) = stream.next().await {
            let batch = batch_result.unwrap();
            batches.push(batch);
        }
        pretty_format_batches(&batches).unwrap().to_string()
    }

    let first_iter = get_result(&service, handle).await;
    let second_iter = get_result(&service, handle).await;

    assert_eq!(first_iter, second_iter);

    first_iter
}

#[tokio::test]
async fn test_url_prefix() {
    let sql = "select COUNT(*) from hits where \"URL\" like 'https://%'";
    let eager = run_sql(sql, CacheMode::LiquidEagerTranscode, 8192).await;
    insta::assert_snapshot!(eager);

    let arrow = run_sql(sql, CacheMode::Arrow, usize::MAX).await;
    assert_eq!(eager, arrow);

    let lazy = run_sql(sql, CacheMode::Liquid, 8192).await;
    assert_eq!(eager, lazy);
}

#[tokio::test]
async fn test_url() {
    let sql = "select \"URL\" from hits where \"URL\" like '%tours%' order by \"URL\" desc";
    // 573960 is the first batch size of URL
    let eager = run_sql(sql, CacheMode::LiquidEagerTranscode, 573960).await;
    insta::assert_snapshot!(eager);

    let arrow = run_sql(sql, CacheMode::Arrow, usize::MAX).await;
    assert_eq!(eager, arrow);

    let lazy = run_sql(sql, CacheMode::Liquid, 573960).await;
    assert_eq!(eager, lazy);
}

#[tokio::test]
async fn test_os() {
    let sql = "select \"OS\" from hits where \"URL\" like '%tours%' order by \"OS\" desc";
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
