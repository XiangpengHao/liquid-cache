use std::{path::PathBuf, sync::Arc};

use arrow::util::pretty::pretty_format_batches;
use datafusion::{
    physical_plan::{ExecutionPlan, collect},
    prelude::SessionContext,
};
use liquid_cache_common::CacheEvictionStrategy::Discard;
use liquid_cache_common::CacheMode;
use liquid_cache_parquet::policies::DiscardPolicy;
use uuid::Uuid;

mod cases;

use crate::{LiquidCacheService, LiquidCacheServiceInner};

const TEST_FILE: &str = "../../examples/nano_hits.parquet";

async fn get_physical_plan(sql: &str, ctx: &SessionContext) -> Arc<dyn ExecutionPlan> {
    let df = ctx.sql(sql).await.unwrap();
    let (state, plan) = df.into_parts();
    state.create_physical_plan(&plan).await.unwrap()
}

async fn run_sql(sql: &str, mode: CacheMode, cache_size_bytes: usize, file_path: &str) -> String {
    let ctx = Arc::new(LiquidCacheService::context().unwrap());
    ctx.register_parquet("hits", file_path, Default::default())
        .await
        .unwrap();
    let service = LiquidCacheServiceInner::new(
        ctx.clone(),
        Some(cache_size_bytes),
        PathBuf::from("test"),
        mode,
        Discard,
    );
    async fn get_result(service: &LiquidCacheServiceInner, sql: &str) -> String {
        let handle = Uuid::new_v4();
        let ctx = service.get_ctx();
        let plan = get_physical_plan(sql, &ctx).await;
        service.register_plan(handle, plan);
        let plan = service.get_plan(&handle).unwrap();
        let batches = collect(plan, ctx.task_ctx()).await.unwrap();
        pretty_format_batches(&batches).unwrap().to_string()
    }

    let first_iter = get_result(&service, sql).await;
    let second_iter = get_result(&service, sql).await;

    assert_eq!(first_iter, second_iter);

    first_iter
}

async fn test_runner(sql: &str, reference: &str) {
    let modes = [
        CacheMode::LiquidEagerTranscode,
        CacheMode::Arrow,
        CacheMode::Liquid,
    ];

    // 573960 is the first batch size of URL
    let sizes = [10, 573960, usize::MAX];

    for mode in modes {
        for size in sizes {
            let result = run_sql(sql, mode, size, TEST_FILE).await;
            assert_eq!(result, reference);
        }
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn test_url_prefix() {
    let sql = r#"select COUNT(*) from hits where "URL" like 'https://%'"#;
    let reference = run_sql(sql, CacheMode::LiquidEagerTranscode, 573960, TEST_FILE).await;
    insta::assert_snapshot!(reference);
    test_runner(sql, &reference).await;
}

#[tokio::test(flavor = "multi_thread")]
async fn test_url() {
    let sql = r#"select "URL" from hits where "URL" like '%tours%' order by "URL" desc"#;
    let reference = run_sql(sql, CacheMode::LiquidEagerTranscode, 573960, TEST_FILE).await;
    insta::assert_snapshot!(reference);
    test_runner(sql, &reference).await;
}

#[tokio::test(flavor = "multi_thread")]
async fn test_os() {
    let sql = r#"select "OS" from hits where "URL" like '%tours%' order by "OS" desc"#;
    let reference = run_sql(sql, CacheMode::LiquidEagerTranscode, 573960, TEST_FILE).await;
    insta::assert_snapshot!(reference);
    test_runner(sql, &reference).await;
}

#[tokio::test(flavor = "multi_thread")]
async fn test_referer() {
    let sql = r#"select "Referer" from hits where "Referer" <> '' AND "URL" like '%tours%' order by "Referer" desc"#;
    let reference = run_sql(sql, CacheMode::LiquidEagerTranscode, 573960, TEST_FILE).await;
    insta::assert_snapshot!(reference);
    test_runner(sql, &reference).await;
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "Wait for https://github.com/apache/datafusion/pull/15827 to be merged"]
async fn test_min_max() {
    let sql = r#"select min("Referer"), max("Referer") from hits where "Referer" <> '' AND "URL" like '%tours%'"#;
    let reference = run_sql(sql, CacheMode::LiquidEagerTranscode, 573960, TEST_FILE).await;
    insta::assert_snapshot!(reference);
    test_runner(sql, &reference).await;
}
