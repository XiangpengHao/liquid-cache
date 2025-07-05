use std::{path::Path, sync::Arc};

use datafusion::{
    physical_plan::{ExecutionPlan, display::DisplayableExecutionPlan},
    prelude::{SessionConfig, SessionContext},
};

use crate::LiquidCacheBuilder;

async fn setup_tpch_ctx() -> Arc<SessionContext> {
    let mut session_config = SessionConfig::from_env().unwrap();
    // To get deterministic results
    session_config.options_mut().execution.target_partitions = 8;
    let ctx = LiquidCacheBuilder::new("http://localhost:50051")
        .build(session_config)
        .unwrap();
    let tables = [
        "customer", "lineitem", "nation", "orders", "part", "partsupp", "region", "supplier",
    ];

    for table in tables {
        let table_url = format!("../../benchmark/tpch/data/sf0.001/{table}.parquet");
        ctx.register_parquet(table, table_url, Default::default())
            .await
            .unwrap();
    }
    Arc::new(ctx)
}

/// One query file can contain multiple queries, separated by `;`
fn get_query_by_id(query_dir: impl AsRef<Path>, query_id: u32) -> Vec<String> {
    let query_dir = query_dir.as_ref();
    let mut path = query_dir.to_owned();
    path.push(format!("q{query_id}.sql"));
    let content = std::fs::read_to_string(&path).unwrap();
    content
        .split(';')
        .map(|s| s.trim())
        .filter(|s| !s.is_empty())
        .map(|s| s.to_string())
        .collect()
}

async fn get_query_plan_by_id(query_id: u32) -> Arc<dyn ExecutionPlan> {
    let query_dir = Path::new("../../benchmark/tpch/queries");
    let ctx = setup_tpch_ctx().await;

    let query = if query_id != 15 {
        &get_query_by_id(query_dir, query_id)[0]
    } else {
        &get_query_by_id(query_dir, query_id)[1]
    };
    let result = ctx.sql(query).await.unwrap();
    let (state, plan) = result.into_parts();
    state.create_physical_plan(&plan).await.unwrap()
}

#[tokio::test]
async fn test_tpch_q1() {
    let physical_plan = get_query_plan_by_id(1).await;
    let displayable = DisplayableExecutionPlan::new(physical_plan.as_ref());
    insta::assert_snapshot!(displayable.tree_render().to_string());
}

#[tokio::test]
async fn test_tpch_q2() {
    let physical_plan = get_query_plan_by_id(2).await;
    let displayable = DisplayableExecutionPlan::new(physical_plan.as_ref());
    insta::assert_snapshot!(displayable.tree_render().to_string());
}

#[tokio::test]
async fn test_tpch_q3() {
    let physical_plan = get_query_plan_by_id(3).await;
    let displayable = DisplayableExecutionPlan::new(physical_plan.as_ref());
    insta::assert_snapshot!(displayable.tree_render().to_string());
}

#[tokio::test]
async fn test_tpch_q4() {
    let physical_plan = get_query_plan_by_id(4).await;
    let displayable = DisplayableExecutionPlan::new(physical_plan.as_ref());
    insta::assert_snapshot!(displayable.tree_render().to_string());
}

#[tokio::test]
async fn test_tpch_q5() {
    let physical_plan = get_query_plan_by_id(5).await;
    let displayable = DisplayableExecutionPlan::new(physical_plan.as_ref());
    insta::assert_snapshot!(displayable.tree_render().to_string());
}

#[tokio::test]
async fn test_tpch_q6() {
    let physical_plan = get_query_plan_by_id(6).await;
    let displayable = DisplayableExecutionPlan::new(physical_plan.as_ref());
    insta::assert_snapshot!(displayable.tree_render().to_string());
}

#[tokio::test]
async fn test_tpch_q7() {
    let physical_plan = get_query_plan_by_id(7).await;
    let displayable = DisplayableExecutionPlan::new(physical_plan.as_ref());
    insta::assert_snapshot!(displayable.tree_render().to_string());
}

#[tokio::test]
async fn test_tpch_q8() {
    let physical_plan = get_query_plan_by_id(8).await;
    let displayable = DisplayableExecutionPlan::new(physical_plan.as_ref());
    insta::assert_snapshot!(displayable.tree_render().to_string());
}

#[tokio::test]
async fn test_tpch_q9() {
    let physical_plan = get_query_plan_by_id(9).await;
    let displayable = DisplayableExecutionPlan::new(physical_plan.as_ref());
    insta::assert_snapshot!(displayable.tree_render().to_string());
}

#[tokio::test]
async fn test_tpch_q10() {
    let physical_plan = get_query_plan_by_id(10).await;
    let displayable = DisplayableExecutionPlan::new(physical_plan.as_ref());
    insta::assert_snapshot!(displayable.tree_render().to_string());
}

#[tokio::test]
async fn test_tpch_q11() {
    let physical_plan = get_query_plan_by_id(11).await;
    let displayable = DisplayableExecutionPlan::new(physical_plan.as_ref());
    insta::assert_snapshot!(displayable.tree_render().to_string());
}

#[tokio::test]
async fn test_tpch_q12() {
    let physical_plan = get_query_plan_by_id(12).await;
    let displayable = DisplayableExecutionPlan::new(physical_plan.as_ref());
    insta::assert_snapshot!(displayable.tree_render().to_string());
}

#[tokio::test]
async fn test_tpch_q13() {
    let physical_plan = get_query_plan_by_id(13).await;
    let displayable = DisplayableExecutionPlan::new(physical_plan.as_ref());
    insta::assert_snapshot!(displayable.tree_render().to_string());
}

#[tokio::test]
async fn test_tpch_q14() {
    let physical_plan = get_query_plan_by_id(14).await;
    let displayable = DisplayableExecutionPlan::new(physical_plan.as_ref());
    insta::assert_snapshot!(displayable.tree_render().to_string());
}

#[tokio::test]
/// Q15 needs to actually run the first query to get the correct plan
#[ignore]
async fn test_tpch_q15() {
    let physical_plan = get_query_plan_by_id(15).await;
    let displayable = DisplayableExecutionPlan::new(physical_plan.as_ref());
    insta::assert_snapshot!(displayable.tree_render().to_string());
}

#[tokio::test]
async fn test_tpch_q16() {
    let physical_plan = get_query_plan_by_id(16).await;
    let displayable = DisplayableExecutionPlan::new(physical_plan.as_ref());
    insta::assert_snapshot!(displayable.tree_render().to_string());
}

#[tokio::test]
async fn test_tpch_q17() {
    let physical_plan = get_query_plan_by_id(17).await;
    let displayable = DisplayableExecutionPlan::new(physical_plan.as_ref());
    insta::assert_snapshot!(displayable.tree_render().to_string());
}

#[tokio::test]
async fn test_tpch_q18() {
    let physical_plan = get_query_plan_by_id(18).await;
    let displayable = DisplayableExecutionPlan::new(physical_plan.as_ref());
    insta::assert_snapshot!(displayable.tree_render().to_string());
}

#[tokio::test]
async fn test_tpch_q19() {
    let physical_plan = get_query_plan_by_id(19).await;
    let displayable = DisplayableExecutionPlan::new(physical_plan.as_ref());
    insta::assert_snapshot!(displayable.tree_render().to_string());
}

#[tokio::test]
async fn test_tpch_q20() {
    let physical_plan = get_query_plan_by_id(20).await;
    let displayable = DisplayableExecutionPlan::new(physical_plan.as_ref());
    insta::assert_snapshot!(displayable.tree_render().to_string());
}

#[tokio::test]
async fn test_tpch_q21() {
    let physical_plan = get_query_plan_by_id(21).await;
    let displayable = DisplayableExecutionPlan::new(physical_plan.as_ref());
    insta::assert_snapshot!(displayable.tree_render().to_string());
}

#[tokio::test]
async fn test_tpch_q22() {
    let physical_plan = get_query_plan_by_id(22).await;
    let displayable = DisplayableExecutionPlan::new(physical_plan.as_ref());
    insta::assert_snapshot!(displayable.tree_render().to_string());
}
