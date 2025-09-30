use datafusion::common::tree_node::TreeNode;
use datafusion::execution::SessionStateBuilder;
use datafusion::execution::context::{SessionConfig, SessionContext};
use datafusion::execution::runtime_env::RuntimeEnv;
use datafusion::logical_expr::{Expr, LogicalPlan};
use datafusion::optimizer::Optimizer;
use datafusion::physical_plan::ExecutionPlan;
use datafusion_sql::unparser::expr_to_sql;
use std::collections::HashSet;
use std::sync::Arc;

fn collect_table_names(plan: &LogicalPlan, tables: &mut Vec<String>) {
    match plan {
        LogicalPlan::TableScan(scan) => {
            tables.push(scan.table_name.clone().to_string());
        }
        _ => {
            for input in plan.inputs() {
                collect_table_names(input, tables);
            }
        }
    }
}

/// Extract filter expressions from logical plan
fn get_filter_expressions(plan: &LogicalPlan) -> Vec<String> {
    let mut filters = Vec::new();

    plan.apply(|node| {
        if let LogicalPlan::Filter(filter) = node {
            let sql = expr_to_sql(&filter.predicate).unwrap().to_string();
            filters.push(sql);
        }
        Ok(datafusion::common::tree_node::TreeNodeRecursion::Continue)
    })
    .unwrap();

    filters
}

/// Extract underlying columns from expressions (including those in aggregates)
fn extract_underlying_columns(expr: &Expr) -> HashSet<String> {
    let mut columns = HashSet::new();

    expr.apply(|expr| {
        match expr {
            Expr::Column(column) => {
                columns.insert(column.name.to_string());
            }
            Expr::AggregateFunction(agg) => {
                // Extract ONLY the underlying columns from aggregate function arguments
                // Do NOT include the aggregate function name itself
                for arg in &agg.params.args {
                    let arg_columns = extract_underlying_columns(arg);
                    columns.extend(arg_columns);
                }
                // Skip traversing children since we handled them manually
                return Ok(datafusion::common::tree_node::TreeNodeRecursion::Jump);
            }
            _ => {
                // Continue for other expression types but don't add them to columns
                // This ensures we only get base column names
            }
        }
        Ok(datafusion::common::tree_node::TreeNodeRecursion::Continue)
    })
    .unwrap();

    columns
}

/// Get projected columns from logical plan (finds underlying columns accessed)
fn get_projected_columns(logical_plan: &LogicalPlan) -> Vec<String> {
    let mut all_columns = HashSet::new();
    let mut has_aggregate = false;

    // First check if this query has aggregates
    logical_plan
        .apply(|node| {
            if let LogicalPlan::Aggregate(_) = node {
                has_aggregate = true;
                return Ok(datafusion::common::tree_node::TreeNodeRecursion::Stop);
            }
            Ok(datafusion::common::tree_node::TreeNodeRecursion::Continue)
        })
        .unwrap();

    // Extract all columns from logical plan (includes underlying columns in aggregates)
    logical_plan
        .apply(|node| {
            match node {
                LogicalPlan::Projection(projection) => {
                    // Skip projection nodes if there are aggregates below them
                    // Because they just project aggregate results with names like "count(hits.URL)"
                    if !has_aggregate {
                        for expr in &projection.expr {
                            let columns = extract_underlying_columns(expr);
                            all_columns.extend(columns);
                        }
                    }
                }
                LogicalPlan::Aggregate(aggregate) => {
                    // Extract underlying columns from group expressions
                    for expr in &aggregate.group_expr {
                        let columns = extract_underlying_columns(expr);
                        all_columns.extend(columns);
                    }
                    // Extract underlying columns from aggregate expressions
                    for expr in &aggregate.aggr_expr {
                        let columns = extract_underlying_columns(expr);
                        all_columns.extend(columns);
                    }
                }
                LogicalPlan::Filter(filter) => {
                    // Also include columns from filter expressions
                    let columns = extract_underlying_columns(&filter.predicate);
                    all_columns.extend(columns);
                }
                _ => {}
            }
            Ok(datafusion::common::tree_node::TreeNodeRecursion::Continue)
        })
        .unwrap();

    all_columns.into_iter().collect()
}

/// Main function: returns (filter_expressions, projected_columns)
fn analyze_query(
    logical_plan: &LogicalPlan,
    _physical_plan: &Arc<dyn ExecutionPlan>,
) -> (Vec<String>, Vec<String>) {
    let filter_expressions = get_filter_expressions(logical_plan);
    let projected_columns = get_projected_columns(logical_plan);

    (filter_expressions, projected_columns)
}

#[tokio::main]
async fn main() -> datafusion::error::Result<()> {
    // Create an optimizer with no logical optimization rules
    let optimizer = Optimizer::with_rules(vec![]);

    // Create session config
    let config = SessionConfig::new();

    // Create session state with no optimization rules (both logical and physical)
    let session_state = SessionStateBuilder::new()
        .with_config(config)
        .with_runtime_env(Arc::new(RuntimeEnv::default()))
        .with_optimizer_rules(optimizer.rules.clone()) // Empty rules list
        .with_default_features()
        .build();

    let ctx = SessionContext::new_with_state(session_state);

    ctx.register_parquet("hits", "benchmark/data/hits_0.parquet", Default::default())
        .await?;

    let query_file = "test.sql";
    let query = std::fs::read_to_string(query_file)?;
    let df = ctx.sql(&query).await?;

    // Get both logical and physical plans
    let logical_plan = df.logical_plan().clone();
    let physical_plan = df.create_physical_plan().await?;

    // Analyze query to get filter expressions and projected columns
    let (filter_expressions, projected_columns) = analyze_query(&logical_plan, &physical_plan);

    let mut tables = Vec::new();
    collect_table_names(&logical_plan, &mut tables);

    //println!("Filter expressions: {:?}", filter_expressions);
    //println!("Projected columns: {:?}", projected_columns);

    let mut query_str = String::new();
    query_str.push_str("SELECT ");
    query_str.push_str(&projected_columns.join(", "));
    query_str.push_str(" FROM ");
    query_str.push_str(&tables.join(", "));
    if !filter_expressions.is_empty() {
        query_str.push_str(" WHERE ");
        query_str.push_str(&filter_expressions.join(" AND "));
    }

    println!("Query: {}", query);
    println!("\tSimplified query: {:?}", query_str);

    Ok(())
}
