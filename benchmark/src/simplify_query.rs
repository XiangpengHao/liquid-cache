use anyhow::{Context, Result};
use clap::Parser;
use datafusion::common::tree_node::TreeNode;
use datafusion::execution::SessionStateBuilder;
use datafusion::execution::context::{SessionConfig, SessionContext};
use datafusion::execution::runtime_env::RuntimeEnv;
use datafusion::logical_expr::{Expr, LogicalPlan};
use datafusion::optimizer::Optimizer;
use datafusion::physical_plan::ExecutionPlan;
use liquid_cache_benchmarks::BenchmarkManifest;
use std::collections::HashSet;
use std::path::{Path, PathBuf};
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
/// Returns a string representation of the filter predicate
fn get_filter_expressions(plan: &LogicalPlan) -> Vec<String> {
    let mut filters = Vec::new();

    plan.apply(|node| {
        if let LogicalPlan::Filter(filter) = node {
            // Format the expression as a string
            // This gives us a readable representation of the filter
            filters.push(format!("{}", filter.predicate));
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

async fn simplify_single_query(
    ctx: &SessionContext,
    query: &str,
    query_name: &str,
) -> Result<String> {
    let df = ctx
        .sql(query)
        .await
        .with_context(|| format!("Failed to parse query from {}", query_name))?;

    // Get both logical and physical plans
    let logical_plan = df.logical_plan().clone();
    let physical_plan = df
        .create_physical_plan()
        .await
        .with_context(|| format!("Failed to create physical plan for {}", query_name))?;

    // Analyze query to get filter expressions and projected columns
    let (filter_expressions, projected_columns) = analyze_query(&logical_plan, &physical_plan);

    let mut tables = Vec::new();
    collect_table_names(&logical_plan, &mut tables);

    if projected_columns.is_empty() {
        // If no columns found, return original query
        return Ok(query.to_string());
    }

    let mut query_str = String::new();
    query_str.push_str("SELECT ");
    query_str.push_str(&projected_columns.join(", "));
    query_str.push_str(" FROM ");
    query_str.push_str(&tables.join(", "));
    if !filter_expressions.is_empty() {
        query_str.push_str(" WHERE ");
        query_str.push_str(&filter_expressions.join(" AND "));
    }

    Ok(query_str)
}

#[derive(Parser)]
#[command(name = "simplify_query")]
#[command(about = "Simplifies SQL queries by extracting filters and projected columns")]
struct Args {
    /// Input: single SQL file or directory containing SQL files
    #[arg(short, long)]
    input: PathBuf,

    /// Output directory for simplified queries (required if input is directory)
    #[arg(short, long)]
    output: Option<PathBuf>,

    /// Manifest file to load table definitions from
    #[arg(short, long)]
    manifest: Option<PathBuf>,

    /// Data directory containing parquet files (alternative to manifest)
    #[arg(short, long)]
    data_dir: Option<PathBuf>,
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();

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

    // Register tables from manifest or data directory
    if let Some(manifest_path) = &args.manifest {
        let manifest = BenchmarkManifest::load_from_file(manifest_path)
            .with_context(|| format!("Failed to load manifest from {:?}", manifest_path))?;
        manifest
            .register_tables(&ctx)
            .await
            .context("Failed to register tables from manifest")?;
        println!("Registered {} tables from manifest", manifest.tables.len());
    } else if let Some(data_dir) = &args.data_dir {
        // Auto-register all parquet files in data directory
        if data_dir.exists() {
            let entries = std::fs::read_dir(data_dir)
                .with_context(|| format!("Failed to read data directory: {:?}", data_dir))?;

            let mut count = 0;
            for entry in entries {
                let entry = entry?;
                let path = entry.path();
                if path.is_file() && path.extension().and_then(|s| s.to_str()) == Some("parquet") {
                    let table_name = path
                        .file_stem()
                        .and_then(|s| s.to_str())
                        .unwrap_or("unknown")
                        .to_lowercase()
                        .replace('-', "_");

                    let path_str = path.to_string_lossy().to_string();
                    ctx.register_parquet(
                        &table_name,
                        &path_str,
                        datafusion::prelude::ParquetReadOptions::default(),
                    )
                    .await
                    .with_context(|| {
                        format!("Failed to register table {} from {:?}", table_name, path)
                    })?;
                    count += 1;
                }
            }
            println!("Registered {} tables from data directory", count);
        }
    }

    // Process input
    if args.input.is_file() {
        // Single file mode
        let query = std::fs::read_to_string(&args.input)
            .with_context(|| format!("Failed to read query file: {:?}", args.input))?;

        let simplified =
            simplify_single_query(&ctx, &query, args.input.to_string_lossy().as_ref()).await?;

        if let Some(output) = &args.output {
            // Write to output file
            std::fs::create_dir_all(output.parent().unwrap_or(Path::new(".")))?;
            std::fs::write(output, simplified)?;
            println!("Simplified query written to: {:?}", output);
        } else {
            // Print to stdout
            println!("Simplified query:\n{}", simplified);
        }
    } else if args.input.is_dir() {
        // Directory mode - process all SQL files
        let output_dir = args
            .output
            .context("Output directory required when processing a directory")?;

        std::fs::create_dir_all(&output_dir)
            .with_context(|| format!("Failed to create output directory: {:?}", output_dir))?;

        let entries = std::fs::read_dir(&args.input)
            .with_context(|| format!("Failed to read input directory: {:?}", args.input))?;

        let mut processed = 0;
        let mut errors = 0;

        for entry in entries {
            let entry = entry?;
            let path = entry.path();

            if path.is_file() && path.extension().and_then(|s| s.to_str()) == Some("sql") {
                let query = std::fs::read_to_string(&path)
                    .with_context(|| format!("Failed to read query file: {:?}", path))?;

                match simplify_single_query(&ctx, &query, path.to_string_lossy().as_ref()).await {
                    Ok(simplified) => {
                        // Generate output filename
                        let file_name = path.file_name().unwrap().to_string_lossy();
                        let output_name = if file_name.ends_with("_simplified.sql") {
                            file_name.to_string()
                        } else {
                            file_name.replace(".sql", "_simplified.sql")
                        };
                        let output_path = output_dir.join(&output_name);

                        std::fs::write(&output_path, simplified)
                            .with_context(|| format!("Failed to write to {:?}", output_path))?;

                        processed += 1;
                        println!("Processed: {} -> {}", path.display(), output_path.display());
                    }
                    Err(e) => {
                        eprintln!("Error processing {:?}: {}", path, e);
                        errors += 1;
                    }
                }
            }
        }

        println!(
            "\nProcessed {} queries successfully, {} errors",
            processed, errors
        );
    } else {
        return Err(anyhow::anyhow!(
            "Input path does not exist: {:?}",
            args.input
        ));
    }

    Ok(())
}
