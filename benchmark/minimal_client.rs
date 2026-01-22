use anyhow::{Result, anyhow};
use clap::Parser;
use liquid_cache_benchmarks::{
    BenchmarkManifest, MinimalClientConfig, build_client_context, manifest_object_store_options,
    run_query, setup_observability,
};
use log::info;
use mimalloc::MiMalloc;
use std::{path::PathBuf, sync::Arc};

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

#[derive(Parser, Clone)]
#[command(name = "Minimal LiquidCache Client")]
struct CliArgs {
    /// LiquidCache server address (Flight gRPC)
    #[arg(long, default_value = "http://127.0.0.1:15214")]
    server: String,

    /// Path to the benchmark manifest file
    #[arg(long)]
    manifest: PathBuf,

    /// Query id to run (defaults to all queries in the manifest)
    #[arg(long)]
    query: Option<u32>,

    /// Number of partitions to use
    #[arg(long)]
    partitions: Option<usize>,

    /// Enable parquet row-group pruning (zone mapping)
    #[arg(long = "zone-mapping", default_value_t = true)]
    zone_mapping: bool,

    /// Enable parquet filter pushdown
    #[arg(long = "filter-pushdown", default_value_t = true)]
    filter_pushdown: bool,

    /// Enable dynamic filter pushdown
    #[arg(long = "dynamic-filtering", default_value_t = true)]
    dynamic_filtering: bool,

    /// Jaeger OTLP gRPC endpoint (for example: http://localhost:4317)
    #[arg(long = "jaeger-endpoint")]
    jaeger_endpoint: Option<String>,
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = CliArgs::parse();
    setup_observability("minimal-liquid-cache-client", args.jaeger_endpoint.as_deref());

    let manifest = BenchmarkManifest::load_from_file(&args.manifest)?;
    let object_store_options = manifest_object_store_options(&manifest)?;
    let ctx = build_client_context(
        &MinimalClientConfig {
            server: args.server.clone(),
            partitions: args.partitions,
            zone_mapping: args.zone_mapping,
            filter_pushdown: args.filter_pushdown,
            dynamic_filtering: args.dynamic_filtering,
        },
        &object_store_options,
    )?;

    manifest.register_object_stores(&ctx).await?;
    manifest.register_tables(&ctx).await?;

    let mut queries = manifest.load_queries(0);
    if let Some(query_id) = args.query {
        queries.retain(|query| query.id() == query_id);
        if queries.is_empty() {
            return Err(anyhow!("No query with id {query_id} found in manifest"));
        }
    }

    let ctx = Arc::new(ctx);
    for query in queries {
        for statement in query.statement() {
            let (results, _plan, _plan_uuids) = run_query(&ctx, statement).await;
            let row_count: usize = results.iter().map(|batch| batch.num_rows()).sum();
            info!(
                "Query {} completed: {} batches, {} rows",
                query.id(),
                results.len(),
                row_count
            );
        }
    }

    Ok(())
}
