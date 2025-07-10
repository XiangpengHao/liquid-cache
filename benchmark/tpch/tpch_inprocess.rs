use anyhow::Result;
use clap::Parser;
use liquid_cache_benchmarks::{
    BenchmarkManifest, InProcessBenchmarkMode, InProcessBenchmarkRunner, setup_observability, tpch,
};
use mimalloc::MiMalloc;
use serde::Serialize;
use std::{collections::HashMap, path::PathBuf};
use url::Url;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

const RELEVANT_QUERIES: &[u32] = &[4, 6, 11, 12, 14, 15, 16, 20];

#[derive(Parser, Serialize, Clone)]
#[command(name = "TPCH In-Process Benchmark")]
struct TpchInProcessBenchmark {
    /// Path to the query directory
    #[arg(long = "query-dir")]
    pub query_dir: PathBuf,

    /// Path to the data directory with TPCH data
    #[arg(long = "data-dir")]
    pub data_dir: PathBuf,

    /// Benchmark mode to use
    #[arg(long = "bench-mode", default_value = "liquid-eager-transcode")]
    pub bench_mode: InProcessBenchmarkMode,

    /// Number of times to run each query
    #[arg(long, default_value = "3")]
    pub iteration: u32,

    /// Path to the output JSON file to save the benchmark results
    #[arg(long)]
    pub output: Option<PathBuf>,

    /// Reset the cache before running a new query
    #[arg(long = "reset-cache", default_value = "false")]
    pub reset_cache: bool,

    /// Number of partitions to use
    #[arg(long)]
    pub partitions: Option<usize>,

    /// Query number to run, if not provided, all queries will be run
    #[arg(long)]
    pub query: Option<u32>,

    /// Maximum cache size in bytes
    #[arg(long = "max-cache-mb")]
    pub max_cache_mb: Option<usize>,

    /// Directory to write flamegraph SVG files to
    #[arg(long)]
    pub flamegraph_dir: Option<PathBuf>,
}

impl TpchInProcessBenchmark {
    fn generate_manifest(&self) -> Result<BenchmarkManifest> {
        let current_dir = std::env::current_dir()?.to_string_lossy().to_string();
        let tables = [
            "customer", "lineitem", "nation", "orders", "part", "partsupp", "region", "supplier",
        ];

        let mut manifest = BenchmarkManifest::new("TPC-H In-Process Benchmark".to_string())
            .with_description("TPC-H benchmark using in-process liquid cache".to_string());

        for table_name in tables.iter() {
            let table_path = Url::parse(&format!(
                "file://{}/{}/{}.parquet",
                current_dir,
                self.data_dir.display(),
                table_name
            ))?
            .to_string();
            manifest = manifest.add_table(table_name.to_string(), table_path);
        }

        // Load queries from TPC-H directory
        let queries = tpch::get_all_queries(&self.query_dir)?;

        // Filter to relevant queries if no specific query is requested
        let query_filter: Vec<u32> = if let Some(query) = self.query {
            vec![query]
        } else {
            RELEVANT_QUERIES.to_vec()
        };

        for query in queries {
            if query_filter.contains(&query.id) {
                // For TPC-H, we add the query SQL directly as inline SQL
                manifest = manifest.add_query(query.sql);
            }
        }

        // Add special handling for Q15
        if query_filter.contains(&15) {
            let mut special_handling = HashMap::new();
            special_handling.insert(15, "tpch_q15".to_string());
            manifest = manifest.with_special_query_handling(special_handling);
        }

        Ok(manifest)
    }

    pub async fn run(self) -> Result<()> {
        let manifest = self.generate_manifest()?;

        // Convert query number to query index for the runner
        let query_filter = if let Some(query_num) = self.query {
            // Find the index of the requested query in the relevant queries
            RELEVANT_QUERIES.iter().position(|&q| q == query_num)
        } else {
            None
        };

        let bench_mode = self.bench_mode;
        let iteration = self.iteration;
        let reset_cache = self.reset_cache;
        let partitions = self.partitions;
        let max_cache_mb = self.max_cache_mb;
        let flamegraph_dir = self.flamegraph_dir.clone();
        let output = self.output.clone();

        let runner = InProcessBenchmarkRunner::new()
            .with_bench_mode(bench_mode)
            .with_iteration(iteration)
            .with_reset_cache(reset_cache)
            .with_partitions(partitions)
            .with_max_cache_mb(max_cache_mb)
            .with_flamegraph_dir(flamegraph_dir)
            .with_query_filter(query_filter);

        runner.run(manifest, self, output).await?;
        Ok(())
    }
}

#[tokio::main]
pub async fn main() -> Result<()> {
    setup_observability(
        "tpch-inprocess",
        opentelemetry::trace::SpanKind::Client,
        None,
    );
    let benchmark = TpchInProcessBenchmark::parse();
    benchmark.run().await?;
    Ok(())
}
