use anyhow::Result;
use clap::Parser;
use liquid_cache_benchmarks::{
    BenchmarkManifest, InProcessBenchmarkMode, InProcessBenchmarkRunner, setup_observability,
};
use mimalloc::MiMalloc;
use serde::Serialize;
use std::path::PathBuf;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

#[derive(Parser, Serialize, Clone)]
#[command(name = "In-Process Benchmark")]
struct InProcessBenchmark {
    /// Path to the benchmark manifest file (JSON)
    #[arg(long = "manifest")]
    pub manifest: PathBuf,

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

    /// Maximum cache size in bytes
    #[arg(long = "max-cache-mb")]
    pub max_cache_mb: Option<usize>,

    /// Directory to write flamegraph SVG files to
    #[arg(long = "flamegraph-dir")]
    pub flamegraph_dir: Option<PathBuf>,

    /// Query index to run (0-based), if not provided, all queries will be run
    #[arg(long)]
    pub query_index: Option<usize>,
}

impl InProcessBenchmark {
    pub async fn run(self) -> Result<()> {
        let manifest = BenchmarkManifest::load_from_file(&self.manifest)?;
        let output = self.output.clone();

        let runner = InProcessBenchmarkRunner::new()
            .with_bench_mode(self.bench_mode)
            .with_iteration(self.iteration)
            .with_reset_cache(self.reset_cache)
            .with_partitions(self.partitions)
            .with_max_cache_mb(self.max_cache_mb)
            .with_flamegraph_dir(self.flamegraph_dir.clone())
            .with_query_filter(self.query_index);

        runner.run(manifest, self, output).await?;
        Ok(())
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    setup_observability("inprocess", opentelemetry::trace::SpanKind::Client, None);
    let benchmark = InProcessBenchmark::parse();
    benchmark.run().await?;
    Ok(())
}
