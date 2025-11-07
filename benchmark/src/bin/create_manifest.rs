use anyhow::{Context, Result};
use clap::Parser;
use liquid_cache_benchmarks::BenchmarkManifest;
use std::collections::HashMap;
use std::path::{Path, PathBuf};

#[derive(Parser)]
#[command(name = "create_manifest")]
#[command(
    about = "Generates manifest.json files for benchmarks by scanning parquet data and query directories"
)]
struct Args {
    /// Benchmark name: "job" or "publicbi"
    #[arg(short, long)]
    benchmark: String,

    /// Data directory containing parquet files
    #[arg(short, long)]
    data_dir: Option<PathBuf>,

    /// Query directory containing SQL files
    #[arg(short, long)]
    query_dir: Option<PathBuf>,

    /// Output manifest file path
    #[arg(short, long)]
    output: Option<PathBuf>,

    /// Description for the benchmark
    #[arg(long)]
    description: Option<String>,
}

fn scan_parquet_files(data_dir: &Path) -> Result<HashMap<String, String>> {
    let mut tables = HashMap::new();

    if !data_dir.exists() {
        eprintln!("Warning: Data directory does not exist: {:?}", data_dir);
        return Ok(tables);
    }

    let entries = std::fs::read_dir(data_dir)
        .with_context(|| format!("Failed to read data directory: {:?}", data_dir))?;

    for entry in entries {
        let entry = entry?;
        let path = entry.path();

        if path.is_file() && path.extension().and_then(|s| s.to_str()) == Some("parquet") {
            let file_name = path
                .file_stem()
                .and_then(|s| s.to_str())
                .context("Invalid file name")?;

            // Convert filename to table name (lowercase, replace special chars)
            let table_name = file_name.to_lowercase().replace('-', "_");

            // Get relative path from current directory
            let current_dir = std::env::current_dir()?;
            let relative_path = path
                .strip_prefix(&current_dir)
                .unwrap_or(&path)
                .to_string_lossy()
                .to_string();

            tables.insert(table_name, relative_path);
        }
    }

    Ok(tables)
}

fn scan_query_files(query_dir: &Path) -> Result<Vec<String>> {
    let mut queries = Vec::new();

    if !query_dir.exists() {
        eprintln!("Warning: Query directory does not exist: {:?}", query_dir);
        return Ok(queries);
    }

    let entries = std::fs::read_dir(query_dir)
        .with_context(|| format!("Failed to read query directory: {:?}", query_dir))?;

    for entry in entries {
        let entry = entry?;
        let path = entry.path();

        if path.is_file() && path.extension().and_then(|s| s.to_str()) == Some("sql") {
            // Get relative path from current directory
            let current_dir = std::env::current_dir()?;
            let relative_path = path
                .strip_prefix(&current_dir)
                .unwrap_or(&path)
                .to_string_lossy()
                .to_string();

            queries.push(relative_path);
        }
    }

    // Sort queries for consistent output
    queries.sort();

    Ok(queries)
}

fn get_default_paths(benchmark: &str) -> (PathBuf, PathBuf, PathBuf) {
    let base = PathBuf::from("benchmark");
    match benchmark.to_lowercase().as_str() {
        "job" => (
            base.join("job").join("parquet_data"),
            base.join("job").join("queries"),
            base.join("job").join("manifest.json"),
        ),
        "publicbi" | "public_bi" => (
            base.join("publicbi").join("data"),
            base.join("publicbi").join("queries"),
            base.join("publicbi").join("manifest.json"),
        ),
        _ => {
            eprintln!("Unknown benchmark: {}. Using default paths.", benchmark);
            (
                base.join(benchmark).join("data"),
                base.join(benchmark).join("queries"),
                base.join(benchmark).join("manifest.json"),
            )
        }
    }
}

fn main() -> Result<()> {
    let args = Args::parse();

    let (default_data_dir, default_query_dir, default_output) = get_default_paths(&args.benchmark);

    let data_dir = args.data_dir.unwrap_or(default_data_dir);
    let query_dir = args.query_dir.unwrap_or(default_query_dir);
    let output_path = args.output.unwrap_or(default_output);

    println!("Creating manifest for benchmark: {}", args.benchmark);
    println!("  Data directory: {:?}", data_dir);
    println!("  Query directory: {:?}", query_dir);
    println!("  Output: {:?}", output_path);

    // Scan parquet files
    println!("Scanning parquet files...");
    let tables = scan_parquet_files(&data_dir)?;
    println!("  Found {} tables", tables.len());

    // Scan query files
    println!("Scanning query files...");
    let queries = scan_query_files(&query_dir)?;
    println!("  Found {} queries", queries.len());

    // Create manifest
    let mut manifest = BenchmarkManifest::new(args.benchmark.clone());

    if let Some(desc) = args.description {
        manifest = manifest.with_description(desc);
    } else {
        let default_desc = match args.benchmark.to_lowercase().as_str() {
            "job" => "Join Order Benchmark (JOB) - IMDB dataset queries".to_string(),
            "publicbi" | "public_bi" => {
                "Public BI Benchmark - Real-world queries from Tableau Public workbooks".to_string()
            }
            _ => format!("Benchmark: {}", args.benchmark),
        };
        manifest = manifest.with_description(default_desc);
    }

    // Add tables
    for (table_name, path) in tables {
        manifest = manifest.add_table(table_name, path);
    }

    // Add queries
    for query_path in queries {
        manifest = manifest.add_query(query_path);
    }

    // Ensure output directory exists
    if let Some(parent) = output_path.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("Failed to create output directory: {:?}", parent))?;
    }

    // Save manifest
    manifest
        .save_to_file(&output_path)
        .with_context(|| format!("Failed to save manifest to: {:?}", output_path))?;

    println!("Manifest created successfully at: {:?}", output_path);

    Ok(())
}
