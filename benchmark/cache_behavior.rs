use std::fs::File;
use std::path::Path;
use std::process::Command;
use std::sync::Arc;
use std::time::{Duration, Instant};

use datafusion::arrow::datatypes::{DataType, Field, Schema};
use datafusion::arrow::record_batch::RecordBatch;
use datafusion::parquet::arrow::{
    ArrowWriter, ProjectionMask, arrow_reader::ParquetRecordBatchReaderBuilder,
};
use datafusion::parquet::file::properties::WriterProperties;
use datafusion::prelude::{ParquetReadOptions, SessionConfig, SessionContext};
use liquid_cache_parquet::{LiquidCacheInProcessBuilder, LiquidCacheRef, common::LiquidCacheMode};
use liquid_cache_store::store::policies::ToDiskPolicy;
use tempfile::TempDir;

#[derive(Debug)]
struct QueryResult {
    query: String,
    duration: Duration,
    cache_memory_usage: usize,
}

impl QueryResult {
    fn log(&self) {
        println!(
            "Query: {} | Duration: {} ms | Cache memory: {} bytes",
            self.query,
            self.duration.as_millis(),
            self.cache_memory_usage
        );
    }
}

async fn prepare_test_data(benchmark_file: &str) -> Result<(), Box<dyn std::error::Error>> {
    if Path::new(benchmark_file).exists() {
        println!("Benchmark file already exists: {benchmark_file}");
        return Ok(());
    }

    println!("Creating benchmark data from hits.parquet...");

    let hits_file = File::open("benchmark/clickbench/data/hits.parquet")?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(hits_file)?;

    // Find the "Title" column index
    let schema = builder.schema();
    let title_column_index = schema
        .fields()
        .iter()
        .position(|field| field.name() == "Title")
        .ok_or("Title column not found in hits.parquet")?;

    println!("Found Title column at index: {title_column_index}");

    // Create projection mask to only read the Title column
    let projection = ProjectionMask::roots(
        builder.metadata().file_metadata().schema_descr(),
        [title_column_index],
    );

    let reader = builder
        .with_projection(projection)
        .with_batch_size(8192)
        .build()?;

    // Create new schema with columns A and B
    let new_schema = Arc::new(Schema::new(vec![
        Field::new("A", DataType::Utf8, true),
        Field::new("B", DataType::Utf8, true),
    ]));

    // Create output file
    let output_file = File::create(benchmark_file)?;
    let props = WriterProperties::builder()
        .set_compression(datafusion::parquet::basic::Compression::SNAPPY)
        .set_statistics_enabled(parquet::file::properties::EnabledStatistics::Chunk)
        .build();
    let mut writer = ArrowWriter::try_new(output_file, new_schema.clone(), Some(props))?;

    // Process batches and duplicate Title column as A and B
    let mut total_rows = 0;
    for batch in reader {
        let batch = batch?;
        let title_array = batch.column(0);

        // Create new batch with A and B columns both containing Title data
        let new_batch = RecordBatch::try_new(
            new_schema.clone(),
            vec![title_array.clone(), title_array.clone()],
        )?;

        writer.write(&new_batch)?;
        total_rows += new_batch.num_rows();
    }

    writer.close()?;
    println!("Created benchmark file with {total_rows} rows: {benchmark_file}");
    Ok(())
}

async fn run_query_with_timing(
    ctx: &SessionContext,
    cache: &LiquidCacheRef,
    query: &str,
) -> Result<QueryResult, Box<dyn std::error::Error>> {
    let start = Instant::now();
    let df = ctx.sql(query).await?;
    let _result = df.collect().await?;
    let duration = start.elapsed();

    let cache_memory_usage = cache.memory_usage_bytes();

    Ok(QueryResult {
        query: query.to_string(),
        duration,
        cache_memory_usage,
    })
}

async fn drop_os_page_cache() -> Result<(), Box<dyn std::error::Error>> {
    println!("--- Dropping OS page cache ---");

    let output = Command::new("sh")
        .arg("-c")
        .arg("echo 1 | sudo tee /proc/sys/vm/drop_caches")
        .output()?;

    if output.status.success() {
        println!("Successfully dropped OS page cache");
    } else {
        let stderr = String::from_utf8_lossy(&output.stderr);
        println!("Warning: Failed to drop OS page cache: {stderr}");
        println!("You may need to run this manually: echo 1 | sudo tee /proc/sys/vm/drop_caches");
    }

    Ok(())
}

async fn run_cache_behavior_benchmark() -> Result<(), Box<dyn std::error::Error>> {
    let benchmark_file = "benchmark/data/cache_test_data.parquet";

    // Step 1: Prepare test data
    prepare_test_data(benchmark_file).await?;

    // Step 2: Setup liquid cache in-process context
    let temp_dir = TempDir::new()?;
    let (ctx, cache) = LiquidCacheInProcessBuilder::new()
        .with_max_cache_bytes(10 * 1024 * 1024 * 1024) // 10GB
        .with_cache_dir(temp_dir.path().to_path_buf())
        .with_cache_mode(LiquidCacheMode::Liquid {
            transcode_in_background: false,
        })
        .with_cache_strategy(Box::new(ToDiskPolicy::new()))
        .build(SessionConfig::new())?;

    // Register the benchmark parquet file
    ctx.register_parquet("test_table", benchmark_file, ParquetReadOptions::default())
        .await?;

    println!("=== Cache Behavior Benchmark ===");

    drop_os_page_cache().await?;

    run_query_with_timing(
        &ctx,
        &cache,
        "SELECT COUNT(*) FROM test_table WHERE \"A\"='Madison'",
    )
    .await?
    .log();

    run_query_with_timing(
        &ctx,
        &cache,
        "SELECT COUNT(*) FROM test_table WHERE \"A\"='Seattle'",
    )
    .await?
    .log();

    run_query_with_timing(
        &ctx,
        &cache,
        "SELECT COUNT(*) FROM test_table WHERE \"A\">'Utah'",
    )
    .await?
    .log();

    println!("--- Flushing cache to disk ---");
    cache.flush_data();
    drop_os_page_cache().await?;

    run_query_with_timing(
        &ctx,
        &cache,
        "SELECT COUNT(*) FROM test_table WHERE \"A\"<>'Chicago'",
    )
    .await?
    .log();

    run_query_with_timing(
        &ctx,
        &cache,
        "SELECT COUNT(*) FROM test_table WHERE \"B\"='Madison'",
    )
    .await?
    .log();

    run_query_with_timing(
        &ctx,
        &cache,
        "SELECT COUNT(*) FROM test_table WHERE \"A\" like 'Pitts%' AND \"B\" <> 'Madison'",
    )
    .await?
    .log();

    run_query_with_timing(
        &ctx,
        &cache,
        "SELECT COUNT(*) FROM test_table WHERE \"A\"='Seattle'",
    )
    .await?
    .log();

    run_query_with_timing(
        &ctx,
        &cache,
        "SELECT COUNT(*) FROM test_table WHERE \"A\"='Seattle'",
    )
    .await?
    .log();

    println!("--- Resetting cache ---");
    unsafe {
        cache.reset();
    }
    drop_os_page_cache().await?;

    run_query_with_timing(
        &ctx,
        &cache,
        "SELECT COUNT(*) FROM test_table WHERE \"A\"='Chicago' OR \"B\"='Madison'",
    )
    .await?
    .log();

    Ok(())
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize logging (using println! for now since env_logger is not available)

    match run_cache_behavior_benchmark().await {
        Ok(()) => println!("\nBenchmark completed successfully!"),
        Err(e) => {
            eprintln!("Benchmark failed: {e}");
            std::process::exit(1);
        }
    }

    Ok(())
}
