use anyhow::{Result, anyhow};
use arrow::{
    array::{
        Array, ArrayRef, BinaryArray, DictionaryArray, FixedSizeBinaryArray, LargeBinaryArray,
        LargeStringArray, StringArray, StructArray,
    },
    datatypes::{
        ArrowDictionaryKeyType, DataType, Date32Type, Date64Type, Decimal128Type, Decimal256Type,
        Float16Type, Float32Type, Float64Type, Int16Type, Int32Type, Int64Type, Int8Type,
        UInt16Type, UInt32Type, UInt64Type, UInt8Type, TimeUnit,
    },
};
use clap::Parser;
use datafusion::execution::object_store::ObjectStoreUrl;
use liquid_cache_benchmarks::{
    BenchmarkManifest, MinimalClientConfig, MinimalServerConfig, Query, build_client_context,
    manifest_object_store_options, run_query, setup_observability, start_minimal_server,
};
use liquid_cache_common::IoMode;
use log::info;
use mimalloc::MiMalloc;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use std::{
    collections::HashMap, net::SocketAddr, path::{Path, PathBuf}, sync::Arc, time::Duration,
};

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

#[derive(Parser, Clone)]
#[command(name = "Data Pruning Runner")]
struct CliArgs {
    /// Flight server address
    #[arg(long, default_value = "127.0.0.1:15214")]
    address: SocketAddr,

    /// Benchmark manifest file
    #[arg(long, default_value = "benchmark/clickbench/manifest.json")]
    manifest: PathBuf,

    /// Query id to run (defaults to all queries)
    #[arg(long)]
    query: Option<u32>,

    /// Number of partitions to use
    #[arg(long)]
    partitions: Option<usize>,

    /// Maximum cache size in MB
    #[arg(long = "max-cache-mb")]
    max_cache_mb: Option<usize>,

    /// Path to disk cache directory
    #[arg(long = "disk-cache-dir")]
    disk_cache_dir: Option<PathBuf>,

    /// Jaeger OTLP gRPC endpoint (for example: http://localhost:4317)
    #[arg(long = "jaeger-endpoint")]
    jaeger_endpoint: Option<String>,

    /// IO mode, available options: uring, uring-direct, std-blocking, tokio, std-spawn-blocking
    #[arg(long = "io-mode", default_value = "uring-multi-async")]
    io_mode: IoMode,
}

struct LayerConfig {
    name: &'static str,
    zone_mapping: bool,
    filter_pushdown: bool,
    dynamic_filtering: bool,
    squeeze: bool,
}

struct SqueezeUsage {
    mem_before: u64,
    disk_before: u64,
    mem_after: u64,
    disk_after: u64,
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = CliArgs::parse();
    setup_observability("data-pruning-runner", args.jaeger_endpoint.as_deref());

    let manifest = BenchmarkManifest::load_from_file(&args.manifest)?;
    let object_store_options = manifest_object_store_options(&manifest)?;
    let server_url = format!("http://{}", args.address);
    let raw_data_bytes = raw_data_bytes_for_manifest_tables(&manifest)?;

    let layers = [
        LayerConfig {
            name: "modern-encodings",
            zone_mapping: false,
            filter_pushdown: false,
            dynamic_filtering: false,
            squeeze: false,
        },
        LayerConfig {
            name: "modern-encodings+zone-mapping",
            zone_mapping: true,
            filter_pushdown: false,
            dynamic_filtering: false,
            squeeze: false,
        },
        LayerConfig {
            name: "modern-encodings+zone-mapping+filter-pushdown",
            zone_mapping: true,
            filter_pushdown: true,
            dynamic_filtering: false,
            squeeze: false,
        },
        LayerConfig {
            name: "modern-encodings+zone-mapping+filter-pushdown+dynamic-filtering",
            zone_mapping: true,
            filter_pushdown: true,
            dynamic_filtering: true,
            squeeze: false,
        },
        /*LayerConfig {
            name: "zone-mapping+filter-pushdown+dynamic-filtering+squeeze",
            zone_mapping: true,
            filter_pushdown: true,
            dynamic_filtering: true,
            squeeze: true,
        },*/
    ];

    let mut queries = manifest.load_queries(0);
    if let Some(query_id) = args.query {
        queries.retain(|query| query.id() == query_id);
        if queries.is_empty() {
            return Err(anyhow!("No query with id {query_id} found in manifest"));
        }
    }

    let header = layers
        .iter()
        .map(|layer| layer.name)
        .collect::<Vec<_>>()
        .join(",");
    println!(
        "query,raw_data_bytes,{},squeeze_mem_before_bytes,squeeze_disk_before_bytes,squeeze_mem_after_bytes,squeeze_disk_after_bytes",
        header
    );
    for query in &queries {
        eprintln!("Starting query {}", query.id());
        let mut values = Vec::with_capacity(layers.len());
        for layer in &layers {
            eprintln!("Running query {} layer {}", query.id(), layer.name);
            let server = start_minimal_server(MinimalServerConfig {
                address: args.address,
                max_cache_mb: args.max_cache_mb,
                disk_cache_dir: args.disk_cache_dir.clone(),
                enable_squeeze: layer.squeeze,
                io_mode: args.io_mode,
                zone_mapping: layer.zone_mapping,
                filter_pushdown: layer.filter_pushdown,
            })?;

            tokio::time::sleep(Duration::from_millis(200)).await;
            let (mem_before, disk_before) = server.cache_usage_bytes();
            if mem_before != 0 || disk_before != 0 {
                eprintln!(
                    "Warning: cache not empty before query {} layer {} (mem={}, disk={})",
                    query.id(),
                    layer.name,
                    mem_before,
                    disk_before
                );
            }

            let ctx = build_client_context(
                &MinimalClientConfig {
                    server: server_url.clone(),
                    partitions: args.partitions,
                    zone_mapping: layer.zone_mapping,
                    filter_pushdown: layer.filter_pushdown,
                    dynamic_filtering: layer.dynamic_filtering,
                },
                &object_store_options,
            )?;

            manifest.register_object_stores(&ctx).await?;
            manifest.register_tables(&ctx).await?;

            let ctx = Arc::new(ctx);
            let mut total_batches = 0usize;
            let mut total_rows = 0usize;
            for statement in query.statement() {
                let (results, _plan, _plan_uuids) = run_query(&ctx, statement).await;
                /*
                let bytes = flight_data_bytes_from_plan(&plan);
                if args.query.is_some() {
                    let record_bytes =
                        flight_metric_from_plan(&plan, "flight_record_batch_bytes");
                    let dictionary_bytes =
                        flight_metric_from_plan(&plan, "flight_dictionary_bytes");
                    eprintln!(
                        "Query {} layer {} flight_bytes total={} record={} dictionary={}",
                        query.id(),
                        layer.name,
                        bytes,
                        record_bytes,
                        dictionary_bytes
                    );
                    if query.id() == 25 {
                        eprintln!(
                            "Query {} layer {} plan:\n{}",
                            query.id(),
                            layer.name,
                            datafusion::physical_plan::display::DisplayableExecutionPlan::with_metrics(plan.as_ref()).indent(true)
                        );
                    }
                }
                total_received = total_received.saturating_add(bytes);
                */
                total_batches += results.len();
                total_rows += results.iter().map(|batch| batch.num_rows()).sum::<usize>();
            }

            let (mem_after, disk_after) = server.cache_usage_bytes();
            info!(
                "Layer {}, query {}: cache_mem_bytes={}, cache_disk_bytes={}, batches={}, rows={}",
                layer.name, query.id(), mem_after, disk_after, total_batches, total_rows
            );
            values.push(mem_after);

            server.shutdown().await?;
            tokio::time::sleep(Duration::from_millis(200)).await;
        }

        eprintln!("Running query {} squeeze usage", query.id());
        let squeeze = run_squeeze_usage_for_query(
            &args,
            &manifest,
            &object_store_options,
            &server_url,
            query,
        )
        .await?;

        let values = values
            .into_iter()
            .map(|value| value.to_string())
            .collect::<Vec<_>>()
            .join(",");
        println!(
            "{},{},{},{},{},{},{}",
            query.id(),
            raw_data_bytes,
            values,
            squeeze.mem_before,
            squeeze.disk_before,
            squeeze.mem_after,
            squeeze.disk_after
        );
    }
    Ok(())
}

async fn run_squeeze_usage_for_query(
    args: &CliArgs,
    manifest: &BenchmarkManifest,
    object_store_options: &[(ObjectStoreUrl, HashMap<String, String>)],
    server_url: &str,
    query: &Query,
) -> Result<SqueezeUsage> {
    let server = start_minimal_server(MinimalServerConfig {
        address: args.address,
        max_cache_mb: args.max_cache_mb,
        disk_cache_dir: args.disk_cache_dir.clone(),
        enable_squeeze: true,
        io_mode: args.io_mode,
        zone_mapping: true,
        filter_pushdown: true,
    })?;

    tokio::time::sleep(Duration::from_millis(200)).await;
    let (mem_before, disk_before) = server.cache_usage_bytes();
    if mem_before != 0 || disk_before != 0 {
        eprintln!(
            "Warning: cache not empty before squeeze query {} (mem={}, disk={})",
            query.id(),
            mem_before,
            disk_before
        );
    }

    let ctx = build_client_context(
        &MinimalClientConfig {
            server: server_url.to_string(),
            partitions: args.partitions,
            zone_mapping: true,
            filter_pushdown: true,
            dynamic_filtering: true,
        },
        object_store_options,
    )?;

    manifest.register_object_stores(&ctx).await?;
    manifest.register_tables(&ctx).await?;

    let ctx = Arc::new(ctx);
    for statement in query.statement() {
        let _ = run_query(&ctx, statement).await;
    }

    let (mem_before, disk_before) = server.cache_usage_bytes();
    server.squeeze_all_entries().await;
    let (mem_after, disk_after) = server.cache_usage_bytes();

    server.shutdown().await?;
    tokio::time::sleep(Duration::from_millis(200)).await;

    Ok(SqueezeUsage {
        mem_before,
        disk_before,
        mem_after,
        disk_after,
    })
}

/*
fn flight_data_bytes_from_plan(plan: &Arc<dyn ExecutionPlan>) -> u64 {
    fn sum_for_plan(plan: &Arc<dyn ExecutionPlan>) -> u64 {
        let mut total = 0u64;
        if let Some(metrics) = plan.metrics() {
            if let Some(metric) = metrics.sum_by_name("flight_data_bytes") {
                if let MetricValue::Count { count, .. } = metric {
                    total += count.value() as u64;
                }
            }
        }
        for child in plan.children() {
            total += sum_for_plan(child);
        }
        total
    }
    sum_for_plan(plan)
}

fn flight_metric_from_plan(plan: &Arc<dyn ExecutionPlan>, name: &str) -> u64 {
    fn sum_for_plan(plan: &Arc<dyn ExecutionPlan>, name: &str) -> u64 {
        let mut total = 0u64;
        if let Some(metrics) = plan.metrics() {
            if let Some(metric) = metrics.sum_by_name(name) {
                if let MetricValue::Count { count, .. } = metric {
                    total += count.value() as u64;
                }
            }
        }
        for child in plan.children() {
            total += sum_for_plan(child, name);
        }
        total
    }
    sum_for_plan(plan, name)
}
*/

fn raw_data_bytes_for_manifest_tables(manifest: &BenchmarkManifest) -> Result<u64> {
    let mut total = 0u64;
    for table_path in manifest.tables.values() {
        let path = resolve_local_table_path(table_path)?;
        let metadata = std::fs::metadata(&path)?;
        if metadata.is_dir() {
            for entry in std::fs::read_dir(&path)? {
                let entry = entry?;
                let entry_path = entry.path();
                if entry_path.extension().and_then(|ext| ext.to_str()) == Some("parquet") {
                    total += raw_data_bytes_for_parquet(&entry_path)?;
                }
            }
        } else {
            total += raw_data_bytes_for_parquet(&path)?;
        }
    }
    Ok(total)
}

fn resolve_local_table_path(table_path: &str) -> Result<PathBuf> {
    if table_path.starts_with("s3://")
        || table_path.starts_with("http://")
        || table_path.starts_with("https://")
    {
        return Err(anyhow!(
            "raw data size only supports local files, got {table_path}"
        ));
    }
    if let Some(stripped) = table_path.strip_prefix("file://") {
        return Ok(PathBuf::from(stripped));
    }
    let current_dir = std::env::current_dir()?;
    Ok(current_dir.join(table_path))
}

fn raw_data_bytes_for_parquet(path: &Path) -> Result<u64> {
    let file = std::fs::File::open(path)?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file)?;
    let total_rows = builder.metadata().file_metadata().num_rows() as u64;
    let mut reader = builder.build()?;
    let mut sample_rows = 0u64;
    let mut sample_bytes = 0u64;
    for _ in 0..100 {
        let Some(batch) = reader.next() else {
            break;
        };
        let batch = batch?;
        sample_rows += batch.num_rows() as u64;
        sample_bytes += logical_value_bytes_for_batch(&batch);
    }
    if sample_rows == 0 {
        return Ok(0);
    }
    let avg_per_row = sample_bytes as f64 / sample_rows as f64;
    Ok((avg_per_row * total_rows as f64).round() as u64)
}

fn logical_value_bytes_for_batch(batch: &arrow::record_batch::RecordBatch) -> u64 {
    batch
        .columns()
        .iter()
        .map(|column| logical_value_bytes(column))
        .sum()
}

fn logical_value_bytes(array: &ArrayRef) -> u64 {
    let valid_len = (array.len() - array.null_count()) as u64;
    match array.data_type() {
        DataType::Boolean => (valid_len + 7) / 8,
        DataType::Int8 => valid_len * byte_width::<Int8Type>(),
        DataType::UInt8 => valid_len * byte_width::<UInt8Type>(),
        DataType::Int16 => valid_len * byte_width::<Int16Type>(),
        DataType::UInt16 => valid_len * byte_width::<UInt16Type>(),
        DataType::Int32 => valid_len * byte_width::<Int32Type>(),
        DataType::UInt32 => valid_len * byte_width::<UInt32Type>(),
        DataType::Int64 => valid_len * byte_width::<Int64Type>(),
        DataType::UInt64 => valid_len * byte_width::<UInt64Type>(),
        DataType::Float16 => valid_len * byte_width::<Float16Type>(),
        DataType::Float32 => valid_len * byte_width::<Float32Type>(),
        DataType::Float64 => valid_len * byte_width::<Float64Type>(),
        DataType::Date32 => valid_len * byte_width::<Date32Type>(),
        DataType::Date64 => valid_len * byte_width::<Date64Type>(),
        DataType::Time32(TimeUnit::Second) | DataType::Time32(TimeUnit::Millisecond) => {
            valid_len * 4
        }
        DataType::Time64(TimeUnit::Microsecond) | DataType::Time64(TimeUnit::Nanosecond) => {
            valid_len * 8
        }
        DataType::Timestamp(_, _) => valid_len * 8,
        DataType::Decimal128(_, _) => valid_len * byte_width::<Decimal128Type>(),
        DataType::Decimal256(_, _) => valid_len * byte_width::<Decimal256Type>(),
        DataType::FixedSizeBinary(byte_len) => {
            let array = array.as_any().downcast_ref::<FixedSizeBinaryArray>().unwrap();
            let mut total = 0u64;
            for i in 0..array.len() {
                if array.is_valid(i) {
                    total += *byte_len as u64;
                }
            }
            total
        }
        DataType::Binary => {
            let array = array.as_any().downcast_ref::<BinaryArray>().unwrap();
            sum_binary_lengths(array)
        }
        DataType::LargeBinary => {
            let array = array.as_any().downcast_ref::<LargeBinaryArray>().unwrap();
            sum_large_binary_lengths(array)
        }
        DataType::Utf8 => {
            let array = array.as_any().downcast_ref::<StringArray>().unwrap();
            sum_string_lengths(array)
        }
        DataType::LargeUtf8 => {
            let array = array.as_any().downcast_ref::<LargeStringArray>().unwrap();
            sum_large_string_lengths(array)
        }
        DataType::Dictionary(key_type, value_type) => {
            match &**key_type {
                DataType::Int8 => dictionary_value_bytes::<Int8Type>(array, value_type),
                DataType::Int16 => dictionary_value_bytes::<Int16Type>(array, value_type),
                DataType::Int32 => dictionary_value_bytes::<Int32Type>(array, value_type),
                DataType::Int64 => dictionary_value_bytes::<Int64Type>(array, value_type),
                DataType::UInt8 => dictionary_value_bytes::<UInt8Type>(array, value_type),
                DataType::UInt16 => dictionary_value_bytes::<UInt16Type>(array, value_type),
                DataType::UInt32 => dictionary_value_bytes::<UInt32Type>(array, value_type),
                DataType::UInt64 => dictionary_value_bytes::<UInt64Type>(array, value_type),
                _ => 0,
            }
        }
        DataType::Struct(fields) => {
            fields
                .iter()
                .enumerate()
                .map(|(index, _field)| {
                    let child = array.as_any().downcast_ref::<StructArray>().unwrap();
                    logical_value_bytes(child.column(index))
                })
                .sum()
        }
        _ => 0,
    }
}

fn byte_width<T: arrow::datatypes::ArrowPrimitiveType>() -> u64 {
    std::mem::size_of::<T::Native>() as u64
}

fn sum_string_lengths(array: &StringArray) -> u64 {
    let mut total = 0u64;
    for i in 0..array.len() {
        if array.is_valid(i) {
            total += array.value_length(i) as u64;
        }
    }
    total
}

fn sum_large_string_lengths(array: &LargeStringArray) -> u64 {
    let mut total = 0u64;
    for i in 0..array.len() {
        if array.is_valid(i) {
            total += array.value_length(i) as u64;
        }
    }
    total
}

fn sum_binary_lengths(array: &BinaryArray) -> u64 {
    let mut total = 0u64;
    for i in 0..array.len() {
        if array.is_valid(i) {
            total += array.value_length(i) as u64;
        }
    }
    total
}

fn sum_large_binary_lengths(array: &LargeBinaryArray) -> u64 {
    let mut total = 0u64;
    for i in 0..array.len() {
        if array.is_valid(i) {
            total += array.value_length(i) as u64;
        }
    }
    total
}

fn dictionary_value_bytes<K: arrow::datatypes::ArrowPrimitiveType + ArrowDictionaryKeyType>(
    array: &ArrayRef,
    value_type: &DataType,
) -> u64 {
    let dict = array
        .as_any()
        .downcast_ref::<DictionaryArray<K>>()
        .unwrap();
    let values = dict.values();
    let valid_len = (dict.len() - dict.null_count()) as u64;
    match value_type {
        DataType::Utf8 => {
            let values = values.as_any().downcast_ref::<StringArray>().unwrap();
            let mut total = 0u64;
            for i in 0..dict.len() {
                if let Some(key) = dict.key(i) {
                    total += values.value_length(key) as u64;
                }
            }
            total
        }
        DataType::LargeUtf8 => {
            let values = values.as_any().downcast_ref::<LargeStringArray>().unwrap();
            let mut total = 0u64;
            for i in 0..dict.len() {
                if let Some(key) = dict.key(i) {
                    total += values.value_length(key) as u64;
                }
            }
            total
        }
        DataType::Binary => {
            let values = values.as_any().downcast_ref::<BinaryArray>().unwrap();
            let mut total = 0u64;
            for i in 0..dict.len() {
                if let Some(key) = dict.key(i) {
                    total += values.value_length(key) as u64;
                }
            }
            total
        }
        DataType::LargeBinary => {
            let values = values.as_any().downcast_ref::<LargeBinaryArray>().unwrap();
            let mut total = 0u64;
            for i in 0..dict.len() {
                if let Some(key) = dict.key(i) {
                    total += values.value_length(key) as u64;
                }
            }
            total
        }
        DataType::FixedSizeBinary(byte_len) => {
            valid_len * (*byte_len as u64)
        }
        DataType::Int8 => valid_len * byte_width::<Int8Type>(),
        DataType::UInt8 => valid_len * byte_width::<UInt8Type>(),
        DataType::Int16 => valid_len * byte_width::<Int16Type>(),
        DataType::UInt16 => valid_len * byte_width::<UInt16Type>(),
        DataType::Int32 => valid_len * byte_width::<Int32Type>(),
        DataType::UInt32 => valid_len * byte_width::<UInt32Type>(),
        DataType::Int64 => valid_len * byte_width::<Int64Type>(),
        DataType::UInt64 => valid_len * byte_width::<UInt64Type>(),
        DataType::Float16 => valid_len * byte_width::<Float16Type>(),
        DataType::Float32 => valid_len * byte_width::<Float32Type>(),
        DataType::Float64 => valid_len * byte_width::<Float64Type>(),
        DataType::Date32 => valid_len * byte_width::<Date32Type>(),
        DataType::Date64 => valid_len * byte_width::<Date64Type>(),
        DataType::Decimal128(_, _) => valid_len * byte_width::<Decimal128Type>(),
        DataType::Decimal256(_, _) => valid_len * byte_width::<Decimal256Type>(),
        _ => 0,
    }
}
