use std::collections::HashSet;
use std::fmt::Display;
use std::fs::File;
use std::io::{Cursor, Write};
use std::sync::Arc;
use std::time::Instant;

use clap::Parser;

use arrow::array::{Array, AsArray, RecordBatch, StringViewArray};
use arrow::compute::{cast, sort_to_indices};
use arrow::datatypes::{DataType, Float64Type};
use arrow_schema::{Field, Schema};
use datafusion::logical_expr::Operator;
use datafusion::prelude::*;
use liquid_cache_parquet::liquid_array::{
    ByteViewArrayMemoryUsage, LiquidArray, LiquidByteArray, LiquidByteViewArray,
};
use rand::{rng, seq::SliceRandom};
use serde::{Deserialize, Serialize};

#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

#[derive(Debug, Clone, PartialEq)]
enum WorkloadType {
    EncodeDecodeOnly,
    FindNeedleOnly,
    SortOnly,
}

impl WorkloadType {
    fn from_str(s: &str) -> Result<Self, String> {
        match s {
            "encode_decode" => Ok(Self::EncodeDecodeOnly),
            "find_needle" => Ok(Self::FindNeedleOnly),
            "sort" => Ok(Self::SortOnly),
            _ => Err(format!("Unknown workload: {}", s)),
        }
    }

    fn parse_workloads(s: &str) -> Result<Vec<Self>, String> {
        if s == "all" {
            Ok(vec![
                Self::EncodeDecodeOnly,
                Self::FindNeedleOnly,
                Self::SortOnly,
            ])
        } else {
            s.split(',')
                .map(|w| Self::from_str(w.trim()))
                .collect::<Result<Vec<_>, _>>()
        }
    }
}

#[derive(Parser)]
#[command(name = "FSST View Benchmark")]
#[command(about = "A benchmark tool for comparing different array compression techniques")]
struct CliArgs {
    /// Workload type to run
    #[arg(long, default_value = "all")]
    #[arg(
        help = "Workload to run: encode_decode, find_needle, sort, all, or comma-separated list (e.g., encode_decode,sort)"
    )]
    workload: String,

    /// Benchmark type to run
    #[arg(long)]
    #[arg(
        help = "Benchmark type to run: fsst_view, byte_array, string_array, string_array_lz4, or all"
    )]
    benchmark: Option<String>,

    /// Columns to process
    #[arg(long)]
    #[arg(
        help = "Comma-separated list of columns to process. Available: Title,URL,SearchPhrase,Referer,OriginalURL"
    )]
    columns: Option<String>,

    #[arg(long, default_value = "false")]
    #[arg(help = "make cargo happy")]
    bench: bool,
}

#[derive(Clone)]
struct ColumnData {
    data: Vec<StringViewArray>,
    avg_str_length: f64,
}

async fn download_clickbench_column(column: &str) -> ColumnData {
    let config = SessionConfig::default().with_batch_size(8192 * 2);
    let ctx = SessionContext::new_with_config(config);
    ctx.register_parquet(
        "hits",
        "../../benchmark/clickbench/data/hits.parquet",
        Default::default(),
    )
    .await
    .unwrap();

    // Load the column data
    let df = ctx
        .sql(&format!("SELECT \"{column}\" from \"hits\""))
        .await
        .unwrap();
    let batches = df.collect().await.unwrap();

    let column_data = batches
        .iter()
        .map(|batch| {
            batch
                .column_by_name(column)
                .unwrap()
                .as_string_view()
                .clone()
        })
        .collect::<Vec<_>>();

    // Get average string length
    let avg_length_batches = ctx
        .sql(&format!(
            "SELECT AVG(LENGTH(\"{column}\")) AS \"{column}\" FROM \"hits\""
        ))
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();

    let avg_str_length = avg_length_batches[0]
        .column_by_name(column)
        .unwrap()
        .as_primitive::<Float64Type>()
        .clone()
        .value(0);

    ColumnData {
        data: column_data,
        avg_str_length,
    }
}

fn load_column_data(column: &str) -> ColumnData {
    let rt = tokio::runtime::Runtime::new().unwrap();
    rt.block_on(download_clickbench_column(column))
}

#[derive(Serialize, Deserialize, Clone)]
struct EncodeResult {
    total_size: usize,
    encode_time_sec: f64,
    decode_time_sec: f64,
    workload: String,
}

#[derive(Serialize, Deserialize, Clone)]
struct FindNeedleResult {
    needle_count: usize,
    total_search_time_sec: f64,
    avg_search_time_per_needle_sec: f64,
    avg_search_time_per_needle_ms: f64,
    workload: String,
}

#[derive(Serialize, Deserialize, Clone)]
struct SortResult {
    total_sort_time_sec: f64,
    workload: String,
}

#[derive(Serialize, Deserialize, Clone)]
struct BenchmarkResults {
    encode_result: Option<EncodeResult>,
    find_needle_result: Option<FindNeedleResult>,
    sort_result: Option<SortResult>,
}

/// Trait for running benchmarks on different array types
trait ArrayBenchmark {
    fn run_encode_decode(&self, arrays: &[StringViewArray]) -> EncodeResult;
    fn run_find_needle(&self, arrays: &[StringViewArray], needles: &[String]) -> FindNeedleResult;
    fn run_sort(&self, arrays: &[StringViewArray]) -> SortResult;
    fn workload_name(&self) -> String;
}

struct BenchmarkRunner;

impl BenchmarkRunner {
    fn get_benchmarks() -> Vec<Box<dyn ArrayBenchmark>> {
        vec![
            Box::new(FsstViewBenchmark),
            Box::new(ByteArrayBenchmark),
            Box::new(StringArrayBenchmark),
            Box::new(StringArrayLz4Benchmark),
        ]
    }

    fn get_filtered_benchmarks(filter: Option<&str>) -> Vec<Box<dyn ArrayBenchmark>> {
        let all_benchmarks = Self::get_benchmarks();

        match filter {
            Some("fsst_view") => vec![Box::new(FsstViewBenchmark)],
            Some("byte_array") => vec![Box::new(ByteArrayBenchmark)],
            Some("string_array") => vec![Box::new(StringArrayBenchmark)],
            Some("string_array_lz4") => vec![Box::new(StringArrayLz4Benchmark)],
            Some("all") | None => all_benchmarks,
            Some(unknown) => {
                eprintln!("Unknown benchmark type: {}. Using all benchmarks.", unknown);
                all_benchmarks
            }
        }
    }

    fn run_workloads(
        &self,
        arrays: &[StringViewArray],
        workloads: &[WorkloadType],
        benchmark_filter: Option<&str>,
    ) -> Vec<BenchmarkResults> {
        let benchmarks = Self::get_filtered_benchmarks(benchmark_filter);
        let needles = select_random_needles(&arrays[0]);

        benchmarks
            .into_iter()
            .map(|benchmark| {
                let mut encode_result = None;
                let mut find_needle_result = None;
                let mut sort_result = None;

                for workload in workloads {
                    match workload {
                        WorkloadType::EncodeDecodeOnly => {
                            let result = benchmark.run_encode_decode(arrays);
                            println!("{} encode/decode: {}", benchmark.workload_name(), result);
                            encode_result = Some(result);
                        }
                        WorkloadType::FindNeedleOnly => {
                            let result = benchmark.run_find_needle(arrays, &needles);
                            println!("{} find needle: {}", benchmark.workload_name(), result);
                            find_needle_result = Some(result);
                        }
                        WorkloadType::SortOnly => {
                            let result = benchmark.run_sort(arrays);
                            println!("{} sort: {}", benchmark.workload_name(), result);
                            sort_result = Some(result);
                        }
                    }
                }
                BenchmarkResults {
                    encode_result,
                    find_needle_result,
                    sort_result,
                }
            })
            .collect()
    }
}

impl Display for EncodeResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{} -- total_size: {} bytes, encode: {} s, decode: {} s",
            self.workload, self.total_size, self.encode_time_sec, self.decode_time_sec
        )
    }
}

impl Display for FindNeedleResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{} -- needles: {}, total: {:.4} s, avg: {:.3} ms per needle",
            self.workload,
            self.needle_count,
            self.total_search_time_sec,
            self.avg_search_time_per_needle_ms
        )
    }
}

impl Display for SortResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{} -- total: {:.4} s",
            self.workload, self.total_sort_time_sec
        )
    }
}

/// Select 10 random different strings from the first batch of the array
fn select_random_needles(first_batch: &StringViewArray) -> Vec<String> {
    let mut unique_strings = HashSet::new();
    let mut all_strings = Vec::new();

    // Collect all non-null unique strings from the first batch
    for i in 0..first_batch.len() {
        if !first_batch.is_null(i) {
            let s = first_batch.value(i).to_string();
            if unique_strings.insert(s.clone()) {
                all_strings.push(s);
            }
        }
    }

    // Shuffle and take up to 10 strings
    let mut rng = rng();
    all_strings.shuffle(&mut rng);
    all_strings.into_iter().take(10).collect()
}

struct FsstViewBenchmark;

impl ArrayBenchmark for FsstViewBenchmark {
    fn workload_name(&self) -> String {
        "FSSTView".to_string()
    }

    fn run_encode_decode(&self, arrays: &[StringViewArray]) -> EncodeResult {
        let (compressor, _) = LiquidByteViewArray::train_from_string_view(&arrays[0]);
        let mut total_size = 0;
        let mut encode_time: f64 = 0.0;
        let mut decode_time: f64 = 0.0;

        for a in arrays {
            let start = Instant::now();
            let array = LiquidByteViewArray::from_string_view_array(a, compressor.clone());
            encode_time += start.elapsed().as_secs_f64();

            total_size += array.get_array_memory_size();

            let start = Instant::now();
            let _array = array.to_arrow_array();
            decode_time += start.elapsed().as_secs_f64();
        }

        EncodeResult {
            total_size,
            encode_time_sec: encode_time,
            decode_time_sec: decode_time,
            workload: self.workload_name(),
        }
    }

    fn run_find_needle(&self, arrays: &[StringViewArray], needles: &[String]) -> FindNeedleResult {
        let (compressor, _) = LiquidByteViewArray::train_from_string_view(&arrays[0]);
        let needle_count = needles.len();
        let mut total_search_time_sec = 0.0;

        for array in arrays {
            let fsst_array = LiquidByteViewArray::from_string_view_array(array, compressor.clone());
            let start = Instant::now();
            for needle in needles {
                let needle_bytes = needle.as_bytes();
                let _result = fsst_array
                    .compare_with(needle_bytes, &Operator::Eq)
                    .unwrap();
            }
            total_search_time_sec += start.elapsed().as_secs_f64();
        }

        let avg_search_time_per_needle_sec = total_search_time_sec / needle_count as f64;
        let avg_search_time_per_needle_ms = avg_search_time_per_needle_sec * 1000.0;

        FindNeedleResult {
            needle_count,
            total_search_time_sec,
            avg_search_time_per_needle_sec,
            avg_search_time_per_needle_ms,
            workload: self.workload_name(),
        }
    }

    fn run_sort(&self, arrays: &[StringViewArray]) -> SortResult {
        let (compressor, _) = LiquidByteViewArray::train_from_string_view(&arrays[0]);
        let mut total_sort_time_sec = 0.0;

        for array in arrays {
            let fsst_array = LiquidByteViewArray::from_string_view_array(array, compressor.clone());
            let start = Instant::now();
            let _indices = fsst_array.sort_to_indices().unwrap();
            total_sort_time_sec += start.elapsed().as_secs_f64();
        }

        SortResult {
            total_sort_time_sec,
            workload: self.workload_name(),
        }
    }
}

struct ByteArrayBenchmark;

impl ArrayBenchmark for ByteArrayBenchmark {
    fn workload_name(&self) -> String {
        "ByteArray".to_string()
    }

    fn run_encode_decode(&self, arrays: &[StringViewArray]) -> EncodeResult {
        let (compressor, _) = LiquidByteArray::train_from_string_view(&arrays[0]);
        let mut total_size = 0;
        let mut encode_time: f64 = 0.0;
        let mut decode_time: f64 = 0.0;

        for a in arrays {
            let start = Instant::now();
            let array = LiquidByteArray::from_string_view_array(a, compressor.clone());
            encode_time += start.elapsed().as_secs_f64();

            total_size += array.get_array_memory_size();

            let start = Instant::now();
            let _array = array.to_arrow_array();
            decode_time += start.elapsed().as_secs_f64();
        }

        EncodeResult {
            total_size,
            encode_time_sec: encode_time,
            decode_time_sec: decode_time,
            workload: self.workload_name(),
        }
    }

    fn run_find_needle(&self, arrays: &[StringViewArray], needles: &[String]) -> FindNeedleResult {
        let (compressor, _) = LiquidByteArray::train_from_string_view(&arrays[0]);
        let needle_count = needles.len();
        let mut total_search_time_sec = 0.0;

        for array in arrays {
            let byte_array = LiquidByteArray::from_string_view_array(array, compressor.clone());
            let start = Instant::now();
            for needle in needles {
                let _result = byte_array.compare_equals(needle);
            }
            total_search_time_sec += start.elapsed().as_secs_f64();
        }

        let avg_search_time_per_needle_sec = total_search_time_sec / needle_count as f64;
        let avg_search_time_per_needle_ms = avg_search_time_per_needle_sec * 1000.0;

        FindNeedleResult {
            needle_count,
            total_search_time_sec,
            avg_search_time_per_needle_sec,
            avg_search_time_per_needle_ms,
            workload: self.workload_name(),
        }
    }

    fn run_sort(&self, arrays: &[StringViewArray]) -> SortResult {
        let (compressor, _) = LiquidByteArray::train_from_string_view(&arrays[0]);
        let mut total_sort_time_sec = 0.0;

        for array in arrays {
            let byte_array = LiquidByteArray::from_string_view_array(array, compressor.clone());
            let start = Instant::now();
            let arrow_array = byte_array.to_arrow_array();
            let _indices = sort_to_indices(&arrow_array, None, None).unwrap();
            total_sort_time_sec += start.elapsed().as_secs_f64();
        }

        SortResult {
            total_sort_time_sec,
            workload: self.workload_name(),
        }
    }
}

struct StringArrayBenchmark;

impl ArrayBenchmark for StringArrayBenchmark {
    fn workload_name(&self) -> String {
        "StringArray".to_string()
    }

    fn run_encode_decode(&self, arrays: &[StringViewArray]) -> EncodeResult {
        let mut encode_time: f64 = 0.0;
        let mut decode_time: f64 = 0.0;
        let mut total_size = 0;

        for a in arrays {
            let start = Instant::now();
            let v = cast(a, &DataType::Utf8).unwrap();
            let v = v.as_string::<i32>().clone();
            encode_time += start.elapsed().as_secs_f64();

            total_size += v.get_array_memory_size();

            let start = Instant::now();
            let _v = cast(&v, &DataType::Utf8View).unwrap();
            decode_time += start.elapsed().as_secs_f64();
        }

        EncodeResult {
            total_size,
            encode_time_sec: encode_time,
            decode_time_sec: decode_time,
            workload: self.workload_name(),
        }
    }

    fn run_find_needle(&self, arrays: &[StringViewArray], needles: &[String]) -> FindNeedleResult {
        if arrays.is_empty() || needles.is_empty() {
            return FindNeedleResult {
                needle_count: 0,
                total_search_time_sec: 0.0,
                avg_search_time_per_needle_sec: 0.0,
                avg_search_time_per_needle_ms: 0.0,
                workload: self.workload_name(),
            };
        }

        let needle_count = needles.len();
        let start = Instant::now();

        for needle in needles {
            for string_view_array in arrays {
                let needle_scalar = arrow::array::StringViewArray::new_scalar(needle.clone());
                let _result =
                    arrow::compute::kernels::cmp::eq(&string_view_array, &needle_scalar).unwrap();
            }
        }

        let total_search_time_sec = start.elapsed().as_secs_f64();
        let avg_search_time_per_needle_sec = total_search_time_sec / needle_count as f64;
        let avg_search_time_per_needle_ms = avg_search_time_per_needle_sec * 1000.0;

        FindNeedleResult {
            needle_count,
            total_search_time_sec,
            avg_search_time_per_needle_sec,
            avg_search_time_per_needle_ms,
            workload: self.workload_name(),
        }
    }

    fn run_sort(&self, arrays: &[StringViewArray]) -> SortResult {
        let mut total_sort_time_sec = 0.0;

        for array in arrays {
            let start = Instant::now();
            let _indices = sort_to_indices(array, None, None).unwrap();
            total_sort_time_sec += start.elapsed().as_secs_f64();
        }

        SortResult {
            total_sort_time_sec,
            workload: self.workload_name(),
        }
    }
}

struct StringArrayLz4Benchmark;

impl ArrayBenchmark for StringArrayLz4Benchmark {
    fn workload_name(&self) -> String {
        "StringArrayLZ4".to_string()
    }

    fn run_encode_decode(&self, arrays: &[StringViewArray]) -> EncodeResult {
        let schema = Arc::new(Schema::new(vec![Field::new("a", DataType::Utf8, false)]));
        let compression = arrow::ipc::CompressionType::LZ4_FRAME;
        let options = arrow::ipc::writer::IpcWriteOptions::default()
            .try_with_compression(Some(compression))
            .unwrap();

        let mut encode_time: f64 = 0.0;
        let mut decode_time: f64 = 0.0;
        let mut total_size = 0;

        for a in arrays {
            let v = cast(a, &DataType::Utf8).unwrap();

            let mut file = vec![];
            let mut writer = arrow::ipc::writer::FileWriter::try_new_with_options(
                &mut file,
                &schema,
                options.clone(),
            )
            .unwrap();

            let start = Instant::now();
            let batch = RecordBatch::try_new(schema.clone(), vec![v]).unwrap();
            writer.write(&batch).unwrap();
            writer.finish().unwrap();
            encode_time += start.elapsed().as_secs_f64();
            total_size += file.len();

            let start = Instant::now();
            let mut file = Cursor::new(file);
            let mut reader = arrow::ipc::reader::FileReader::try_new(&mut file, None).unwrap();
            let batch = reader.next().unwrap().unwrap();
            let _v = batch.column(0).as_string::<i32>().clone();
            decode_time += start.elapsed().as_secs_f64();
        }

        EncodeResult {
            total_size,
            encode_time_sec: encode_time,
            decode_time_sec: decode_time,
            workload: self.workload_name(),
        }
    }

    fn run_find_needle(&self, arrays: &[StringViewArray], needles: &[String]) -> FindNeedleResult {
        if arrays.is_empty() || needles.is_empty() {
            return FindNeedleResult {
                needle_count: 0,
                total_search_time_sec: 0.0,
                avg_search_time_per_needle_sec: 0.0,
                avg_search_time_per_needle_ms: 0.0,
                workload: self.workload_name(),
            };
        }

        let schema = Arc::new(Schema::new(vec![Field::new("a", DataType::Utf8, false)]));
        let compression = arrow::ipc::CompressionType::LZ4_FRAME;
        let options = arrow::ipc::writer::IpcWriteOptions::default()
            .try_with_compression(Some(compression))
            .unwrap();

        let needle_count = needles.len();
        let mut total_search_time_sec = 0.0;

        for array in arrays {
            let v = cast(array, &DataType::Utf8).unwrap();
            let mut file = vec![];
            let mut writer = arrow::ipc::writer::FileWriter::try_new_with_options(
                &mut file,
                &schema,
                options.clone(),
            )
            .unwrap();
            let batch = RecordBatch::try_new(schema.clone(), vec![v]).unwrap();
            writer.write(&batch).unwrap();
            writer.finish().unwrap();

            for needle in needles {
                let start = Instant::now();
                // Decompress and search
                let mut file = Cursor::new(&file);
                let mut reader = arrow::ipc::reader::FileReader::try_new(&mut file, None).unwrap();
                let batch = reader.next().unwrap().unwrap();
                let string_array = batch.column(0).as_string::<i32>();

                let needle_scalar = arrow::array::StringArray::new_scalar(needle.clone());
                let _result =
                    arrow::compute::kernels::cmp::eq(&string_array, &needle_scalar).unwrap();
                total_search_time_sec += start.elapsed().as_secs_f64();
            }
        }
        let avg_search_time_per_needle_sec = total_search_time_sec / needle_count as f64;
        let avg_search_time_per_needle_ms = avg_search_time_per_needle_sec * 1000.0;

        FindNeedleResult {
            needle_count,
            total_search_time_sec,
            avg_search_time_per_needle_sec,
            avg_search_time_per_needle_ms,
            workload: self.workload_name(),
        }
    }

    fn run_sort(&self, arrays: &[StringViewArray]) -> SortResult {
        let schema = Arc::new(Schema::new(vec![Field::new("a", DataType::Utf8, false)]));
        let compression = arrow::ipc::CompressionType::LZ4_FRAME;
        let options = arrow::ipc::writer::IpcWriteOptions::default()
            .try_with_compression(Some(compression))
            .unwrap();

        let mut total_sort_time_sec = 0.0;

        for array in arrays {
            let v = cast(array, &DataType::Utf8).unwrap();
            let mut file = vec![];
            let mut writer = arrow::ipc::writer::FileWriter::try_new_with_options(
                &mut file,
                &schema,
                options.clone(),
            )
            .unwrap();
            let batch = RecordBatch::try_new(schema.clone(), vec![v]).unwrap();
            writer.write(&batch).unwrap();
            writer.finish().unwrap();

            let start = Instant::now();
            // Decompress and sort
            let mut file = Cursor::new(&file);
            let mut reader = arrow::ipc::reader::FileReader::try_new(&mut file, None).unwrap();
            let batch = reader.next().unwrap().unwrap();
            let string_array = batch.column(0).as_string::<i32>();
            let _indices = sort_to_indices(&string_array, None, None).unwrap();
            total_sort_time_sec += start.elapsed().as_secs_f64();
        }

        SortResult {
            total_sort_time_sec,
            workload: self.workload_name(),
        }
    }
}

#[derive(Serialize, Deserialize, Clone)]
struct SerializableMemoryUsage {
    dictionary_views: usize,
    offsets: usize,
    nulls: usize,
    fsst_buffer: usize,
    shared_prefix: usize,
    struct_size: usize,
    total: usize,
}

impl From<ByteViewArrayMemoryUsage> for SerializableMemoryUsage {
    fn from(usage: ByteViewArrayMemoryUsage) -> Self {
        Self {
            dictionary_views: usage.dictionary_views,
            offsets: usage.offsets,
            nulls: usage.nulls,
            fsst_buffer: usage.fsst_buffer,
            shared_prefix: usage.shared_prefix,
            struct_size: usage.struct_size,
            total: usage.total(),
        }
    }
}

#[derive(Serialize, Deserialize)]
struct BenchmarkResult {
    column_name: String,
    avg_string_length: f64,
    benchmark_results: Vec<BenchmarkResults>,
    fsst_view_memory_usage: SerializableMemoryUsage,
}

#[derive(Serialize, Deserialize)]
struct CompleteResults {
    benchmark_name: String,
    timestamp: String,
    columns: Vec<BenchmarkResult>,
}

fn main() {
    let args = CliArgs::parse();

    // Parse and validate workloads
    let workloads_to_run = WorkloadType::parse_workloads(&args.workload).unwrap();

    let all_columns = ["Title", "URL", "SearchPhrase", "Referer", "OriginalURL"];
    let columns_to_process: Vec<&str> = if let Some(ref columns_str) = args.columns {
        columns_str.split(',').map(|s| s.trim()).collect()
    } else {
        all_columns.to_vec()
    };

    let runner = BenchmarkRunner;
    let mut all_column_results = Vec::new();

    println!("Running workloads: {:?}", workloads_to_run);
    if let Some(ref benchmark) = args.benchmark {
        println!("Benchmark filter: {}", benchmark);
    }
    println!("Columns: {}", columns_to_process.join(", "));
    println!();

    for c in columns_to_process {
        println!("Loading column: {c}");
        let column_data = load_column_data(c);
        println!("{c} average length: {}", column_data.avg_str_length);

        println!("Running benchmarks for {c}");
        let benchmark_results = runner.run_workloads(
            &column_data.data,
            &workloads_to_run,
            args.benchmark.as_deref(),
        );

        let (compressor, _) = LiquidByteViewArray::train_from_string_view(&column_data.data[0]);
        let mut total_detailed_memory_usage = ByteViewArrayMemoryUsage {
            dictionary_views: 0,
            offsets: 0,
            nulls: 0,
            fsst_buffer: 0,
            shared_prefix: 0,
            struct_size: 0,
        };

        for a in &column_data.data {
            let array = LiquidByteViewArray::from_string_view_array(a, compressor.clone());
            total_detailed_memory_usage += array.get_detailed_memory_usage();
        }

        all_column_results.push(BenchmarkResult {
            column_name: c.to_string(),
            avg_string_length: column_data.avg_str_length,
            benchmark_results,
            fsst_view_memory_usage: total_detailed_memory_usage.into(),
        });

        println!("Finished processing {c}\n");
    }

    // Write all results to JSON file once at the end
    let complete_results = CompleteResults {
        benchmark_name: "FSST View Study".to_string(),
        timestamp: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs()
            .to_string(),
        columns: all_column_results,
    };

    let json_output = serde_json::to_string_pretty(&complete_results).unwrap();
    let filename = "../../target/benchmark_results.json";
    let mut file = File::create(&filename).unwrap();
    file.write_all(json_output.as_bytes()).unwrap();

    println!("Benchmark results written to {}", filename);

    println!("Benchmark completed!");
}
