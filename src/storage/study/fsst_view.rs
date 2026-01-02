use std::collections::HashSet;
use std::fmt::Display;
use std::fs::File;
use std::io::{Cursor, Write};
use std::sync::Arc;
use std::time::Instant;

use clap::Parser;

use arrow::array::{Array, AsArray, RecordBatch, StringViewArray};
use arrow::compute::cast;
use arrow::datatypes::{DataType, Float64Type};
use arrow_schema::{Field, Schema};
use datafusion::logical_expr::Operator;
use datafusion::prelude::*;
use fsst::Compressor;
use liquid_cache_storage::liquid_array::byte_view_array::ByteViewArrayMemoryUsage;
use liquid_cache_storage::liquid_array::raw::FsstArray;
use liquid_cache_storage::liquid_array::{LiquidArray, LiquidByteArray, LiquidByteViewArray};
use rand::{rng, seq::SliceRandom};
use serde::{Deserialize, Serialize};

#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

#[derive(Debug, Clone, PartialEq)]
enum WorkloadType {
    EncodeDecode,
    FindNeedle,
    CmpNeedle,
}

impl WorkloadType {
    fn from_str(s: &str) -> Result<Self, String> {
        match s {
            "encode_decode" => Ok(Self::EncodeDecode),
            "find_needle" => Ok(Self::FindNeedle),
            "cmp_needle" => Ok(Self::CmpNeedle),
            _ => Err(format!("Unknown workload: {s}")),
        }
    }

    fn parse_workloads(s: &str) -> Result<Vec<Self>, String> {
        if s == "all" {
            Ok(vec![Self::EncodeDecode, Self::FindNeedle, Self::CmpNeedle])
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
        help = "Workload to run: encode_decode, find_needle, cmp_needle, sort, all, or comma-separated list (e.g., encode_decode,sort)"
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
    distinct_count_ratio: f64,
    non_empty_ratio: f64,
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
        .sql(&format!("SELECT \"{column}\" from \"hits\" limit 10000000"))
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

    // Get average string length, distinct count ratio, and non-empty ratio in one query
    let stats_batches = ctx
        .sql(&format!(
            "SELECT
                AVG(LENGTH(\"{column}\")) AS avg_length,
                COUNT(DISTINCT \"{column}\") * 1.0 / COUNT(\"{column}\") AS distinct_ratio,
                COUNT(CASE WHEN \"{column}\" IS NOT NULL AND \"{column}\" != '' THEN 1 END) * 1.0 / COUNT(*) AS non_empty_ratio
            FROM \"hits\" limit 10000000"
        ))
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();

    let avg_str_length = stats_batches[0]
        .column_by_name("avg_length")
        .unwrap()
        .as_primitive::<Float64Type>()
        .clone()
        .value(0);

    let distinct_count_ratio = stats_batches[0]
        .column_by_name("distinct_ratio")
        .unwrap()
        .as_primitive::<Float64Type>()
        .clone()
        .value(0);

    let non_empty_ratio = stats_batches[0]
        .column_by_name("non_empty_ratio")
        .unwrap()
        .as_primitive::<Float64Type>()
        .clone()
        .value(0);

    ColumnData {
        data: column_data,
        avg_str_length,
        distinct_count_ratio,
        non_empty_ratio,
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
struct CmpNeedleResult {
    needle_count: usize,
    total_cmp_time_sec: f64,
    avg_cmp_time_per_needle_sec: f64,
    avg_cmp_time_per_needle_ms: f64,
    workload: String,
}

#[derive(Serialize, Deserialize, Clone)]
struct BenchmarkResults {
    encode_results: Vec<EncodeResult>,
    find_needle_results: Vec<FindNeedleResult>,
    cmp_needle_results: Vec<CmpNeedleResult>,
}

/// Trait for running benchmarks on different array types
trait ArrayBenchmark {
    type EncodedData;

    fn encode(&mut self, array: &StringViewArray) -> (Self::EncodedData, f64, usize);
    fn run_decode(&self, encoded_data: &Self::EncodedData) -> f64;
    fn run_find_needle(&self, encoded_data: &Self::EncodedData, needles: &[String]) -> f64;
    fn run_cmp_needle(&self, encoded_data: &Self::EncodedData, needles: &[String]) -> f64;
    fn workload_name(&self) -> String;
}

struct BenchmarkRunner;

impl BenchmarkRunner {
    fn run_benchmark<T: ArrayBenchmark>(
        mut benchmark: T,
        arrays: &[StringViewArray],
        workloads: &[WorkloadType],
        needles: &[String],
    ) -> BenchmarkResults {
        let mut encode_results = Vec::new();
        let mut find_needle_results = Vec::new();
        let mut cmp_needle_results = Vec::new();

        // Repeat each workload 3 times
        for iteration in 0..3 {
            let mut total_encode_time = 0.0;
            let mut total_decode_time = 0.0;
            let mut total_size = 0;
            let mut total_find_needle_time = 0.0;
            let mut total_cmp_needle_time = 0.0;

            // First, encode all arrays (this is common for all workloads)
            let mut encoded_arrays = Vec::new();
            for array in arrays {
                let (encoded_data, encode_time, size) = benchmark.encode(array);
                total_encode_time += encode_time;
                total_size += size;
                encoded_arrays.push(encoded_data);
            }

            // Then run each specific workload
            for workload in workloads {
                match workload {
                    WorkloadType::EncodeDecode => {
                        for encoded_data in &encoded_arrays {
                            let decode_time = benchmark.run_decode(encoded_data);
                            total_decode_time += decode_time;
                        }

                        let result = EncodeResult {
                            total_size,
                            encode_time_sec: total_encode_time,
                            decode_time_sec: total_decode_time,
                            workload: benchmark.workload_name(),
                        };
                        println!(
                            "{} encode/decode (iteration {}): {}",
                            benchmark.workload_name(),
                            iteration + 1,
                            result
                        );
                        encode_results.push(result);
                    }
                    WorkloadType::FindNeedle => {
                        for encoded_data in &encoded_arrays {
                            let search_time = benchmark.run_find_needle(encoded_data, needles);
                            total_find_needle_time += search_time;
                        }

                        let needle_count = needles.len();
                        let avg_search_time_per_needle_sec =
                            total_find_needle_time / needle_count as f64;
                        let result = FindNeedleResult {
                            needle_count,
                            total_search_time_sec: total_find_needle_time,
                            avg_search_time_per_needle_sec,
                            avg_search_time_per_needle_ms: avg_search_time_per_needle_sec * 1000.0,
                            workload: benchmark.workload_name(),
                        };
                        println!(
                            "{} find needle (iteration {}): {}",
                            benchmark.workload_name(),
                            iteration + 1,
                            result
                        );
                        find_needle_results.push(result);
                    }
                    WorkloadType::CmpNeedle => {
                        for encoded_data in &encoded_arrays {
                            let cmp_time = benchmark.run_cmp_needle(encoded_data, needles);
                            total_cmp_needle_time += cmp_time;
                        }

                        let needle_count = needles.len();
                        let avg_cmp_time_per_needle_sec =
                            total_cmp_needle_time / needle_count as f64;
                        let result = CmpNeedleResult {
                            needle_count,
                            total_cmp_time_sec: total_cmp_needle_time,
                            avg_cmp_time_per_needle_sec,
                            avg_cmp_time_per_needle_ms: avg_cmp_time_per_needle_sec * 1000.0,
                            workload: benchmark.workload_name(),
                        };
                        println!(
                            "{} cmp needle (iteration {}): {}",
                            benchmark.workload_name(),
                            iteration + 1,
                            result
                        );
                        cmp_needle_results.push(result);
                    }
                }
            }
        }

        BenchmarkResults {
            encode_results,
            find_needle_results,
            cmp_needle_results,
        }
    }

    fn run_workloads(
        &self,
        arrays: &[StringViewArray],
        workloads: &[WorkloadType],
        benchmark_filter: Option<&str>,
    ) -> Vec<BenchmarkResults> {
        let needles = select_random_needles(&arrays[0]);
        let mut results = Vec::new();

        match benchmark_filter {
            Some("fsst_view") => {
                results.push(Self::run_benchmark(
                    FsstViewBenchmark { compressor: None },
                    arrays,
                    workloads,
                    &needles,
                ));
            }
            Some("byte_array") => {
                results.push(Self::run_benchmark(
                    ByteArrayBenchmark { compressor: None },
                    arrays,
                    workloads,
                    &needles,
                ));
            }
            Some("string_array") => {
                results.push(Self::run_benchmark(
                    StringArrayBenchmark,
                    arrays,
                    workloads,
                    &needles,
                ));
            }
            Some("string_array_lz4") => {
                results.push(Self::run_benchmark(
                    StringArrayLz4Benchmark,
                    arrays,
                    workloads,
                    &needles,
                ));
            }
            Some("all") | None => {
                results.push(Self::run_benchmark(
                    FsstViewBenchmark { compressor: None },
                    arrays,
                    workloads,
                    &needles,
                ));
                results.push(Self::run_benchmark(
                    ByteArrayBenchmark { compressor: None },
                    arrays,
                    workloads,
                    &needles,
                ));
                results.push(Self::run_benchmark(
                    StringArrayBenchmark,
                    arrays,
                    workloads,
                    &needles,
                ));
                results.push(Self::run_benchmark(
                    StringArrayLz4Benchmark,
                    arrays,
                    workloads,
                    &needles,
                ));
            }
            Some(unknown) => {
                eprintln!("Unknown benchmark type: {unknown}. Using all benchmarks.");
                results.push(Self::run_benchmark(
                    FsstViewBenchmark { compressor: None },
                    arrays,
                    workloads,
                    &needles,
                ));
                results.push(Self::run_benchmark(
                    ByteArrayBenchmark { compressor: None },
                    arrays,
                    workloads,
                    &needles,
                ));
                results.push(Self::run_benchmark(
                    StringArrayBenchmark,
                    arrays,
                    workloads,
                    &needles,
                ));
                results.push(Self::run_benchmark(
                    StringArrayLz4Benchmark,
                    arrays,
                    workloads,
                    &needles,
                ));
            }
        }

        results
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

impl Display for CmpNeedleResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{} -- needles: {}, total: {:.4} s, avg: {:.3} ms per needle",
            self.workload,
            self.needle_count,
            self.total_cmp_time_sec,
            self.avg_cmp_time_per_needle_ms
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

struct FsstViewBenchmark {
    compressor: Option<Arc<Compressor>>,
}

impl ArrayBenchmark for FsstViewBenchmark {
    type EncodedData = LiquidByteViewArray<FsstArray>;

    fn workload_name(&self) -> String {
        "FSSTView".to_string()
    }

    fn encode(&mut self, array: &StringViewArray) -> (Self::EncodedData, f64, usize) {
        // Train compressor only on the first call
        let compressor = if let Some(cached_compressor) = &self.compressor {
            cached_compressor.clone()
        } else {
            let (trained_compressor, _) =
                LiquidByteViewArray::<FsstArray>::train_from_string_view(array);
            self.compressor = Some(trained_compressor.clone());
            trained_compressor
        };

        let start = Instant::now();
        let encoded_array =
            LiquidByteViewArray::<FsstArray>::from_string_view_array(array, compressor);
        let encode_time = start.elapsed().as_secs_f64();
        let size = encoded_array.get_array_memory_size();
        (encoded_array, encode_time, size)
    }

    fn run_decode(&self, encoded_data: &Self::EncodedData) -> f64 {
        let start = Instant::now();
        let _array = encoded_data.to_arrow_array();
        start.elapsed().as_secs_f64()
    }

    fn run_find_needle(&self, encoded_data: &Self::EncodedData, needles: &[String]) -> f64 {
        let start = Instant::now();
        for needle in needles {
            let needle_bytes = needle.as_bytes();
            let _result = encoded_data.compare_with(needle_bytes, &Operator::Eq);
        }
        start.elapsed().as_secs_f64()
    }

    fn run_cmp_needle(&self, encoded_data: &Self::EncodedData, needles: &[String]) -> f64 {
        let start = Instant::now();
        for needle in needles {
            let needle_bytes = needle.as_bytes();
            let _result = encoded_data.compare_with(needle_bytes, &Operator::Gt);
        }
        start.elapsed().as_secs_f64()
    }
}

struct ByteArrayBenchmark {
    compressor: Option<Arc<Compressor>>,
}

impl ArrayBenchmark for ByteArrayBenchmark {
    type EncodedData = LiquidByteArray;

    fn workload_name(&self) -> String {
        "ByteArray".to_string()
    }

    fn encode(&mut self, array: &StringViewArray) -> (Self::EncodedData, f64, usize) {
        // Train compressor only on the first call
        let compressor = if let Some(cached_compressor) = &self.compressor {
            cached_compressor.clone()
        } else {
            let (trained_compressor, _) = LiquidByteArray::train_from_string_view(array);
            self.compressor = Some(trained_compressor.clone());
            trained_compressor
        };

        let start = Instant::now();
        let encoded_array = LiquidByteArray::from_string_view_array(array, compressor);
        let encode_time = start.elapsed().as_secs_f64();
        let size = encoded_array.get_array_memory_size();
        (encoded_array, encode_time, size)
    }

    fn run_decode(&self, encoded_data: &Self::EncodedData) -> f64 {
        let start = Instant::now();
        let _array = encoded_data.to_arrow_array();
        start.elapsed().as_secs_f64()
    }

    fn run_find_needle(&self, encoded_data: &Self::EncodedData, needles: &[String]) -> f64 {
        let start = Instant::now();
        for needle in needles {
            let _result = encoded_data.compare_equals(needle);
        }
        start.elapsed().as_secs_f64()
    }

    fn run_cmp_needle(&self, encoded_data: &Self::EncodedData, needles: &[String]) -> f64 {
        let start = Instant::now();
        for needle in needles {
            let arrow_array = encoded_data.to_dict_arrow();
            let needle_scalar = arrow::array::StringArray::new_scalar(needle.clone());
            let _result = arrow::compute::kernels::cmp::gt(&arrow_array, &needle_scalar).unwrap();
        }
        start.elapsed().as_secs_f64()
    }
}

struct StringArrayBenchmark;

impl ArrayBenchmark for StringArrayBenchmark {
    type EncodedData = StringViewArray;

    fn workload_name(&self) -> String {
        "StringArray".to_string()
    }

    fn encode(&mut self, array: &StringViewArray) -> (Self::EncodedData, f64, usize) {
        let start = Instant::now();
        let v = cast(array, &DataType::Utf8).unwrap();
        let v = v.as_string::<i32>().clone();
        let encode_time = start.elapsed().as_secs_f64();
        let size = v.get_array_memory_size();
        (array.clone(), encode_time, size)
    }

    fn run_decode(&self, encoded_data: &Self::EncodedData) -> f64 {
        let start = Instant::now();
        let _v = cast(encoded_data, &DataType::Utf8View).unwrap();
        start.elapsed().as_secs_f64()
    }

    fn run_find_needle(&self, encoded_data: &Self::EncodedData, needles: &[String]) -> f64 {
        let start = Instant::now();
        for needle in needles {
            let needle_scalar = arrow::array::StringViewArray::new_scalar(needle.clone());
            let _result = arrow::compute::kernels::cmp::eq(encoded_data, &needle_scalar).unwrap();
        }
        start.elapsed().as_secs_f64()
    }

    fn run_cmp_needle(&self, encoded_data: &Self::EncodedData, needles: &[String]) -> f64 {
        let start = Instant::now();
        for needle in needles {
            let needle_scalar = arrow::array::StringViewArray::new_scalar(needle.clone());
            let _result = arrow::compute::kernels::cmp::gt(encoded_data, &needle_scalar).unwrap();
        }
        start.elapsed().as_secs_f64()
    }
}

struct StringArrayLz4Benchmark;

impl ArrayBenchmark for StringArrayLz4Benchmark {
    type EncodedData = Vec<u8>;

    fn workload_name(&self) -> String {
        "StringArrayLZ4".to_string()
    }

    fn encode(&mut self, array: &StringViewArray) -> (Self::EncodedData, f64, usize) {
        let schema = Arc::new(Schema::new(vec![Field::new("a", DataType::Utf8, false)]));
        let compression = arrow::ipc::CompressionType::LZ4_FRAME;
        let options = arrow::ipc::writer::IpcWriteOptions::default()
            .try_with_compression(Some(compression))
            .unwrap();

        let v = cast(array, &DataType::Utf8).unwrap();
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
        let encode_time = start.elapsed().as_secs_f64();
        let size = file.len();
        (file, encode_time, size)
    }

    fn run_decode(&self, encoded_data: &Self::EncodedData) -> f64 {
        let start = Instant::now();
        let mut file = Cursor::new(encoded_data);
        let mut reader = arrow::ipc::reader::FileReader::try_new(&mut file, None).unwrap();
        let batch = reader.next().unwrap().unwrap();
        let _v = batch.column(0).as_string::<i32>().clone();
        start.elapsed().as_secs_f64()
    }

    fn run_find_needle(&self, encoded_data: &Self::EncodedData, needles: &[String]) -> f64 {
        let start = Instant::now();
        for needle in needles {
            // Decompress and search
            let mut file = Cursor::new(encoded_data);
            let mut reader = arrow::ipc::reader::FileReader::try_new(&mut file, None).unwrap();
            let batch = reader.next().unwrap().unwrap();
            let string_array = batch.column(0).as_string::<i32>();

            let needle_scalar = arrow::array::StringArray::new_scalar(needle.clone());
            let _result = arrow::compute::kernels::cmp::eq(&string_array, &needle_scalar).unwrap();
        }
        start.elapsed().as_secs_f64()
    }

    fn run_cmp_needle(&self, encoded_data: &Self::EncodedData, needles: &[String]) -> f64 {
        let start = Instant::now();
        for needle in needles {
            // Decompress and search
            let mut file = Cursor::new(encoded_data);
            let mut reader = arrow::ipc::reader::FileReader::try_new(&mut file, None).unwrap();
            let batch = reader.next().unwrap().unwrap();
            let string_array = batch.column(0).as_string::<i32>();

            let needle_scalar = arrow::array::StringArray::new_scalar(needle.clone());
            let _result = arrow::compute::kernels::cmp::gt(&string_array, &needle_scalar).unwrap();
        }
        start.elapsed().as_secs_f64()
    }
}

#[derive(Serialize, Deserialize, Clone)]
struct SerializableMemoryUsage {
    dictionary_views: usize,
    offsets: usize,
    fsst_buffer: usize,
    shared_prefix: usize,
    struct_size: usize,
    total: usize,
}

impl From<ByteViewArrayMemoryUsage> for SerializableMemoryUsage {
    fn from(usage: ByteViewArrayMemoryUsage) -> Self {
        Self {
            dictionary_views: usage.dictionary_key,
            offsets: usage.offsets,
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
    distinct_count_ratio: f64,
    non_empty_ratio: f64,
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

    println!("Running workloads: {workloads_to_run:?}");
    if let Some(ref benchmark) = args.benchmark {
        println!("Benchmark filter: {benchmark}");
    }
    println!("Columns: {}", columns_to_process.join(", "));
    println!();

    for c in columns_to_process {
        println!("Loading column: {c}");
        let column_data = load_column_data(c);
        println!(
            "{c} average length: {:.2}, distinct ratio: {:.4}, non-empty ratio: {:.4}",
            column_data.avg_str_length,
            column_data.distinct_count_ratio,
            column_data.non_empty_ratio
        );

        let benchmark_results = runner.run_workloads(
            &column_data.data,
            &workloads_to_run,
            args.benchmark.as_deref(),
        );

        let (compressor, _) =
            LiquidByteViewArray::<FsstArray>::train_from_string_view(&column_data.data[0]);
        let mut total_detailed_memory_usage = ByteViewArrayMemoryUsage {
            dictionary_key: 0,
            offsets: 0,
            prefix_keys: 0,
            fsst_buffer: 0,
            string_fingerprints: 0,
            shared_prefix: 0,
            struct_size: 0,
        };

        for a in &column_data.data {
            let array =
                LiquidByteViewArray::<FsstArray>::from_string_view_array(a, compressor.clone());
            total_detailed_memory_usage += array.get_detailed_memory_usage();
        }

        all_column_results.push(BenchmarkResult {
            column_name: c.to_string(),
            avg_string_length: column_data.avg_str_length,
            distinct_count_ratio: column_data.distinct_count_ratio,
            non_empty_ratio: column_data.non_empty_ratio,
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
    let mut file = File::create(filename).unwrap();
    file.write_all(json_output.as_bytes()).unwrap();

    println!("Benchmark results written to {filename}");

    println!("Benchmark completed!");
}
