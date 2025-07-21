use std::collections::{HashMap, HashSet};
use std::fmt::Display;
use std::fs::File;
use std::io::{Cursor, Write};
use std::sync::Arc;
use std::time::Instant;

use arrow::array::{Array, AsArray, RecordBatch, StringViewArray};
use arrow::compute::cast;
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

#[derive(Clone)]
struct Dataset {
    data: HashMap<String, ColumnData>,
}

#[derive(Clone)]
struct ColumnData {
    data: Vec<StringViewArray>,
    avg_str_length: f64,
}

async fn download_clickbench(string_columns: &[&str]) -> Dataset {
    let config = SessionConfig::default().with_batch_size(8192 * 2);
    let ctx = SessionContext::new_with_config(config);
    ctx.register_parquet(
        "hits",
        "../../benchmark/clickbench/data/hits.parquet",
        Default::default(),
    )
    .await
    .unwrap();

    let quoted_columns = string_columns
        .iter()
        .map(|c| format!("\"{c}\""))
        .collect::<Vec<_>>();

    let df = ctx
        .sql(&format!(
            "SELECT {} from \"hits\"",
            quoted_columns.join(", ")
        ))
        .await
        .unwrap();
    let batches = df.collect().await.unwrap();

    let avg_length = string_columns
        .iter()
        .map(|c| format!("AVG(LENGTH(\"{c}\")) AS \"{c}\""))
        .collect::<Vec<_>>()
        .join(", ");
    let avg_length_batches = ctx
        .sql(&format!("SELECT {avg_length} FROM \"hits\""))
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();

    let mut data = HashMap::new();
    for column in string_columns {
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
        let avg_str_length = avg_length_batches[0]
            .column_by_name(column)
            .unwrap()
            .as_primitive::<Float64Type>()
            .clone()
            .value(0);
        data.insert(
            column.to_string(),
            ColumnData {
                data: column_data,
                avg_str_length,
            },
        );
    }

    Dataset { data }
}

fn setup_data(columns: &[&str]) -> Dataset {
    let rt = tokio::runtime::Runtime::new().unwrap();
    rt.block_on(download_clickbench(columns))
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
struct BenchmarkResults {
    encode_result: EncodeResult,
    find_needle_result: FindNeedleResult,
}

/// Trait for running benchmarks on different array types
trait ArrayBenchmark {
    fn run_encode_decode(&self, arrays: &[StringViewArray]) -> EncodeResult;
    fn run_find_needle(&self, arrays: &[StringViewArray], needles: &[String]) -> FindNeedleResult;
    fn workload_name(&self) -> String;
}

struct BenchmarkRunner;

impl BenchmarkRunner {
    fn run_all_benchmarks(&self, arrays: &[StringViewArray]) -> Vec<BenchmarkResults> {
        let needles = select_random_needles(&arrays[0]);

        let benchmarks: Vec<Box<dyn ArrayBenchmark>> = vec![
            Box::new(FsstViewBenchmark),
            Box::new(ByteArrayBenchmark),
            Box::new(StringArrayBenchmark),
            Box::new(StringArrayLz4Benchmark),
        ];

        benchmarks
            .into_iter()
            .map(|benchmark| {
                let encode_result = benchmark.run_encode_decode(arrays);
                println!(
                    "{} encode/decode: {}",
                    benchmark.workload_name(),
                    encode_result
                );
                let find_needle_result = benchmark.run_find_needle(arrays, &needles);
                println!(
                    "{} find needle: {}",
                    benchmark.workload_name(),
                    find_needle_result
                );
                BenchmarkResults {
                    encode_result,
                    find_needle_result,
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
    let columns = ["Title", "URL", "SearchPhrase", "Referer", "OriginalURL"];
    let dataset = setup_data(&columns);
    let runner = BenchmarkRunner;

    let mut results = Vec::new();

    for c in columns {
        let array = dataset.data.get(c).unwrap();
        println!("{c} average length: {}", array.avg_str_length);

        println!("Running all benchmarks for {c}");
        let benchmark_results = runner.run_all_benchmarks(&array.data);

        // Print results
        for result in &benchmark_results {
            println!(
                "{} {}: {}",
                c, result.encode_result.workload, result.encode_result
            );
            println!(
                "{} {} find needle: {}",
                c, result.find_needle_result.workload, result.find_needle_result
            );
        }

        // Get FSST view memory usage for detailed reporting
        let (compressor, _) = LiquidByteViewArray::train_from_string_view(&array.data[0]);
        let mut total_detailed_memory_usage = ByteViewArrayMemoryUsage {
            dictionary_views: 0,
            offsets: 0,
            nulls: 0,
            fsst_buffer: 0,
            shared_prefix: 0,
            struct_size: 0,
        };

        for a in &array.data {
            let array = LiquidByteViewArray::from_string_view_array(a, compressor.clone());
            total_detailed_memory_usage += array.get_detailed_memory_usage();
        }

        results.push(BenchmarkResult {
            column_name: c.to_string(),
            avg_string_length: array.avg_str_length,
            benchmark_results,
            fsst_view_memory_usage: total_detailed_memory_usage.into(),
        });
    }

    let complete_results = CompleteResults {
        benchmark_name: "FSST View Study".to_string(),
        timestamp: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs()
            .to_string(),
        columns: results,
    };

    let json_output = serde_json::to_string_pretty(&complete_results).unwrap();
    let mut file = File::create("../../target/benchmark_results.json").unwrap();
    file.write_all(json_output.as_bytes()).unwrap();

    println!("Benchmark results written to target/benchmark_results.json");
}
