use std::collections::HashMap;
use std::fmt::Display;
use std::fs::File;
use std::io::{Cursor, Write};
use std::sync::Arc;
use std::time::Instant;

use arrow::array::{Array, AsArray, RecordBatch, StringViewArray};
use arrow::compute::cast;
use arrow::datatypes::{DataType, Float64Type};
use arrow_schema::{Field, Schema};
use datafusion::prelude::*;
use liquid_cache_parquet::liquid_array::{
    ByteViewArrayMemoryUsage, LiquidArray, LiquidByteArray, LiquidByteViewArray,
};
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
        .map(|c| format!("\"{}\"", c))
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
        .map(|c| format!("AVG(LENGTH(\"{}\")) AS \"{}\"", c, c))
        .collect::<Vec<_>>()
        .join(", ");
    let avg_length_batches = ctx
        .sql(&format!("SELECT {} FROM \"hits\"", avg_length))
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

impl Display for EncodeResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "total_size: {} bytes, encode: {} s, decode: {} s",
            self.total_size, self.encode_time_sec, self.decode_time_sec
        )
    }
}

fn roundtrip_fsst_view(array: &Vec<StringViewArray>) -> (EncodeResult, ByteViewArrayMemoryUsage) {
    let (compressor, _) = LiquidByteViewArray::train_from_string_view(&array[0]);
    let mut total_size = 0;
    let mut encode_time: f64 = 0.0;
    let mut decode_time: f64 = 0.0;
    let mut total_detailed_memory_usage = ByteViewArrayMemoryUsage {
        dictionary_views: 0,
        offsets: 0,
        nulls: 0,
        fsst_buffer: 0,
        shared_prefix: 0,
        struct_size: 0,
    };

    for a in array {
        let start = Instant::now();
        let array = LiquidByteViewArray::from_string_view_array(a, compressor.clone());
        encode_time += start.elapsed().as_secs_f64();

        total_size += array.get_array_memory_size();
        total_detailed_memory_usage += array.get_detailed_memory_usage();

        let start = Instant::now();
        let _array = array.to_arrow_array();
        decode_time += start.elapsed().as_secs_f64();
    }

    let encode_result = EncodeResult {
        total_size,
        encode_time_sec: encode_time,
        decode_time_sec: decode_time,
        workload: "FSSTView".to_string(),
    };

    (encode_result, total_detailed_memory_usage)
}

fn roundtrip_byte_array(array: &Vec<StringViewArray>) -> EncodeResult {
    let (compressor, _) = LiquidByteArray::train_from_string_view(&array[0]);
    let mut total_size = 0;
    let mut encode_time: f64 = 0.0;
    let mut decode_time: f64 = 0.0;

    for a in array {
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
        workload: "ByteArray".to_string(),
    }
}

fn roundtrip_string_array(array: &Vec<StringViewArray>) -> EncodeResult {
    let mut encode_time: f64 = 0.0;
    let mut decode_time: f64 = 0.0;
    let mut total_size = 0;

    for a in array {
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
        workload: "StringArray".to_string(),
    }
}

fn roundtrip_string_array_lz4(array: &Vec<StringViewArray>) -> EncodeResult {
    let schema = Arc::new(Schema::new(vec![Field::new("a", DataType::Utf8, false)]));
    let compression = arrow::ipc::CompressionType::LZ4_FRAME;
    let options = arrow::ipc::writer::IpcWriteOptions::default()
        .try_with_compression(Some(compression))
        .unwrap();

    let mut encode_time: f64 = 0.0;
    let mut decode_time: f64 = 0.0;
    let mut total_size = 0;

    for a in array {
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

        let start = Instant::now();
        let mut file = Cursor::new(file);
        let mut reader = arrow::ipc::reader::FileReader::try_new(&mut file, None).unwrap();
        let batch = reader.next().unwrap().unwrap();
        let v = batch.column(0).as_string::<i32>().clone();
        decode_time += start.elapsed().as_secs_f64();

        total_size += v.get_array_memory_size();
    }

    EncodeResult {
        total_size,
        encode_time_sec: encode_time,
        decode_time_sec: decode_time,
        workload: "StringArrayLZ4".to_string(),
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
    results: Vec<EncodeResult>,
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

    let mut results = Vec::new();

    for c in columns {
        let array = dataset.data.get(c).unwrap();
        println!("{} average length: {}", c, array.avg_str_length);
        let (fsst_view_result, memory_usage) = roundtrip_fsst_view(&array.data);
        println!("{} to/from FSST View: {}", c, fsst_view_result);
        let byte_array_result = roundtrip_byte_array(&array.data);
        println!("{} to/from byte array: {}", c, byte_array_result);
        let string_array_result = roundtrip_string_array(&array.data);
        println!("{} to/from string array: {}", c, string_array_result);
        let string_array_lz4_result = roundtrip_string_array_lz4(&array.data);
        println!(
            "{} to/from string array lz4: {}",
            c, string_array_lz4_result
        );

        results.push(BenchmarkResult {
            column_name: c.to_string(),
            avg_string_length: array.avg_str_length,
            results: vec![
                fsst_view_result,
                byte_array_result,
                string_array_result,
                string_array_lz4_result,
            ],
            fsst_view_memory_usage: memory_usage.into(),
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
    let mut file = File::create("target/benchmark_results.json").unwrap();
    file.write_all(json_output.as_bytes()).unwrap();

    println!("Benchmark results written to target/benchmark_results.json");
}
