use criterion::*;
use std::sync::Arc;
use std::time::Duration;

extern crate arrow;

use arrow::{
    array::{Array, StringArray, StringBuilder},
    datatypes::Utf8Type,
};
use liquid_cache_parquet::liquid_array::raw::FsstArray;
use std::fs;

const CHUNK_SIZE: [usize; 5] = [12, 32, 64, 128, 256];

fn create_string_arrays_from_file() -> Vec<(usize, StringArray)> {
    const TEST_FILE_PATH: &str = "../../README.md";
    const LICENSE_FILE_PATH: &str = "../../LICENSE";

    let readme = fs::read_to_string(TEST_FILE_PATH).expect("Failed to read file");
    let license = fs::read_to_string(LICENSE_FILE_PATH).expect("Failed to read file");
    let content = format!("{}\n\n{}", readme, license);

    let mut result = Vec::new();

    let chars: Vec<char> = content.chars().collect();

    for &chunk_size in &CHUNK_SIZE {
        let mut builder = StringBuilder::new();
        for chunk in chars.chunks(chunk_size) {
            let chunk_str: String = chunk.iter().collect();
            builder.append_value(chunk_str);
        }
        result.push((chunk_size, builder.finish()));
    }

    result
}

// Benchmark for training the FSST compressor
fn compressor_benchmark(c: &mut Criterion) {
    let string_arrays = create_string_arrays_from_file();

    let mut group = c.benchmark_group("fsst");
    for (chunk_size, string_array) in string_arrays {
        let total_size = chunk_size * string_array.len();
        // Set the measurement time for the benchmark
        group.measurement_time(Duration::new(10, 0));

        // Set the throughput for the benchmark
        group.throughput(Throughput::Bytes(total_size as u64));

        // Benchmark the FSST compressor training
        group.bench_function(
            format!("train_compressor - chunk_size: {}", chunk_size),
            |b| {
                b.iter(|| {
                    let input = criterion::black_box(
                        string_array.iter().flat_map(|s| s.map(|a| a.as_bytes())),
                    );
                    FsstArray::train_compressor(input)
                });
            },
        );
    }
    group.finish();
}

// Benchmark for creating an FSST array from a byte array using a pre-trained compressor
fn from_byte_array_with_compressor_benchmark(c: &mut Criterion) {
    let string_arrays = create_string_arrays_from_file();

    let mut group = c.benchmark_group(format!("fsst"));
    for (chunk_size, string_array) in string_arrays {
        // Train the FSST compressor
        let compressor =
            FsstArray::train_compressor(string_array.iter().flat_map(|s| s.map(|s| s.as_bytes())));

        let compressed =
            FsstArray::from_byte_array_with_compressor(&string_array, Arc::new(compressor.clone()));
        let compressed_size = compressed.get_array_memory_size();
        let uncompressed_size = chunk_size * string_array.len();
        println!(
            "compressed_size: {}, uncompressed_size: {}, compression_ratio: {}",
            compressed_size,
            uncompressed_size,
            compressed_size as f64 / uncompressed_size as f64
        );

        // Set the throughput for the benchmark
        group.throughput(Throughput::Bytes(uncompressed_size as u64));

        // Benchmark the creation of an FSST array from a byte array
        group.bench_function(format!("compress - chunk_size: {}", chunk_size), |b| {
            b.iter(|| {
                criterion::black_box(FsstArray::from_byte_array_with_compressor(
                    &string_array,
                    Arc::new(compressor.clone()),
                ))
            });
        });
    }
    group.finish();
}

// Benchmark for converting an FSST array to an Arrow byte array
fn to_arrow_byte_array_benchmark(c: &mut Criterion) {
    let string_arrays = create_string_arrays_from_file();

    let mut group = c.benchmark_group(format!("fsst"));
    for (chunk_size, string_array) in string_arrays {
        // Train the FSST compressor
        let compressor =
            FsstArray::train_compressor(string_array.iter().flat_map(|s| s.map(|s| s.as_bytes())));

        // Create an FSST array using the trained compressor
        let fsst_values =
            FsstArray::from_byte_array_with_compressor(&string_array, Arc::new(compressor));

        let total_size = chunk_size * string_array.len();

        // Set the throughput for the benchmark
        group.throughput(Throughput::Bytes(total_size as u64));

        // Benchmark the conversion of FSST array to Arrow byte array
        group.bench_function(format!("decompress - chunk_size: {}", chunk_size), |b| {
            b.iter(|| criterion::black_box(fsst_values.to_arrow_byte_array::<Utf8Type>()));
        });
    }
    group.finish();
}

// Define the benchmark group
criterion_group!(
    benches,
    compressor_benchmark,
    from_byte_array_with_compressor_benchmark,
    to_arrow_byte_array_benchmark
);

// Entry point for Criterion benchmarking
criterion_main!(benches);
