use criterion::*;
use std::sync::Arc;
use std::time::Duration;

extern crate arrow;

use arrow::{
    array::{StringArray, StringBuilder},
    datatypes::Utf8Type,
};
use liquid_cache_parquet::liquid_array::raw::FsstArray;
use std::fs;

const FOLDER_PATH: &str = "./bench/training_texts/";

// Function to create a vector of StringArray from files in a folder
fn create_string_arrays_from_files(folder_path: &str) -> Vec<StringArray> {
    let mut string_arrays = Vec::new();

    // Read all entries in the specified folder
    let entries = fs::read_dir(folder_path).expect("Failed to read directory");

    // Collect and sort entries by file name
    let mut entries: Vec<_> = entries.map(|e| e.expect("Failed to read entry")).collect();
    entries.sort_by_key(|e| e.file_name());

    for entry in entries {
        let path = entry.path();

        // Skip non-file entries or hidden files (starting with '.')
        if !path.is_file() || path.file_name().unwrap().to_str().unwrap().starts_with('.') {
            continue;
        }

        let content = fs::read_to_string(&path).expect("Failed to read file");

        let mut builder = StringBuilder::new();

        builder.append_value(&content);

        string_arrays.push(builder.finish());
    }

    string_arrays
}

// Benchmark for training the FSST compressor
fn compressor_benchmark(c: &mut Criterion) {
    let string_arrays = create_string_arrays_from_files(FOLDER_PATH);

    let mut group = c.benchmark_group("fsst_array_benchmark");
    for i in 0..string_arrays.len() {
        // Set the measurement time for the benchmark
        group.measurement_time(Duration::new(10, 0));

        // Calculate the total byte size of the training text
        let string_byte_size: usize = string_arrays[i]
            .iter()
            .filter_map(|s| s) // Skip null values
            .map(|s| s.len()) // Get the byte length of each string
            .sum();

        // Set the throughput for the benchmark
        group.throughput(Throughput::Bytes(string_byte_size as u64));

        // Benchmark the FSST compressor training
        group.bench_function(
            format!("train_compressor - str_size: {}", string_byte_size),
            |b| {
                b.iter(|| {
                    let input = criterion::black_box(
                        string_arrays[i]
                            .iter()
                            .flat_map(|s| s.map(|a| a.as_bytes())),
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
    let string_arrays = create_string_arrays_from_files(FOLDER_PATH);

    let mut group = c.benchmark_group(format!("compressor_benchmark"));
    for i in 0..string_arrays.len() {
        // Train the FSST compressor
        let compressor = FsstArray::train_compressor(
            string_arrays[i]
                .iter()
                .flat_map(|s| s.map(|s| s.as_bytes())),
        );

        // Calculate the total byte size of the training text
        let string_byte_size: usize = string_arrays[i]
            .iter()
            .filter_map(|s| s) // Skip null values
            .map(|s| s.len()) // Get the byte length of each string
            .sum();

        // Set the throughput for the benchmark
        group.throughput(Throughput::Bytes(string_byte_size as u64));

        // Benchmark the creation of an FSST array from a byte array
        group.bench_function(
            format!(
                "from_byte_array_with_compressor - str_size: {}",
                string_byte_size
            ),
            |b| {
                b.iter(|| {
                    criterion::black_box(FsstArray::from_byte_array_with_compressor(
                        &string_arrays[i],
                        Arc::new(compressor.clone()),
                    ))
                });
            },
        );
    }
    group.finish();
}

// Benchmark for converting an FSST array to an Arrow byte array
fn to_arrow_byte_array_benchmark(c: &mut Criterion) {
    let string_arrays = create_string_arrays_from_files(FOLDER_PATH);

    let mut group = c.benchmark_group(format!("compressor_benchmark"));
    for i in 0..string_arrays.len() {
        // Train the FSST compressor
        let compressor = FsstArray::train_compressor(
            string_arrays[i]
                .iter()
                .flat_map(|s| s.map(|s| s.as_bytes())),
        );

        // Create an FSST array using the trained compressor
        let fsst_values =
            FsstArray::from_byte_array_with_compressor(&string_arrays[i], Arc::new(compressor));

        // Calculate the total byte size of the training text
        let string_byte_size: usize = string_arrays[i]
            .iter()
            .filter_map(|s| s) // Skip null values
            .map(|s| s.len()) // Get the byte length of each string
            .sum();

        // Set the throughput for the benchmark
        group.throughput(Throughput::Bytes(string_byte_size as u64));

        // Benchmark the conversion of FSST array to Arrow byte array
        group.bench_function(
            format!("to_arrow_byte_array - str_size: {}", string_byte_size),
            |b| {
                b.iter(|| criterion::black_box(fsst_values.to_arrow_byte_array::<Utf8Type>()));
            },
        );
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
