use divan::Bencher;
use std::sync::Arc;

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

#[divan::bench(args = CHUNK_SIZE)]
fn compressor_benchmark(bencher: Bencher, chunk_size: usize) {
    let string_arrays = create_string_arrays_from_file();
    let (_, string_array) = string_arrays
        .into_iter()
        .find(|(s, _)| *s == chunk_size)
        .unwrap();
    let total_size = chunk_size * string_array.len();

    bencher
        .with_inputs(|| string_array.clone())
        .input_counter(move |_| divan::counter::BytesCount::new(total_size))
        .bench_values(|string_array| {
            let input = string_array.iter().flat_map(|s| s.map(|a| a.as_bytes()));
            FsstArray::train_compressor(input)
        });
}

#[divan::bench(args = CHUNK_SIZE)]
fn from_byte_array_with_compressor_benchmark(bencher: Bencher, chunk_size: usize) {
    let string_arrays = create_string_arrays_from_file();
    let (_, string_array) = string_arrays
        .into_iter()
        .find(|(s, _)| *s == chunk_size)
        .unwrap();
    let compressor =
        FsstArray::train_compressor(string_array.iter().flat_map(|s| s.map(|s| s.as_bytes())));
    let uncompressed_size = chunk_size * string_array.len();

    bencher
        .with_inputs(|| (string_array.clone(), Arc::new(compressor.clone())))
        .input_counter(move |_| divan::counter::BytesCount::new(uncompressed_size))
        .bench_values(|(string_array, compressor)| {
            FsstArray::from_byte_array_with_compressor(&string_array, compressor)
        });
}

#[divan::bench(args = CHUNK_SIZE)]
fn to_arrow_byte_array_benchmark(bencher: Bencher, chunk_size: usize) {
    let string_arrays = create_string_arrays_from_file();
    let (_, string_array) = string_arrays
        .into_iter()
        .find(|(s, _)| *s == chunk_size)
        .unwrap();
    let compressor =
        FsstArray::train_compressor(string_array.iter().flat_map(|s| s.map(|s| s.as_bytes())));
    let fsst_values =
        FsstArray::from_byte_array_with_compressor(&string_array, Arc::new(compressor));
    let total_size = chunk_size * string_array.len();

    bencher
        .with_inputs(|| fsst_values.clone())
        .input_counter(move |_| divan::counter::BytesCount::new(total_size))
        .bench_values(|fsst_values| fsst_values.to_arrow_byte_array::<Utf8Type>());
}

fn main() {
    divan::main();
}
