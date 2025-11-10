use divan::Bencher;
use std::sync::Arc;

extern crate arrow;

use arrow::array::{Array, StringArray, StringBuilder};
use datafusion::common::ScalarValue;
use liquid_cache_storage::liquid_array::byte_view_array::MemoryBuffer;
use liquid_cache_storage::liquid_array::{LiquidArray, LiquidByteArray, LiquidByteViewArray};
use std::fs;

const CHUNK_SIZE: [usize; 5] = [12, 32, 64, 128, 256];

fn create_string_arrays_from_file() -> Vec<(usize, StringArray)> {
    const TEST_FILE_PATH: &str = "../../README.md";
    const LICENSE_FILE_PATH: &str = "../../LICENSE";

    let readme = fs::read_to_string(TEST_FILE_PATH).expect("Failed to read file");
    let license = fs::read_to_string(LICENSE_FILE_PATH).expect("Failed to read file");
    let content = format!("{readme}\n\n{license}");

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
fn byte_array_eq_operation(bencher: Bencher, chunk_size: usize) {
    let string_arrays = create_string_arrays_from_file();
    let (_, string_array) = string_arrays
        .into_iter()
        .find(|(s, _)| *s == chunk_size)
        .unwrap();

    let compressor = LiquidByteArray::train_compressor(string_array.iter());
    let liquid_array = LiquidByteArray::from_string_array(&string_array, compressor);

    // Use the first string as the needle for comparison
    let needle = string_array.value(0);

    bencher
        .with_inputs(|| (liquid_array.clone(), needle))
        .bench_values(|(arr, needle)| arr.compare_equals(needle));
}

#[divan::bench(args = CHUNK_SIZE)]
fn byte_view_array_eq_operation(bencher: Bencher, chunk_size: usize) {
    let string_arrays = create_string_arrays_from_file();
    let (_, string_array) = string_arrays
        .into_iter()
        .find(|(s, _)| *s == chunk_size)
        .unwrap();

    let compressor = LiquidByteViewArray::<MemoryBuffer>::train_compressor(string_array.iter());
    let liquid_array =
        LiquidByteViewArray::<MemoryBuffer>::from_string_array(&string_array, compressor);

    // Use the first string as the needle for comparison
    let needle = string_array.value(0).as_bytes();

    bencher
        .with_inputs(|| (liquid_array.clone(), needle))
        .bench_values(|(arr, needle)| {
            arr.compare_with(needle, &datafusion::logical_expr::Operator::Eq)
        });
}

#[divan::bench(args = CHUNK_SIZE)]
fn byte_array_gt_operation(bencher: Bencher, chunk_size: usize) {
    let string_arrays = create_string_arrays_from_file();
    let (_, string_array) = string_arrays
        .into_iter()
        .find(|(s, _)| *s == chunk_size)
        .unwrap();

    let compressor = LiquidByteArray::train_compressor(string_array.iter());
    let liquid_array = LiquidByteArray::from_string_array(&string_array, compressor);

    // Use a middle string as the needle for gt comparison
    let needle_idx = string_array.len() / 2;
    let needle = string_array.value(needle_idx);

    bencher
        .with_inputs(|| (liquid_array.clone(), needle))
        .bench_values(|(arr, needle)| {
            // For byte_array, we need to convert to dict_arrow and use arrow compute
            let dict = arr.to_dict_arrow();
            let needle_scalar = ScalarValue::Utf8(Some(needle.to_string()));
            let lhs = datafusion::logical_expr::ColumnarValue::Array(Arc::new(dict));
            let rhs = datafusion::logical_expr::ColumnarValue::Scalar(needle_scalar);
            datafusion::physical_expr_common::datum::apply_cmp(
                datafusion::logical_expr::Operator::Gt,
                &lhs,
                &rhs,
            )
        });
}

#[divan::bench(args = CHUNK_SIZE)]
fn byte_view_array_gt_operation(bencher: Bencher, chunk_size: usize) {
    let string_arrays = create_string_arrays_from_file();
    let (_, string_array) = string_arrays
        .into_iter()
        .find(|(s, _)| *s == chunk_size)
        .unwrap();

    let compressor = LiquidByteViewArray::<MemoryBuffer>::train_compressor(string_array.iter());
    let liquid_array =
        LiquidByteViewArray::<MemoryBuffer>::from_string_array(&string_array, compressor);

    // Use a middle string as the needle for gt comparison
    let needle_idx = string_array.len() / 2;
    let needle = string_array.value(needle_idx).as_bytes();

    bencher
        .with_inputs(|| (liquid_array.clone(), needle))
        .bench_values(|(arr, needle)| {
            arr.compare_with(needle, &datafusion::logical_expr::Operator::Gt)
        });
}

#[divan::bench(args = CHUNK_SIZE)]
fn byte_array_to_arrow_conversion(bencher: Bencher, chunk_size: usize) {
    let string_arrays = create_string_arrays_from_file();
    let (_, string_array) = string_arrays
        .into_iter()
        .find(|(s, _)| *s == chunk_size)
        .unwrap();

    let compressor = LiquidByteArray::train_compressor(string_array.iter());
    let liquid_array = LiquidByteArray::from_string_array(&string_array, compressor);

    bencher
        .with_inputs(|| liquid_array.clone())
        .bench_values(|arr| arr.to_arrow_array());
}

#[divan::bench(args = CHUNK_SIZE)]
fn byte_view_array_to_arrow_conversion(bencher: Bencher, chunk_size: usize) {
    let string_arrays = create_string_arrays_from_file();
    let (_, string_array) = string_arrays
        .into_iter()
        .find(|(s, _)| *s == chunk_size)
        .unwrap();

    let compressor = LiquidByteViewArray::<MemoryBuffer>::train_compressor(string_array.iter());
    let liquid_array =
        LiquidByteViewArray::<MemoryBuffer>::from_string_array(&string_array, compressor);

    bencher
        .with_inputs(|| liquid_array.clone())
        .bench_values(|arr| arr.to_arrow_array());
}

#[divan::bench(args = CHUNK_SIZE)]
fn array_size(_bencher: Bencher, chunk_size: usize) {
    let string_arrays = create_string_arrays_from_file();
    let (_, string_array) = string_arrays
        .into_iter()
        .find(|(s, _)| *s == chunk_size)
        .unwrap();

    let byte_view_compressor =
        LiquidByteViewArray::<MemoryBuffer>::train_compressor(string_array.iter());
    let byte_view_array =
        LiquidByteViewArray::<MemoryBuffer>::from_string_array(&string_array, byte_view_compressor);

    let byte_array_compressor = LiquidByteArray::train_compressor(string_array.iter());
    let byte_array_array = LiquidByteArray::from_string_array(&string_array, byte_array_compressor);

    println!("\narrow size: {}", string_array.get_array_memory_size());
    println!(
        "byte_view_array size: {}",
        byte_view_array.get_array_memory_size()
    );
    println!(
        "byte_array_array size: {}",
        byte_array_array.get_array_memory_size()
    );
}

fn main() {
    divan::main();
}
