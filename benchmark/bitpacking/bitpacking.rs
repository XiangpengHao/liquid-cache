use criterion::*;

use std::num::NonZero;

use arrow::array::PrimitiveArray;
use liquid_cache_parquet::liquid_array::raw::BitPackedArray;
use rand::Rng;

const MAX_BIT_WIDTH: u8 = 32;
const MAX_ARRAY_MULTIPLIER: usize = 20;
const BASE_ARRAY_SIZE: usize = 8192;

// Function to create a random vector of u32 values with a given size and bit width
fn create_random_vec(array_size: usize, bit_width: u8) -> Vec<u32> {
    let max_value = (1u32 << bit_width) - 1;
    let mut rng = rand::rng();
    let values: Vec<u32> = (0..array_size)
        .map(|_| rng.random_range(0..=max_value))
        .collect();
    values
}

// Benchmark function to measure the performance of from_primitive
fn from_primitive_benchmark(c: &mut Criterion) {
    use arrow::datatypes::UInt32Type;

    // `bit_widths` represents the range of bit widths to test (1 through MAX_BIT_WIDTH).
    // Each bit width determines the maximum value that can be represented in the random vector.
    // For example, a bit width of 8 allows values in the range [0, 255].
    let bit_widths: Vec<u8> = (1..=MAX_BIT_WIDTH).collect();
    for bit_width in bit_widths {
        // `array_sizes` represents the range of array sizes to test.
        // Each size is a multiple of BASE_ARRAY_SIZE (e.g., 8192, 16384, etc.).
        let array_sizes: Vec<usize> = (1..=MAX_ARRAY_MULTIPLIER)
            .map(|i| BASE_ARRAY_SIZE * i)
            .collect();
        for array_size in array_sizes {
            let values: Vec<u32> = create_random_vec(array_size, bit_width);

            // Convert the random vector into a PrimitiveArray
            let array = PrimitiveArray::<UInt32Type>::from(values);
            let bitwidth = NonZero::new(bit_width).unwrap();

            // Benchmark from_primitive() - the conversion from PrimitiveArray to BitPackedArray
            c.bench_function(
                &format!(
                    "from_primitive(bitwidth={}, array size: {})",
                    bit_width, array_size
                ),
                |b| {
                    b.iter(|| {
                        criterion::black_box(BitPackedArray::from_primitive(
                            array.clone(),
                            bitwidth,
                        ))
                    })
                },
            );
        }
    }
}

// Benchmark function to measure the performance of to_primitive
fn to_primitive_benchmark(c: &mut Criterion) {
    use arrow::datatypes::UInt32Type;

    let bit_widths: Vec<u8> = (1..=MAX_BIT_WIDTH).collect();
    for bit_width in bit_widths {
        let array_sizes: Vec<usize> = (1..=MAX_ARRAY_MULTIPLIER)
            .map(|i| BASE_ARRAY_SIZE * i)
            .collect();
        for array_size in array_sizes {
            let values: Vec<u32> = create_random_vec(array_size, bit_width);

            // Convert the random vector into a PrimitiveArray
            let array = PrimitiveArray::<UInt32Type>::from(values);
            let bitwidth = NonZero::new(bit_width).unwrap();
            let bit_packed = BitPackedArray::from_primitive(array, bitwidth);

            // Benchmark to_primitive() - the conversion from a BitPackedArray to PrimitiveArray
            c.bench_function(
                &format!(
                    "to_primitive(bitwidth={}, array size: {})",
                    bit_width, array_size
                ),
                |b| b.iter(|| criterion::black_box(bit_packed.to_primitive())),
            );
        }
    }
}

criterion_group!(benches, from_primitive_benchmark, to_primitive_benchmark);

// Entry point for Criterion benchmarking
criterion_main!(benches);
