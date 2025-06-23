use criterion::Throughput;
use criterion::*;

use std::num::NonZero;

use arrow::array::{BooleanArray, PrimitiveArray};
use liquid_cache_parquet::liquid_array::raw::BitPackedArray;
use liquid_cache_parquet::liquid_array::{LiquidArray, LiquidPrimitiveArray};
use rand::Rng;

const ARRAY_SIZES: [usize; 4] = [8192, 16384, 32768, 65536];
const BIT_WIDTHS: [u8; 6] = [1, 3, 7, 11, 19, 27];

fn create_random_vec(array_size: usize, bit_width: u8) -> Vec<u32> {
    let max_value = (1u32 << bit_width) - 1;
    let mut rng = rand::rng();
    let values: Vec<u32> = (0..array_size)
        .map(|_| rng.random_range(0..=max_value))
        .collect();
    values
}

fn create_selection_array(array_size: usize, selectivity: f64) -> BooleanArray {
    let mut rng = rand::rng();
    let values: Vec<bool> = (0..array_size)
        .map(|_| rng.random::<f64>() < selectivity)
        .collect();
    BooleanArray::from(values)
}

fn from_primitive_benchmark(c: &mut Criterion) {
    use arrow::datatypes::UInt32Type;

    for bit_width in BIT_WIDTHS {
        for array_size in ARRAY_SIZES {
            let values: Vec<u32> = create_random_vec(array_size, bit_width);

            let array = PrimitiveArray::<UInt32Type>::from(values);
            let bit_width = NonZero::new(bit_width).unwrap();

            let mut group = c.benchmark_group(format!("from_primitive_bw_{}", bit_width));
            group.throughput(Throughput::Bytes(
                (array_size * std::mem::size_of::<u32>()) as u64,
            ));
            group.bench_function(format!("size_{}", array_size), |b| {
                b.iter(|| {
                    std::hint::black_box(BitPackedArray::from_primitive(array.clone(), bit_width))
                })
            });
            group.finish();
        }
    }
}

fn to_primitive_benchmark(c: &mut Criterion) {
    use arrow::datatypes::UInt32Type;
    for bit_width in BIT_WIDTHS {
        for array_size in ARRAY_SIZES {
            let values: Vec<u32> = create_random_vec(array_size, bit_width);

            let array = PrimitiveArray::<UInt32Type>::from(values);
            let bit_width = NonZero::new(bit_width).unwrap();
            let bit_packed = BitPackedArray::from_primitive(array, bit_width);

            let mut group = c.benchmark_group(format!("to_primitive_bw_{}", bit_width));
            group.throughput(Throughput::Bytes(
                (array_size * std::mem::size_of::<u32>()) as u64,
            ));
            group.bench_function(format!("size_{}", array_size), |b| {
                b.iter(|| std::hint::black_box(bit_packed.to_primitive()))
            });
            group.finish();
        }
    }
}

fn filter_benchmark(c: &mut Criterion) {
    use arrow::datatypes::UInt32Type;

    let selectivities = vec![0.01, 0.1, 0.3, 0.7, 0.9];

    for selectivity in selectivities {
        for bit_width in BIT_WIDTHS {
            for array_size in ARRAY_SIZES {
                let values: Vec<u32> = create_random_vec(array_size, bit_width);

                let array = PrimitiveArray::<UInt32Type>::from(values);
                let liquid_array = LiquidPrimitiveArray::<UInt32Type>::from_arrow_array(array);

                let selection = create_selection_array(array_size, selectivity);

                let mut group = c.benchmark_group(format!(
                    "primitive_filter_sel_{}_bw_{}",
                    (selectivity * 100.0) as u32,
                    bit_width
                ));
                group.throughput(Throughput::Bytes(
                    (array_size * std::mem::size_of::<u32>()) as u64,
                ));
                group.bench_function(format!("size_{}", array_size), |b| {
                    b.iter(|| std::hint::black_box(liquid_array.filter(&selection)))
                });
                group.finish();
            }
        }
    }
}

criterion_group!(
    benches,
    from_primitive_benchmark,
    to_primitive_benchmark,
    filter_benchmark
);

// Entry point for Criterion benchmarking
criterion_main!(benches);
