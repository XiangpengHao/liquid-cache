use arrow::buffer::BooleanBuffer;
use divan::Bencher;

use std::num::NonZero;

use arrow::array::PrimitiveArray;
use liquid_cache_storage::liquid_array::raw::BitPackedArray;
use liquid_cache_storage::liquid_array::{LiquidArray, LiquidPrimitiveArray};
use rand::RngExt as _;

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

fn create_selection_array(array_size: usize, selectivity: f64) -> BooleanBuffer {
    let mut rng = rand::rng();
    let values: Vec<bool> = (0..array_size)
        .map(|_| rng.random::<f64>() < selectivity)
        .collect();
    BooleanBuffer::from(values)
}

#[divan::bench(args = BIT_WIDTHS, consts = ARRAY_SIZES)]
fn from_primitive_benchmark<const SIZE: usize>(bencher: Bencher, bit_width: u8) {
    use arrow::datatypes::UInt32Type;

    let values: Vec<u32> = create_random_vec(SIZE, bit_width);
    let array = PrimitiveArray::<UInt32Type>::from(values);
    let bit_width = NonZero::new(bit_width).unwrap();

    bencher
        .with_inputs(|| array.clone())
        .input_counter(|_| divan::counter::BytesCount::new(SIZE * std::mem::size_of::<u32>()))
        .bench_values(|array| {
            std::hint::black_box(BitPackedArray::from_primitive(array, bit_width))
        });
}

#[divan::bench(args = BIT_WIDTHS, consts = ARRAY_SIZES)]
fn to_primitive_benchmark<const SIZE: usize>(bencher: Bencher, bit_width: u8) {
    use arrow::datatypes::UInt32Type;

    let values: Vec<u32> = create_random_vec(SIZE, bit_width);
    let array = PrimitiveArray::<UInt32Type>::from(values);
    let bit_width = NonZero::new(bit_width).unwrap();
    let bit_packed = BitPackedArray::from_primitive(array, bit_width);

    bencher
        .with_inputs(|| bit_packed.clone())
        .input_counter(|_| divan::counter::BytesCount::new(SIZE * std::mem::size_of::<u32>()))
        .bench_values(|bit_packed| std::hint::black_box(bit_packed.to_primitive()));
}

#[divan::bench(args = [(0.01, 1), (0.01, 3), (0.01, 7), (0.01, 11), (0.01, 19), (0.01, 27), (0.1, 1), (0.1, 3), (0.1, 7), (0.1, 11), (0.1, 19), (0.1, 27), (0.3, 1), (0.3, 3), (0.3, 7), (0.3, 11), (0.3, 19), (0.3, 27), (0.7, 1), (0.7, 3), (0.7, 7), (0.7, 11), (0.7, 19), (0.7, 27), (0.9, 1), (0.9, 3), (0.9, 7), (0.9, 11), (0.9, 19), (0.9, 27)], consts = ARRAY_SIZES)]
fn filter_benchmark<const SIZE: usize>(bencher: Bencher, (selectivity, bit_width): (f64, u8)) {
    use arrow::datatypes::UInt32Type;

    let values: Vec<u32> = create_random_vec(SIZE, bit_width);
    let array = PrimitiveArray::<UInt32Type>::from(values);
    let liquid_array = LiquidPrimitiveArray::<UInt32Type>::from_arrow_array(array);
    let selection = create_selection_array(SIZE, selectivity);

    bencher
        .with_inputs(|| (&liquid_array, &selection))
        .input_counter(|_| divan::counter::BytesCount::new(SIZE * std::mem::size_of::<u32>()))
        .bench_values(|(liquid_array, selection)| {
            std::hint::black_box(liquid_array.filter(selection))
        });
}

fn main() {
    divan::main();
}
