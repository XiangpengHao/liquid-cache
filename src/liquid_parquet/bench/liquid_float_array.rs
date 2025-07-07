use datafusion::arrow::{
    array::PrimitiveArray,
    buffer::ScalarBuffer,
    datatypes::{Float32Type, Float64Type},
};
use divan::Bencher;
use liquid_cache_parquet::liquid_array::{LiquidArray, LiquidFloatArray};
use rand::Rng;

const BENCH_SIZES: [usize; 3] = [8192, 16384, 24576];

#[divan::bench(consts = BENCH_SIZES)]
fn float32_liquid_encode<const SIZE: usize>(bencher: Bencher) {
    bencher
        .with_inputs(|| {
            let mut rng = rand::rng();
            let mut array: Vec<f32> = vec![];
            for _ in 0..SIZE {
                array.push(rng.random_range(-1.3e3..1.3e3));
            }
            PrimitiveArray::new(ScalarBuffer::from(array), None)
        })
        .input_counter(|_| divan::counter::BytesCount::new(SIZE * std::mem::size_of::<f32>()))
        .bench_values(LiquidFloatArray::<Float32Type>::from_arrow_array);
}

#[divan::bench(consts = BENCH_SIZES)]
fn float64_liquid_encode<const SIZE: usize>(bencher: Bencher) {
    bencher
        .with_inputs(|| {
            let mut rng = rand::rng();
            let mut array: Vec<f64> = vec![];
            for _ in 0..SIZE {
                array.push(rng.random_range(-1.3e3..1.3e3));
            }
            PrimitiveArray::new(ScalarBuffer::from(array), None)
        })
        .input_counter(|_| divan::counter::BytesCount::new(SIZE * std::mem::size_of::<f64>()))
        .bench_values(LiquidFloatArray::<Float64Type>::from_arrow_array);
}

#[divan::bench(consts = BENCH_SIZES)]
fn float32_liquid_decode<const SIZE: usize>(bencher: Bencher) {
    bencher
        .with_inputs(|| {
            let mut rng = rand::rng();
            let mut array: Vec<f32> = vec![];
            for _ in 0..SIZE {
                array.push(rng.random_range(-1.3e3..1.3e3));
            }
            let arrow_array = PrimitiveArray::<Float32Type>::new(ScalarBuffer::from(array), None);
            LiquidFloatArray::<Float32Type>::from_arrow_array(arrow_array)
        })
        .input_counter(|_| divan::counter::BytesCount::new(SIZE * std::mem::size_of::<f32>()))
        .bench_values(|liquid_array| liquid_array.to_arrow_array());
}

#[divan::bench(consts = BENCH_SIZES)]
fn float64_liquid_decode<const SIZE: usize>(bencher: Bencher) {
    bencher
        .with_inputs(|| {
            let mut rng = rand::rng();
            let mut array: Vec<f64> = vec![];
            for _ in 0..SIZE {
                array.push(rng.random_range(-1.3e3..1.3e3));
            }
            let arrow_array = PrimitiveArray::<Float64Type>::new(ScalarBuffer::from(array), None);
            LiquidFloatArray::<Float64Type>::from_arrow_array(arrow_array)
        })
        .input_counter(|_| divan::counter::BytesCount::new(SIZE * std::mem::size_of::<f64>()))
        .bench_values(|liquid_array| liquid_array.to_arrow_array());
}

fn main() {
    divan::main();
}
