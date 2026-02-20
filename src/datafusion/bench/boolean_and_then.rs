use arrow::{array::BooleanBufferBuilder, buffer::BooleanBuffer};
use divan::Bencher;
use liquid_cache_datafusion::boolean_buffer_and_then;
use rand::RngExt as _;

const BUFFER_SIZE: usize = 8192 * 2; // 16384

/// Generate a BooleanBuffer with specified selectivity (percentage of true bits)
fn generate_boolean_buffer(size: usize, selectivity: f64) -> BooleanBuffer {
    let mut rng = rand::rng();
    let mut builder = BooleanBufferBuilder::new(size);

    for _ in 0..size {
        let should_set = rng.random_bool(selectivity);
        builder.append(should_set);
    }

    builder.finish()
}

/// Generate a right BooleanBuffer that has exactly `count_set_bits` bits
fn generate_right_boolean_buffer(count_set_bits: usize, selectivity: f64) -> BooleanBuffer {
    let mut rng = rand::rng();
    let mut builder = BooleanBufferBuilder::new(count_set_bits);

    for _ in 0..count_set_bits {
        let should_set = rng.random_bool(selectivity);
        builder.append(should_set);
    }

    builder.finish()
}

#[divan::bench(args = [(0.1, 0.1), (0.1, 0.5), (0.1, 0.9), (0.5, 0.1), (0.5, 0.5), (0.5, 0.9), (0.9, 0.1), (0.9, 0.5), (0.9, 0.9)])]
fn benchmark_boolean_and_then(bencher: Bencher, (left_selectivity, right_selectivity): (f64, f64)) {
    bencher
        .with_inputs(|| {
            let left = generate_boolean_buffer(BUFFER_SIZE, left_selectivity);
            let count_set_bits = left.count_set_bits();
            let right = generate_right_boolean_buffer(count_set_bits, right_selectivity);
            (left, right)
        })
        .input_counter(|_| divan::counter::BytesCount::new(BUFFER_SIZE / 8))
        .bench_values(|(left, right)| std::hint::black_box(boolean_buffer_and_then(&left, &right)));
}

fn main() {
    divan::main();
}
