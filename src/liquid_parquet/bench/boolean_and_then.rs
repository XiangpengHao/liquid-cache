use arrow::{array::BooleanBufferBuilder, buffer::BooleanBuffer};
use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use liquid_cache_parquet::boolean_buffer_and_then;

use rand::Rng;

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

fn benchmark_boolean_and_then(c: &mut Criterion) {
    // Three selectivity levels: low (10%), medium (50%), high (90%)
    let selectivities = [0.1, 0.5, 0.9];

    for left_selectivity in selectivities {
        for right_selectivity in selectivities {
            let group_name = format!(
                "boolean_and_then_left_{:.0}%_right_{:.0}%",
                left_selectivity * 100.0,
                right_selectivity * 100.0
            );

            let mut group = c.benchmark_group(&group_name);

            // Set throughput based on the buffer size in bytes
            // Each boolean buffer uses approximately size/8 bytes
            group.throughput(Throughput::Bytes((BUFFER_SIZE / 8) as u64));

            group.bench_function("size_16384", |b| {
                // Pre-generate test data
                let left = generate_boolean_buffer(BUFFER_SIZE, left_selectivity);
                let count_set_bits = left.count_set_bits();
                let right = generate_right_boolean_buffer(count_set_bits, right_selectivity);

                b.iter(|| std::hint::black_box(boolean_buffer_and_then(&left, &right)))
            });

            group.finish();
        }
    }
}

criterion_group!(benches, benchmark_boolean_and_then);
criterion_main!(benches);
