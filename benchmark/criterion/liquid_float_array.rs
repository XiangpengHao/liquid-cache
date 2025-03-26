use criterion::{black_box, criterion_group, criterion_main, Criterion};
use datafusion::arrow::{array::PrimitiveArray, buffer::ScalarBuffer, datatypes::{Float32Type, Float64Type}};
use liquid_parquet::liquid_array::LiquidFloatArray;
use rand::Rng;

fn float32_liquid() {
    let size: usize = 1 << 15;
    let mut rng = rand::rng();
    let mut array: Vec<f32> = vec![];
    for _ in 0..size {
        array.push(rng.random_range(-1.3e3..1.3e3));
    }
    let arrow_array = PrimitiveArray::new(
        ScalarBuffer::from(array), None
    );
    black_box((|| {
        let _x = LiquidFloatArray::<Float32Type>::from_arrow_array(arrow_array);
    })())
}

fn float64_liquid() {
    let size: usize = 1 << 15;
    let mut rng = rand::rng();
    let mut array: Vec<f64> = vec![];
    for _ in 0..size {
        array.push(rng.random_range(-1.3e5..1.3e5));
    }
    let arrow_array = PrimitiveArray::new(
        ScalarBuffer::from(array), None
    );
    black_box((|| {
        let _x = LiquidFloatArray::<Float64Type>::from_arrow_array(arrow_array);
    })())
}


fn criterion_benchmark(c: &mut Criterion) {
    c.bench_function("float32_liquid", |b| b.iter(|| float32_liquid()));
    c.bench_function("float64_liquid", |b| b.iter(|| float64_liquid()));
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);