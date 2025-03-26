use criterion::{black_box, criterion_group, criterion_main, Criterion};
use datafusion::arrow::{array::PrimitiveArray, buffer::ScalarBuffer, datatypes::{Float32Type, Float64Type}};
use liquid_parquet::liquid_array::{LiquidArray, LiquidFloatArray};
use rand::Rng;

fn float32_liquid_from_arrow(size: usize) {
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

fn float32_liquid_to_arrow(liquid_array: &LiquidFloatArray<Float32Type>) {
    black_box((|| {
        let _x = liquid_array.to_arrow_array();
    })())
}

fn float64_liquid_from_arrow(size: usize) {
    let mut rng = rand::rng();
    let mut array: Vec<f64> = vec![];
    for _ in 0..size {
        array.push(rng.random_range(-1.3e3..1.3e3));
    }
    let arrow_array = PrimitiveArray::new(
        ScalarBuffer::from(array), None
    );
    black_box((|| {
        let _x = LiquidFloatArray::<Float64Type>::from_arrow_array(arrow_array);
    })())
}

fn float64_liquid_to_arrow(liquid_array: &LiquidFloatArray<Float64Type>) {
    black_box((|| {
        let _x = liquid_array.to_arrow_array();
    })())
}


fn criterion_benchmark(c: &mut Criterion) {    
    // Encoding benchmarks
    c.bench_function("float32_liquid_from_arrow 2048", |b| b.iter(|| float32_liquid_from_arrow(2048)));
    c.bench_function("float32_liquid_from_arrow 4096", |b| b.iter(|| float32_liquid_from_arrow(4096)));
    c.bench_function("float32_liquid_from_arrow 8192", |b| b.iter(|| float32_liquid_from_arrow(8192)));

    c.bench_function("float64_liquid_from_arrow 2048", |b| b.iter(|| float64_liquid_from_arrow(2048)));
    c.bench_function("float64_liquid_from_arrow 4096", |b| b.iter(|| float64_liquid_from_arrow(4096)));
    c.bench_function("float64_liquid_from_arrow 8192", |b| b.iter(|| float64_liquid_from_arrow(8192)));

    //Decoding benchmarks
    for size in [2048, 4096, 8192] {
        let mut rng = rand::rng();
        let mut array: Vec<f32> = vec![];
        for _ in 0..size {
            array.push(rng.random_range(-1.3e3..1.3e3));
        }
        let arrow_array = PrimitiveArray::<Float32Type>::new(
            ScalarBuffer::from(array), None
        );
        let liquid_array = LiquidFloatArray::<Float32Type>::from_arrow_array(arrow_array);
        c.bench_function(&format!("float32_liquid_to_arrow {}", size), |b| b.iter(|| float32_liquid_to_arrow(&liquid_array)));
    }
    
    for size in [2048, 4096, 8192] {
        let mut rng = rand::rng();
        let mut array: Vec<f64> = vec![];
        for _ in 0..size {
            array.push(rng.random_range(-1.3e3..1.3e3));
        }
        let arrow_array = PrimitiveArray::<Float64Type>::new(
            ScalarBuffer::from(array), None
        );
        let liquid_array = LiquidFloatArray::<Float64Type>::from_arrow_array(arrow_array);
        c.bench_function(&format!("float64_liquid_to_arrow {}", size), |b| b.iter(|| float64_liquid_to_arrow(&liquid_array)));
    }
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);