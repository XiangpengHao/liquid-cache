use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use datafusion::arrow::{
    array::PrimitiveArray,
    buffer::ScalarBuffer,
    datatypes::{Float32Type, Float64Type},
};
use liquid_cache_parquet::liquid_array::{LiquidArray, LiquidFloatArray};
use rand::Rng;

fn criterion_benchmark(c: &mut Criterion) {
    // Encoding benchmarks for float32
    let bench_sizes = [8192, 16384, 24576];
    for size in bench_sizes {
        let mut group = c.benchmark_group(format!("float32_liquid_encode"));
        group.throughput(Throughput::Bytes(
            (size * std::mem::size_of::<f32>()) as u64,
        ));
        group.bench_function(format!("size_{}", size), |b| {
            let mut rng = rand::rng();
            let mut array: Vec<f32> = vec![];
            for _ in 0..size {
                array.push(rng.random_range(-1.3e3..1.3e3));
            }
            let arrow_array = PrimitiveArray::new(ScalarBuffer::from(array), None);
            b.iter(|| {
                let _x = LiquidFloatArray::<Float32Type>::from_arrow_array(arrow_array.clone());
            })
        });
        group.finish();
    }

    for size in bench_sizes {
        let mut group = c.benchmark_group(format!("float64_liquid_encode"));
        group.throughput(Throughput::Bytes(
            (size * std::mem::size_of::<f64>()) as u64,
        ));
        group.bench_function(format!("size_{}", size), |b| {
            let mut rng = rand::rng();
            let mut array: Vec<f64> = vec![];
            for _ in 0..size {
                array.push(rng.random_range(-1.3e3..1.3e3));
            }
            let arrow_array = PrimitiveArray::new(ScalarBuffer::from(array), None);
            b.iter(|| {
                let _x = LiquidFloatArray::<Float64Type>::from_arrow_array(arrow_array.clone());
            })
        });
        group.finish();
    }

    // Decoding benchmarks for float32
    for size in bench_sizes {
        let mut rng = rand::rng();
        let mut array: Vec<f32> = vec![];
        for _ in 0..size {
            array.push(rng.random_range(-1.3e3..1.3e3));
        }
        let arrow_array = PrimitiveArray::<Float32Type>::new(ScalarBuffer::from(array), None);
        let liquid_array = LiquidFloatArray::<Float32Type>::from_arrow_array(arrow_array);

        let mut group = c.benchmark_group(format!("float32_liquid_decode"));
        group.throughput(Throughput::Bytes(
            (size * std::mem::size_of::<f32>()) as u64,
        ));
        group.bench_function(format!("size_{}", size), |b| {
            b.iter(|| {
                let _x = liquid_array.to_arrow_array();
            })
        });
        group.finish();
    }

    // Decoding benchmarks for float64
    for size in bench_sizes {
        let mut rng = rand::rng();
        let mut array: Vec<f64> = vec![];
        for _ in 0..size {
            array.push(rng.random_range(-1.3e3..1.3e3));
        }
        let arrow_array = PrimitiveArray::<Float64Type>::new(ScalarBuffer::from(array), None);
        let liquid_array = LiquidFloatArray::<Float64Type>::from_arrow_array(arrow_array);

        let mut group = c.benchmark_group(format!("float64_liquid_decode"));
        group.throughput(Throughput::Bytes(
            (size * std::mem::size_of::<f64>()) as u64,
        ));
        group.bench_function(format!("size_{}", size), |b| {
            b.iter(|| {
                let _x = liquid_array.to_arrow_array();
            })
        });
        group.finish();
    }
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
