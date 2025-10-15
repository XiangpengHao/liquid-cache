use datafusion::arrow::{array::PrimitiveArray, buffer::ScalarBuffer, datatypes::Int32Type};
use liquid_cache_storage::liquid_array::LiquidArray;
use liquid_cache_storage::liquid_array::primitive_array::{
    LiquidPrimitiveArray, LiquidPrimitiveDeltaArray,
};
use rand::{Rng, SeedableRng, rngs::StdRng};
use std::time::Instant;

fn main() {
    let size = 262144;
    let data_size_bytes = size * std::mem::size_of::<i32>();
    let mut rng = StdRng::seed_from_u64(42);

    // Test both sequential and random data
    let datasets = vec![
        (
            "Sequential",
            (0..size).map(|x| x as i32).collect::<Vec<i32>>(),
        ),
        (
            "Random",
            (0..size)
                .map(|_| rng.gen_range(0..1_000_000))
                .collect::<Vec<i32>>(),
        ),
    ];

    for (name, data) in datasets {
        println!("{} Data 1MB ({} integers)", name, size);
        println!("{}", "=".repeat(50));

        let array = PrimitiveArray::<Int32Type>::new(ScalarBuffer::from(data.clone()), None);

        // Memory comparison
        println!("Memory Consumption:");
        let regular = LiquidPrimitiveArray::<Int32Type>::from_arrow_array(array.clone());
        let delta = LiquidPrimitiveDeltaArray::<Int32Type>::from_arrow_array(array.clone());

        let regular_mem_kb = regular.get_array_memory_size() as f64 / 1024.0;
        let delta_mem_kb = delta.get_array_memory_size() as f64 / 1024.0;
        let memory_savings = ((regular_mem_kb - delta_mem_kb) / regular_mem_kb) * 100.0;

        println!("  Regular: {:.1} KB", regular_mem_kb);
        println!("  Delta:   {:.1} KB", delta_mem_kb);
        println!("  Savings: {:.1}%", memory_savings);
        println!();

        // Encoding speed comparison
        println!("Encoding Speed:");

        let encode_start = Instant::now();
        let regular_encoded = LiquidPrimitiveArray::<Int32Type>::from_arrow_array(array.clone());
        let regular_encode_time = encode_start.elapsed();
        let regular_encode_speed =
            data_size_bytes as f64 / regular_encode_time.as_secs_f64() / 1_000_000.0;

        let encode_start = Instant::now();
        let delta_encoded = LiquidPrimitiveDeltaArray::<Int32Type>::from_arrow_array(array.clone());
        let delta_encode_time = encode_start.elapsed();
        let delta_encode_speed =
            data_size_bytes as f64 / delta_encode_time.as_secs_f64() / 1_000_000.0;

        let encode_ratio = regular_encode_speed / delta_encode_speed;

        println!("  Regular: {:.1} MB/s", regular_encode_speed);
        println!("  Delta:   {:.1} MB/s", delta_encode_speed);
        println!("  Ratio:   {:.1}x slower", encode_ratio);
        println!();

        // Decoding speed comparison
        println!("Decoding Speed:");

        let decode_start = Instant::now();
        let _ = regular_encoded.to_arrow_array();
        let regular_decode_time = decode_start.elapsed();
        let regular_decode_speed =
            data_size_bytes as f64 / regular_decode_time.as_secs_f64() / 1_000_000.0;

        let decode_start = Instant::now();
        let _ = delta_encoded.to_arrow_array();
        let delta_decode_time = decode_start.elapsed();
        let delta_decode_speed =
            data_size_bytes as f64 / delta_decode_time.as_secs_f64() / 1_000_000.0;

        let decode_ratio = regular_decode_speed / delta_decode_speed;

        println!("  Regular: {:.1} MB/s", regular_decode_speed);
        println!("  Delta:   {:.1} MB/s", delta_decode_speed);
        println!("  Ratio:   {:.1}x slower", decode_ratio);
        println!();
    }
}
