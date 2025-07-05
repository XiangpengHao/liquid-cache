use divan::Bencher;
use liquid_cache_parquet::cache::LiquidCachedColumn;
use std::sync::Arc;
use tempfile::TempDir;

use arrow::array::{ArrayRef, BooleanArray, Int32Array};
use arrow::datatypes::{DataType, Field};
use liquid_cache_common::LiquidCacheMode;
use liquid_cache_parquet::cache::{BatchID, LiquidCache, policies::DiscardPolicy};
use rand::Rng;

const BATCH_SIZE: usize = 8192 * 2;
const SELECTIVITIES: [f64; 4] = [0.01, 0.1, 0.3, 0.7];

fn create_test_data(array_size: usize) -> ArrayRef {
    let mut rng = rand::rng();
    let values: Vec<i32> = (0..array_size).map(|_| rng.random_range(0..1000)).collect();
    Arc::new(Int32Array::from(values))
}

fn create_boolean_filter(array_size: usize, selectivity: f64) -> BooleanArray {
    let mut rng = rand::rng();
    let values: Vec<bool> = (0..array_size)
        .map(|_| rng.random::<f64>() < selectivity)
        .collect();
    BooleanArray::from(values)
}

fn setup_cache(tmp_dir: &TempDir) -> Arc<LiquidCachedColumn> {
    let cache = LiquidCache::new(
        BATCH_SIZE,
        1024 * 1024 * 1024, // max_cache_bytes (1GB)
        tmp_dir.path().to_path_buf(),
        LiquidCacheMode::Liquid {
            transcode_in_background: false,
        },
        Box::new(DiscardPolicy::default()),
    );
    let file = cache.register_or_get_file("test_file.parquet".to_string());
    let row_group = file.row_group(0);
    let field = Arc::new(Field::new("test_column", DataType::Int32, false));
    let column = row_group.create_column(0, field);
    column
}

#[divan::bench(args = SELECTIVITIES, sample_count = 1000)]
fn get_arrow_array_with_filter_arrow_cache(bencher: Bencher, selectivity: f64) {
    let temp_dir = tempfile::tempdir().unwrap();
    let column = setup_cache(&temp_dir);

    // Create and insert test data
    let test_data = create_test_data(BATCH_SIZE);
    let batch_id = BatchID::from_row_id(0, BATCH_SIZE);
    column
        .insert(batch_id, test_data)
        .expect("Failed to insert data");

    let filter = create_boolean_filter(BATCH_SIZE, selectivity);

    bencher
        .with_inputs(|| (&column, batch_id, &filter))
        .bench_values(|(column, batch_id, filter)| {
            std::hint::black_box(column.get_arrow_array_with_filter(batch_id, filter))
        });
}

#[divan::bench(args = SELECTIVITIES, sample_count = 1000)]
fn get_arrow_array_with_filter_liquid_cache(bencher: Bencher, selectivity: f64) {
    let temp_dir = tempfile::tempdir().unwrap();
    let column = setup_cache(&temp_dir);

    // Create and insert test data
    let test_data = create_test_data(BATCH_SIZE);
    let batch_id = BatchID::from_row_id(0, BATCH_SIZE);
    column
        .insert(batch_id, test_data)
        .expect("Failed to insert data");

    let filter = create_boolean_filter(BATCH_SIZE, selectivity);

    bencher
        .with_inputs(|| (&column, batch_id, &filter))
        .bench_values(|(column, batch_id, filter)| {
            std::hint::black_box(column.get_arrow_array_with_filter(batch_id, filter))
        });
}

fn main() {
    divan::main();
}
