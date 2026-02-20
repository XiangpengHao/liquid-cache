use arrow::buffer::BooleanBuffer;
use divan::Bencher;
use liquid_cache::cache::AlwaysHydrate;
use liquid_cache::cache::squeeze_policies::TranscodeSqueezeEvict;
use liquid_cache::cache_policies::LiquidPolicy;
use liquid_cache_common::IoMode;
use liquid_cache_datafusion::cache::CachedColumn;
use liquid_cache_datafusion::{FilterCandidateBuilder, LiquidPredicate};
use std::sync::Arc;
use tempfile::TempDir;

use arrow::array::{ArrayRef, Int32Array, RecordBatch};
use arrow::datatypes::{DataType, Field, Schema};
use datafusion::common::ScalarValue;
use datafusion::logical_expr::Operator;
use datafusion::physical_expr::PhysicalExpr;
use datafusion::physical_expr::expressions::{BinaryExpr, Literal};
use datafusion::physical_plan::expressions::Column;
use datafusion::physical_plan::metrics;
use liquid_cache_datafusion::cache::{BatchID, LiquidCacheParquet};
use parquet::arrow::ArrowWriter;
use parquet::arrow::arrow_reader::{ArrowReaderMetadata, ArrowReaderOptions};
use rand::RngExt as _;

const BATCH_SIZE: usize = 8192 * 2;
const SELECTIVITIES: [f64; 4] = [0.01, 0.1, 0.3, 0.7];

fn create_test_data(array_size: usize) -> ArrayRef {
    let mut rng = rand::rng();
    let values: Vec<i32> = (0..array_size).map(|_| rng.random_range(0..1000)).collect();
    Arc::new(Int32Array::from(values))
}

fn create_boolean_filter(array_size: usize, selectivity: f64) -> BooleanBuffer {
    let mut rng = rand::rng();
    let values: Vec<bool> = (0..array_size)
        .map(|_| rng.random::<f64>() < selectivity)
        .collect();
    BooleanBuffer::from(values)
}

fn setup_cache(tmp_dir: &TempDir) -> Arc<CachedColumn> {
    let cache = LiquidCacheParquet::new(
        BATCH_SIZE,
        1024 * 1024 * 1024, // max_cache_bytes (1GB)
        tmp_dir.path().to_path_buf(),
        Box::new(LiquidPolicy::new()),
        Box::new(TranscodeSqueezeEvict),
        Box::new(AlwaysHydrate::new()),
        IoMode::Uring,
    );
    let field = Arc::new(Field::new("test_column", DataType::Int32, false));
    let schema = Arc::new(Schema::new(vec![field.clone()]));
    let file = cache.register_or_get_file("test_file.parquet".to_string(), schema);
    let row_group = file.create_row_group(0, vec![]);
    row_group.get_column(0).unwrap()
}

#[divan::bench(args = SELECTIVITIES, sample_count = 1000)]
fn get_arrow_array_with_filter_arrow_cache(bencher: Bencher, selectivity: f64) {
    let temp_dir = tempfile::tempdir().unwrap();
    let column = setup_cache(&temp_dir);

    // Create and insert test data
    let test_data = create_test_data(BATCH_SIZE);
    let batch_id = BatchID::from_row_id(0, BATCH_SIZE);
    tokio_test::block_on(async {
        column.insert(batch_id, test_data.clone()).await.unwrap();
    });

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
    tokio_test::block_on(async {
        column.insert(batch_id, test_data.clone()).await.unwrap();
    });

    let filter = create_boolean_filter(BATCH_SIZE, selectivity);

    // Create a simple predicate for benchmarking (column = 500)
    let schema = Arc::new(Schema::new(vec![Field::new(
        "test_column",
        DataType::Int32,
        false,
    )]));

    // build parquet metadata for predicate construction
    let tmp_meta = tempfile::NamedTempFile::new().unwrap();
    let mut writer =
        ArrowWriter::try_new(tmp_meta.reopen().unwrap(), Arc::clone(&schema), None).unwrap();
    let batch = RecordBatch::try_new(Arc::clone(&schema), vec![test_data]).unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();
    let file_reader = std::fs::File::open(tmp_meta.path()).unwrap();
    let metadata = ArrowReaderMetadata::load(&file_reader, ArrowReaderOptions::new()).unwrap();

    // Create a simple predicate: column = 500
    let expr: Arc<dyn PhysicalExpr> = Arc::new(BinaryExpr::new(
        Arc::new(Column::new("test_column", 0)),
        Operator::Eq,
        Arc::new(Literal::new(ScalarValue::Int32(Some(500)))),
    ));

    let builder = FilterCandidateBuilder::new(expr, Arc::clone(&schema));
    let candidate = builder.build(metadata.metadata()).unwrap().unwrap();
    let projection = candidate.projection(metadata.metadata());
    let predicate = LiquidPredicate::try_new_with_metrics(
        candidate,
        projection,
        metrics::Count::new(),
        metrics::Count::new(),
        metrics::Time::new(),
    )
    .unwrap();

    bencher
        .with_inputs(|| (&column, batch_id, &filter))
        .bench_values(|(column, batch_id, filter)| {
            let mut predicate = predicate.clone();
            tokio::task::block_in_place(|| async move {
                std::hint::black_box(
                    column
                        .eval_predicate_with_filter(batch_id, filter, &mut predicate)
                        .await,
                )
            })
        });
}

fn main() {
    divan::main();
}
