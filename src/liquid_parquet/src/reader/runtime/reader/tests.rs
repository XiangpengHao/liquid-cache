use crate::{
    LiquidCachedFileRef, LiquidPredicate,
    cache::LiquidCache,
    liquid_array::LiquidArrayRef,
    reader::{
        plantime::CachedMetaReaderFactory,
        runtime::{
            ArrowReaderBuilderBridge, LiquidRowFilter,
            liquid_stream::{LiquidStream, LiquidStreamBuilder},
        },
    },
};
use arrow::{
    array::{AsArray, BooleanArray},
    datatypes::{Int16Type, Schema},
    record_batch::RecordBatch,
};
use arrow_schema::{ArrowError, SchemaRef};
use datafusion::physical_plan::metrics::ExecutionPlanMetricsSet;
use futures::StreamExt;
use liquid_cache_common::{
    LiquidCacheMode, coerce_binary_to_string, coerce_string_to_view, coerce_to_liquid_cache_types,
};
use object_store::ObjectMeta;
use parquet::arrow::{
    ParquetRecordBatchStreamBuilder, ProjectionMask,
    arrow_reader::{
        ArrowPredicate, ArrowReaderMetadata, ArrowReaderOptions, ParquetRecordBatchReaderBuilder,
    },
};
use std::{fs::File, path::PathBuf, sync::Arc};

const TEST_FILE_PATH: &str = "../../examples/nano_hits.parquet";

fn test_output_schema(cache_mode: &LiquidCacheMode) -> SchemaRef {
    let file = File::open(TEST_FILE_PATH).unwrap();
    let builder = ArrowReaderMetadata::load(&file, Default::default()).unwrap();
    let schema = builder.schema().clone();
    Arc::new(coerce_to_liquid_cache_types(schema.as_ref(), cache_mode))
}

pub fn generate_test_parquet() -> (File, String) {
    return (
        File::open(TEST_FILE_PATH).unwrap(),
        TEST_FILE_PATH.to_string(),
    );
}

fn get_baseline_record_batch(batch_size: usize, projection: &[usize]) -> Vec<RecordBatch> {
    let (file, _path) = generate_test_parquet();
    let metadata = ArrowReaderMetadata::load(&file, Default::default()).unwrap();
    let builder = ParquetRecordBatchReaderBuilder::new_with_metadata(file, metadata.clone())
        .with_batch_size(batch_size)
        .with_projection(ProjectionMask::roots(
            metadata.parquet_schema(),
            projection.iter().cloned(),
        ));
    let reader = builder.build().unwrap();
    reader
        .collect::<Vec<_>>()
        .into_iter()
        .map(|batch| batch.unwrap())
        .collect()
}

const BATCH_SIZE: usize = 8192;

async fn get_test_reader() -> (LiquidStreamBuilder, File) {
    let (file, path) = generate_test_parquet();
    let object_store = Arc::new(object_store::local::LocalFileSystem::new());

    let file_metadata = file.metadata().unwrap();
    let object_meta = ObjectMeta {
        location: object_store::path::Path::from_filesystem_path(path).unwrap(),
        size: file_metadata.len(),
        last_modified: file_metadata.modified().unwrap().into(),
        e_tag: None,
        version: None,
    };

    let metrics = ExecutionPlanMetricsSet::new();
    let reader_factory = CachedMetaReaderFactory::new(object_store);
    let mut async_reader =
        reader_factory.create_liquid_reader(0, object_meta.into(), None, &metrics);

    let metadata = ArrowReaderMetadata::load_async(&mut async_reader, Default::default())
        .await
        .unwrap();
    let schema = Arc::clone(metadata.schema());
    let schema = coerce_binary_to_string(&schema);
    let reader_schema = Arc::new(coerce_string_to_view(&schema));

    let options = ArrowReaderOptions::new().with_schema(Arc::clone(&reader_schema));
    let metadata = ArrowReaderMetadata::try_new(Arc::clone(metadata.metadata()), options).unwrap();

    let builder = ParquetRecordBatchStreamBuilder::new_with_metadata(async_reader, metadata)
        .with_batch_size(BATCH_SIZE);

    let liquid_builder =
        unsafe { ArrowReaderBuilderBridge::from_parquet(builder).into_liquid_builder() };

    let metadata = &liquid_builder.metadata;
    assert_eq!(metadata.num_row_groups(), 2);
    assert_eq!(
        metadata.file_metadata().num_rows(),
        BATCH_SIZE as i64 * 3 + 10
    );
    (liquid_builder, file)
}

/// We could directly assert_eq!(left, right) but this is more debugging friendly
fn assert_batch_eq(left: &RecordBatch, right: &RecordBatch) {
    assert_eq!(left.num_rows(), right.num_rows());
    assert_eq!(left.columns().len(), right.columns().len());
    for (c_l, c_r) in left.columns().iter().zip(right.columns().iter()) {
        let casted = arrow::compute::cast(c_l, c_r.data_type()).unwrap();
        assert_eq!(&casted, c_r);
    }
}

fn assert_schema_eq(left: &Schema, right: &Schema) {
    assert_eq!(left.fields().len(), right.fields().len());
    for (f_l, f_r) in left.fields().iter().zip(right.fields().iter()) {
        assert_eq!(f_l, f_r);
    }
}

fn get_test_cache(
    bath_size: usize,
    cache_dir: PathBuf,
    cache_mode: &LiquidCacheMode,
) -> LiquidCachedFileRef {
    let lq = LiquidCache::new(bath_size, usize::MAX, cache_dir, *cache_mode);

    lq.register_or_get_file("".to_string())
}

async fn basic_stuff(cache_mode: &LiquidCacheMode) {
    let tmp_dir = tempfile::tempdir().unwrap();
    let (builder, _file) = get_test_reader().await;
    let batch_size = builder.batch_size;
    let liquid_cache = get_test_cache(batch_size, tmp_dir.path().to_path_buf(), cache_mode);
    let reader = builder.build(liquid_cache).unwrap();

    let schema = reader.schema();
    assert_schema_eq(&schema, test_output_schema(cache_mode).as_ref());

    let projection = (0..schema.fields().len()).collect::<Vec<_>>();

    let batches = reader
        .collect::<Vec<_>>()
        .await
        .into_iter()
        .map(|batch| batch.unwrap())
        .collect::<Vec<_>>();

    let baseline_batches = get_baseline_record_batch(batch_size, &projection);

    for (i, batch) in batches.iter().enumerate() {
        assert_batch_eq(&baseline_batches[i], batch);
    }
}

const CACHE_MODES: &[LiquidCacheMode] = &[
    LiquidCacheMode::InMemoryArrow,
    LiquidCacheMode::InMemoryLiquid {
        transcode_in_background: false,
    },
    LiquidCacheMode::InMemoryLiquid {
        transcode_in_background: true,
    },
];

#[tokio::test(flavor = "multi_thread")]
async fn test_basic() {
    for cache_mode in CACHE_MODES {
        basic_stuff(cache_mode).await;
    }
}

async fn read_with_projection(cache_mode: &LiquidCacheMode) {
    let tmp_dir = tempfile::tempdir().unwrap();
    let column_projections = vec![0, 3, 6, 8];
    let (mut builder, _file) = get_test_reader().await;
    builder.projection = ProjectionMask::roots(
        builder.metadata.file_metadata().schema_descr(),
        column_projections.iter().cloned(),
    );
    let batch_size = builder.batch_size;
    let liquid_cache = get_test_cache(batch_size, tmp_dir.path().to_path_buf(), &cache_mode);
    let reader = builder.build(liquid_cache).unwrap();

    let batches = reader
        .collect::<Vec<_>>()
        .await
        .into_iter()
        .map(|batch| batch.unwrap())
        .collect::<Vec<_>>();
    let baseline_batches = get_baseline_record_batch(batch_size, &column_projections);

    for (i, batch) in batches.iter().enumerate() {
        assert_batch_eq(&baseline_batches[i], batch);
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn test_read_with_projection() {
    for cache_mode in CACHE_MODES {
        read_with_projection(cache_mode).await;
    }
}

async fn read_warm(cache_mode: &LiquidCacheMode) {
    let tmp_dir = tempfile::tempdir().unwrap();
    let column_projections = vec![0, 3, 6, 8];
    let (mut builder, _file) = get_test_reader().await;
    let batch_size = builder.batch_size;
    let liquid_cache = get_test_cache(batch_size, tmp_dir.path().to_path_buf(), cache_mode);
    builder.projection = ProjectionMask::roots(
        builder.metadata.file_metadata().schema_descr(),
        column_projections.iter().cloned(),
    );
    let reader = builder.build(liquid_cache.clone()).unwrap();
    let _batches = reader.collect::<Vec<_>>().await;

    let (mut builder, _file) = get_test_reader().await;
    builder.projection = ProjectionMask::roots(
        builder.metadata.file_metadata().schema_descr(),
        column_projections.iter().cloned(),
    );
    let reader = builder.build(liquid_cache.clone()).unwrap();

    let batches = reader
        .collect::<Vec<_>>()
        .await
        .into_iter()
        .map(|batch| batch.unwrap())
        .collect::<Vec<_>>();
    let baseline_batches = get_baseline_record_batch(batch_size, &column_projections);

    for (i, batch) in batches.iter().enumerate() {
        assert_batch_eq(&baseline_batches[i], batch);
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn test_reading_warm() {
    for cache_mode in CACHE_MODES {
        read_warm(cache_mode).await;
    }
}

struct TwoColumnsPredicate {
    projection_mask: ProjectionMask,
}

impl LiquidPredicate for TwoColumnsPredicate {
    fn evaluate_liquid(
        &mut self,
        _array: &LiquidArrayRef,
    ) -> Result<Option<BooleanArray>, ArrowError> {
        unimplemented!()
    }
}

impl ArrowPredicate for TwoColumnsPredicate {
    fn evaluate(&mut self, batch: RecordBatch) -> Result<BooleanArray, ArrowError> {
        assert_eq!(batch.num_columns(), 2);
        let column1 = batch.column(0);
        let column2 = batch.column(1);
        let mask = arrow::compute::kernels::cmp::gt(column1, column2).unwrap();
        Ok(mask)
    }

    fn projection(&self) -> &ProjectionMask {
        &self.projection_mask
    }
}

async fn get_projected_reader(projection: &[usize], cache: LiquidCachedFileRef) -> LiquidStream {
    let (mut builder, _file) = get_test_reader().await;
    builder.projection = ProjectionMask::roots(
        builder.metadata.file_metadata().schema_descr(),
        projection.iter().cloned(),
    );
    let predicate = TwoColumnsPredicate {
        projection_mask: ProjectionMask::roots(
            builder.metadata.file_metadata().schema_descr(),
            projection.iter().cloned(),
        ),
    };
    builder.filter = Some(LiquidRowFilter::new(vec![Box::new(predicate)]));
    builder.build(cache).unwrap()
}

async fn reading_with_filter_two_columns(cache_mode: &LiquidCacheMode) {
    let projection = vec![11, 12]; // OS and UserAgent
    let tmp_dir = tempfile::tempdir().unwrap();
    let liquid_cache = get_test_cache(BATCH_SIZE, tmp_dir.path().to_path_buf(), cache_mode);
    let reader = get_projected_reader(&projection, liquid_cache.clone()).await;

    let batches = reader
        .collect::<Vec<_>>()
        .await
        .into_iter()
        .map(|batch| batch.unwrap())
        .collect::<Vec<_>>();

    for (_i, batch) in batches.iter().enumerate() {
        let col_1 = batch.column(0).as_primitive::<Int16Type>();
        let col_2 = batch.column(1).as_primitive::<Int16Type>();
        for (c1, c2) in col_1.iter().zip(col_2.iter()) {
            assert!(c1 > c2);
        }
    }

    // now run again with the same cache
    let reader = get_projected_reader(&projection, liquid_cache).await;
    let warm_batches = reader
        .collect::<Vec<_>>()
        .await
        .into_iter()
        .map(|batch| batch.unwrap())
        .collect::<Vec<_>>();
    assert_eq!(batches.len(), warm_batches.len());
    for (batch, warm_batch) in batches.iter().zip(warm_batches.iter()) {
        assert_batch_eq(&batch, &warm_batch);
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn test_reading_with_filter_two_columns() {
    for cache_mode in CACHE_MODES {
        reading_with_filter_two_columns(cache_mode).await;
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn test_reading_with_full_cache() {
    let column_projections = vec![0, 3, 6, 8];
    let (mut builder, _file) = get_test_reader().await;
    let batch_size = builder.batch_size;
    let tmp_dir = tempfile::tempdir().unwrap();
    // Create a cache with a very small max size to force cache misses
    let lq = LiquidCache::new(
        batch_size,
        1,
        tmp_dir.path().to_path_buf(),
        LiquidCacheMode::InMemoryLiquid {
            transcode_in_background: false,
        },
    );
    let lq_file = lq.register_or_get_file("".to_string());

    builder.projection = ProjectionMask::roots(
        builder.metadata.file_metadata().schema_descr(),
        column_projections.iter().cloned(),
    );
    let reader = builder.build(lq_file.clone()).unwrap();

    let batches = reader
        .collect::<Vec<_>>()
        .await
        .into_iter()
        .map(|batch| batch.unwrap())
        .collect::<Vec<_>>();

    let baseline_batches = get_baseline_record_batch(batch_size, &column_projections);

    // Verify we got the expected results even with a full cache
    for (i, batch) in batches.iter().enumerate() {
        assert_batch_eq(&baseline_batches[i], batch);
    }
    assert_eq!(lq.memory_usage_bytes(), 0);
}
