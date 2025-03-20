use crate::{
    LiquidCacheMode, LiquidCachedFileRef, LiquidPredicate,
    cache::{CacheConfig, LiquidCachedFile},
    liquid_array::LiquidArrayRef,
    reader::{
        plantime::{
            CachedMetaReaderFactory, coerce_binary_to_string, coerce_string_to_view,
            coerce_to_liquid_cache_types,
        },
        runtime::{ArrowReaderBuilderBridge, LiquidRowFilter, LiquidStreamBuilder},
    },
};
use arrow::{
    array::{AsArray, BooleanArray, BooleanBuilder},
    buffer::BooleanBuffer,
    compute::filter,
    datatypes::{
        DataType, Field, Int8Type, Int16Type, Int32Type, Int64Type, Schema, UInt8Type, UInt16Type,
        UInt32Type, UInt64Type,
    },
    record_batch::RecordBatch,
};
use arrow_schema::{ArrowError, SchemaRef};
use datafusion::physical_plan::metrics::ExecutionPlanMetricsSet;
use futures::StreamExt;
use object_store::ObjectMeta;
use parquet::{
    arrow::{
        ParquetRecordBatchStreamBuilder, ProjectionMask,
        arrow_reader::{
            ArrowPredicate, ArrowReaderMetadata, ArrowReaderOptions,
            ParquetRecordBatchReaderBuilder,
        },
    },
    file::metadata::ParquetMetaData,
};
use std::{fs::File, path::PathBuf, sync::Arc};

const TEST_FILE_PATH: &str = "../../examples/nano_hits.parquet";

fn test_output_schema() -> SchemaRef {
    let file = File::open(TEST_FILE_PATH).unwrap();
    let builder = ArrowReaderMetadata::load(&file, Default::default()).unwrap();
    let schema = builder.schema().clone();
    Arc::new(coerce_to_liquid_cache_types(schema.as_ref()))
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

async fn get_test_reader() -> (LiquidStreamBuilder, File) {
    let (file, path) = generate_test_parquet();
    let object_store = Arc::new(object_store::local::LocalFileSystem::new());

    let file_metadata = file.metadata().unwrap();
    let object_meta = ObjectMeta {
        location: object_store::path::Path::from_filesystem_path(path).unwrap(),
        size: file_metadata.len() as usize,
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
        .with_batch_size(8192);

    let liquid_builder =
        unsafe { ArrowReaderBuilderBridge::from_parquet(builder).into_liquid_builder() };

    let metadata = &liquid_builder.metadata;
    assert_eq!(metadata.num_row_groups(), 2);
    assert_eq!(metadata.file_metadata().num_rows(), 8192 * 3 + 10);
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

fn get_test_cached_file(bath_size: usize, cache_dir: PathBuf) -> LiquidCachedFileRef {
    let liquid_cache = LiquidCachedFile::new(
        LiquidCacheMode::InMemoryLiquid {
            transcode_in_background: false,
        },
        Arc::new(CacheConfig::new(bath_size, usize::MAX)),
        cache_dir,
    );
    Arc::new(liquid_cache)
}
#[tokio::test]
async fn basic_stuff() {
    let tmp_dir = tempfile::tempdir().unwrap();
    let (builder, _file) = get_test_reader().await;
    let batch_size = builder.batch_size;
    let liquid_cache = get_test_cached_file(batch_size, tmp_dir.path().to_path_buf());
    let reader = builder.build(liquid_cache).unwrap();

    let schema = &reader.schema;
    assert_eq!(schema.as_ref(), test_output_schema().as_ref());

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

#[tokio::test]
async fn test_reading_with_projection() {
    let tmp_dir = tempfile::tempdir().unwrap();
    let column_projections = vec![0, 3, 6, 8];
    let (mut builder, _file) = get_test_reader().await;
    builder.projection = ProjectionMask::roots(
        builder.metadata.file_metadata().schema_descr(),
        column_projections.iter().cloned(),
    );
    let batch_size = builder.batch_size;
    let liquid_cache = get_test_cached_file(batch_size, tmp_dir.path().to_path_buf());
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

#[tokio::test]
async fn test_reading_warm() {
    let tmp_dir = tempfile::tempdir().unwrap();
    let column_projections = vec![0, 3, 6, 8];
    let (mut builder, _file) = get_test_reader().await;
    let batch_size = builder.batch_size;
    let liquid_cache = get_test_cached_file(batch_size, tmp_dir.path().to_path_buf());
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

struct TestPredicate {
    projection_mask: ProjectionMask,
    strategy: FilterStrategy,
}

impl TestPredicate {
    fn new(parquet_meta: &ParquetMetaData, column_id: usize, strategy: FilterStrategy) -> Self {
        Self {
            projection_mask: ProjectionMask::roots(
                parquet_meta.file_metadata().schema_descr(),
                [column_id],
            ),
            strategy,
        }
    }
}

impl LiquidPredicate for TestPredicate {
    fn evaluate_liquid(&mut self, array: &LiquidArrayRef) -> Result<BooleanArray, ArrowError> {
        let batch = array.to_arrow_array();

        let schema = Schema::new(vec![Field::new(
            "_",
            batch.data_type().clone(),
            batch.is_nullable(),
        )]);
        let batch = RecordBatch::try_new(Arc::new(schema), vec![Arc::new(batch)]).unwrap();
        self.evaluate(batch)
    }
}

impl ArrowPredicate for TestPredicate {
    fn evaluate(&mut self, batch: RecordBatch) -> Result<BooleanArray, ArrowError> {
        assert_eq!(batch.num_columns(), 1);
        let column = batch.column(0);

        let mut builder = BooleanBuilder::new();

        // A helper macro to reduce code duplication:
        macro_rules! filter_values {
            ($ARRAY:ty, $CAST:ty) => {{
                let typed = column.as_primitive::<$CAST>();
                for v in typed {
                    match v {
                        Some(v) => {
                            let v = v as i64;
                            let keep = match self.strategy {
                                FilterStrategy::NoOdd => v % 2 == 0,
                                FilterStrategy::NoSmallerThan(min) => v >= min,
                                FilterStrategy::NoLargerThan(max) => v <= max,
                            };
                            builder.append_value(keep);
                        }
                        None => builder.append_null(),
                    }
                }
            }};
        }

        match column.data_type() {
            DataType::Int8 => filter_values!(Int8Array, Int8Type),
            DataType::Int16 => filter_values!(Int16Array, Int16Type),
            DataType::Int32 => filter_values!(Int32Array, Int32Type),
            DataType::Int64 => filter_values!(Int64Array, Int64Type),
            DataType::UInt8 => filter_values!(UInt8Array, UInt8Type),
            DataType::UInt16 => filter_values!(UInt16Array, UInt16Type),
            DataType::UInt32 => filter_values!(UInt32Array, UInt32Type),
            DataType::UInt64 => filter_values!(UInt64Array, UInt64Type),
            _ => panic!("not supported {:?}", column.data_type()),
        }

        Ok(builder.finish())
    }

    fn projection(&self) -> &ProjectionMask {
        &self.projection_mask
    }
}

enum FilterStrategy {
    NoOdd,
    NoSmallerThan(i64),
    NoLargerThan(i64),
}

#[tokio::test]
async fn test_reading_with_filter() {
    let projection = vec![0, 3, 5, 6, 8];
    let (mut builder, _file) = get_test_reader().await;
    let batch_size = builder.batch_size;

    builder.projection = ProjectionMask::roots(
        builder.metadata.file_metadata().schema_descr(),
        projection.iter().cloned(),
    );

    fn get_filters(metadata: &ParquetMetaData) -> Vec<Box<dyn LiquidPredicate>> {
        let filter1 = TestPredicate::new(metadata, 0, FilterStrategy::NoOdd);
        let filter2 = TestPredicate::new(metadata, 5, FilterStrategy::NoSmallerThan(10_000));
        let filter3 = TestPredicate::new(metadata, 6, FilterStrategy::NoLargerThan(20_000));
        let filters = vec![
            Box::new(filter1) as Box<dyn LiquidPredicate>,
            Box::new(filter2),
            Box::new(filter3),
        ];
        filters
    }
    builder.filter = Some(LiquidRowFilter::new(get_filters(&builder.metadata)));

    let tmp_dir = tempfile::tempdir().unwrap();
    let liquid_cache = get_test_cached_file(batch_size, tmp_dir.path().to_path_buf());

    let reader = builder.build(liquid_cache.clone()).unwrap();

    let batches = reader
        .collect::<Vec<_>>()
        .await
        .into_iter()
        .map(|batch| batch.unwrap())
        .collect::<Vec<_>>();
    let baseline_batches = get_baseline_record_batch(batch_size, &projection);

    for (i, batch) in batches.iter().enumerate() {
        let expected = &baseline_batches[i];

        let col_i64 = expected.column(0).as_primitive::<Int64Type>();
        let mask1 = BooleanBuffer::from_iter(
            col_i64
                .iter()
                .map(|val| val.map(|v| v % 2 == 0).unwrap_or(false)),
        );

        // 1373872581 is the average value of that column
        let col_i32 = expected.column(4).as_primitive::<Int32Type>();
        let mask2 = BooleanBuffer::from_iter(
            col_i32
                .iter()
                .map(|val| val.map(|v| v <= 1373872581).unwrap_or(false)),
        );

        let combined_mask = &mask1 & &mask2;

        let expected = filter_record_batch(&expected, combined_mask);

        assert_batch_eq(&expected, batch);
    }

    // now run again with the same cache
    let (mut builder, _file) = get_test_reader().await;
    builder.projection = ProjectionMask::roots(
        builder.metadata.file_metadata().schema_descr(),
        projection.iter().cloned(),
    );
    builder.filter = Some(LiquidRowFilter::new(get_filters(&builder.metadata)));
    let reader = builder.build(liquid_cache.clone()).unwrap();
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

#[tokio::test]
async fn test_reading_with_filter_two_columns() {
    let projection = vec![11, 12]; // OS and UserAgent
    let (mut builder, _file) = get_test_reader().await;
    let batch_size = builder.batch_size;

    builder.projection = ProjectionMask::roots(
        builder.metadata.file_metadata().schema_descr(),
        projection.iter().cloned(),
    );

    struct TwoColumnsPredicate {
        projection_mask: ProjectionMask,
    }

    impl LiquidPredicate for TwoColumnsPredicate {
        fn evaluate_liquid(&mut self, _array: &LiquidArrayRef) -> Result<BooleanArray, ArrowError> {
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
    let predicate = TwoColumnsPredicate {
        projection_mask: ProjectionMask::roots(
            builder.metadata.file_metadata().schema_descr(),
            projection.iter().cloned(),
        ),
    };

    builder.filter = Some(LiquidRowFilter::new(vec![Box::new(predicate)]));

    let tmp_dir = tempfile::tempdir().unwrap();
    let liquid_cache = get_test_cached_file(batch_size, tmp_dir.path().to_path_buf());

    let reader = builder.build(liquid_cache.clone()).unwrap();

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
    let reader = builder.build(liquid_cache.clone()).unwrap();
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

fn filter_record_batch(batch: &RecordBatch, mask: BooleanBuffer) -> RecordBatch {
    let mask = BooleanArray::new(mask, None);
    let filtered_columns = batch
        .columns()
        .iter()
        .map(|col| filter(col, &mask).unwrap())
        .collect::<Vec<_>>();

    RecordBatch::try_new(batch.schema(), filtered_columns).unwrap()
}

#[tokio::test]
async fn test_reading_with_full_cache() {
    let column_projections = vec![0, 3, 6, 8];
    let (mut builder, _file) = get_test_reader().await;
    let batch_size = builder.batch_size;
    let tmp_dir = tempfile::tempdir().unwrap();
    // Create a cache with a very small max size to force cache misses
    let liquid_cache = Arc::new(LiquidCachedFile::new(
        LiquidCacheMode::InMemoryLiquid {
            transcode_in_background: false,
        },
        Arc::new(CacheConfig::new(batch_size, 1)),
        tmp_dir.path().to_path_buf(),
    ));

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

    // Verify we got the expected results even with a full cache
    for (i, batch) in batches.iter().enumerate() {
        assert_batch_eq(&baseline_batches[i], batch);
    }
    assert_eq!(liquid_cache.memory_usage(), 0);
}
