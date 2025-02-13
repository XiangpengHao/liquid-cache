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
use arrow_schema::ArrowError;
use bytes::Bytes;
use futures::future::BoxFuture;
use futures::{FutureExt, StreamExt};
use parquet::{
    arrow::{
        ParquetRecordBatchStreamBuilder, ProjectionMask,
        arrow_reader::{ArrowPredicate, ArrowReaderMetadata, ArrowReaderOptions},
        async_reader::AsyncFileReader,
    },
    errors::ParquetError,
    file::metadata::{ParquetMetaData, ParquetMetaDataReader},
};
use std::{ops::Range, sync::Arc};

use crate::{
    LiquidCacheMode, LiquidPredicate,
    cache::LiquidCachedFile,
    liquid_array::LiquidArrayRef,
    reader::{
        plantime::coerce_to_parquet_reader_types,
        runtime::{ArrowReaderBuilderBridge, LiquidRowFilter, LiquidStreamBuilder},
    },
};

use std::fs;

fn test_output_schema() -> Schema {
    Schema::new(vec![
        Field::new("WatchID", DataType::Int64, false),
        Field::new("JavaEnable", DataType::Int16, false),
        Field::new(
            "Title",
            DataType::Dictionary(Box::new(DataType::UInt16), Box::new(DataType::Utf8)),
            false,
        ),
        Field::new("GoodEvent", DataType::Int16, false),
        Field::new("EventTime", DataType::Int64, false),
        Field::new("EventDate", DataType::UInt16, false),
        Field::new("CounterID", DataType::Int32, false),
        Field::new("ClientIP", DataType::Int32, false),
        Field::new("RegionID", DataType::Int32, false),
        Field::new("UserID", DataType::Int64, false),
        Field::new("CounterClass", DataType::Int16, false),
        Field::new("OS", DataType::Int16, false),
        Field::new("UserAgent", DataType::Int16, false),
        Field::new(
            "URL",
            DataType::Dictionary(Box::new(DataType::UInt16), Box::new(DataType::Utf8)),
            false,
        ),
        Field::new(
            "Referer",
            DataType::Dictionary(Box::new(DataType::UInt16), Box::new(DataType::Utf8)),
            false,
        ),
        Field::new("IsRefresh", DataType::Int16, false),
        Field::new("RefererCategoryID", DataType::Int16, false),
        Field::new("RefererRegionID", DataType::Int32, false),
        Field::new("URLCategoryID", DataType::Int16, false),
        Field::new("URLRegionID", DataType::Int32, false),
        Field::new("ResolutionWidth", DataType::Int16, false),
        Field::new("ResolutionHeight", DataType::Int16, false),
        Field::new("ResolutionDepth", DataType::Int16, false),
        Field::new("FlashMajor", DataType::Int16, false),
        Field::new("FlashMinor", DataType::Int16, false),
        Field::new(
            "FlashMinor2",
            DataType::Dictionary(Box::new(DataType::UInt16), Box::new(DataType::Utf8)),
            false,
        ),
        Field::new("NetMajor", DataType::Int16, false),
        Field::new("NetMinor", DataType::Int16, false),
        Field::new("UserAgentMajor", DataType::Int16, false),
        Field::new(
            "UserAgentMinor",
            DataType::Dictionary(Box::new(DataType::UInt16), Box::new(DataType::Utf8)),
            false,
        ),
        Field::new("CookieEnable", DataType::Int16, false),
        Field::new("JavascriptEnable", DataType::Int16, false),
        Field::new("IsMobile", DataType::Int16, false),
        Field::new("MobilePhone", DataType::Int16, false),
        Field::new(
            "MobilePhoneModel",
            DataType::Dictionary(Box::new(DataType::UInt16), Box::new(DataType::Utf8)),
            false,
        ),
        Field::new(
            "Params",
            DataType::Dictionary(Box::new(DataType::UInt16), Box::new(DataType::Utf8)),
            false,
        ),
        Field::new("IPNetworkID", DataType::Int32, false),
        Field::new("TraficSourceID", DataType::Int16, false),
        Field::new("SearchEngineID", DataType::Int16, false),
        Field::new(
            "SearchPhrase",
            DataType::Dictionary(Box::new(DataType::UInt16), Box::new(DataType::Utf8)),
            false,
        ),
        Field::new("AdvEngineID", DataType::Int16, false),
        Field::new("IsArtifical", DataType::Int16, false),
        Field::new("WindowClientWidth", DataType::Int16, false),
        Field::new("WindowClientHeight", DataType::Int16, false),
        Field::new("ClientTimeZone", DataType::Int16, false),
        Field::new("ClientEventTime", DataType::Int64, false),
        Field::new("SilverlightVersion1", DataType::Int16, false),
        Field::new("SilverlightVersion2", DataType::Int16, false),
        Field::new("SilverlightVersion3", DataType::Int32, false),
        Field::new("SilverlightVersion4", DataType::Int16, false),
        Field::new(
            "PageCharset",
            DataType::Dictionary(Box::new(DataType::UInt16), Box::new(DataType::Utf8)),
            false,
        ),
        Field::new("CodeVersion", DataType::Int32, false),
        Field::new("IsLink", DataType::Int16, false),
        Field::new("IsDownload", DataType::Int16, false),
        Field::new("IsNotBounce", DataType::Int16, false),
        Field::new("FUniqID", DataType::Int64, false),
        Field::new(
            "OriginalURL",
            DataType::Dictionary(Box::new(DataType::UInt16), Box::new(DataType::Utf8)),
            false,
        ),
        Field::new("HID", DataType::Int32, false),
        Field::new("IsOldCounter", DataType::Int16, false),
        Field::new("IsEvent", DataType::Int16, false),
        Field::new("IsParameter", DataType::Int16, false),
        Field::new("DontCountHits", DataType::Int16, false),
        Field::new("WithHash", DataType::Int16, false),
        Field::new(
            "HitColor",
            DataType::Dictionary(Box::new(DataType::UInt16), Box::new(DataType::Utf8)),
            false,
        ),
        Field::new("LocalEventTime", DataType::Int64, false),
        Field::new("Age", DataType::Int16, false),
        Field::new("Sex", DataType::Int16, false),
        Field::new("Income", DataType::Int16, false),
        Field::new("Interests", DataType::Int16, false),
        Field::new("Robotness", DataType::Int16, false),
        Field::new("RemoteIP", DataType::Int32, false),
        Field::new("WindowName", DataType::Int32, false),
        Field::new("OpenerName", DataType::Int32, false),
        Field::new("HistoryLength", DataType::Int16, false),
        Field::new(
            "BrowserLanguage",
            DataType::Dictionary(Box::new(DataType::UInt16), Box::new(DataType::Utf8)),
            false,
        ),
        Field::new(
            "BrowserCountry",
            DataType::Dictionary(Box::new(DataType::UInt16), Box::new(DataType::Utf8)),
            false,
        ),
        Field::new(
            "SocialNetwork",
            DataType::Dictionary(Box::new(DataType::UInt16), Box::new(DataType::Utf8)),
            false,
        ),
        Field::new(
            "SocialAction",
            DataType::Dictionary(Box::new(DataType::UInt16), Box::new(DataType::Utf8)),
            false,
        ),
        Field::new("HTTPError", DataType::Int16, false),
        Field::new("SendTiming", DataType::Int32, false),
        Field::new("DNSTiming", DataType::Int32, false),
        Field::new("ConnectTiming", DataType::Int32, false),
        Field::new("ResponseStartTiming", DataType::Int32, false),
        Field::new("ResponseEndTiming", DataType::Int32, false),
        Field::new("FetchTiming", DataType::Int32, false),
        Field::new("SocialSourceNetworkID", DataType::Int16, false),
        Field::new(
            "SocialSourcePage",
            DataType::Dictionary(Box::new(DataType::UInt16), Box::new(DataType::Utf8)),
            false,
        ),
        Field::new("ParamPrice", DataType::Int64, false),
        Field::new(
            "ParamOrderID",
            DataType::Dictionary(Box::new(DataType::UInt16), Box::new(DataType::Utf8)),
            false,
        ),
        Field::new(
            "ParamCurrency",
            DataType::Dictionary(Box::new(DataType::UInt16), Box::new(DataType::Utf8)),
            false,
        ),
        Field::new("ParamCurrencyID", DataType::Int16, false),
        Field::new(
            "OpenstatServiceName",
            DataType::Dictionary(Box::new(DataType::UInt16), Box::new(DataType::Utf8)),
            false,
        ),
        Field::new(
            "OpenstatCampaignID",
            DataType::Dictionary(Box::new(DataType::UInt16), Box::new(DataType::Utf8)),
            false,
        ),
        Field::new(
            "OpenstatAdID",
            DataType::Dictionary(Box::new(DataType::UInt16), Box::new(DataType::Utf8)),
            false,
        ),
        Field::new(
            "OpenstatSourceID",
            DataType::Dictionary(Box::new(DataType::UInt16), Box::new(DataType::Utf8)),
            false,
        ),
        Field::new(
            "UTMSource",
            DataType::Dictionary(Box::new(DataType::UInt16), Box::new(DataType::Utf8)),
            false,
        ),
        Field::new(
            "UTMMedium",
            DataType::Dictionary(Box::new(DataType::UInt16), Box::new(DataType::Utf8)),
            false,
        ),
        Field::new(
            "UTMCampaign",
            DataType::Dictionary(Box::new(DataType::UInt16), Box::new(DataType::Utf8)),
            false,
        ),
        Field::new(
            "UTMContent",
            DataType::Dictionary(Box::new(DataType::UInt16), Box::new(DataType::Utf8)),
            false,
        ),
        Field::new(
            "UTMTerm",
            DataType::Dictionary(Box::new(DataType::UInt16), Box::new(DataType::Utf8)),
            false,
        ),
        Field::new(
            "FromTag",
            DataType::Dictionary(Box::new(DataType::UInt16), Box::new(DataType::Utf8)),
            false,
        ),
        Field::new("HasGCLID", DataType::Int16, false),
        Field::new("RefererHash", DataType::Int64, false),
        Field::new("URLHash", DataType::Int64, false),
        Field::new("CLID", DataType::Int32, false),
    ])
}

pub fn generate_test_parquet() -> Vec<u8> {
    println!("{}", std::env::current_dir().unwrap().display().to_string());
    return fs::read("../../benchmark/data/nano_hits.parquet").unwrap();
}

async fn create_record_batch(batch_size: usize, i: usize) -> RecordBatch {
    let mut reader = get_test_reader().await;
    reader.batch_size = batch_size;
    let reader = reader
        .build(Arc::new(LiquidCachedFile::new(
            LiquidCacheMode::InMemoryLiquid,
            batch_size,
        )))
        .unwrap();

    let mut batches = reader.collect::<Vec<_>>().await;
    let batch = batches.remove(i).unwrap();
    return batch;
}

struct TestReader {
    data: Bytes,
    metadata: Arc<ParquetMetaData>,
}

impl AsyncFileReader for TestReader {
    fn get_bytes(&mut self, range: Range<usize>) -> BoxFuture<'_, Result<Bytes, ParquetError>> {
        futures::future::ready(Ok(self.data.slice(range))).boxed()
    }

    fn get_metadata(&mut self) -> BoxFuture<'_, Result<Arc<ParquetMetaData>, ParquetError>> {
        futures::future::ready(Ok(self.metadata.clone())).boxed()
    }
}

impl TestReader {
    fn new_dyn(data: Bytes) -> Box<dyn AsyncFileReader> {
        Box::new(TestReader {
            metadata: Arc::new(
                ParquetMetaDataReader::new()
                    .parse_and_finish(&data)
                    .unwrap(),
            ),
            data,
        })
    }
}

async fn get_test_reader() -> LiquidStreamBuilder {
    let file = generate_test_parquet();
    let data = Bytes::from(file);
    let mut async_reader = TestReader::new_dyn(data);

    let metadata = ArrowReaderMetadata::load_async(&mut async_reader, Default::default())
        .await
        .unwrap();
    let schema = Arc::clone(metadata.schema());

    println!("Schema:");
    for field in schema.fields() {
        println!("  {} - {}", field.name(), field.data_type());
    }

    let reader_schema = Arc::new(coerce_to_parquet_reader_types(&schema));

    let options = ArrowReaderOptions::new().with_schema(Arc::clone(&reader_schema));
    let metadata = ArrowReaderMetadata::try_new(Arc::clone(metadata.metadata()), options).unwrap();

    let builder = ParquetRecordBatchStreamBuilder::new_with_metadata(async_reader, metadata)
        .with_batch_size(8192);

    let liquid_builder =
        unsafe { ArrowReaderBuilderBridge::from_parquet(builder).into_liquid_builder() };

    let metadata = &liquid_builder.metadata;
    assert_eq!(metadata.num_row_groups(), 2);
    assert_eq!(metadata.file_metadata().num_rows(), 8192 * 3 + 10);
    liquid_builder
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

#[tokio::test]
async fn basic_stuff() {
    let builder = get_test_reader().await;
    let batch_size = builder.batch_size;
    let liquid_cache = LiquidCachedFile::new(LiquidCacheMode::InMemoryLiquid, batch_size);
    let reader = builder.build(Arc::new(liquid_cache)).unwrap();

    let schema = &reader.schema;
    //println!("{:?}", schema);
    //println!("{:?}", test_output_schema());
    //println!("===========================");
    assert_eq!(schema.as_ref(), &test_output_schema());

    let batches = reader
        .collect::<Vec<_>>()
        .await
        .into_iter()
        .map(|batch| batch.unwrap())
        .collect::<Vec<_>>();

    for (i, batch) in batches.iter().enumerate() {
        let expected = create_record_batch(batch_size, i).await;
        assert_batch_eq(&expected, batch);
    }
}

#[tokio::test]
async fn test_reading_with_projection() {
    let column_projections = vec![0, 3, 6, 8];
    let mut builder = get_test_reader().await;
    builder.projection = ProjectionMask::roots(
        builder.metadata.file_metadata().schema_descr(),
        column_projections.iter().cloned(),
    );
    let batch_size = builder.batch_size;
    let liquid_cache = LiquidCachedFile::new(LiquidCacheMode::InMemoryLiquid, batch_size);
    let reader = builder.build(Arc::new(liquid_cache)).unwrap();

    let batches = reader
        .collect::<Vec<_>>()
        .await
        .into_iter()
        .map(|batch| batch.unwrap())
        .collect::<Vec<_>>();

    for (i, batch) in batches.iter().enumerate() {
        let expected = create_record_batch(batch_size, i)
            .await
            .project(&column_projections)
            .unwrap();
        assert_batch_eq(&expected, batch);
    }
}

#[tokio::test]
async fn test_reading_warm() {
    let column_projections = vec![0, 3, 6, 8];
    let mut builder = get_test_reader().await;
    let batch_size = builder.batch_size;
    let liquid_cache = Arc::new(LiquidCachedFile::new(
        LiquidCacheMode::InMemoryLiquid,
        batch_size,
    ));
    builder.projection = ProjectionMask::roots(
        builder.metadata.file_metadata().schema_descr(),
        column_projections.iter().cloned(),
    );
    let reader = builder.build(liquid_cache.clone()).unwrap();
    let _batches = reader.collect::<Vec<_>>().await;

    let mut builder = get_test_reader().await;
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

    for (i, batch) in batches.iter().enumerate() {
        let expected = create_record_batch(batch_size, i)
            .await
            .project(&column_projections)
            .unwrap();
        assert_batch_eq(&expected, batch);
    }
}

struct TestPredicate {
    projection_mask: ProjectionMask,
    strategy: FilterStrategy,
}

impl TestPredicate {
    fn new(parquet_meta: &ParquetMetaData, column_id: usize, strategy: FilterStrategy) -> Self {
        Self {
            projection_mask: ProjectionMask::roots(parquet_meta.file_metadata().schema_descr(), [
                column_id,
            ]),
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
    let mut builder = get_test_reader().await;
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

    let liquid_cache = Arc::new(LiquidCachedFile::new(
        LiquidCacheMode::InMemoryLiquid,
        batch_size,
    ));

    let reader = builder.build(liquid_cache.clone()).unwrap();

    let batches = reader
        .collect::<Vec<_>>()
        .await
        .into_iter()
        .map(|batch| batch.unwrap())
        .collect::<Vec<_>>();

    for (i, batch) in batches.iter().enumerate() {
        let expected = create_record_batch(batch_size, i)
            .await
            .project(&projection)
            .unwrap();

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
    let mut builder = get_test_reader().await;
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

fn filter_record_batch(batch: &RecordBatch, mask: BooleanBuffer) -> RecordBatch {
    let mask = BooleanArray::new(mask, None);
    let filtered_columns = batch
        .columns()
        .iter()
        .map(|col| filter(col, &mask).unwrap())
        .collect::<Vec<_>>();

    RecordBatch::try_new(batch.schema(), filtered_columns).unwrap()
}
