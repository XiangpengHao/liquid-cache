use arrow::{
    array::{
        BooleanArray, Int8Array, Int16Array, Int32Array, Int64Array, StringViewBuilder, UInt8Array,
        UInt16Array, UInt32Array, UInt64Array,
    },
    datatypes::{DataType, Field, Schema},
    record_batch::RecordBatch,
};
use arrow_schema::ArrowError;
use bytes::Bytes;
use futures::future::BoxFuture;
use futures::{FutureExt, StreamExt};
use parquet::{
    arrow::{
        ArrowWriter, ParquetRecordBatchStreamBuilder, ProjectionMask,
        arrow_reader::{ArrowPredicate, ArrowReaderMetadata, ArrowReaderOptions},
        async_reader::AsyncFileReader,
    },
    errors::ParquetError,
    file::{
        metadata::{ParquetMetaData, ParquetMetaDataReader},
        properties::WriterProperties,
    },
};
use std::{ops::Range, sync::Arc};
use tempfile::NamedTempFile;

use crate::{
    LiquidCacheMode,
    cache::LiquidCachedFile,
    reader::runtime::{ArrowReaderBuilderBridge, LiquidRowFilter, LiquidStreamBuilder},
};

fn test_schema() -> Schema {
    Schema::new(vec![
        Field::new("u8_col", DataType::UInt8, false),
        Field::new("u16_col", DataType::UInt16, false),
        Field::new("u32_col", DataType::UInt32, false),
        Field::new("u64_col", DataType::UInt64, false),
        Field::new("i8_col", DataType::Int8, false),
        Field::new("i16_col", DataType::Int16, false),
        Field::new("i32_col", DataType::Int32, false),
        Field::new("i64_col", DataType::Int64, false),
        Field::new("string_view_col", DataType::Utf8View, false),
    ])
}

pub fn generate_test_parquet() -> NamedTempFile {
    let schema = test_schema();
    let temp_file = NamedTempFile::new().unwrap();
    let props = WriterProperties::builder()
        .set_max_row_group_size(16384) // 8192 * 2
        .build();

    let mut writer =
        ArrowWriter::try_new(temp_file.reopen().unwrap(), Arc::new(schema), Some(props)).unwrap();

    let mut batch_id = 0;
    for _ in 0..2 {
        for _ in 0..2 {
            let batch = create_record_batch(8192, batch_id);
            writer.write(&batch).unwrap();
            batch_id += 1;
        }
    }

    writer.close().unwrap();
    temp_file
}

fn create_record_batch(batch_size: usize, batch_id: usize) -> RecordBatch {
    let mut u8_builder = UInt8Array::builder(batch_size);
    let mut u16_builder = UInt16Array::builder(batch_size);
    let mut u32_builder = UInt32Array::builder(batch_size);
    let mut u64_builder = UInt64Array::builder(batch_size);
    let mut i8_builder = Int8Array::builder(batch_size);
    let mut i16_builder = Int16Array::builder(batch_size);
    let mut i32_builder = Int32Array::builder(batch_size);
    let mut i64_builder = Int64Array::builder(batch_size);
    let mut string_builder = StringViewBuilder::new();

    for i in batch_id * batch_size..(batch_id + 1) * batch_size {
        // Numeric values
        u8_builder.append_value((i % u8::MAX as usize) as u8);
        u16_builder.append_value(i as u16);
        u32_builder.append_value(i as u32);
        u64_builder.append_value(i as u64);
        i8_builder.append_value((i as i8).wrapping_neg());
        i16_builder.append_value(-(i as i16));
        i32_builder.append_value(-(i as i32));
        i64_builder.append_value(-(i as i64));

        // String values with varying lengths and repetitions
        let s = match i % 10 {
            0 => "short".to_string(),
            1 => "long_string_".repeat(50),
            _ => format!("value_{}", i % 100), // Repeating patterns
        };
        string_builder.append_value(s);
    }

    RecordBatch::try_new(Arc::new(test_schema()), vec![
        Arc::new(u8_builder.finish()),
        Arc::new(u16_builder.finish()),
        Arc::new(u32_builder.finish()),
        Arc::new(u64_builder.finish()),
        Arc::new(i8_builder.finish()),
        Arc::new(i16_builder.finish()),
        Arc::new(i32_builder.finish()),
        Arc::new(i64_builder.finish()),
        Arc::new(string_builder.finish()),
    ])
    .unwrap()
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
    let data = Bytes::from(std::fs::read(file.path()).unwrap());
    let mut async_reader = TestReader::new_dyn(data);

    let options = ArrowReaderOptions::new().with_page_index(true);
    let metadata = ArrowReaderMetadata::load_async(&mut async_reader, options)
        .await
        .unwrap();
    let builder = ParquetRecordBatchStreamBuilder::new_with_metadata(async_reader, metadata)
        .with_batch_size(8192);

    let liquid_builder =
        unsafe { ArrowReaderBuilderBridge::from_parquet(builder).into_liquid_builder() };

    let metadata = &liquid_builder.metadata;
    assert_eq!(metadata.num_row_groups(), 2);
    assert_eq!(metadata.file_metadata().num_rows(), 8192 * 2 * 2);
    liquid_builder
}

/// We could directly assert_eq!(left, right) but this is more debugging friendly
fn assert_batch_eq(left: &RecordBatch, right: &RecordBatch) {
    assert_eq!(left.num_rows(), right.num_rows());
    assert_eq!(left.schema(), right.schema());
    assert_eq!(left.columns().len(), right.columns().len());
    for (c_l, c_r) in left.columns().iter().zip(right.columns().iter()) {
        assert_eq!(c_l, c_r);
    }
}

#[tokio::test]
async fn basic_stuff() {
    let builder = get_test_reader().await;
    let batch_size = builder.batch_size;
    let liquid_cache = LiquidCachedFile::new(LiquidCacheMode::InMemoryLiquid, batch_size);
    let reader = builder.build(Arc::new(liquid_cache)).unwrap();

    let schema = &reader.schema;
    assert_eq!(schema.as_ref(), &test_schema());

    let batches = reader
        .collect::<Vec<_>>()
        .await
        .into_iter()
        .map(|batch| batch.unwrap())
        .collect::<Vec<_>>();

    for (i, batch) in batches.iter().enumerate() {
        let expected = create_record_batch(batch_size, i);
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

    let expected_schema = test_schema().project(&column_projections).unwrap();
    for (i, batch) in batches.iter().enumerate() {
        assert_eq!(batch.schema().as_ref(), &expected_schema);
        let expected = create_record_batch(batch_size, i)
            .project(&column_projections)
            .unwrap();
        assert_batch_eq(&expected, batch);
    }
}

struct TestPredicate {
    projection_mask: ProjectionMask,
}

impl TestPredicate {
    fn new(parquet_meta: Arc<ParquetMetaData>, projection: Vec<usize>) -> Self {
        Self {
            projection_mask: ProjectionMask::roots(
                parquet_meta.file_metadata().schema_descr(),
                projection.iter().cloned(),
            ),
        }
    }
}

impl ArrowPredicate for TestPredicate {
    fn evaluate(&mut self, batch: RecordBatch) -> Result<BooleanArray, ArrowError> {
        Ok(BooleanArray::from(vec![false; batch.num_rows()]))
    }

    fn projection(&self) -> &ProjectionMask {
        &self.projection_mask
    }
}

#[tokio::test]
async fn test_reading_with_filter() {
    let projection = vec![0, 3, 5, 6, 8];
    let mut builder = get_test_reader().await;
    builder.projection = ProjectionMask::roots(
        builder.metadata.file_metadata().schema_descr(),
        projection.iter().cloned(),
    );

    {
        let filter1 = TestPredicate::new(builder.metadata.clone(), vec![0]);
        let filter2 = TestPredicate::new(builder.metadata.clone(), vec![5]);
        let filter3 = TestPredicate::new(builder.metadata.clone(), vec![6]);
        let filters = vec![
            Box::new(filter1) as Box<dyn ArrowPredicate>,
            Box::new(filter2),
            Box::new(filter3),
        ];
        builder.filter = Some(LiquidRowFilter::new(filters));
    }
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
            .project(&projection)
            .unwrap();
        assert_batch_eq(&expected, batch);
    }
}
