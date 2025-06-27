use std::sync::Arc;

use bytes::{Buf, Bytes};
use parquet::{
    arrow::{ProjectionMask, arrow_reader::RowGroups, async_reader::AsyncFileReader},
    column::page::{PageIterator, PageReader},
    errors::ParquetError,
    file::{
        metadata::RowGroupMetaData,
        page_index::offset_index::OffsetIndexMetaData,
        reader::{ChunkReader, Length, SerializedPageReader},
    },
};

use super::reader::cached_page::{CachedPageReader, PredicatePageCache};
use crate::cache::{BatchID, LiquidCachedRowGroupRef};
use crate::reader::plantime::ParquetMetadataCacheReader;

/// An in-memory collection of column chunks
pub(super) struct InMemoryRowGroup<'a> {
    metadata: &'a RowGroupMetaData,
    offset_index: Option<&'a [OffsetIndexMetaData]>,
    column_chunks: Vec<Option<Arc<ColumnChunkData>>>,
    row_count: usize,
    cache: Arc<PredicatePageCache>,
    projection_to_cache: Option<ProjectionMask>,
    liquid_cache: LiquidCachedRowGroupRef,
}

impl<'a> InMemoryRowGroup<'a> {
    pub(super) fn new(
        metadata: &'a RowGroupMetaData,
        offset_index: Option<&'a [OffsetIndexMetaData]>,
        projection_to_cache: Option<ProjectionMask>,
        liquid_cache: LiquidCachedRowGroupRef,
    ) -> Self {
        Self {
            metadata,
            offset_index,
            column_chunks: vec![None; metadata.columns().len()],
            row_count: metadata.num_rows() as usize,
            projection_to_cache,
            cache: Arc::new(PredicatePageCache::new()),
            liquid_cache,
        }
    }
}

impl InMemoryRowGroup<'_> {
    /// Fetches the necessary column data into memory
    /// If any batch is not cached, fetches entire row group. If all cached, skips fetching.
    pub(crate) async fn fetch(
        &mut self,
        reader: &mut ParquetMetadataCacheReader,
        projection: &ProjectionMask,
    ) -> Result<(), parquet::errors::ParquetError> {
        // Check if we need to fetch any data
        for idx in 0..self.column_chunks.len() {
            if !projection.leaf_included(idx) {
                // don't read this column
                continue;
            }

            if self.column_chunks[idx].is_some() {
                // this column is already fetched
                continue;
            }

            if self.is_column_fully_cached(idx, &self.liquid_cache) {
                // this column is fully cached
                self.column_chunks[idx] = Some(Arc::new(ColumnChunkData::Cached {
                    length: self.metadata.column(idx).byte_range().1,
                }));
            } else {
                // this column is not fully cached, fetch it
                let (start, length) = self.metadata.column(idx).byte_range();
                let range = start..start + length;
                let data = reader.get_bytes(range).await?;
                let data_len = data.len();
                if data_len != length as usize {
                    return Err(ParquetError::General(format!(
                        "Data length mismatch: expected {length} bytes, got {data_len} bytes",
                    )));
                }
                self.column_chunks[idx] = Some(Arc::new(ColumnChunkData::Materialized {
                    offset: start,
                    data,
                }));
            }
        }
        Ok(())
    }

    /// Check if all batches for a column are cached
    fn is_column_fully_cached(
        &self,
        column_idx: usize,
        liquid_cache: &LiquidCachedRowGroupRef,
    ) -> bool {
        if let Some(cached_column) = liquid_cache.get_column(column_idx as u64) {
            let num_rows = self.row_count;
            let batch_size = cached_column.batch_size();
            let num_batches = num_rows.div_ceil(batch_size);

            // Check if all batches are cached
            for batch_idx in 0..num_batches {
                let batch_id = BatchID::from_row_id(batch_idx * batch_size, batch_size);
                if !cached_column.is_cached(batch_id) {
                    return false;
                }
            }
            true
        } else {
            // Column doesn't exist in cache, so it's not cached
            false
        }
    }
}

impl RowGroups for InMemoryRowGroup<'_> {
    fn num_rows(&self) -> usize {
        self.row_count
    }

    fn column_chunks(
        &self,
        i: usize,
    ) -> Result<Box<dyn PageIterator>, parquet::errors::ParquetError> {
        match &self.column_chunks[i] {
            None => Err(ParquetError::General(format!(
                "Invalid column index {i}, column was not fetched"
            ))),
            Some(data) => {
                let page_locations = self
                    .offset_index
                    // filter out empty offset indexes (old versions specified Some(vec![]) when no present)
                    .filter(|index| !index.is_empty())
                    .map(|index| index[i].page_locations.clone());

                let cached_reader = if let Some(projection_to_cache) = &self.projection_to_cache {
                    projection_to_cache.leaf_included(i)
                } else {
                    false
                };

                let page_reader: Box<dyn PageReader> = if cached_reader {
                    Box::new(CachedPageReader::new(
                        SerializedPageReader::new(
                            data.clone(),
                            self.metadata.column(i),
                            self.row_count,
                            page_locations,
                        )?,
                        self.cache.clone(),
                        i,
                    ))
                } else {
                    Box::new(SerializedPageReader::new(
                        data.clone(),
                        self.metadata.column(i),
                        self.row_count,
                        page_locations,
                    )?)
                };

                Ok(Box::new(ColumnChunkIterator {
                    reader: Some(Ok(page_reader)),
                }))
            }
        }
    }
}

/// An in-memory column chunk with prefetched data
#[derive(Clone)]
pub(super) enum ColumnChunkData {
    /// Data has been materialized into memory
    Materialized { offset: u64, data: Bytes },
    /// Data is available in cache, no need to materialize
    Cached { length: u64 },
}

impl ColumnChunkData {
    fn get(&self, start: u64) -> Result<Bytes, parquet::errors::ParquetError> {
        match self {
            ColumnChunkData::Materialized { offset, data } => {
                let start = start as usize - *offset as usize;
                Ok(data.slice(start..))
            }
            ColumnChunkData::Cached { .. } => {
                unreachable!("Cached column chunks should not be accessed directly")
            }
        }
    }
}

impl Length for ColumnChunkData {
    fn len(&self) -> u64 {
        match self {
            ColumnChunkData::Materialized { data, .. } => data.len() as u64,
            ColumnChunkData::Cached { length, .. } => *length,
        }
    }
}

impl ChunkReader for ColumnChunkData {
    type T = bytes::buf::Reader<Bytes>;

    fn get_read(&self, start: u64) -> Result<Self::T, parquet::errors::ParquetError> {
        Ok(self.get(start)?.reader())
    }

    fn get_bytes(&self, start: u64, length: usize) -> Result<Bytes, parquet::errors::ParquetError> {
        Ok(self.get(start)?.slice(..length))
    }
}

/// Implements [`PageIterator`] for a single column chunk, yielding a single [`PageReader`]
struct ColumnChunkIterator {
    reader: Option<Result<Box<dyn PageReader>, parquet::errors::ParquetError>>,
}

impl Iterator for ColumnChunkIterator {
    type Item = Result<Box<dyn PageReader>, parquet::errors::ParquetError>;

    fn next(&mut self) -> Option<Self::Item> {
        self.reader.take()
    }
}

impl PageIterator for ColumnChunkIterator {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        LiquidCache,
        cache::policies::DiscardPolicy,
        reader::{
            plantime::CachedMetaReaderFactory,
            runtime::{
                ArrowReaderBuilderBridge, liquid_stream::LiquidStreamBuilder,
                reader::build_cached_array_reader,
            },
        },
    };
    use arrow::array::AsArray;
    use datafusion::physical_plan::metrics::ExecutionPlanMetricsSet;
    use liquid_cache_common::{LiquidCacheMode, ParquetReaderSchema};
    use object_store::{ObjectStore, local::LocalFileSystem};
    use parquet::arrow::{
        ParquetRecordBatchStreamBuilder,
        arrow_reader::{ArrowReaderMetadata, ArrowReaderOptions},
    };
    use std::path::PathBuf;

    async fn get_test_stream_builder(batch_size: usize) -> LiquidStreamBuilder {
        let mut reader = {
            let local_fs = Arc::new(LocalFileSystem::new());
            let reader_factory = CachedMetaReaderFactory::new(local_fs.clone());
            let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("../../examples/nano_hits.parquet")
                .canonicalize()
                .unwrap();
            let file_meta = local_fs
                .head(&object_store::path::Path::parse(path.to_string_lossy()).unwrap())
                .await
                .unwrap();
            reader_factory.create_liquid_reader(
                0,
                file_meta.into(),
                None,
                &ExecutionPlanMetricsSet::new(),
            )
        };
        let reader_metadata = ArrowReaderMetadata::load_async(&mut reader, Default::default())
            .await
            .unwrap();
        let mut physical_file_schema = Arc::clone(reader_metadata.schema());
        physical_file_schema = ParquetReaderSchema::from(&physical_file_schema);
        let mut options = ArrowReaderOptions::new().with_page_index(true);
        options = options.with_schema(Arc::clone(&physical_file_schema));
        let reader_metadata =
            ArrowReaderMetadata::try_new(Arc::clone(reader_metadata.metadata()), options).unwrap();

        let builder =
            ParquetRecordBatchStreamBuilder::new_with_metadata(reader, reader_metadata.clone())
                .with_batch_size(batch_size)
                .with_row_groups(vec![0]);
        unsafe { ArrowReaderBuilderBridge::from_parquet(builder).into_liquid_builder() }
    }

    #[tokio::test]
    async fn test_cache_reset_after_fetch() {
        let tmp_dir = tempfile::tempdir().unwrap();
        let batch_size = 8192 * 2;
        let liquid_cache = LiquidCache::new(
            batch_size,
            batch_size,
            tmp_dir.path().to_path_buf(),
            LiquidCacheMode::Liquid {
                transcode_in_background: false,
            },
            Box::new(DiscardPolicy),
        );
        let liquid_cache_file = liquid_cache.register_or_get_file("whatever".into());

        let mut builder = get_test_stream_builder(batch_size).await;
        let fields = builder.fields.clone();

        let row_group_metadata = &builder.metadata.row_groups()[0];
        let value_cnt = row_group_metadata.num_rows() as usize;
        let liquid_cache_rg = liquid_cache_file.row_group(0);

        {
            let mut row_group =
                InMemoryRowGroup::new(&row_group_metadata, None, None, liquid_cache_rg.clone());

            row_group
                .fetch(&mut builder.input, &ProjectionMask::all())
                .await
                .unwrap();

            let mut array_reader = build_cached_array_reader(
                fields.as_ref().map(|f| f.as_ref()),
                &ProjectionMask::all(),
                &row_group,
                liquid_cache_rg.clone(),
            )
            .unwrap();

            array_reader.read_records(value_cnt).unwrap();
            array_reader.consume_batch().unwrap();
        }

        // by now everything should be cached
        {
            let mut row_group =
                InMemoryRowGroup::new(&row_group_metadata, None, None, liquid_cache_rg.clone());
            row_group
                .fetch(&mut builder.input, &ProjectionMask::all())
                .await
                .unwrap();
            let mut array_reader = build_cached_array_reader(
                fields.as_ref().map(|f| f.as_ref()),
                &ProjectionMask::all(),
                &row_group,
                liquid_cache_rg.clone(),
            )
            .unwrap();
            array_reader.read_records(value_cnt).unwrap();
            array_reader.consume_batch().unwrap();
        }
    }

    #[tokio::test]
    async fn test_partial_projection_with_mixed_cache_state() {
        let tmp_dir = tempfile::tempdir().unwrap();
        let batch_size = 8192 * 2;
        let liquid_cache = LiquidCache::new(
            batch_size,
            batch_size,
            tmp_dir.path().to_path_buf(),
            LiquidCacheMode::Liquid {
                transcode_in_background: false,
            },
            Box::new(DiscardPolicy),
        );
        let liquid_cache_file = liquid_cache.register_or_get_file("test_partial".into());

        let mut builder = get_test_stream_builder(batch_size).await;
        let fields = builder.fields.clone();
        let row_group_metadata = &builder.metadata.row_groups()[0];
        let value_cnt = row_group_metadata.num_rows() as usize;
        let liquid_cache_rg = liquid_cache_file.row_group(0);

        // Get the number of columns to create meaningful projections
        let num_columns = row_group_metadata.columns().len();
        assert!(num_columns >= 3, "Need at least 3 columns for this test");

        // Step 1: Pre-populate cache with only first 2 columns using full projection
        {
            let first_two_projection =
                ProjectionMask::roots(row_group_metadata.schema_descr(), vec![0, 1]);

            let mut row_group =
                InMemoryRowGroup::new(&row_group_metadata, None, None, liquid_cache_rg.clone());

            row_group
                .fetch(&mut builder.input, &first_two_projection)
                .await
                .unwrap();

            let mut array_reader = build_cached_array_reader(
                fields.as_ref().map(|f| f.as_ref()),
                &first_two_projection,
                &row_group,
                liquid_cache_rg.clone(),
            )
            .unwrap();

            array_reader.read_records(value_cnt).unwrap();
            array_reader.consume_batch().unwrap();
        }

        // Step 2: Verify that first 2 columns are cached
        for col_idx in 0..2 {
            assert!(
                liquid_cache_rg.get_column(col_idx as u64).is_some(),
                "Column {} should be in cache after step 1",
                col_idx
            );
        }

        // Step 3: Create a projection that includes cached (0,1) and non-cached (2) columns
        let mixed_projection = if num_columns > 2 {
            ProjectionMask::roots(row_group_metadata.schema_descr(), vec![0, 2])
        } else {
            ProjectionMask::roots(row_group_metadata.schema_descr(), vec![0, 1])
        };

        // Step 4: Test fetch with mixed cache state
        {
            let mut row_group =
                InMemoryRowGroup::new(&row_group_metadata, None, None, liquid_cache_rg.clone());

            // This should succeed - should use cached data for column 0 and fetch column 2
            row_group
                .fetch(&mut builder.input, &mixed_projection)
                .await
                .unwrap();

            let mut array_reader = build_cached_array_reader(
                fields.as_ref().map(|f| f.as_ref()),
                &mixed_projection,
                &row_group,
                liquid_cache_rg.clone(),
            )
            .unwrap();

            array_reader.read_records(value_cnt).unwrap();
            let batch = array_reader.consume_batch().unwrap();

            // Verify we got the expected number of columns
            if num_columns > 2 {
                assert_eq!(batch.as_struct().columns().len(), 2);
            }
        }

        // Step 5: Test accessing unfetched column should fail
        {
            let mut row_group =
                InMemoryRowGroup::new(&row_group_metadata, None, None, liquid_cache_rg.clone());

            // Only fetch column 0
            let single_col_projection =
                ProjectionMask::roots(row_group_metadata.schema_descr(), vec![0]);
            row_group
                .fetch(&mut builder.input, &single_col_projection)
                .await
                .unwrap();

            // Try to access unfetched column - should fail
            if num_columns > 1 {
                let result = row_group.column_chunks(1);
                assert!(result.is_err(), "Accessing unfetched column should fail");
                if let Err(err) = result {
                    assert!(err.to_string().contains("column was not fetched"));
                }
            }
        }
    }

    #[tokio::test]
    async fn test_predicate_projection_caching_flow() {
        let tmp_dir = tempfile::tempdir().unwrap();
        let batch_size = 8192 * 2;
        let liquid_cache = LiquidCache::new(
            batch_size,
            batch_size,
            tmp_dir.path().to_path_buf(),
            LiquidCacheMode::Liquid {
                transcode_in_background: false,
            },
            Box::new(DiscardPolicy),
        );
        let liquid_cache_file = liquid_cache.register_or_get_file("test_predicate".into());

        let mut builder = get_test_stream_builder(batch_size).await;
        let fields = builder.fields.clone();
        let row_group_metadata = &builder.metadata.row_groups()[0];
        let value_cnt = row_group_metadata.num_rows() as usize;
        let liquid_cache_rg = liquid_cache_file.row_group(0);

        let num_columns = row_group_metadata.columns().len();
        assert!(num_columns >= 3, "Need at least 3 columns for this test");

        // Simulate the liquid_stream.rs predicate flow:
        // 1. Create predicate projection (columns 0, 1)
        // 2. Create main projection (columns 1, 2)
        // 3. projection_to_cache should be intersection: (column 1)

        let predicate_projection =
            ProjectionMask::roots(row_group_metadata.schema_descr(), vec![0, 1]);

        let main_projection = ProjectionMask::roots(row_group_metadata.schema_descr(), vec![1, 2]);

        // Calculate projection_to_cache as intersection
        let mut projection_to_cache = predicate_projection.clone();
        projection_to_cache.intersect(&main_projection);

        // Step 1: Create row group with projection_to_cache (simulating filter scenario)
        let mut row_group = InMemoryRowGroup::new(
            &row_group_metadata,
            None,
            Some(projection_to_cache),
            liquid_cache_rg.clone(),
        );

        // Step 2: Fetch predicate columns first (simulating filter evaluation)
        row_group
            .fetch(&mut builder.input, &predicate_projection)
            .await
            .unwrap();

        // Build array reader for predicate columns
        let mut predicate_reader = build_cached_array_reader(
            fields.as_ref().map(|f| f.as_ref()),
            &predicate_projection,
            &row_group,
            liquid_cache_rg.clone(),
        )
        .unwrap();

        // Simulate predicate evaluation
        predicate_reader.read_records(value_cnt).unwrap();
        predicate_reader.consume_batch().unwrap();

        // Step 3: Fetch main projection columns (simulating final projection)
        row_group
            .fetch(&mut builder.input, &main_projection)
            .await
            .unwrap();

        let mut main_reader = build_cached_array_reader(
            fields.as_ref().map(|f| f.as_ref()),
            &main_projection,
            &row_group,
            liquid_cache_rg.clone(),
        )
        .unwrap();

        main_reader.read_records(value_cnt).unwrap();
        let final_batch = main_reader.consume_batch().unwrap();

        // Verify final result has correct number of columns
        assert_eq!(final_batch.as_struct().columns().len(), 2);

        // Step 4: Verify caching behavior - column 1 should be cached (in intersection)
        // while column 0 and 2 should only be cached if fetched
        assert!(
            liquid_cache_rg.get_column(1).is_some(),
            "Column 1 should be cached as it's in projection_to_cache"
        );
    }

    #[tokio::test]
    async fn test_error_handling_and_data_integrity() {
        let tmp_dir = tempfile::tempdir().unwrap();
        let batch_size = 8192 * 2;
        let liquid_cache = LiquidCache::new(
            batch_size,
            batch_size,
            tmp_dir.path().to_path_buf(),
            LiquidCacheMode::Liquid {
                transcode_in_background: false,
            },
            Box::new(DiscardPolicy),
        );
        let liquid_cache_file = liquid_cache.register_or_get_file("test_errors".into());

        let mut builder = get_test_stream_builder(batch_size).await;
        let fields = builder.fields.clone();
        let row_group_metadata = &builder.metadata.row_groups()[0];
        let liquid_cache_rg = liquid_cache_file.row_group(0);

        let num_columns = row_group_metadata.columns().len();

        // Test 1: Access unfetched but valid column index
        {
            let mut row_group =
                InMemoryRowGroup::new(&row_group_metadata, None, None, liquid_cache_rg.clone());

            // Fetch only column 0
            let single_projection =
                ProjectionMask::roots(row_group_metadata.schema_descr(), vec![0]);
            row_group
                .fetch(&mut builder.input, &single_projection)
                .await
                .unwrap();

            // Try to access column 1 which exists but wasn't fetched
            if num_columns > 1 {
                let result = row_group.column_chunks(1);
                assert!(
                    result.is_err(),
                    "Should fail when accessing unfetched column"
                );
                if let Err(err) = result {
                    assert!(err.to_string().contains("column was not fetched"));
                }
            }
        }

        // Test 2: Empty projection should work
        {
            let mut row_group =
                InMemoryRowGroup::new(&row_group_metadata, None, None, liquid_cache_rg.clone());

            let empty_projection = ProjectionMask::roots(row_group_metadata.schema_descr(), vec![]);

            // This should succeed without fetching anything
            row_group
                .fetch(&mut builder.input, &empty_projection)
                .await
                .unwrap();
        }

        // Test 3: Multiple fetches should be idempotent
        {
            let mut row_group =
                InMemoryRowGroup::new(&row_group_metadata, None, None, liquid_cache_rg.clone());

            let projection = ProjectionMask::roots(row_group_metadata.schema_descr(), vec![0]);

            // First fetch
            row_group
                .fetch(&mut builder.input, &projection)
                .await
                .unwrap();

            // Second fetch of same column should be no-op
            row_group
                .fetch(&mut builder.input, &projection)
                .await
                .unwrap();

            // Should still be able to access the column
            let result = row_group.column_chunks(0);
            assert!(
                result.is_ok(),
                "Should be able to access fetched column after multiple fetches"
            );
        }

        // Test 4: Verify is_column_fully_cached logic
        {
            let mut row_group =
                InMemoryRowGroup::new(&row_group_metadata, None, None, liquid_cache_rg.clone());

            // Before fetching, column should not be considered fully cached
            assert!(!row_group.is_column_fully_cached(0, &liquid_cache_rg));

            // After fetching and reading, check if caching logic works
            let projection = ProjectionMask::roots(row_group_metadata.schema_descr(), vec![0]);
            row_group
                .fetch(&mut builder.input, &projection)
                .await
                .unwrap();

            let mut array_reader = build_cached_array_reader(
                fields.as_ref().map(|f| f.as_ref()),
                &projection,
                &row_group,
                liquid_cache_rg.clone(),
            )
            .unwrap();

            array_reader
                .read_records(row_group_metadata.num_rows() as usize)
                .unwrap();
            array_reader.consume_batch().unwrap();

            // Note: The column might now be cached depending on the implementation
            // This test mainly ensures the is_column_fully_cached method doesn't panic
        }
    }
}
