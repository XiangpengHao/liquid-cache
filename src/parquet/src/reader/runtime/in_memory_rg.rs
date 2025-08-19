use std::sync::Arc;

use bytes::{Buf, Bytes};
use parquet::{
    arrow::{ProjectionMask, arrow_reader::RowGroups, async_reader::AsyncFileReader},
    column::page::{PageIterator, PageReader},
    errors::ParquetError,
    file::{
        metadata::RowGroupMetaData,
        page_index::offset_index::OffsetIndexMetaData,
        reader::{ChunkReader, Length},
    },
};

use super::parquet::cached_page::{CachedPageReader, PredicatePageCache};
use crate::{cache::BatchID, reader::plantime::ParquetMetadataCacheReader};
use crate::{cache::LiquidCachedRowGroupRef, reader::runtime::parquet::SerializedPageReader};

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

            let missing_ranges = get_missing_batch_ranges(self.row_count, idx, &self.liquid_cache);

            if missing_ranges.is_empty() {
                self.column_chunks[idx] = Some(Arc::new(ColumnChunkData::Cached {
                    length: self.metadata.column(idx).byte_range().1,
                }));
                continue;
            }
            if let Some(offset_index) = self.offset_index
                && let Some(column_offset_index) = offset_index.get(idx)
                && let Some(page_ranges) =
                    get_pages_for_row_ranges(column_offset_index, &missing_ranges, self.row_count)
            {
                // Calculate if fetching individual pages is more efficient than full column
                let (column_start, column_length) = self.metadata.column(idx).byte_range();
                let total_page_bytes: u64 = page_ranges.iter().map(|(_, len)| len).sum();
                // If we're fetching more than 80% of the column anyway, just fetch the whole thing
                if total_page_bytes * 100 / column_length > 80 {
                    let range = column_start..column_start + column_length;
                    let data = reader.get_bytes(range).await?;
                    if data.len() != column_length as usize {
                        return Err(ParquetError::General(format!(
                            "Data length mismatch: expected {column_length} bytes, got {} bytes",
                            data.len()
                        )));
                    }
                    self.column_chunks[idx] = Some(Arc::new(ColumnChunkData::Dense {
                        offset: column_start,
                        data,
                    }));
                } else {
                    // Fetch individual pages and consolidate them into a sparse buffer
                    let data = reader
                        .get_byte_ranges(
                            page_ranges
                                .iter()
                                .map(|(start, len)| *start..*start + *len)
                                .collect(),
                        )
                        .await?;
                    let mut page_data = Vec::new();
                    for ((page_offset, page_length), data) in page_ranges.into_iter().zip(data) {
                        assert_eq!(data.len(), page_length as usize);
                        page_data.push((page_offset, data));
                    }

                    self.column_chunks[idx] = Some(Arc::new(ColumnChunkData::Sparse {
                        length: column_length,
                        data: page_data,
                    }));
                }
            }

            let (start, length) = self.metadata.column(idx).byte_range();
            let range = start..start + length;
            let data = reader.get_bytes(range).await?;
            let data_len = data.len();
            if data_len != length as usize {
                return Err(ParquetError::General(format!(
                    "Data length mismatch: expected {length} bytes, got {data_len} bytes",
                )));
            }
            self.column_chunks[idx] = Some(Arc::new(ColumnChunkData::Dense {
                offset: start,
                data,
            }));
        }
        Ok(())
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
    /// Data has been materialized into memory (full column chunk)
    Dense { offset: u64, data: Bytes },
    /// Data is available in cache, no need to materialize
    Cached { length: u64 },
    /// Partial data has been materialized using page index (sparse pages)
    Sparse {
        length: u64,
        data: Vec<(u64, Bytes)>, // Sorted (page_offset, page_data) pairs
    },
}

impl ColumnChunkData {
    fn get(&self, start: u64) -> Result<Bytes, parquet::errors::ParquetError> {
        match self {
            ColumnChunkData::Dense { offset, data } => {
                let start = start as usize - *offset as usize;
                Ok(data.slice(start..))
            }
            ColumnChunkData::Cached { .. } => {
                unreachable!("Cached column chunks should not be accessed directly")
            }
            ColumnChunkData::Sparse { data, .. } => {
                // Use binary search to find the page containing the requested offset
                let idx = data.binary_search_by_key(&start, |(offset, _)| *offset);
                let idx = match idx {
                    Ok(idx) => idx,
                    Err(idx) => {
                        if idx == 0 {
                            return Err(ParquetError::General(format!(
                                "Requested offset {start} not found in sparse column data",
                            )));
                        }
                        idx - 1
                    }
                };

                let (page_offset, page_data) = &data[idx];
                if start >= *page_offset && start < *page_offset + page_data.len() as u64 {
                    let relative_start = start - page_offset;
                    Ok(page_data.slice(relative_start as usize..))
                } else {
                    Err(ParquetError::General(format!(
                        "Requested offset {start} not found in page at offset {page_offset}",
                    )))
                }
            }
        }
    }
}

impl Length for ColumnChunkData {
    fn len(&self) -> u64 {
        match self {
            ColumnChunkData::Dense { data, .. } => data.len() as u64,
            ColumnChunkData::Cached { length, .. } => *length,
            ColumnChunkData::Sparse { length, .. } => *length,
        }
    }
}

impl ChunkReader for ColumnChunkData {
    type T = bytes::buf::Reader<Bytes>;

    fn get_read(&self, start: u64) -> Result<Self::T, parquet::errors::ParquetError> {
        Ok(self.get(start)?.reader())
    }

    fn get_bytes(&self, start: u64, length: usize) -> Result<Bytes, parquet::errors::ParquetError> {
        match self {
            ColumnChunkData::Sparse { data, .. } => {
                // For sparse data, we need to handle cross-page reads
                let mut result = Vec::with_capacity(length);
                let mut current_offset = start;
                let mut remaining_length = length;

                // Find the starting page using binary search
                let mut page_idx =
                    match data.binary_search_by_key(&current_offset, |(offset, _)| *offset) {
                        Ok(idx) => idx,
                        Err(idx) => {
                            if idx == 0 {
                                return Err(ParquetError::General(format!(
                                    "Offset {current_offset} not found in sparse column data",
                                )));
                            }
                            idx - 1
                        }
                    };

                while remaining_length > 0 && page_idx < data.len() {
                    let (page_offset, page_bytes) = &data[page_idx];

                    // Check if current_offset is within this page
                    if current_offset >= *page_offset
                        && current_offset < *page_offset + page_bytes.len() as u64
                    {
                        let page_start = (current_offset - page_offset) as usize;
                        let available_in_page = page_bytes.len() - page_start;
                        let to_copy = remaining_length.min(available_in_page);

                        result.extend_from_slice(&page_bytes[page_start..page_start + to_copy]);
                        current_offset += to_copy as u64;
                        remaining_length -= to_copy;
                    } else if current_offset < *page_offset {
                        // Gap between pages - this shouldn't happen in well-formed sparse data
                        return Err(ParquetError::General(format!(
                            "Gap in sparse data: current offset {current_offset} but next page starts at {page_offset}",
                        )));
                    }

                    page_idx += 1;
                }

                if remaining_length > 0 {
                    return Err(ParquetError::General(format!(
                        "Insufficient data in sparse column: {remaining_length} bytes remaining",
                    )));
                }

                Ok(Bytes::from(result))
            }
            _ => Ok(self.get(start)?.slice(..length)),
        }
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

/// Get page ranges that contain the specified row ranges using page index
fn get_pages_for_row_ranges(
    column_offset_index: &OffsetIndexMetaData,
    row_ranges: &[(usize, usize)],
    row_count: usize,
) -> Option<Vec<(u64, u64)>> {
    let page_locations = &column_offset_index.page_locations;

    if page_locations.is_empty() {
        return None;
    }

    let mut page_ranges = Vec::new();
    let mut required_pages = std::collections::HashSet::new();

    // Find which pages contain the missing row ranges
    for &(start_row, end_row) in row_ranges {
        for (page_idx, page_location) in page_locations.iter().enumerate() {
            let page_first_row = page_location.first_row_index as usize;

            // Calculate page last row (next page's first row - 1, or total rows for last page)
            let page_last_row = if page_idx + 1 < page_locations.len() {
                page_locations[page_idx + 1].first_row_index as usize - 1
            } else {
                row_count - 1
            };

            // Check if this page overlaps with the missing row range
            if page_first_row < end_row && page_last_row >= start_row {
                required_pages.insert(page_idx);
            }
        }
    }

    // Convert page indexes to byte ranges
    for page_idx in required_pages {
        if let Some(page_location) = page_locations.get(page_idx) {
            let start_offset = page_location.offset as u64;
            let length = page_location.compressed_page_size as u64;
            page_ranges.push((start_offset, length));
        }
    }

    // Sort by offset for efficient fetching
    page_ranges.sort_by_key(|&(offset, _)| offset);

    if page_ranges.is_empty() {
        None
    } else {
        Some(page_ranges)
    }
}

/// Get the missing batch row ranges for a column
fn get_missing_batch_ranges(
    row_count: usize,
    column_idx: usize,
    liquid_cache: &LiquidCachedRowGroupRef,
) -> Vec<(usize, usize)> {
    let mut missing_ranges = Vec::new();

    if let Some(cached_column) = liquid_cache.get_column(column_idx as u64) {
        let num_rows = row_count;
        let batch_size = cached_column.batch_size();
        let num_batches = num_rows.div_ceil(batch_size);

        for batch_idx in 0..num_batches {
            let batch_start_row = batch_idx * batch_size;
            let batch_end_row = (batch_start_row + batch_size).min(num_rows);
            let batch_id = BatchID::from_row_id(batch_start_row, batch_size);

            if !cached_column.is_cached(batch_id) {
                missing_ranges.push((batch_start_row, batch_end_row));
            }
        }
    } else {
        // Column doesn't exist in cache, so all rows are missing
        missing_ranges.push((0, row_count));
    }

    missing_ranges
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        LiquidCache,
        reader::{
            plantime::CachedMetaReaderFactory,
            runtime::{
                ArrowReaderBuilderBridge, liquid_stream::LiquidStreamBuilder,
                parquet::build_cached_array_reader,
            },
        },
    };
    use arrow::array::{AsArray, Int32Array};
    use arrow::datatypes::{DataType, Field};
    use datafusion::physical_plan::metrics::ExecutionPlanMetricsSet;
    use liquid_cache_common::{LiquidCacheMode, ParquetReaderSchema};
    use liquid_cache_storage::policies::FiloPolicy;
    use object_store::{ObjectStore, local::LocalFileSystem};
    use parquet::arrow::{
        ParquetRecordBatchStreamBuilder,
        arrow_reader::{ArrowReaderMetadata, ArrowReaderOptions},
    };
    use parquet::file::page_index::offset_index::OffsetIndexMetaData;
    use parquet::format::PageLocation;
    use std::path::{Path, PathBuf};

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
        let options = ArrowReaderOptions::new().with_page_index(true);
        let reader_metadata = ArrowReaderMetadata::load_async(&mut reader, options)
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
            LiquidCacheMode::LiquidBlocking,
            Box::new(FiloPolicy::new()),
        );
        let liquid_cache_file = liquid_cache.register_or_get_file("whatever".into());

        let mut builder = get_test_stream_builder(batch_size).await;
        let fields = builder.fields.clone();

        let row_group_metadata = &builder.metadata.row_groups()[0];
        let value_cnt = row_group_metadata.num_rows() as usize;
        let liquid_cache_rg = liquid_cache_file.row_group(0);

        {
            let mut row_group =
                InMemoryRowGroup::new(row_group_metadata, None, None, liquid_cache_rg.clone());

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
                InMemoryRowGroup::new(row_group_metadata, None, None, liquid_cache_rg.clone());
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

    #[test]
    fn test_get_missing_batch_ranges() {
        let tmp_dir = tempfile::tempdir().unwrap();
        let batch_size = 32;
        let liquid_cache = LiquidCache::new(
            batch_size,
            usize::MAX,
            tmp_dir.path().to_path_buf(),
            LiquidCacheMode::LiquidBlocking,
            Box::new(FiloPolicy::new()),
        );
        let liquid_cache_file = liquid_cache.register_or_get_file("test".to_string());
        let liquid_cache_rg = liquid_cache_file.row_group(0);

        // Test Case 1: Column doesn't exist in cache - should return entire range
        let missing_ranges = get_missing_batch_ranges(100, 0, &liquid_cache_rg);
        assert_eq!(missing_ranges, vec![(0, 100)]);

        // Create a column in the cache
        let column = liquid_cache_rg
            .create_column(0, Arc::new(Field::new("test_col", DataType::Int32, false)));

        // Test Case 2: Column exists but no batches are cached - should return all batch ranges
        let missing_ranges = get_missing_batch_ranges(100, 0, &liquid_cache_rg);
        // 100 rows with batch_size 32: batches 0-31, 32-63, 64-95, 96-99
        assert_eq!(missing_ranges, vec![(0, 32), (32, 64), (64, 96), (96, 100)]);

        // Test Case 3: Cache some batches and test partial caching
        // Cache batch 0 (rows 0-31) and batch 2 (rows 64-95)
        let array_data = Arc::new(Int32Array::from((0..32).collect::<Vec<i32>>()));
        column
            .insert(BatchID::from_row_id(0, batch_size), array_data.clone())
            .unwrap();
        column
            .insert(BatchID::from_row_id(64, batch_size), array_data.clone())
            .unwrap();

        let missing_ranges = get_missing_batch_ranges(100, 0, &liquid_cache_rg);
        // Should be missing batch 1 (rows 32-63) and batch 3 (rows 96-99)
        assert_eq!(missing_ranges, vec![(32, 64), (96, 100)]);

        // Test Case 4: Cache all batches
        column
            .insert(BatchID::from_row_id(32, batch_size), array_data.clone())
            .unwrap();
        column
            .insert(BatchID::from_row_id(96, batch_size), array_data.clone())
            .unwrap();

        let missing_ranges = get_missing_batch_ranges(100, 0, &liquid_cache_rg);
        // Should have no missing ranges
        assert_eq!(missing_ranges, vec![]);

        // Test Case 5: Edge case - row count exactly divisible by batch size
        let missing_ranges = get_missing_batch_ranges(96, 0, &liquid_cache_rg);
        // Should have no missing ranges (all batches 0-31, 32-63, 64-95 are cached)
        assert_eq!(missing_ranges, vec![]);

        // Test Case 6: Edge case - row count smaller than batch size
        let missing_ranges = get_missing_batch_ranges(16, 0, &liquid_cache_rg);
        // Should have no missing ranges (batch 0 covers rows 0-31, which includes 0-15)
        assert_eq!(missing_ranges, vec![]);

        // Test Case 7: Test with a different column that doesn't exist
        let missing_ranges = get_missing_batch_ranges(100, 1, &liquid_cache_rg);
        assert_eq!(missing_ranges, vec![(0, 100)]);

        // Test Case 8: Create another column and test partial caching
        let column2 = liquid_cache_rg
            .create_column(1, Arc::new(Field::new("test_col2", DataType::Int32, false)));

        // Cache only the middle batch
        column2
            .insert(BatchID::from_row_id(32, batch_size), array_data.clone())
            .unwrap();

        let missing_ranges = get_missing_batch_ranges(100, 1, &liquid_cache_rg);
        // Should be missing first, third, and fourth batches
        assert_eq!(missing_ranges, vec![(0, 32), (64, 96), (96, 100)]);
    }

    #[test]
    fn test_get_pages_for_row_ranges_synthetic() {
        // Test Case 1: Create synthetic offset index with 4 pages
        let page_locations = vec![
            PageLocation {
                offset: 1000,
                compressed_page_size: 500,
                first_row_index: 0,
            },
            PageLocation {
                offset: 1500,
                compressed_page_size: 600,
                first_row_index: 100,
            },
            PageLocation {
                offset: 2100,
                compressed_page_size: 400,
                first_row_index: 200,
            },
            PageLocation {
                offset: 2500,
                compressed_page_size: 300,
                first_row_index: 300,
            },
        ];

        let offset_index = OffsetIndexMetaData {
            page_locations: page_locations.clone(),
            unencoded_byte_array_data_bytes: None,
        };

        let row_count = 400;

        // Test: Request rows 50-150 (should overlap with pages 0 and 1)
        let row_ranges = vec![(50, 150)];
        let result = get_pages_for_row_ranges(&offset_index, &row_ranges, row_count);

        let expected = vec![(1000, 500), (1500, 600)]; // Page 0 and Page 1
        assert_eq!(result, Some(expected));

        // Test: Request rows 90-110 (should need pages 0 and 1)
        // Page 0: rows 0-99, Page 1: rows 100-199
        // Range 90-110 overlaps with both pages
        let row_ranges = vec![(90, 110)];
        let result = get_pages_for_row_ranges(&offset_index, &row_ranges, row_count);

        let expected = vec![(1000, 500), (1500, 600)]; // Pages 0 and 1
        assert_eq!(result, Some(expected));

        // Test: Request rows within a single page (120-150, should only need page 1)
        let row_ranges = vec![(120, 150)];
        let result = get_pages_for_row_ranges(&offset_index, &row_ranges, row_count);

        let expected = vec![(1500, 600)]; // Only Page 1
        assert_eq!(result, Some(expected));

        // Test: Request rows 250-350 (should overlap with pages 2 and 3)
        let row_ranges = vec![(250, 350)];
        let result = get_pages_for_row_ranges(&offset_index, &row_ranges, row_count);

        let expected = vec![(2100, 400), (2500, 300)]; // Page 2 and Page 3
        assert_eq!(result, Some(expected));

        // Test: Request multiple non-contiguous ranges
        // Range (10, 30) needs Page 0
        // Range (180, 220) needs Page 1 (100-199) and Page 2 (200-299)
        // Range (350, 380) needs Page 3
        let row_ranges = vec![(10, 30), (180, 220), (350, 380)];
        let result = get_pages_for_row_ranges(&offset_index, &row_ranges, row_count);

        let expected = vec![(1000, 500), (1500, 600), (2100, 400), (2500, 300)]; // All pages
        assert_eq!(result, Some(expected));

        // Test: Request entire range (should get all pages)
        let row_ranges = vec![(0, 400)];
        let result = get_pages_for_row_ranges(&offset_index, &row_ranges, row_count);

        let expected = vec![(1000, 500), (1500, 600), (2100, 400), (2500, 300)]; // All pages
        assert_eq!(result, Some(expected));

        // Test: Request range that doesn't overlap with any pages
        let row_ranges = vec![(500, 600)];
        let result = get_pages_for_row_ranges(&offset_index, &row_ranges, row_count);

        assert_eq!(result, None);

        // Test: Empty page locations
        let empty_offset_index = OffsetIndexMetaData {
            page_locations: vec![],
            unencoded_byte_array_data_bytes: None,
        };
        let row_ranges = vec![(0, 100)];
        let result = get_pages_for_row_ranges(&empty_offset_index, &row_ranges, row_count);

        assert_eq!(result, None);
    }

    #[tokio::test]
    async fn test_get_pages_for_row_ranges_real_data() {
        // Test with real offset index from nano_hits.parquet
        let builder = get_test_stream_builder(8192).await;

        let offset_indexes = builder.metadata.offset_index().unwrap();
        // Check if we have offset index data
        let row_group_metadata = &builder.metadata.row_groups()[0];
        let row_count = row_group_metadata.num_rows() as usize;

        // Test with the first column's offset index
        let column_offset_index = &offset_indexes[0][2]; // title column

        println!(
            "Testing with {} pages from nano_hits.parquet",
            column_offset_index.page_locations.len()
        );

        // Test: Request first 100 rows
        let row_ranges = vec![(0, 100)];
        let result = get_pages_for_row_ranges(column_offset_index, &row_ranges, row_count);

        // Should get some pages (exact count depends on page size in the file)
        assert!(result.is_some());
        let pages = result.unwrap();
        assert!(!pages.is_empty());

        // Verify pages are sorted by offset
        for i in 1..pages.len() {
            assert!(
                pages[i - 1].0 <= pages[i].0,
                "Pages should be sorted by offset"
            );
        }

        // Test: Request middle portion of data
        let mid_row = row_count / 2;
        let row_ranges = vec![(mid_row, mid_row + 50)];
        let result = get_pages_for_row_ranges(column_offset_index, &row_ranges, row_count);

        assert!(result.is_some());

        // Test: Request last few rows
        let row_ranges = vec![(mid_row, row_count)];
        let result = get_pages_for_row_ranges(column_offset_index, &row_ranges, row_count);
        let pages = result.unwrap();

        // Verify pages are sorted and unique
        for i in 1..pages.len() {
            assert!(
                pages[i - 1].0 < pages[i].0,
                "Pages should be sorted and unique"
            );
        }

        // Verify all page ranges are valid
        for (offset, length) in &pages {
            assert!(*length > 0, "Page length should be positive");
            assert!(*offset > 0, "Page offset should be positive");
        }
    }

    fn setup_test_lq_cache(
        batch_size: usize,
        dir: &Path,
    ) -> (LiquidCache, LiquidCachedRowGroupRef) {
        let liquid_cache = LiquidCache::new(
            batch_size,
            usize::MAX,
            dir.to_path_buf(),
            LiquidCacheMode::LiquidBlocking,
            Box::new(FiloPolicy::new()),
        );
        let liquid_cache_file = liquid_cache.register_or_get_file("test_file".to_string());
        let liquid_cache_rg = liquid_cache_file.row_group(0);
        (liquid_cache, liquid_cache_rg)
    }

    #[tokio::test]
    async fn test_in_memory_rg_end_to_end_cache() {
        let tmp_dir = tempfile::tempdir().unwrap();
        let batch_size = 1024; // Smaller batch size to ensure multiple batches
        let (_liquid_cache, liquid_cache_rg) = setup_test_lq_cache(batch_size, tmp_dir.path());
        let mut builder = get_test_stream_builder(batch_size).await;
        let fields = builder.fields.clone();

        let row_group_metadata = &builder.metadata.row_groups()[0];

        let value_cnt = row_group_metadata.num_rows() as usize;
        let mut row_group =
            InMemoryRowGroup::new(row_group_metadata, None, None, liquid_cache_rg.clone());
        let mask = ProjectionMask::leaves(builder.metadata.file_metadata().schema_descr(), vec![2]); // title column

        row_group.fetch(&mut builder.input, &mask).await.unwrap();

        let mut array_reader = build_cached_array_reader(
            fields.as_ref().map(|f| f.as_ref()),
            &mask,
            &row_group,
            liquid_cache_rg.clone(),
        )
        .unwrap();

        let mut baseline_results = Vec::new();
        for _i in (0..value_cnt).step_by(batch_size) {
            array_reader.read_records(batch_size).unwrap();
            let array = array_reader.consume_batch().unwrap();
            baseline_results.push(array);
        }

        // ===== Now  insert some arrays to the cache =====
        let (_liquid_cache, liquid_cache_rg) = setup_test_lq_cache(batch_size, tmp_dir.path());
        let mut builder = get_test_stream_builder(batch_size).await;
        let column = liquid_cache_rg.create_column(
            2,
            Arc::new(Field::new(
                "Title",
                DataType::Dictionary(Box::new(DataType::UInt16), Box::new(DataType::Utf8)),
                false,
            )),
        );
        for i in (0..value_cnt).step_by(2 * batch_size) {
            let batch_id = BatchID::from_row_id(i, batch_size);
            let array = baseline_results[i / batch_size].clone();
            let inner_array = array.as_struct().column(0).clone();
            column.insert(batch_id, inner_array).unwrap();
        }
        let mut row_group =
            InMemoryRowGroup::new(row_group_metadata, None, None, liquid_cache_rg.clone());
        row_group
            .fetch(&mut builder.input, &ProjectionMask::all())
            .await
            .unwrap();

        let mut array_reader = build_cached_array_reader(
            fields.as_ref().map(|f| f.as_ref()),
            &mask,
            &row_group,
            liquid_cache_rg.clone(),
        )
        .unwrap();
        let mut mix_cache_results = Vec::new();
        for _i in (0..value_cnt).step_by(batch_size) {
            array_reader.read_records(batch_size).unwrap();
            let array = array_reader.consume_batch().unwrap();
            mix_cache_results.push(array);
        }
        assert_eq!(baseline_results, mix_cache_results);

        // ===== Now test fully cached =====
        let mut array_reader = build_cached_array_reader(
            fields.as_ref().map(|f| f.as_ref()),
            &mask,
            &row_group,
            liquid_cache_rg.clone(),
        )
        .unwrap();
        let mut full_cache_results = Vec::new();
        for _i in (0..value_cnt).step_by(batch_size) {
            array_reader.read_records(batch_size).unwrap();
            let array = array_reader.consume_batch().unwrap();
            full_cache_results.push(array);
        }

        assert_eq!(baseline_results, full_cache_results);
    }
}
