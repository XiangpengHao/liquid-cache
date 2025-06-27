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
}
