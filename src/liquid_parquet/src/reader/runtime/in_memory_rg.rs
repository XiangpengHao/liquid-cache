use std::sync::Arc;

use bytes::{Buf, Bytes};
use parquet::{
    arrow::{
        arrow_reader::{RowGroups, RowSelection},
        async_reader::AsyncFileReader,
        ProjectionMask,
    },
    column::page::{PageIterator, PageReader},
    errors::ParquetError,
    file::{
        metadata::RowGroupMetaData,
        page_index::offset_index::OffsetIndexMetaData,
        reader::{ChunkReader, Length, SerializedPageReader},
    },
};

use super::record_batch_reader::{CachedPageReader, PredicatePageCache};

/// An in-memory collection of column chunks
pub(super) struct InMemoryRowGroup<'a> {
    pub(super) metadata: &'a RowGroupMetaData,
    pub(super) offset_index: Option<&'a [OffsetIndexMetaData]>,
    pub(super) column_chunks: Vec<Option<Arc<ColumnChunkData>>>,
    pub(super) row_count: usize,
    pub(super) cache: Arc<PredicatePageCache>,
    pub(super) projection_to_cache: Option<ProjectionMask>,
}

impl<'a> InMemoryRowGroup<'a> {
    pub(super) fn new(
        metadata: &'a RowGroupMetaData,
        offset_index: Option<&'a [OffsetIndexMetaData]>,
        projection_to_cache: Option<ProjectionMask>,
    ) -> Self {
        Self {
            metadata,
            offset_index,
            column_chunks: vec![None; metadata.columns().len()],
            row_count: metadata.num_rows() as usize,
            projection_to_cache,
            cache: Arc::new(PredicatePageCache::new()),
        }
    }
}

impl InMemoryRowGroup<'_> {
    /// Fetches the necessary column data into memory
    pub(crate) async fn fetch<T: AsyncFileReader + Send>(
        &mut self,
        input: &mut T,
        projection: &ProjectionMask,
        selection: Option<&RowSelection>,
    ) -> Result<(), parquet::errors::ParquetError> {
        if let Some((selection, offset_index)) = selection.zip(self.offset_index) {
            // If we have a `RowSelection` and an `OffsetIndex` then only fetch pages required for the
            // `RowSelection`
            let mut page_start_offsets: Vec<Vec<usize>> = vec![];

            let fetch_ranges = self
                .column_chunks
                .iter()
                .zip(self.metadata.columns())
                .enumerate()
                .filter(|&(idx, (chunk, _chunk_meta))| {
                    chunk.is_none() && projection.leaf_included(idx)
                })
                .flat_map(|(idx, (_chunk, chunk_meta))| {
                    // If the first page does not start at the beginning of the column,
                    // then we need to also fetch a dictionary page.
                    let mut ranges = vec![];
                    let (start, _len) = chunk_meta.byte_range();
                    match offset_index[idx].page_locations.first() {
                        Some(first) if first.offset as u64 != start => {
                            ranges.push(start as usize..first.offset as usize);
                        }
                        _ => (),
                    }

                    ranges.extend(selection.scan_ranges(&offset_index[idx].page_locations));
                    page_start_offsets.push(ranges.iter().map(|range| range.start).collect());

                    ranges
                })
                .collect();

            let mut chunk_data = input.get_byte_ranges(fetch_ranges).await?.into_iter();
            let mut page_start_offsets = page_start_offsets.into_iter();

            for (idx, chunk) in self.column_chunks.iter_mut().enumerate() {
                if chunk.is_some() || !projection.leaf_included(idx) {
                    continue;
                }

                if let Some(offsets) = page_start_offsets.next() {
                    let mut chunks = Vec::with_capacity(offsets.len());
                    for _ in 0..offsets.len() {
                        chunks.push(chunk_data.next().unwrap());
                    }

                    *chunk = Some(Arc::new(ColumnChunkData::Sparse {
                        length: self.metadata.column(idx).byte_range().1 as usize,
                        data: offsets.into_iter().zip(chunks.into_iter()).collect(),
                    }))
                }
            }
        } else {
            let fetch_ranges = self
                .column_chunks
                .iter()
                .enumerate()
                .filter(|&(idx, chunk)| chunk.is_none() && projection.leaf_included(idx))
                .map(|(idx, _chunk)| {
                    let column = self.metadata.column(idx);
                    let (start, length) = column.byte_range();
                    start as usize..(start + length) as usize
                })
                .collect();

            let mut chunk_data = input.get_byte_ranges(fetch_ranges).await?.into_iter();

            for (idx, chunk) in self.column_chunks.iter_mut().enumerate() {
                if chunk.is_some() || !projection.leaf_included(idx) {
                    continue;
                }

                if let Some(data) = chunk_data.next() {
                    *chunk = Some(Arc::new(ColumnChunkData::Dense {
                        offset: self.metadata.column(idx).byte_range().0 as usize,
                        data,
                    }));
                }
            }
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

/// An in-memory column chunk
#[derive(Clone)]
pub(super) enum ColumnChunkData {
    /// Column chunk data representing only a subset of data pages
    Sparse {
        /// Length of the full column chunk
        length: usize,
        /// Set of data pages included in this sparse chunk. Each element is a tuple
        /// of (page offset, page data)
        data: Vec<(usize, Bytes)>,
    },
    /// Full column chunk and its offset
    Dense { offset: usize, data: Bytes },
}

impl ColumnChunkData {
    fn get(&self, start: u64) -> Result<Bytes, parquet::errors::ParquetError> {
        match &self {
            ColumnChunkData::Sparse { data, .. } => data
                .binary_search_by_key(&start, |(offset, _)| *offset as u64)
                .map(|idx| data[idx].1.clone())
                .map_err(|_| {
                    ParquetError::General(format!(
                        "Invalid offset in sparse column chunk data: {start}"
                    ))
                }),
            ColumnChunkData::Dense { offset, data } => {
                let start = start as usize - *offset;
                Ok(data.slice(start..))
            }
        }
    }
}

impl Length for ColumnChunkData {
    fn len(&self) -> u64 {
        match &self {
            ColumnChunkData::Sparse { length, .. } => *length as u64,
            ColumnChunkData::Dense { data, .. } => data.len() as u64,
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
