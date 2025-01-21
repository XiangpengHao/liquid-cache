use std::{
    collections::VecDeque,
    fmt::Formatter,
    pin::Pin,
    sync::Arc,
    task::{Context, Poll},
};

use arrow::array::RecordBatch;
use arrow_schema::{DataType, Fields, Schema, SchemaRef};
use cached_array_reader::build_array_reader;
use futures::{future::BoxFuture, ready, FutureExt, Stream};
use in_memory_rg::InMemoryRowGroup;
use parquet::{
    arrow::{
        arrow_reader::{ArrowPredicate, RowSelection, RowSelector},
        async_reader::AsyncFileReader,
        ProjectionMask,
    },
    errors::ParquetError,
    file::metadata::ParquetMetaData,
};
pub(crate) use parquet_bridge::ArrowReaderBuilderBridge;
use parquet_bridge::{
    intersect_projection_mask, limit_row_selection, offset_row_selection, union_projection_mask,
    ParquetField,
};
use record_batch_reader::LiquidRecordBatchReader;

use crate::liquid_parquet::cache::LiquidCacheRef;

mod cached_array_reader;
mod in_memory_rg;
mod parquet_bridge;
mod record_batch_reader;

pub struct LiquidRowFilter {
    pub(crate) predicates: Vec<Box<dyn ArrowPredicate>>,
}

impl LiquidRowFilter {
    pub fn new(predicates: Vec<Box<dyn ArrowPredicate>>) -> Self {
        Self { predicates }
    }
}

type ReadResult = Result<(ReaderFactory, Option<LiquidRecordBatchReader>), ParquetError>;

struct ReaderFactory {
    metadata: Arc<ParquetMetaData>,

    fields: Option<Arc<ParquetField>>,

    input: Box<dyn AsyncFileReader>,

    filter: Option<LiquidRowFilter>,

    limit: Option<usize>,

    offset: Option<usize>,

    liquid_cache: LiquidCacheRef,
}

impl ReaderFactory {
    /// Reads the next row group with the provided `selection`, `projection` and `batch_size`
    ///
    /// Note: this captures self so that the resulting future has a static lifetime
    async fn read_row_group(
        mut self,
        row_group_idx: usize,
        selection: Option<RowSelection>,
        projection: ProjectionMask,
        batch_size: usize,
    ) -> ReadResult {
        let meta = self.metadata.row_group(row_group_idx);
        let offset_index = self
            .metadata
            .offset_index()
            .filter(|index| index.first().map(|v| !v.is_empty()).unwrap_or(false))
            .map(|x| x[row_group_idx].as_slice());

        let mut predicate_projection: Option<ProjectionMask> = None;
        if let Some(filter) = self.filter.as_mut() {
            for predicate in filter.predicates.iter_mut() {
                let p_projection = predicate.projection();
                if let Some(ref mut p) = predicate_projection {
                    union_projection_mask(p, p_projection);
                } else {
                    predicate_projection = Some(p_projection.clone());
                }
            }
        }
        let projection_to_cache = predicate_projection.map(|mut p| {
            intersect_projection_mask(&mut p, &projection);
            p
        });

        let mut row_group = InMemoryRowGroup::new(meta, offset_index, projection_to_cache);

        let mut selection =
            selection.unwrap_or_else(|| vec![RowSelector::select(row_group.row_count)].into());

        let mut filter_readers = Vec::new();
        if let Some(filter) = self.filter.as_mut() {
            for predicate in filter.predicates.iter_mut() {
                if !selection.selects_any() {
                    return Ok((self, None));
                }

                let p_projection = predicate.projection();
                row_group
                    .fetch(&mut self.input, p_projection, Some(&selection))
                    .await?;

                let array_reader = build_array_reader(
                    self.fields.as_deref(),
                    p_projection,
                    &row_group,
                    row_group_idx,
                    self.liquid_cache.clone(),
                )?;
                filter_readers.push(array_reader);
            }
        }

        let rows_before = selection.row_count();

        if rows_before == 0 {
            return Ok((self, None));
        }

        if let Some(offset) = self.offset {
            selection = offset_row_selection(selection, offset);
        }

        if let Some(limit) = self.limit {
            selection = limit_row_selection(selection, limit);
        }

        let rows_after = selection.row_count();

        // Update offset if necessary
        if let Some(offset) = &mut self.offset {
            // Reduction is either because of offset or limit, as limit is applied
            // after offset has been "exhausted" can just use saturating sub here
            *offset = offset.saturating_sub(rows_before - rows_after)
        }

        if rows_after == 0 {
            return Ok((self, None));
        }

        if let Some(limit) = &mut self.limit {
            *limit -= rows_after;
        }

        row_group
            .fetch(&mut self.input, &projection, Some(&selection))
            .await?;

        let array_reader = build_array_reader(
            self.fields.as_deref(),
            &projection,
            &row_group,
            row_group_idx,
            self.liquid_cache.clone(),
        )?;

        let reader = LiquidRecordBatchReader::new(
            batch_size,
            array_reader,
            selection,
            filter_readers,
            self.filter.take(),
            self.liquid_cache.clone(),
        );

        Ok((self, Some(reader)))
    }
}

enum StreamState {
    /// At the start of a new row group, or the end of the parquet stream
    Init,
    /// Decoding a batch
    Decoding(LiquidRecordBatchReader),
    /// Reading data from input
    Reading(BoxFuture<'static, ReadResult>),
    /// Error
    Error,
}

impl std::fmt::Debug for StreamState {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            StreamState::Init => write!(f, "StreamState::Init"),
            StreamState::Decoding(_) => write!(f, "StreamState::Decoding"),
            StreamState::Reading(_) => write!(f, "StreamState::Reading"),
            StreamState::Error => write!(f, "StreamState::Error"),
        }
    }
}

pub struct LiquidStreamBuilder {
    pub(crate) input: Box<dyn AsyncFileReader>,

    pub(crate) metadata: Arc<ParquetMetaData>,

    pub(crate) fields: Option<Arc<ParquetField>>,

    pub(crate) batch_size: usize,

    pub(crate) row_groups: Option<Vec<usize>>,

    pub(crate) projection: ProjectionMask,

    pub(crate) filter: Option<LiquidRowFilter>,

    pub(crate) selection: Option<RowSelection>,

    pub(crate) limit: Option<usize>,

    pub(crate) offset: Option<usize>,
}

impl LiquidStreamBuilder {
    pub fn with_row_filter(mut self, filter: LiquidRowFilter) -> Self {
        self.filter = Some(filter);
        self
    }

    pub fn build(self, liquid_cache: LiquidCacheRef) -> Result<LiquidStream, ParquetError> {
        let num_row_groups = self.metadata.row_groups().len();

        let row_groups: VecDeque<usize> = match self.row_groups {
            Some(row_groups) => {
                if let Some(col) = row_groups.iter().find(|x| **x >= num_row_groups) {
                    return Err(ParquetError::ArrowError(format!(
                        "row group {} out of bounds 0..{}",
                        col, num_row_groups
                    )));
                }
                row_groups.into()
            }
            None => (0..self.metadata.row_groups().len()).collect(),
        };

        let batch_size = self
            .batch_size
            .min(self.metadata.file_metadata().num_rows() as usize);

        let reader = ReaderFactory {
            input: self.input,
            filter: self.filter,
            metadata: self.metadata.clone(),
            fields: self.fields,
            limit: self.limit,
            offset: self.offset,
            liquid_cache,
        };

        // Ensure schema of ParquetRecordBatchStream respects projection, and does
        // not store metadata (same as for ParquetRecordBatchReader and emitted RecordBatches)
        let projected_fields = match reader.fields.as_deref().map(|pf| &pf.arrow_type) {
            Some(DataType::Struct(fields)) => {
                fields.filter_leaves(|idx, _| self.projection.leaf_included(idx))
            }
            None => Fields::empty(),
            _ => unreachable!("Must be Struct for root type"),
        };
        let schema = Arc::new(Schema::new(projected_fields));

        Ok(LiquidStream {
            metadata: self.metadata.clone(),
            schema: schema.clone(),
            row_groups,
            projection: self.projection,
            batch_size,
            selection: self.selection,
            reader: Some(reader),
            state: StreamState::Init,
        })
    }
}

pub struct LiquidStream {
    metadata: Arc<ParquetMetaData>,

    schema: SchemaRef,

    row_groups: VecDeque<usize>,

    projection: ProjectionMask,

    batch_size: usize,

    selection: Option<RowSelection>,

    /// This is an option so it can be moved into a future
    reader: Option<ReaderFactory>,

    state: StreamState,
}

impl std::fmt::Debug for LiquidStream {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ParquetRecordBatchStream")
            .field("metadata", &self.metadata)
            .field("schema", &self.schema)
            .field("batch_size", &self.batch_size)
            .field("projection", &self.projection)
            .field("state", &self.state)
            .finish()
    }
}

impl Stream for LiquidStream {
    type Item = Result<RecordBatch, parquet::errors::ParquetError>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        loop {
            match &mut self.state {
                StreamState::Decoding(batch_reader) => match batch_reader.next() {
                    Some(Ok(batch)) => {
                        return Poll::Ready(Some(Ok(batch)));
                    }
                    Some(Err(e)) => {
                        self.state = StreamState::Error;
                        return Poll::Ready(Some(Err(ParquetError::ArrowError(e.to_string()))));
                    }
                    None => {
                        // this is ugly, but works for now.
                        let filter = batch_reader.take_filter();
                        self.reader.as_mut().unwrap().filter = filter;
                        self.state = StreamState::Init
                    }
                },
                StreamState::Init => {
                    let row_group_idx = match self.row_groups.pop_front() {
                        Some(idx) => idx,
                        None => return Poll::Ready(None),
                    };

                    let reader = self.reader.take().expect("lost reader");

                    let row_count = self.metadata.row_group(row_group_idx).num_rows() as usize;

                    let selection = self.selection.as_mut().map(|s| s.split_off(row_count));

                    let fut = reader
                        .read_row_group(
                            row_group_idx,
                            selection,
                            self.projection.clone(),
                            self.batch_size,
                        )
                        .boxed();

                    self.state = StreamState::Reading(fut)
                }
                StreamState::Reading(f) => match ready!(f.poll_unpin(cx)) {
                    Ok((reader_factory, maybe_reader)) => {
                        self.reader = Some(reader_factory);
                        match maybe_reader {
                            // Read records from [`ParquetRecordBatchReader`]
                            Some(reader) => self.state = StreamState::Decoding(reader),
                            // All rows skipped, read next row group
                            None => self.state = StreamState::Init,
                        }
                    }
                    Err(e) => {
                        self.state = StreamState::Error;
                        return Poll::Ready(Some(Err(e)));
                    }
                },
                StreamState::Error => return Poll::Ready(None), // Ends the stream as error happens.
            }
        }
    }
}
