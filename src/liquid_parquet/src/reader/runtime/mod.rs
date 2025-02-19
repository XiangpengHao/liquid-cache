use crate::{LiquidCacheMode, cache::LiquidCachedFileRef, liquid_array::LiquidArrayRef};
use arrow::array::{BooleanArray, RecordBatch};
use arrow_schema::{ArrowError, DataType, Fields, Schema, SchemaRef};
use futures::{FutureExt, Stream, future::BoxFuture, ready};
use in_memory_rg::InMemoryRowGroup;
use parquet::{
    arrow::{
        ProjectionMask,
        arrow_reader::{ArrowPredicate, RowSelection, RowSelector},
    },
    errors::ParquetError,
    file::metadata::ParquetMetaData,
};
pub(crate) use parquet_bridge::ArrowReaderBuilderBridge;
use parquet_bridge::{
    ParquetField, intersect_projection_mask, limit_row_selection, offset_row_selection,
    union_projection_mask,
};
use reader::LiquidBatchReader;
use reader::build_cached_array_reader;
use std::{
    collections::VecDeque,
    fmt::Formatter,
    pin::Pin,
    sync::Arc,
    task::{Context, Poll},
};
use tokio::sync::Mutex;

use super::plantime::{ParquetMetadataCacheReader, coerce_from_reader_to_liquid_types};
mod in_memory_rg;
mod parquet_bridge;
mod reader;
mod utils;

pub trait LiquidPredicate: ArrowPredicate {
    fn evaluate_liquid(&mut self, array: &LiquidArrayRef) -> Result<BooleanArray, ArrowError>;
    fn evaluate_arrow(&mut self, array: RecordBatch) -> Result<BooleanArray, ArrowError> {
        self.evaluate(array)
    }
}

pub struct LiquidRowFilter {
    pub(crate) predicates: Vec<Box<dyn LiquidPredicate>>,
}

impl LiquidRowFilter {
    pub fn new(predicates: Vec<Box<dyn LiquidPredicate>>) -> Self {
        Self { predicates }
    }
}

type ReadResult = Result<(ReaderFactory, Option<LiquidBatchReader>), ParquetError>;

struct ReaderFactory {
    metadata: Arc<ParquetMetaData>,

    fields: Option<Arc<ParquetField>>,

    input: ClonableAsyncFileReader,

    filter: Option<LiquidRowFilter>,

    limit: Option<usize>,

    offset: Option<usize>,

    liquid_cache: LiquidCachedFileRef,
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
            selection.unwrap_or_else(|| vec![RowSelector::select(meta.num_rows() as usize)].into());

        let mut filter_readers = Vec::new();
        if let Some(filter) = self.filter.as_mut() {
            for predicate in filter.predicates.iter_mut() {
                if !selection.selects_any() {
                    return Ok((self, None));
                }

                let p_projection = predicate.projection();
                row_group
                    .fetch(&self.input, p_projection, &selection)
                    .await?;

                let array_reader = build_cached_array_reader(
                    self.fields.as_deref(),
                    p_projection,
                    &row_group,
                    self.liquid_cache.row_group(row_group_idx).clone(),
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
            .fetch(&self.input, &projection, &selection)
            .await?;

        let cached_row_group = self.liquid_cache.row_group(row_group_idx);
        let array_reader = build_cached_array_reader(
            self.fields.as_deref(),
            &projection,
            &row_group,
            cached_row_group.clone(),
        )?;

        let reader = LiquidBatchReader::new(
            batch_size,
            array_reader,
            selection,
            filter_readers,
            self.filter.take(),
            cached_row_group,
        );

        Ok((self, Some(reader)))
    }
}

enum StreamState {
    /// At the start of a new row group, or the end of the parquet stream
    Init,
    /// Decoding a batch
    Decoding(LiquidBatchReader),
    /// Reading data from input
    Reading(BoxFuture<'static, ReadResult>),
}

impl std::fmt::Debug for StreamState {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            StreamState::Init => write!(f, "StreamState::Init"),
            StreamState::Decoding(_) => write!(f, "StreamState::Decoding"),
            StreamState::Reading(_) => write!(f, "StreamState::Reading"),
        }
    }
}

#[derive(Clone)]
pub struct ClonableAsyncFileReader(Arc<Mutex<ParquetMetadataCacheReader>>);

pub struct LiquidStreamBuilder {
    pub(crate) input: ClonableAsyncFileReader,

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

    pub fn build(self, liquid_cache: LiquidCachedFileRef) -> Result<LiquidStream, ParquetError> {
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

        let liquid_cache_mode = liquid_cache.cache_mode();
        let reader = ReaderFactory {
            input: self.input,
            filter: self.filter,
            metadata: Arc::clone(&self.metadata),
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
        let schema = if matches!(liquid_cache_mode, LiquidCacheMode::InMemoryLiquid { .. }) {
            Arc::new(coerce_from_reader_to_liquid_types(&schema))
        } else {
            schema
        };
        Ok(LiquidStream {
            metadata: self.metadata,
            schema,
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
                        panic!("Decoding next batch error: {:?}", e);
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
                        panic!("Reading next batch error: {:?}", e);
                    }
                },
            }
        }
    }
}
