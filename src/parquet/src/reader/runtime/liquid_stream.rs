use crate::cache::{BatchID, InsertArrowArrayError, LiquidCachedFileRef, LiquidCachedRowGroupRef};
use crate::reader::plantime::{LiquidRowFilter, ParquetMetadataCacheReader};
use crate::reader::runtime::parquet_bridge::{
    ParquetField, limit_row_selection, offset_row_selection,
};
use arrow::array::{AsArray, RecordBatch};
use arrow_schema::{ArrowError, DataType, Fields, Schema, SchemaRef};
use fastrace::Event;
use fastrace::local::LocalSpan;
use futures::{FutureExt, Stream, future::BoxFuture, ready};
use parquet::arrow::arrow_reader::ArrowPredicate;
use parquet::{
    arrow::{
        ProjectionMask,
        arrow_reader::{RowSelection, RowSelector},
    },
    errors::ParquetError,
    file::metadata::ParquetMetaData,
};
use std::{
    collections::{BTreeSet, VecDeque},
    fmt::Formatter,
    pin::Pin,
    sync::Arc,
    task::{Context, Poll},
};

use super::InMemoryRowGroup;
use super::liquid_batch_reader::LiquidBatchReader;
use super::parquet::{
    ArrayReaderColumn, PlainArrayReader, build_plain_array_reader, get_column_ids,
};

type ReadResult = Result<(ReaderFactory, Option<LiquidBatchReader>), ParquetError>;

struct ReaderFactory {
    metadata: Arc<ParquetMetaData>,

    fields: Option<Arc<ParquetField>>,

    schema: SchemaRef,

    input: ParquetMetadataCacheReader,

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
            for predicate in filter.predicates_mut() {
                let p_projection = predicate.projection();
                if let Some(ref mut p) = predicate_projection {
                    p.union(p_projection);
                } else {
                    predicate_projection = Some(p_projection.clone());
                }
            }
        }
        let projection_to_cache = predicate_projection.as_ref().map(|pred| {
            let mut p = pred.clone();
            p.intersect(&projection);
            p
        });

        let cached_row_group = self.liquid_cache.row_group(row_group_idx as u64);
        let mut row_group = InMemoryRowGroup::new(
            meta,
            offset_index,
            projection_to_cache,
            cached_row_group.clone(),
        );

        let mut selection =
            selection.unwrap_or_else(|| vec![RowSelector::select(meta.num_rows() as usize)].into());

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

        let row_count = meta.num_rows() as usize;
        let cache_batch_size = cached_row_group.batch_size();

        let mut cache_projection = projection.clone();
        if let Some(ref predicate_projection) = predicate_projection {
            cache_projection.union(predicate_projection);
        }

        let selection_for_cache = selection.clone();
        let selection_batches =
            collect_selection_batches(&selection_for_cache, cache_batch_size, row_count);

        let cache_column_ids = get_column_ids(self.fields.as_deref(), &cache_projection);
        let projection_column_ids = get_column_ids(self.fields.as_deref(), &projection);

        let missing_batches =
            compute_missing_batches(&cached_row_group, &cache_column_ids, &selection_batches);

        if !cache_column_ids.is_empty() {
            row_group
                .fetch_with_batches(&mut self.input, &cache_projection, Some(&missing_batches))
                .await?;
        }

        if !missing_batches.is_empty() && !cache_column_ids.is_empty() {
            let PlainArrayReader {
                mut reader,
                columns,
            } = build_plain_array_reader(self.fields.as_deref(), &cache_projection, &row_group)?;

            let backfill_selection =
                build_selection_for_batches(&missing_batches, cache_batch_size, row_count);

            if backfill_selection.selects_any() {
                let record_batch =
                    read_record_batch_from_plain_reader(&mut reader, &backfill_selection)
                        .map_err(|e| ParquetError::ArrowError(e.to_string()))?;

                insert_batches_into_cache(
                    &record_batch,
                    &columns,
                    &missing_batches,
                    cache_batch_size,
                    row_count,
                    &cached_row_group,
                )?;
            }
        }

        let reader = LiquidBatchReader::new(
            batch_size,
            selection,
            self.filter.take(),
            cached_row_group,
            projection_column_ids,
            self.schema.clone(),
        );

        Ok((self, Some(reader)))
    }
}

fn collect_selection_batches(
    selection: &RowSelection,
    batch_size: usize,
    row_count: usize,
) -> BTreeSet<BatchID> {
    let mut batches = BTreeSet::new();
    let mut current_row = 0usize;
    let selectors: Vec<RowSelector> = selection.clone().into();

    for selector in selectors {
        if selector.skip {
            current_row += selector.row_count;
            continue;
        }

        let start = current_row;
        let end = (current_row + selector.row_count).min(row_count);
        if start >= end {
            current_row = current_row.saturating_add(selector.row_count);
            continue;
        }

        let start_batch = start / batch_size;
        let end_batch = (end - 1) / batch_size;
        for batch_idx in start_batch..=end_batch {
            batches.insert(BatchID::from_raw(batch_idx as u16));
        }
        current_row += selector.row_count;
    }

    batches
}

fn compute_missing_batches(
    cached_row_group: &LiquidCachedRowGroupRef,
    column_ids: &[usize],
    selection_batches: &BTreeSet<BatchID>,
) -> Vec<BatchID> {
    if column_ids.is_empty() || selection_batches.is_empty() {
        return Vec::new();
    }

    let mut missing = BTreeSet::new();

    for &column_idx in column_ids {
        match cached_row_group.get_column(column_idx as u64) {
            Some(column) => {
                for batch_id in selection_batches {
                    if !column.is_cached(*batch_id) {
                        missing.insert(*batch_id);
                    }
                }
            }
            None => {
                missing.extend(selection_batches.iter().copied());
            }
        }
    }

    missing.into_iter().collect()
}

fn build_selection_for_batches(
    batches: &[BatchID],
    batch_size: usize,
    row_count: usize,
) -> RowSelection {
    if batches.is_empty() {
        return RowSelection::from(Vec::<RowSelector>::new());
    }

    let mut selectors = Vec::new();
    let mut current_row = 0usize;

    for batch_id in batches {
        let batch_idx = usize::from(**batch_id);
        let start = batch_idx * batch_size;
        if start >= row_count {
            continue;
        }
        let end = ((batch_idx + 1) * batch_size).min(row_count);

        if start > current_row {
            selectors.push(RowSelector::skip(start - current_row));
        }

        selectors.push(RowSelector::select(end - start));
        current_row = end;
    }

    RowSelection::from(selectors)
}

fn read_record_batch_from_plain_reader(
    reader: &mut Box<dyn parquet::arrow::array_reader::ArrayReader>,
    selection: &RowSelection,
) -> Result<RecordBatch, ArrowError> {
    let selectors: Vec<RowSelector> = selection.clone().into();
    for selector in &selectors {
        if selector.skip {
            reader.skip_records(selector.row_count)?;
        } else {
            reader.read_records(selector.row_count)?;
        }
    }
    let array = reader.consume_batch()?;
    Ok(RecordBatch::from(array.as_struct()))
}

fn insert_batches_into_cache(
    record_batch: &RecordBatch,
    columns: &[ArrayReaderColumn],
    batches: &[BatchID],
    batch_size: usize,
    row_count: usize,
    cached_row_group: &LiquidCachedRowGroupRef,
) -> Result<(), ParquetError> {
    if batches.is_empty() || columns.is_empty() || record_batch.num_rows() == 0 {
        return Ok(());
    }

    debug_assert_eq!(record_batch.num_columns(), columns.len());

    let mut offset = 0usize;
    for batch_id in batches {
        let batch_idx = usize::from(**batch_id);
        let start = batch_idx * batch_size;
        if start >= row_count {
            continue;
        }
        let end = ((batch_idx + 1) * batch_size).min(row_count);
        let len = end - start;

        for (col_idx, column_meta) in columns.iter().enumerate() {
            let array = record_batch.column(col_idx).slice(offset, len);
            let column = cached_row_group.create_column(
                column_meta.column_idx as u64,
                Arc::clone(&column_meta.field),
            );
            if let Err(err) = column.insert(*batch_id, array) {
                if !matches!(err, InsertArrowArrayError::AlreadyCached) {
                    return Err(ParquetError::General(format!(
                        "Failed to insert batch {} for column {} into cache: {err:?}",
                        batch_idx, column_meta.column_idx
                    )));
                }
            }
            debug_assert!(column.is_cached(*batch_id));
        }

        offset += len;
    }

    Ok(())
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

pub struct LiquidStreamBuilder {
    pub(crate) input: ParquetMetadataCacheReader,

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
                        "row group {col} out of bounds 0..{num_row_groups}"
                    )));
                }
                row_groups.into()
            }
            None => (0..self.metadata.row_groups().len()).collect(),
        };

        let batch_size = self
            .batch_size
            .min(self.metadata.file_metadata().num_rows() as usize);

        // Ensure schema of ParquetRecordBatchStream respects projection, and does
        // not store metadata (same as for ParquetRecordBatchReader and emitted RecordBatches)
        let projected_fields = match self.fields.as_deref().map(|pf| &pf.arrow_type) {
            Some(DataType::Struct(fields)) => {
                fields.filter_leaves(|idx, _| self.projection.leaf_included(idx))
            }
            None => Fields::empty(),
            _ => unreachable!("Must be Struct for root type"),
        };
        let schema = Arc::new(Schema::new(projected_fields));

        let reader = ReaderFactory {
            metadata: Arc::clone(&self.metadata),
            fields: self.fields,
            schema: Arc::clone(&schema),
            input: self.input,
            filter: self.filter,
            limit: self.limit,
            offset: self.offset,
            liquid_cache,
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
                        panic!("Decoding next batch error: {e:?}");
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

                    LocalSpan::add_event(Event::new("LiquidStream::read_row_group"));
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
                        panic!("Reading next batch error: {e:?}");
                    }
                },
            }
        }
    }
}
