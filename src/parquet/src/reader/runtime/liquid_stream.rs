use crate::cache::{BatchID, InsertArrowArrayError, LiquidCachedFileRef, LiquidCachedRowGroupRef};
use crate::reader::plantime::{LiquidRowFilter, ParquetMetadataCacheReader};
use crate::reader::runtime::parquet_bridge::{
    ParquetField, limit_row_selection, offset_row_selection,
};
use arrow::array::RecordBatch;
use arrow_schema::{DataType, Fields, Schema, SchemaRef};
use fastrace::Event;
use fastrace::local::LocalSpan;
use futures::{FutureExt, Stream, StreamExt, future::BoxFuture};
use parquet::arrow::arrow_reader::{ArrowPredicate, ArrowReaderMetadata, ArrowReaderOptions};
use parquet::{
    arrow::{
        ParquetRecordBatchStreamBuilder, ProjectionMask,
        arrow_reader::{RowSelection, RowSelector},
    },
    errors::ParquetError,
    file::metadata::ParquetMetaData,
};
use std::{
    collections::VecDeque,
    fmt::Formatter,
    pin::Pin,
    sync::Arc,
    task::{Context, Poll},
};

use super::liquid_cache_reader::LiquidCacheReader;
use super::utils::{ArrayReaderColumn, get_column_ids};

type PlanResult = Option<PlanningContext>;
type FillCacheResult = Result<(ReaderFactory, PlanningContext), ParquetError>;

/// Build column metadata from projection, fields, and record batch schema for cache insertion
///
/// The record batch schema contains the projected fields in order, and we use get_column_ids
/// to get the original column indices in the same order.
fn build_column_metadata_from_batch(
    projection: &ProjectionMask,
    fields: Option<&ParquetField>,
    record_batch: &RecordBatch,
) -> Result<Vec<ArrayReaderColumn>, ParquetError> {
    // Get the original column indices in projection order
    let column_ids = get_column_ids(fields, projection);

    // The record batch schema has projected fields in the same order as column_ids
    // Zip them together to create ArrayReaderColumn with correct indices and field names
    let columns = column_ids
        .into_iter()
        .zip(record_batch.schema().fields().iter())
        .map(|(column_idx, field)| ArrayReaderColumn {
            column_idx,
            field: Arc::clone(field),
        })
        .collect();

    Ok(columns)
}

struct ReaderFactory {
    metadata: Arc<ParquetMetaData>,

    fields: Option<Arc<ParquetField>>,

    input: ParquetMetadataCacheReader,

    filter: Option<LiquidRowFilter>,

    limit: Option<usize>,

    offset: Option<usize>,

    liquid_cache: LiquidCachedFileRef,
}

impl ReaderFactory {
    /// Plans what to read from cache vs parquet for the next row group
    fn plan_row_group(
        &mut self,
        row_group_idx: usize,
        selection: Option<RowSelection>,
        projection: ProjectionMask,
        batch_size: usize,
    ) -> PlanResult {
        let meta = self.metadata.row_group(row_group_idx);

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

        let cached_row_group = self.liquid_cache.row_group(row_group_idx as u64);

        let mut selection =
            selection.unwrap_or_else(|| vec![RowSelector::select(meta.num_rows() as usize)].into());

        let rows_before = selection.row_count();

        if rows_before == 0 {
            return None;
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
            return None;
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

        let context = PlanningContext {
            row_group_idx,
            selection,
            batch_size,
            cached_row_group,
            cache_projection,
            projection_column_ids,
            cache_column_ids,
            missing_batches,
        };

        Some(context)
    }

    /// Fills the cache by reading missing batches from parquet using official parquet reader
    async fn fill_cache_from_parquet(self, context: PlanningContext) -> FillCacheResult {
        let row_count = self.metadata.row_group(context.row_group_idx).num_rows() as usize;
        let cache_batch_size = context.cached_row_group.batch_size();

        if context.cache_column_ids.is_empty() || context.missing_batches.is_empty() {
            return Ok((self, context));
        }

        // Build row selection for the missing batches
        let backfill_selection =
            build_selection_for_batches(&context.missing_batches, cache_batch_size, row_count);

        if !backfill_selection.selects_any() {
            return Ok((self, context));
        }

        // Clone the reader for this operation (cheap since it's Arc-based)
        let reader_clone: ParquetMetadataCacheReader = self.input.clone();

        // Use official parquet async reader
        let options = ArrowReaderOptions::new();
        let reader_metadata = ArrowReaderMetadata::try_new(Arc::clone(&self.metadata), options)?;

        let mut stream =
            ParquetRecordBatchStreamBuilder::new_with_metadata(reader_clone, reader_metadata)
                .with_projection(context.cache_projection.clone())
                .with_row_groups(vec![context.row_group_idx])
                .with_row_selection(backfill_selection)
                .with_batch_size(cache_batch_size)
                .build()?;

        let mut processed_batches = 0usize;
        let mut column_metadata: Option<Vec<ArrayReaderColumn>> = None;

        while let Some(batch_result) = stream.next().await {
            let record_batch = batch_result?;
            if record_batch.num_rows() == 0 {
                continue;
            }

            if let Some(column_metadata) = &column_metadata {
                debug_assert_eq!(
                    record_batch.num_columns(),
                    column_metadata.len(),
                    "record batch schema changed while filling cache"
                );
            } else {
                column_metadata = Some(build_column_metadata_from_batch(
                    &context.cache_projection,
                    self.fields.as_deref(),
                    &record_batch,
                )?);
            }

            let Some(batch_id) = context.missing_batches.get(processed_batches) else {
                return Err(ParquetError::General(
                    "parquet stream produced more batches than expected".to_string(),
                ));
            };

            let batch_index = usize::from(**batch_id);
            let batch_start = batch_index * cache_batch_size;
            let expected_len = ((batch_index + 1) * cache_batch_size)
                .min(row_count)
                .saturating_sub(batch_start.min(row_count));

            debug_assert!(
                record_batch.num_rows() <= cache_batch_size,
                "parquet batch larger than cache batch size"
            );
            debug_assert_eq!(
                record_batch.num_rows(),
                expected_len,
                "parquet batch length does not match expected cache slice"
            );

            let batch_id = *batch_id;
            insert_batch_into_cache(
                &record_batch,
                column_metadata.as_ref().unwrap(),
                batch_id,
                cache_batch_size,
                row_count,
                &context.cached_row_group,
            )
            .await?;

            processed_batches += 1;
        }

        if processed_batches != context.missing_batches.len() {
            return Err(ParquetError::General(format!(
                "expected {} batches from parquet stream, received {}",
                context.missing_batches.len(),
                processed_batches
            )));
        }

        Ok((self, context))
    }
}

fn collect_selection_batches(
    selection: &RowSelection,
    batch_size: usize,
    row_count: usize,
) -> Vec<BatchID> {
    let mut batches = Vec::new();
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
            let batch_id = BatchID::from_raw(batch_idx as u16);
            let is_duplicate = batches.last().is_some_and(|last| last == &batch_id);
            if !is_duplicate {
                batches.push(batch_id);
            }
        }
        current_row += selector.row_count;
    }

    batches
}

fn compute_missing_batches(
    cached_row_group: &LiquidCachedRowGroupRef,
    column_ids: &[usize],
    selection_batches: &[BatchID],
) -> Vec<BatchID> {
    if column_ids.is_empty() || selection_batches.is_empty() {
        return Vec::new();
    }

    let mut columns = Vec::with_capacity(column_ids.len());
    for &column_idx in column_ids {
        columns.push(cached_row_group.get_column(column_idx as u64));
    }

    let mut missing = Vec::new();

    'batch: for &batch_id in selection_batches {
        for column in &columns {
            match column {
                Some(column) => {
                    if !column.is_cached(batch_id) {
                        if missing.last().is_some_and(|last| last == &batch_id) {
                            continue 'batch;
                        }
                        missing.push(batch_id);
                        continue 'batch;
                    }
                }
                None => {
                    if missing.last().is_some_and(|last| last == &batch_id) {
                        continue 'batch;
                    }
                    missing.push(batch_id);
                    continue 'batch;
                }
            }
        }
    }

    missing
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

async fn insert_batch_into_cache(
    record_batch: &RecordBatch,
    columns: &[ArrayReaderColumn],
    batch_id: BatchID,
    batch_size: usize,
    row_count: usize,
    cached_row_group: &LiquidCachedRowGroupRef,
) -> Result<(), ParquetError> {
    if columns.is_empty() || record_batch.num_rows() == 0 {
        return Ok(());
    }

    debug_assert_eq!(record_batch.num_columns(), columns.len());

    let batch_idx = usize::from(*batch_id);
    let start = batch_idx * batch_size;
    if start >= row_count {
        return Ok(());
    }
    let end = ((batch_idx + 1) * batch_size).min(row_count);
    let len = end - start;

    debug_assert!(
        len <= batch_size,
        "cache batch length exceeded configured batch size"
    );
    debug_assert_eq!(
        record_batch.num_rows(),
        len,
        "record batch length does not match cache batch window"
    );

    for (col_idx, column_meta) in columns.iter().enumerate() {
        let column = cached_row_group.create_column(
            column_meta.column_idx as u64,
            Arc::clone(&column_meta.field),
        );
        let array = Arc::clone(record_batch.column(col_idx));

        if let Err(err) = column.insert(batch_id, array).await
            && !matches!(err, InsertArrowArrayError::AlreadyCached)
        {
            return Err(ParquetError::General(format!(
                "Failed to insert batch {} for column {} into cache: {err:?}",
                batch_idx, column_meta.column_idx
            )));
        }
        debug_assert!(column.is_cached(batch_id));
    }

    Ok(())
}

/// Context for planning what to read from cache vs parquet
struct PlanningContext {
    row_group_idx: usize,
    selection: RowSelection,
    batch_size: usize,
    cached_row_group: LiquidCachedRowGroupRef,
    cache_projection: ProjectionMask,
    projection_column_ids: Vec<usize>,
    cache_column_ids: Vec<usize>,
    missing_batches: Vec<BatchID>,
}

enum StreamState {
    /// At the start of a new row group, or the end of the parquet stream
    Init,
    /// Reading from parquet and filling cache
    FillCache(BoxFuture<'static, FillCacheResult>),
    /// Decoding a batch from cache
    ReadFromCache(LiquidCacheReader),
}

impl std::fmt::Debug for StreamState {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            StreamState::Init => write!(f, "StreamState::Init"),
            StreamState::FillCache(_) => write!(f, "StreamState::FillingCache"),
            StreamState::ReadFromCache(_) => write!(f, "StreamState::Decoding"),
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

    pub(crate) span: Option<fastrace::Span>,
}

impl LiquidStreamBuilder {
    pub fn with_row_filter(mut self, filter: LiquidRowFilter) -> Self {
        self.filter = Some(filter);
        self
    }

    pub fn with_span(mut self, span: fastrace::Span) -> Self {
        self.span = Some(span);
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
            span: self.span,
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

    span: Option<fastrace::Span>,
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
        let _guard = self.span.as_ref().map(|s| s.set_local_parent());
        loop {
            let state = std::mem::replace(&mut self.state, StreamState::Init);

            match state {
                StreamState::ReadFromCache(mut batch_reader) => {
                    match Pin::new(&mut batch_reader).poll_next(cx) {
                        Poll::Ready(Some(Ok(batch))) => {
                            self.state = StreamState::ReadFromCache(batch_reader);
                            return Poll::Ready(Some(Ok(batch)));
                        }
                        Poll::Ready(Some(Err(e))) => {
                            panic!("Decoding next batch error: {e:?}");
                        }
                        Poll::Ready(None) => {
                            let filter = batch_reader.into_filter();
                            self.reader.as_mut().unwrap().filter = filter;
                            // state left as Init, continue loop to plan next row group
                        }
                        Poll::Pending => {
                            self.state = StreamState::ReadFromCache(batch_reader);
                            return Poll::Pending;
                        }
                    }
                }
                StreamState::Init => {
                    let row_group_idx = match self.row_groups.pop_front() {
                        Some(idx) => idx,
                        None => return Poll::Ready(None),
                    };

                    let row_count = self.metadata.row_group(row_group_idx).num_rows() as usize;

                    let selection = self.selection.as_mut().map(|s| s.split_off(row_count));

                    LocalSpan::add_event(Event::new("LiquidStream::plan_row_group"));
                    let projection = self.projection.clone();
                    let batch_size = self.batch_size;
                    let maybe_context = self.reader.as_mut().expect("lost reader").plan_row_group(
                        row_group_idx,
                        selection,
                        projection,
                        batch_size,
                    );
                    match maybe_context {
                        Some(context) => {
                            if !context.missing_batches.is_empty()
                                && !context.cache_column_ids.is_empty()
                            {
                                LocalSpan::add_event(Event::new("LiquidStream::fill_cache"));
                                let reader = self.reader.take().expect("lost reader");
                                let fut = reader.fill_cache_from_parquet(context).boxed();
                                self.state = StreamState::FillCache(fut);
                            } else {
                                LocalSpan::add_event(Event::new("LiquidStream::read_from_cache"));
                                let reader_factory = self.reader.as_mut().unwrap();
                                let batch_reader = LiquidCacheReader::new(
                                    context.batch_size,
                                    context.selection,
                                    reader_factory.filter.take(),
                                    context.cached_row_group,
                                    context.projection_column_ids,
                                    self.schema.clone(),
                                );
                                self.state = StreamState::ReadFromCache(batch_reader);
                            }
                        }
                        None => {
                            self.state = StreamState::Init;
                        }
                    }
                }
                StreamState::FillCache(mut f) => match f.as_mut().poll(cx) {
                    Poll::Pending => {
                        self.state = StreamState::FillCache(f);
                        return Poll::Pending;
                    }
                    Poll::Ready(result) => match result {
                        Ok((reader_factory, context)) => {
                            self.reader = Some(reader_factory);
                            LocalSpan::add_event(Event::new("LiquidStream::read_from_cache"));
                            let reader_factory = self.reader.as_mut().unwrap();
                            let batch_reader = LiquidCacheReader::new(
                                context.batch_size,
                                context.selection,
                                reader_factory.filter.take(),
                                context.cached_row_group,
                                context.projection_column_ids,
                                self.schema.clone(),
                            );
                            self.state = StreamState::ReadFromCache(batch_reader);
                        }
                        Err(e) => {
                            panic!("Filling cache error: {e:?}");
                        }
                    },
                },
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cache::LiquidCache;
    use arrow::array::{ArrayRef, Int32Array};
    use liquid_cache_common::IoMode;
    use liquid_cache_storage::cache::squeeze_policies::Evict;
    use liquid_cache_storage::cache_policies::LiquidPolicy;
    use parquet::arrow::arrow_reader::RowSelection;
    use std::sync::Arc;

    fn make_cache(batch_size: usize) -> LiquidCachedRowGroupRef {
        let tmp_dir = tempfile::tempdir().unwrap();
        let cache = LiquidCache::new(
            batch_size,
            usize::MAX,
            tmp_dir.path().to_path_buf(),
            Box::new(LiquidPolicy::new()),
            Box::new(Evict),
            IoMode::Uring,
        );
        let file = cache.register_or_get_file("test.parquet".to_string());
        file.row_group(0)
    }

    async fn insert_batches(
        row_group: &LiquidCachedRowGroupRef,
        column_id: usize,
        batch_payloads: &[(u16, &[i32])],
    ) {
        let field = Arc::new(arrow_schema::Field::new(
            format!("col_{column_id}"),
            arrow_schema::DataType::Int32,
            false,
        ));
        let column = row_group.create_column(column_id as u64, field);
        for (batch_idx, values) in batch_payloads.iter() {
            let array: ArrayRef = Arc::new(Int32Array::from(values.to_vec()));
            column
                .insert(BatchID::from_raw(*batch_idx), array)
                .await
                .unwrap();
        }
    }

    #[test]
    fn collect_selection_batches_marks_all_selected_batches() {
        let selection = RowSelection::from(vec![
            RowSelector::select(3),
            RowSelector::skip(2),
            RowSelector::select(5),
        ]);
        let batches = collect_selection_batches(&selection, 4, 10);
        let expected = vec![
            BatchID::from_raw(0),
            BatchID::from_raw(1),
            BatchID::from_raw(2),
        ];
        assert_eq!(batches, expected);
    }

    #[test]
    fn collect_selection_batches_handles_empty_selection() {
        let selection = RowSelection::from(vec![]);
        let batches = collect_selection_batches(&selection, 4, 10);
        let expected: Vec<BatchID> = vec![];
        assert_eq!(batches, expected);
    }

    #[test]
    fn collect_selection_batches_handles_selection_beyond_row_count() {
        let selection = RowSelection::from(vec![
            RowSelector::select(5),  // Select 5 rows
            RowSelector::skip(2),    // Skip 2 rows
            RowSelector::select(10), // Select 10 rows (but only 3 rows left)
        ]);
        let batches = collect_selection_batches(&selection, 4, 8);
        // Total rows: 8
        // First selector: select 5 rows (rows 0-4) -> batches 0, 1
        // Skip 2 rows (rows 5-6)
        // Third selector: select 10 rows from row 7, but only 1 row left -> batch 1
        let expected = vec![BatchID::from_raw(0), BatchID::from_raw(1)];
        assert_eq!(batches, expected);
    }

    #[tokio::test]
    async fn compute_missing_batches_identifies_partial_columns() {
        let row_group = make_cache(4);
        insert_batches(&row_group, 0, &[(0, &[1, 2, 3, 4]), (2, &[9, 9, 9, 9])]).await;
        insert_batches(&row_group, 2, &[(0, &[5, 6, 7, 8])]).await;

        let selection_batches = vec![
            BatchID::from_raw(0),
            BatchID::from_raw(1),
            BatchID::from_raw(2),
        ];

        let missing_for_col0 = compute_missing_batches(&row_group, &[0], &selection_batches);
        assert_eq!(missing_for_col0, vec![BatchID::from_raw(1)]);

        let missing_for_col2 = compute_missing_batches(&row_group, &[2], &selection_batches);
        assert_eq!(
            missing_for_col2,
            vec![BatchID::from_raw(1), BatchID::from_raw(2),]
        );

        let missing_for_col1 = compute_missing_batches(&row_group, &[1], &selection_batches);
        assert_eq!(
            missing_for_col1,
            vec![
                BatchID::from_raw(0),
                BatchID::from_raw(1),
                BatchID::from_raw(2),
            ]
        );
    }

    #[test]
    fn build_selection_for_batches_generates_sparse_selectors() {
        let selection =
            build_selection_for_batches(&[BatchID::from_raw(1), BatchID::from_raw(3)], 4, 20);
        let selectors: Vec<RowSelector> = selection.into();
        assert_eq!(
            selectors,
            vec![
                RowSelector::skip(4),
                RowSelector::select(4),
                RowSelector::skip(4),
                RowSelector::select(4),
            ]
        );
    }

    #[test]
    fn build_selection_for_batches_handles_empty_batches() {
        let selection = build_selection_for_batches(&[], 4, 20);
        let selectors: Vec<RowSelector> = selection.into();
        assert_eq!(selectors, vec![]);
    }

    #[test]
    fn build_selection_for_batches_handles_batch_beyond_row_count() {
        let selection =
            build_selection_for_batches(&[BatchID::from_raw(5), BatchID::from_raw(6)], 4, 16);
        let selectors: Vec<RowSelector> = selection.into();
        // Total rows: 16, so valid batches are 0-3 (rows 0-15)
        // Batch 5: start=20, end=min(24,16)=16, but 20 >= 16, so skipped
        // Batch 6: start=24, end=min(28,16)=16, but 24 >= 16, so skipped
        // Result should be empty selection
        assert_eq!(selectors, vec![]);
    }

    #[test]
    fn build_selection_for_batches_handles_single_batch() {
        let selection = build_selection_for_batches(&[BatchID::from_raw(2)], 4, 20);
        let selectors: Vec<RowSelector> = selection.into();
        // Batch 2: rows 8-11
        // Should skip 8 rows then select 4 rows
        assert_eq!(
            selectors,
            vec![RowSelector::skip(8), RowSelector::select(4),]
        );
    }

    #[test]
    fn build_selection_for_batches_handles_partial_last_batch() {
        let selection = build_selection_for_batches(&[BatchID::from_raw(4)], 4, 18);
        let selectors: Vec<RowSelector> = selection.into();
        // Batch 4: start=16, end=min(20,18)=18
        // Should skip 16 rows then select 2 rows (18-16=2)
        assert_eq!(
            selectors,
            vec![RowSelector::skip(16), RowSelector::select(2),]
        );
    }
}
