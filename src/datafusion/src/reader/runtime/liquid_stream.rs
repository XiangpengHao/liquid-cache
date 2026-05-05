use crate::cache::{CachedFileRef, CachedRowGroupRef};
use crate::reader::plantime::{LiquidRowFilter, ParquetMetadataCacheReader};
use arrow::array::RecordBatch;
use arrow_schema::{Schema, SchemaRef};
use fastrace::Event;
use fastrace::local::LocalSpan;
use futures::Stream;
use parquet::{
    arrow::{
        ProjectionMask,
        arrow_reader::{ArrowPredicate, RowSelection, RowSelector},
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

use super::liquid_cache_reader::{
    LiquidCacheReader, LiquidCacheReaderConfig, ParquetFallbackConfig,
};
use super::utils::{get_root_column_ids, limit_row_selection, offset_row_selection};

type PlanResult = Option<PlanningContext>;

struct ReaderFactory {
    metadata: Arc<ParquetMetaData>,

    input: ParquetMetadataCacheReader,

    filter: Option<LiquidRowFilter>,

    limit: Option<usize>,

    offset: Option<usize>,

    cached_file: CachedFileRef,
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

        let mut cache_projection = projection.clone();
        if let Some(ref predicate_projection) = predicate_projection {
            cache_projection.union(predicate_projection);
        }

        let schema_descr = self.metadata.file_metadata().schema_descr();
        let cache_column_ids = get_root_column_ids(schema_descr, &cache_projection);
        let predicate_column_ids = if let Some(ref predicate_projection) = predicate_projection {
            get_root_column_ids(schema_descr, predicate_projection)
        } else {
            Vec::new()
        };
        let cached_row_group = self
            .cached_file
            .create_row_group(row_group_idx as u64, predicate_column_ids);

        let projection_column_ids = get_root_column_ids(schema_descr, &projection);

        let context = PlanningContext {
            row_group_idx,
            selection,
            batch_size,
            cached_row_group,
            cache_projection,
            projection_column_ids,
            cache_column_ids,
        };

        Some(context)
    }
}

fn build_projection_schema(file_schema: &SchemaRef, projection_column_ids: &[usize]) -> SchemaRef {
    let fields: Vec<_> = projection_column_ids
        .iter()
        .filter_map(|column_id| file_schema.fields().get(*column_id))
        .map(|field_ref| field_ref.as_ref().clone())
        .collect();
    Arc::new(Schema::new(fields))
}

/// Context for planning what to read from cache vs parquet
struct PlanningContext {
    row_group_idx: usize,
    selection: RowSelection,
    batch_size: usize,
    cached_row_group: CachedRowGroupRef,
    cache_projection: ProjectionMask,
    projection_column_ids: Vec<usize>,
    cache_column_ids: Vec<usize>,
}

fn build_liquid_cache_reader(
    reader_factory: &mut ReaderFactory,
    context: PlanningContext,
    schema: SchemaRef,
) -> LiquidCacheReader {
    let row_count = reader_factory
        .metadata
        .row_group(context.row_group_idx)
        .num_rows() as usize;
    let cache_batch_size = context.cached_row_group.batch_size();
    LiquidCacheReader::new(LiquidCacheReaderConfig {
        batch_size: context.batch_size,
        selection: context.selection,
        row_filter: reader_factory.filter.take(),
        cached_row_group: context.cached_row_group,
        projection_columns: context.projection_column_ids,
        schema,
        parquet_fallback: ParquetFallbackConfig {
            row_group_idx: context.row_group_idx,
            metadata: Arc::clone(&reader_factory.metadata),
            input: reader_factory.input.clone(),
            cache_projection: context.cache_projection,
            cache_column_ids: context.cache_column_ids,
            cache_batch_size,
            row_count,
        },
    })
}

enum StreamState {
    /// At the start of a new row group, or the end of the parquet stream
    Init,
    /// Decoding a batch from cache
    ReadFromCache(Box<LiquidCacheReader>),
}

impl std::fmt::Debug for StreamState {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            StreamState::Init => write!(f, "StreamState::Init"),
            StreamState::ReadFromCache(_) => write!(f, "StreamState::Decoding"),
        }
    }
}

pub struct LiquidStreamBuilder {
    pub(crate) input: ParquetMetadataCacheReader,

    pub(crate) metadata: Arc<ParquetMetaData>,

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
    pub fn new(input: ParquetMetadataCacheReader, metadata: Arc<ParquetMetaData>) -> Self {
        Self {
            input,
            metadata,
            batch_size: 1024,
            row_groups: None,
            projection: ProjectionMask::all(),
            filter: None,
            selection: None,
            limit: None,
            offset: None,
            span: None,
        }
    }

    pub fn with_batch_size(mut self, batch_size: usize) -> Self {
        self.batch_size = batch_size;
        self
    }

    pub fn with_row_groups(mut self, row_groups: Vec<usize>) -> Self {
        self.row_groups = Some(row_groups);
        self
    }

    pub fn with_projection(mut self, projection: ProjectionMask) -> Self {
        self.projection = projection;
        self
    }

    pub fn with_selection(mut self, selection: Option<RowSelection>) -> Self {
        self.selection = selection;
        self
    }

    pub fn with_limit(mut self, limit: Option<usize>) -> Self {
        self.limit = limit;
        self
    }

    pub fn with_row_filter(mut self, filter: LiquidRowFilter) -> Self {
        self.filter = Some(filter);
        self
    }

    pub fn with_span(mut self, span: fastrace::Span) -> Self {
        self.span = Some(span);
        self
    }

    pub fn build(self, liquid_cache: CachedFileRef) -> Result<LiquidStream, ParquetError> {
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

        let schema_descr = self.metadata.file_metadata().schema_descr();
        let projection_column_ids = get_root_column_ids(schema_descr, &self.projection);
        let file_schema = liquid_cache.schema();
        let schema = build_projection_schema(&file_schema, &projection_column_ids);

        let reader = ReaderFactory {
            metadata: Arc::clone(&self.metadata),
            input: self.input,
            filter: self.filter,
            limit: self.limit,
            offset: self.offset,
            cached_file: liquid_cache,
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

impl LiquidStream {
    pub fn schema(&self) -> &SchemaRef {
        &self.schema
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
                    match Pin::new(&mut *batch_reader).poll_next(cx) {
                        Poll::Ready(Some(Ok(batch))) => {
                            self.state = StreamState::ReadFromCache(batch_reader);
                            return Poll::Ready(Some(Ok(batch)));
                        }
                        Poll::Ready(Some(Err(e))) => {
                            panic!("Decoding next batch error: {e:?}");
                        }
                        Poll::Ready(None) => {
                            let batch_reader = *batch_reader;
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
                            LocalSpan::add_event(Event::new("LiquidStream::read_from_cache"));
                            let schema = Arc::clone(&self.schema);
                            let reader_factory = self.reader.as_mut().unwrap();
                            let batch_reader =
                                build_liquid_cache_reader(reader_factory, context, schema);
                            self.state = StreamState::ReadFromCache(Box::new(batch_reader));
                        }
                        None => {
                            self.state = StreamState::Init;
                        }
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cache::{BatchID, CachedFileRef, LiquidCacheParquet};
    use crate::reader::plantime::{
        CachedMetaReaderFactory, FilterCandidateBuilder, LiquidPredicate,
    };
    use arrow::array::{Array, ArrayRef, Int32Array};
    use arrow_schema::{DataType, Field, Schema};
    use datafusion::common::ScalarValue;
    use datafusion::datasource::listing::PartitionedFile;
    use datafusion::logical_expr::Operator;
    use datafusion::physical_expr::PhysicalExpr;
    use datafusion::physical_expr::expressions::{BinaryExpr, Column, Literal};
    use datafusion::physical_plan::metrics::ExecutionPlanMetricsSet;
    use futures::StreamExt;
    use liquid_cache::cache::AlwaysHydrate;
    use liquid_cache::cache::squeeze_policies::Evict;
    use liquid_cache::cache_policies::LiquidPolicy;
    use object_store::local::LocalFileSystem;
    use parquet::arrow::ArrowWriter;
    use parquet::arrow::arrow_reader::{ArrowReaderMetadata, ArrowReaderOptions};
    use std::fs::File;
    use std::sync::Arc;

    fn write_two_row_group_file(path: &std::path::Path, schema: SchemaRef) {
        let file = File::create(path).unwrap();
        let mut writer = ArrowWriter::try_new(file, schema.clone(), None).unwrap();
        let batch0 = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(Int32Array::from(vec![0, 1, 2, 3])),
                Arc::new(Int32Array::from(vec![10, 11, 12, 13])),
            ],
        )
        .unwrap();
        let batch1 = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Int32Array::from(vec![4, 5, 6, 7])),
                Arc::new(Int32Array::from(vec![14, 15, 16, 17])),
            ],
        )
        .unwrap();
        writer.write(&batch0).unwrap();
        writer.flush().unwrap();
        writer.write(&batch1).unwrap();
        writer.close().unwrap();
    }

    fn write_single_row_group_file(path: &std::path::Path, schema: SchemaRef, a: Vec<i32>) {
        let file = File::create(path).unwrap();
        let mut writer = ArrowWriter::try_new(file, schema.clone(), None).unwrap();
        let b: Vec<_> = a.iter().map(|value| value + 1000).collect();
        let batch = RecordBatch::try_new(
            schema,
            vec![Arc::new(Int32Array::from(a)), Arc::new(Int32Array::from(b))],
        )
        .unwrap();
        writer.write(&batch).unwrap();
        writer.close().unwrap();
    }

    async fn make_liquid_stream(
        max_memory_bytes: usize,
        max_disk_bytes: usize,
        row_filter: Option<LiquidRowFilter>,
    ) -> (
        LiquidStream,
        Arc<LiquidCacheParquet>,
        CachedFileRef,
        tempfile::TempDir,
    ) {
        let schema = Arc::new(Schema::new(vec![
            Field::new("a", DataType::Int32, false),
            Field::new("b", DataType::Int32, false),
        ]));
        let tmp_dir = tempfile::tempdir().unwrap();
        let parquet_path = tmp_dir.path().join("data.parquet");
        write_two_row_group_file(&parquet_path, schema.clone());
        let metadata_file = File::open(&parquet_path).unwrap();
        let reader_metadata =
            ArrowReaderMetadata::load(&metadata_file, ArrowReaderOptions::new()).unwrap();
        let object_store = Arc::new(LocalFileSystem::new_with_prefix(tmp_dir.path()).unwrap());
        let partitioned_file = PartitionedFile::new(
            "data.parquet",
            std::fs::metadata(&parquet_path).unwrap().len(),
        );
        let metrics = ExecutionPlanMetricsSet::new();
        let input = CachedMetaReaderFactory::new(object_store).create_liquid_reader(
            0,
            partitioned_file,
            None,
            &metrics,
        );

        let store = t4::mount(tmp_dir.path().join("liquid_cache.t4"))
            .await
            .unwrap();
        let cache = Arc::new(
            LiquidCacheParquet::new(
                4,
                max_memory_bytes,
                max_disk_bytes,
                store,
                Box::new(LiquidPolicy::new()),
                Box::new(Evict),
                Box::new(AlwaysHydrate::new()),
            )
            .await,
        );
        let cached_file = cache.register_or_get_file("data.parquet".to_string(), schema);
        let projection = ProjectionMask::roots(
            reader_metadata.metadata().file_metadata().schema_descr(),
            [0, 1],
        );
        let mut builder = LiquidStreamBuilder::new(input, Arc::clone(reader_metadata.metadata()))
            .with_batch_size(4)
            .with_row_groups(vec![0, 1])
            .with_projection(projection);
        if let Some(row_filter) = row_filter {
            builder = builder.with_row_filter(row_filter);
        }
        let stream = builder.build(cached_file.clone()).unwrap();
        (stream, cache, cached_file, tmp_dir)
    }

    async fn collect_liquid_values(stream: LiquidStream) -> (Vec<i32>, Vec<i32>) {
        let batches = stream
            .map(|batch| batch.expect("valid liquid stream batch"))
            .collect::<Vec<_>>()
            .await;
        let mut a = Vec::new();
        let mut b = Vec::new();
        for batch in batches {
            let a_array = batch
                .column(0)
                .as_any()
                .downcast_ref::<Int32Array>()
                .unwrap();
            let b_array = batch
                .column(1)
                .as_any()
                .downcast_ref::<Int32Array>()
                .unwrap();
            a.extend(a_array.iter().map(|value| value.unwrap()));
            b.extend(b_array.iter().map(|value| value.unwrap()));
        }
        (a, b)
    }

    async fn collect_projected_a(stream: LiquidStream) -> Vec<i32> {
        let batches = stream
            .map(|batch| batch.expect("valid liquid stream batch"))
            .collect::<Vec<_>>()
            .await;
        let mut a = Vec::new();
        for batch in batches {
            let a_array = batch
                .column(0)
                .as_any()
                .downcast_ref::<Int32Array>()
                .unwrap();
            a.extend(a_array.iter().map(|value| value.unwrap()));
        }
        a
    }

    fn gt_filter(schema: SchemaRef, literal: i32) -> LiquidRowFilter {
        gt_filter_on(schema, "a", 0, literal)
    }

    fn gt_filter_on(
        schema: SchemaRef,
        col_name: &str,
        col_idx: usize,
        literal: i32,
    ) -> LiquidRowFilter {
        let expr: Arc<dyn PhysicalExpr> = Arc::new(BinaryExpr::new(
            Arc::new(Column::new(col_name, col_idx)),
            Operator::Gt,
            Arc::new(Literal::new(ScalarValue::Int32(Some(literal)))),
        ));
        let tmp_meta = tempfile::NamedTempFile::new().unwrap();
        write_two_row_group_file(tmp_meta.path(), schema.clone());
        let file = File::open(tmp_meta.path()).unwrap();
        let metadata = ArrowReaderMetadata::load(&file, ArrowReaderOptions::new()).unwrap();
        let builder = FilterCandidateBuilder::new(expr, schema);
        let candidate = builder.build(metadata.metadata()).unwrap().unwrap();
        let projection = candidate.projection(metadata.metadata());
        let predicate = LiquidPredicate::try_new(candidate, projection).unwrap();
        LiquidRowFilter::new(vec![predicate])
    }

    async fn make_liquid_stream_with_projection(
        max_memory_bytes: usize,
        max_disk_bytes: usize,
        row_filter: Option<LiquidRowFilter>,
        projection_columns: Vec<usize>,
    ) -> (
        LiquidStream,
        Arc<LiquidCacheParquet>,
        CachedFileRef,
        tempfile::TempDir,
    ) {
        let schema = Arc::new(Schema::new(vec![
            Field::new("a", DataType::Int32, false),
            Field::new("b", DataType::Int32, false),
        ]));
        let tmp_dir = tempfile::tempdir().unwrap();
        let parquet_path = tmp_dir.path().join("data.parquet");
        write_two_row_group_file(&parquet_path, schema.clone());
        let metadata_file = File::open(&parquet_path).unwrap();
        let reader_metadata =
            ArrowReaderMetadata::load(&metadata_file, ArrowReaderOptions::new()).unwrap();
        let object_store = Arc::new(LocalFileSystem::new_with_prefix(tmp_dir.path()).unwrap());
        let partitioned_file = PartitionedFile::new(
            "data.parquet",
            std::fs::metadata(&parquet_path).unwrap().len(),
        );
        let metrics = ExecutionPlanMetricsSet::new();
        let input = CachedMetaReaderFactory::new(object_store).create_liquid_reader(
            0,
            partitioned_file,
            None,
            &metrics,
        );

        let store = t4::mount(tmp_dir.path().join("liquid_cache.t4"))
            .await
            .unwrap();
        let cache = Arc::new(
            LiquidCacheParquet::new(
                4,
                max_memory_bytes,
                max_disk_bytes,
                store,
                Box::new(LiquidPolicy::new()),
                Box::new(Evict),
                Box::new(AlwaysHydrate::new()),
            )
            .await,
        );
        let cached_file = cache.register_or_get_file("data.parquet".to_string(), schema);
        let projection = ProjectionMask::roots(
            reader_metadata.metadata().file_metadata().schema_descr(),
            projection_columns,
        );
        let mut builder = LiquidStreamBuilder::new(input, Arc::clone(reader_metadata.metadata()))
            .with_batch_size(4)
            .with_row_groups(vec![0, 1])
            .with_projection(projection);
        if let Some(row_filter) = row_filter {
            builder = builder.with_row_filter(row_filter);
        }
        let stream = builder.build(cached_file.clone()).unwrap();
        (stream, cache, cached_file, tmp_dir)
    }

    async fn make_single_row_group_stream(
        parquet_a: Vec<i32>,
        projection_columns: Vec<usize>,
    ) -> (LiquidStream, CachedFileRef, tempfile::TempDir) {
        let schema = Arc::new(Schema::new(vec![
            Field::new("a", DataType::Int32, false),
            Field::new("b", DataType::Int32, false),
        ]));
        let tmp_dir = tempfile::tempdir().unwrap();
        let parquet_path = tmp_dir.path().join("data.parquet");
        write_single_row_group_file(&parquet_path, schema.clone(), parquet_a);
        let metadata_file = File::open(&parquet_path).unwrap();
        let reader_metadata =
            ArrowReaderMetadata::load(&metadata_file, ArrowReaderOptions::new()).unwrap();
        let object_store = Arc::new(LocalFileSystem::new_with_prefix(tmp_dir.path()).unwrap());
        let partitioned_file = PartitionedFile::new(
            "data.parquet",
            std::fs::metadata(&parquet_path).unwrap().len(),
        );
        let metrics = ExecutionPlanMetricsSet::new();
        let input = CachedMetaReaderFactory::new(object_store).create_liquid_reader(
            0,
            partitioned_file,
            None,
            &metrics,
        );

        let store = t4::mount(tmp_dir.path().join("liquid_cache.t4"))
            .await
            .unwrap();
        let cache = Arc::new(
            LiquidCacheParquet::new(
                4,
                usize::MAX,
                usize::MAX,
                store,
                Box::new(LiquidPolicy::new()),
                Box::new(Evict),
                Box::new(AlwaysHydrate::new()),
            )
            .await,
        );
        let cached_file = cache.register_or_get_file("data.parquet".to_string(), schema);
        let projection = ProjectionMask::roots(
            reader_metadata.metadata().file_metadata().schema_descr(),
            projection_columns,
        );
        let stream = LiquidStreamBuilder::new(input, Arc::clone(reader_metadata.metadata()))
            .with_batch_size(4)
            .with_row_groups(vec![0])
            .with_projection(projection)
            .build(cached_file.clone())
            .unwrap();
        (stream, cached_file, tmp_dir)
    }

    async fn insert_batches(
        row_group: &CachedRowGroupRef,
        column_id: usize,
        batch_payloads: &[(u16, &[i32])],
    ) {
        let column = row_group.get_column(column_id as u64).unwrap();
        for (batch_idx, values) in batch_payloads.iter() {
            let array: ArrayRef = Arc::new(Int32Array::from(values.to_vec()));
            column
                .insert(BatchID::from_raw(*batch_idx), array)
                .await
                .unwrap();
        }
    }

    async fn is_cached(row_group: &CachedRowGroupRef, column_id: usize, batch_idx: u16) -> bool {
        row_group
            .get_column(column_id as u64)
            .unwrap()
            .get_arrow_array_test_only(BatchID::from_raw(batch_idx))
            .await
            .is_some()
    }

    #[tokio::test]
    async fn cache_full_keeps_inserted_batches_and_skips_failed_inserts() {
        let one_array_memory = Arc::new(Int32Array::from(vec![0, 1, 2, 3])).get_array_memory_size();
        let (stream, _cache, cached_file, _tmp_dir) =
            make_liquid_stream(one_array_memory * 3, 0, None).await;

        let (a, b) = collect_liquid_values(stream).await;

        assert_eq!(a, vec![0, 1, 2, 3, 4, 5, 6, 7]);
        assert_eq!(b, vec![10, 11, 12, 13, 14, 15, 16, 17]);

        let row_group0 = cached_file.create_row_group(0, vec![]);
        let row_group1 = cached_file.create_row_group(1, vec![]);
        assert!(is_cached(&row_group0, 0, 0).await);
        assert!(is_cached(&row_group0, 1, 0).await);
        assert!(is_cached(&row_group1, 0, 0).await);
        assert!(!is_cached(&row_group1, 1, 0).await);
    }

    #[tokio::test]
    async fn cache_full_with_row_filter_keeps_lookaside_results_correct() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("a", DataType::Int32, false),
            Field::new("b", DataType::Int32, false),
        ]));
        let one_array_memory = Arc::new(Int32Array::from(vec![0, 1, 2, 3])).get_array_memory_size();
        let filter = gt_filter(schema, 2);
        let (stream, _cache, cached_file, _tmp_dir) =
            make_liquid_stream(one_array_memory * 3, 0, Some(filter)).await;

        let (a, b) = collect_liquid_values(stream).await;

        assert_eq!(a, vec![3, 4, 5, 6, 7]);
        assert_eq!(b, vec![13, 14, 15, 16, 17]);

        let row_group0 = cached_file.create_row_group(0, vec![]);
        let row_group1 = cached_file.create_row_group(1, vec![]);
        assert!(is_cached(&row_group0, 0, 0).await);
        assert!(is_cached(&row_group0, 1, 0).await);
        assert!(is_cached(&row_group1, 0, 0).await);
        assert!(!is_cached(&row_group1, 1, 0).await);
    }

    #[tokio::test]
    async fn mid_scan_eviction_recovers() {
        let (stream, _cache, cached_file, _tmp_dir) = make_liquid_stream(0, 0, None).await;

        let (a, b) = collect_liquid_values(stream).await;

        assert_eq!(a, vec![0, 1, 2, 3, 4, 5, 6, 7]);
        assert_eq!(b, vec![10, 11, 12, 13, 14, 15, 16, 17]);

        let row_group0 = cached_file.create_row_group(0, vec![]);
        let row_group1 = cached_file.create_row_group(1, vec![]);
        assert!(!is_cached(&row_group0, 0, 0).await);
        assert!(!is_cached(&row_group0, 1, 0).await);
        assert!(!is_cached(&row_group1, 0, 0).await);
        assert!(!is_cached(&row_group1, 1, 0).await);
    }

    #[tokio::test]
    async fn predicate_fallback_uses_predicate_projection() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("a", DataType::Int32, false),
            Field::new("b", DataType::Int32, false),
        ]));
        let one_array_memory = Arc::new(Int32Array::from(vec![0, 1, 2, 3])).get_array_memory_size();
        let filter = gt_filter_on(schema, "b", 1, 13);
        let (stream, _cache, cached_file, _tmp_dir) =
            make_liquid_stream_with_projection(one_array_memory * 3, 0, Some(filter), vec![0])
                .await;

        let a_values = collect_projected_a(stream).await;

        assert_eq!(a_values, vec![4, 5, 6, 7]);

        let row_group0 = cached_file.create_row_group(0, vec![]);
        let row_group1 = cached_file.create_row_group(1, vec![]);
        assert!(is_cached(&row_group0, 0, 0).await);
        assert!(is_cached(&row_group0, 1, 0).await);
        assert!(is_cached(&row_group1, 0, 0).await);
        assert!(!is_cached(&row_group1, 1, 0).await);
    }

    #[tokio::test]
    async fn missing_column_falls_back_to_parquet() {
        let (stream, _cache, cached_file, _tmp_dir) =
            make_liquid_stream(usize::MAX, usize::MAX, None).await;
        let row_group0 = cached_file.create_row_group(0, vec![]);
        let row_group1 = cached_file.create_row_group(1, vec![]);
        insert_batches(&row_group0, 0, &[(0, &[0, 1, 2, 3])]).await;
        insert_batches(&row_group1, 0, &[(0, &[4, 5, 6, 7])]).await;

        let (a, b) = collect_liquid_values(stream).await;

        assert_eq!(a, vec![0, 1, 2, 3, 4, 5, 6, 7]);
        assert_eq!(b, vec![10, 11, 12, 13, 14, 15, 16, 17]);
        assert!(is_cached(&row_group0, 1, 0).await);
        assert!(is_cached(&row_group1, 1, 0).await);
    }

    #[tokio::test]
    async fn fallback_stream_advances_across_misses() {
        let parquet_a = vec![
            100, 101, 102, 103, 4, 5, 6, 7, 200, 201, 202, 203, 12, 13, 14, 15,
        ];
        let (stream, cached_file, _tmp_dir) =
            make_single_row_group_stream(parquet_a, vec![0]).await;
        let row_group = cached_file.create_row_group(0, vec![]);
        insert_batches(&row_group, 0, &[(0, &[0, 1, 2, 3]), (2, &[8, 9, 10, 11])]).await;

        let a_values = collect_projected_a(stream).await;

        assert_eq!(a_values, (0..16).collect::<Vec<_>>());
        assert!(is_cached(&row_group, 0, 0).await);
        assert!(is_cached(&row_group, 0, 1).await);
        assert!(is_cached(&row_group, 0, 2).await);
        assert!(is_cached(&row_group, 0, 3).await);
    }
}
