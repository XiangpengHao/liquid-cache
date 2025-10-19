use std::collections::VecDeque;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};

use arrow::array::{Array, RecordBatch};
use arrow::buffer::BooleanBuffer;
use arrow::compute::prep_null_mask_filter;
use arrow::record_batch::RecordBatchOptions;
use arrow_schema::{ArrowError, SchemaRef};
use futures::{Stream, future::BoxFuture};
use parquet::arrow::arrow_reader::{RowSelection, RowSelector};

use crate::cache::{BatchID, LiquidCachedRowGroupRef};
use crate::reader::plantime::LiquidRowFilter;
use crate::reader::runtime::utils::take_next_batch;
use crate::utils::{boolean_buffer_and_then, row_selector_to_boolean_buffer};

pub(crate) struct LiquidCacheReader {
    inner: Option<LiquidCacheReaderInner>,
    pending: Option<ReaderPending>,
}

enum ReaderPending {
    Process(BoxFuture<'static, (LiquidCacheReaderInner, ProcessResult)>),
}

enum ProcessResult {
    Emit(Result<RecordBatch, ArrowError>),
    Skip,
}

struct LiquidCacheReaderInner {
    liquid_cache: LiquidCachedRowGroupRef,
    current_batch_id: BatchID,
    selection: VecDeque<RowSelector>,
    schema: SchemaRef,
    batch_size: usize,
    row_filter: Option<LiquidRowFilter>,
    projection_columns: Vec<usize>,
}

impl LiquidCacheReader {
    pub(crate) fn new(
        batch_size: usize,
        selection: RowSelection,
        row_filter: Option<LiquidRowFilter>,
        liquid_cache: LiquidCachedRowGroupRef,
        projection_columns: Vec<usize>,
        schema: SchemaRef,
    ) -> Self {
        let inner = LiquidCacheReaderInner::new(
            batch_size,
            selection,
            row_filter,
            liquid_cache,
            projection_columns,
            Arc::clone(&schema),
        );
        Self {
            inner: Some(inner),
            pending: None,
        }
    }

    pub(crate) fn take_filter(&mut self) -> Option<LiquidRowFilter> {
        debug_assert!(
            self.pending.is_none(),
            "cannot take filter while pending work exists"
        );
        self.inner.as_mut().and_then(|inner| inner.take_filter())
    }
}

impl Stream for LiquidCacheReader {
    type Item = Result<RecordBatch, ArrowError>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        loop {
            if let Some(pending) = self.pending.as_mut() {
                match pending {
                    ReaderPending::Process(fut) => match fut.as_mut().poll(cx) {
                        Poll::Pending => return Poll::Pending,
                        Poll::Ready((inner, result)) => {
                            self.inner = Some(inner);
                            self.pending = None;

                            match result {
                                ProcessResult::Emit(item) => return Poll::Ready(Some(item)),
                                ProcessResult::Skip => continue,
                            }
                        }
                    },
                }
            } else {
                let mut inner = self.inner.take().expect("reader state should be available");

                match take_next_batch(&mut inner.selection, inner.batch_size) {
                    Some(selection) => {
                        let future = inner.process_selection(selection);
                        self.pending = Some(ReaderPending::Process(future));
                        continue;
                    }
                    None => {
                        self.inner = Some(inner);
                        return Poll::Ready(None);
                    }
                }
            }
        }
    }
}

impl LiquidCacheReaderInner {
    fn new(
        batch_size: usize,
        selection: RowSelection,
        row_filter: Option<LiquidRowFilter>,
        liquid_cache: LiquidCachedRowGroupRef,
        projection_columns: Vec<usize>,
        schema: SchemaRef,
    ) -> Self {
        Self {
            liquid_cache,
            current_batch_id: BatchID::from_raw(0),
            selection: selection.into(),
            schema,
            batch_size,
            row_filter,
            projection_columns,
        }
    }

    fn take_filter(&mut self) -> Option<LiquidRowFilter> {
        self.row_filter.take()
    }

    fn process_selection(
        self,
        selection: Vec<RowSelector>,
    ) -> BoxFuture<'static, (Self, ProcessResult)> {
        Box::pin(async move {
            let mut inner = self;

            let result = match inner.build_predicate_filter(selection).await {
                Ok(buffer) => match inner.read_from_cache(&buffer).await {
                    Ok(Some(batch)) => {
                        inner.current_batch_id.inc();
                        ProcessResult::Emit(Ok(batch))
                    }
                    Ok(None) => {
                        inner.current_batch_id.inc();
                        ProcessResult::Skip
                    }
                    Err(e) => ProcessResult::Emit(Err(e)),
                },
                Err(e) => ProcessResult::Emit(Err(e)),
            };

            (inner, result)
        })
    }

    async fn build_predicate_filter(
        &mut self,
        selection: Vec<RowSelector>,
    ) -> Result<BooleanBuffer, ArrowError> {
        let mut input_selection = row_selector_to_boolean_buffer(&selection);

        let Some(filter) = &mut self.row_filter else {
            return Ok(input_selection);
        };

        for predicate in filter.predicates_mut() {
            if input_selection.count_set_bits() == 0 {
                break;
            }

            let boolean_array = self
                .liquid_cache
                .evaluate_selection_with_predicate(
                    self.current_batch_id,
                    &input_selection,
                    predicate,
                )
                .expect("item must be in cache")?;

            let boolean_mask = if boolean_array.null_count() == 0 {
                boolean_array.into_parts().0
            } else {
                prep_null_mask_filter(&boolean_array).into_parts().0
            };

            input_selection = boolean_buffer_and_then(&input_selection, &boolean_mask);
        }

        Ok(input_selection)
    }

    async fn read_from_cache(
        &self,
        selection: &BooleanBuffer,
    ) -> Result<Option<RecordBatch>, ArrowError> {
        let selected_rows = selection.count_set_bits();
        if selected_rows == 0 {
            return Ok(None);
        }

        if self.projection_columns.is_empty() {
            let options = RecordBatchOptions::new().with_row_count(Some(selected_rows));
            let batch =
                RecordBatch::try_new_with_options(self.schema.clone(), Vec::new(), &options)?;
            return Ok(Some(batch));
        }

        let mut arrays = Vec::with_capacity(self.projection_columns.len());
        for &column_idx in &self.projection_columns {
            let column = self
                .liquid_cache
                .get_column(column_idx as u64)
                .ok_or_else(|| {
                    ArrowError::ComputeError(format!(
                        "column {column_idx} not present in liquid cache"
                    ))
                })?;

            let array = column
                .get_arrow_array_with_filter(self.current_batch_id, selection)
                .ok_or_else(|| {
                    ArrowError::ComputeError(format!(
                        "column {column_idx} batch {} not cached",
                        *self.current_batch_id as usize
                    ))
                })?;

            arrays.push(array);
        }

        RecordBatch::try_new(self.schema.clone(), arrays).map(Some)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        cache::LiquidCache,
        reader::{FilterCandidateBuilder, LiquidPredicate, LiquidRowFilter},
    };
    use arrow::array::{ArrayRef, Int32Array};
    use arrow::record_batch::RecordBatch;
    use arrow_schema::{DataType, Field, Schema, SchemaRef};
    use datafusion::{
        datasource::schema_adapter::DefaultSchemaAdapterFactory,
        logical_expr::Operator,
        physical_expr::PhysicalExpr,
        physical_expr::expressions::{BinaryExpr, Column, Literal},
        scalar::ScalarValue,
    };
    use futures::{StreamExt, pin_mut};
    use liquid_cache_storage::cache::squeeze_policies::Evict;
    use liquid_cache_storage::cache_policies::LiquidPolicy;
    use parquet::arrow::{
        ArrowWriter,
        arrow_reader::{ArrowReaderMetadata, ArrowReaderOptions, RowSelection, RowSelector},
    };
    use std::sync::Arc;

    fn make_row_group(
        batch_size: usize,
        batches: &[Vec<i32>],
    ) -> (LiquidCachedRowGroupRef, SchemaRef) {
        let tmp_dir = tempfile::tempdir().unwrap();
        let cache = LiquidCache::new(
            batch_size,
            usize::MAX,
            tmp_dir.path().to_path_buf(),
            Box::new(LiquidPolicy::new()),
            Box::new(Evict),
        );
        let file = cache.register_or_get_file("test".to_string());
        let row_group = file.row_group(0);

        let field = Arc::new(Field::new("col0", DataType::Int32, false));
        let column = row_group.create_column(0, field.clone());

        for (idx, values) in batches.iter().enumerate() {
            let array: ArrayRef = Arc::new(Int32Array::from(values.clone()));
            column
                .insert(BatchID::from_raw(idx as u16), array)
                .expect("cache insert");
        }

        let schema = Arc::new(Schema::new(vec![Field::new(
            "col0",
            DataType::Int32,
            false,
        )]));

        (row_group, schema)
    }

    fn flatten_batches(batches: &[Vec<i32>]) -> Vec<i32> {
        batches.iter().flat_map(|b| b.iter().copied()).collect()
    }

    fn collect_batches(reader: LiquidCacheReader) -> Vec<RecordBatch> {
        futures::executor::block_on(
            reader
                .map(|batch| batch.expect("valid record batch"))
                .collect::<Vec<_>>(),
        )
    }

    fn as_i32_values(batch: &RecordBatch) -> Vec<i32> {
        let array = batch
            .column(0)
            .as_any()
            .downcast_ref::<Int32Array>()
            .expect("int column");
        array.iter().map(|v| v.expect("non-null")).collect()
    }

    fn build_filter(
        schema: SchemaRef,
        values: &[i32],
        expr: Arc<dyn PhysicalExpr>,
    ) -> LiquidRowFilter {
        let tmp_meta = tempfile::NamedTempFile::new().unwrap();
        let array: ArrayRef = Arc::new(Int32Array::from(values.to_vec()));
        let batch = RecordBatch::try_new(Arc::clone(&schema), vec![array.clone()]).unwrap();
        let mut writer =
            ArrowWriter::try_new(tmp_meta.reopen().unwrap(), Arc::clone(&schema), None).unwrap();
        writer.write(&batch).unwrap();
        writer.close().unwrap();

        let file = std::fs::File::open(tmp_meta.path()).unwrap();
        let metadata = ArrowReaderMetadata::load(&file, ArrowReaderOptions::new()).unwrap();

        let adapter_factory = Arc::new(DefaultSchemaAdapterFactory);
        let builder = FilterCandidateBuilder::new(
            expr,
            Arc::clone(&schema),
            Arc::clone(&schema),
            adapter_factory,
        );
        let candidate = builder.build(metadata.metadata()).unwrap().unwrap();
        let projection = candidate.projection(metadata.metadata());
        let predicate = LiquidPredicate::try_new(candidate, projection).unwrap();

        LiquidRowFilter::new(vec![predicate])
    }

    fn make_gt_filter(schema: SchemaRef, values: &[i32], literal: i32) -> LiquidRowFilter {
        let expr: Arc<dyn PhysicalExpr> = Arc::new(BinaryExpr::new(
            Arc::new(Column::new("col0", 0)),
            Operator::Gt,
            Arc::new(Literal::new(ScalarValue::Int32(Some(literal)))),
        ));
        build_filter(schema, values, expr)
    }

    #[test]
    fn reads_batches_in_order() {
        let batch_size = 2;
        let (row_group, schema) = make_row_group(batch_size, &[vec![1, 2], vec![3, 4]]);
        let selection = RowSelection::from(vec![RowSelector::select(4)]);

        let reader =
            LiquidCacheReader::new(batch_size, selection, None, row_group, vec![0], schema);

        let batches = collect_batches(reader);
        assert_eq!(batches.len(), 2);
        assert_eq!(as_i32_values(&batches[0]), vec![1, 2]);
        assert_eq!(as_i32_values(&batches[1]), vec![3, 4]);
    }

    #[test]
    fn skips_unselected_batches() {
        let batch_size = 2;
        let (row_group, schema) = make_row_group(batch_size, &[vec![1, 2], vec![3, 4]]);
        let selection = RowSelection::from(vec![RowSelector::skip(2), RowSelector::select(2)]);

        let reader =
            LiquidCacheReader::new(batch_size, selection, None, row_group, vec![0], schema);

        let batches = collect_batches(reader);
        assert_eq!(batches.len(), 1);
        assert_eq!(as_i32_values(&batches[0]), vec![3, 4]);
    }

    #[test]
    fn empty_projection_emits_schema_only_batches() {
        let batch_size = 2;
        let (row_group, _) = make_row_group(batch_size, &[vec![10, 11]]);
        let selection = RowSelection::from(vec![RowSelector::select(2)]);

        let reader = LiquidCacheReader::new(
            batch_size,
            selection,
            None,
            row_group,
            Vec::new(),
            Arc::new(Schema::new(Vec::<Field>::new())),
        );

        let batches = collect_batches(reader);
        assert_eq!(batches.len(), 1);
        let batch = &batches[0];
        assert_eq!(batch.num_columns(), 0);
        assert_eq!(batch.num_rows(), 2);
    }

    #[test]
    fn take_filter_returns_stored_filter() {
        let batch_size = 2;
        let (row_group, schema) = make_row_group(batch_size, &[vec![1, 2]]);
        let selection = RowSelection::from(vec![RowSelector::select(2)]);
        let filter = LiquidRowFilter::new(Vec::new());

        let mut reader = LiquidCacheReader::new(
            batch_size,
            selection,
            Some(filter),
            row_group,
            vec![0],
            schema,
        );

        assert!(reader.take_filter().is_some());
        assert!(reader.take_filter().is_none());
    }

    #[test]
    fn predicate_filters_rows_across_batches() {
        let batches = vec![vec![1, 2], vec![3, 4]];
        let batch_size = 2;
        let all_values = flatten_batches(&batches);
        let (row_group, schema) = make_row_group(batch_size, &batches);
        let filter = make_gt_filter(Arc::clone(&schema), &all_values, 2);
        let selection = RowSelection::from(vec![RowSelector::select(4)]);

        let reader = LiquidCacheReader::new(
            batch_size,
            selection,
            Some(filter),
            row_group,
            vec![0],
            schema,
        );

        let batches = collect_batches(reader);
        assert_eq!(batches.len(), 1);
        assert_eq!(as_i32_values(&batches[0]), vec![3, 4]);
    }

    fn make_or_filter(
        schema: SchemaRef,
        values: &[i32],
        gt_literal: i32,
        lt_literal: i32,
    ) -> LiquidRowFilter {
        let cond1: Arc<dyn PhysicalExpr> = Arc::new(BinaryExpr::new(
            Arc::new(Column::new("col0", 0)),
            Operator::Gt,
            Arc::new(Literal::new(ScalarValue::Int32(Some(gt_literal)))),
        ));
        let cond2: Arc<dyn PhysicalExpr> = Arc::new(BinaryExpr::new(
            Arc::new(Column::new("col0", 0)),
            Operator::Lt,
            Arc::new(Literal::new(ScalarValue::Int32(Some(lt_literal)))),
        ));
        let expr: Arc<dyn PhysicalExpr> = Arc::new(BinaryExpr::new(cond1, Operator::Or, cond2));
        build_filter(schema, values, expr)
    }

    #[test]
    fn predicate_filters_or_rows() {
        let batches = vec![vec![1, 2], vec![3, 4], vec![5, 6]];
        let batch_size = 2;
        let all_values = flatten_batches(&batches);
        let (row_group, schema) = make_row_group(batch_size, &batches);
        let filter = make_or_filter(Arc::clone(&schema), &all_values, 4, 2);
        let selection = RowSelection::from(vec![RowSelector::select(6)]);

        let reader = LiquidCacheReader::new(
            batch_size,
            selection,
            Some(filter),
            row_group,
            vec![0],
            schema,
        );

        let batches = collect_batches(reader);
        assert_eq!(batches.len(), 2);
        assert_eq!(as_i32_values(&batches[0]), vec![1]);
        assert_eq!(as_i32_values(&batches[1]), vec![5, 6]);
    }

    #[test]
    fn predicate_combines_with_selection() {
        let batches = vec![vec![1, 2, 3, 4]];
        let batch_size = 4;
        let all_values = flatten_batches(&batches);
        let (row_group, schema) = make_row_group(batch_size, &batches);
        let filter = make_gt_filter(Arc::clone(&schema), &all_values, 2);
        let selection = RowSelection::from(vec![
            RowSelector::skip(1),
            RowSelector::select(2),
            RowSelector::skip(1),
        ]);

        let reader = LiquidCacheReader::new(
            batch_size,
            selection,
            Some(filter),
            row_group,
            vec![0],
            schema,
        );

        let mut batches = collect_batches(reader);
        assert_eq!(batches.len(), 1);
        assert_eq!(as_i32_values(&batches.pop().unwrap()), vec![3]);
    }

    #[test]
    fn predicate_can_filter_all_rows() {
        let batches = vec![vec![1, 2]];
        let batch_size = 2;
        let all_values = flatten_batches(&batches);
        let (row_group, schema) = make_row_group(batch_size, &batches);
        let filter = make_gt_filter(Arc::clone(&schema), &all_values, 10);
        let selection = RowSelection::from(vec![RowSelector::select(2)]);

        let reader = LiquidCacheReader::new(
            batch_size,
            selection,
            Some(filter),
            row_group,
            vec![0],
            schema,
        );

        let next_batch = futures::executor::block_on(async {
            pin_mut!(reader);
            reader.next().await
        });

        assert!(next_batch.is_none());
    }
}
