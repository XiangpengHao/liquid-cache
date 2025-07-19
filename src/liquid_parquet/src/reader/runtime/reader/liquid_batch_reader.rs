use std::collections::VecDeque;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};

use arrow::array::{Array, AsArray, BooleanArray, RecordBatch, RecordBatchReader};
use arrow::compute::prep_null_mask_filter;
use arrow_schema::{ArrowError, DataType, Schema, SchemaRef};
use parquet::arrow::array_reader::ArrayReader;
use parquet::arrow::arrow_reader::{RowSelection, RowSelector};

use crate::cache::{BatchID, LiquidCachedRowGroupRef};
use crate::reader::runtime::utils::{
    boolean_buffer_and_then, row_selector_to_boolean_buffer, take_next_batch,
};
use crate::{ABLATION_STUDY_MODE, AblationStudyMode};

use super::LiquidRowFilter;
use futures::{FutureExt, Stream};

fn read_record_batch_from_parquet<'a>(
    reader: &mut Box<dyn ArrayReader>,
    selection: impl Iterator<Item = &'a RowSelector>,
) -> Result<RecordBatch, ArrowError> {
    for selector in selection {
        if selector.skip {
            reader.skip_records(selector.row_count)?;
        } else {
            reader.read_records(selector.row_count)?;
        }
    }
    let array = reader.consume_batch()?;
    // TODO:
    // If consume_batch always returns a struct array, why don't we just return StrutArray instead of ArrayRef?
    // This is code smell. We need to fix this.
    let record_batch = RecordBatch::from(array.as_struct());
    Ok(record_batch)
}

enum LiquidBatchReaderState {
    Init,
    ReadingPredicate(Pin<Box<dyn Future<Output = Result<RowSelection, ArrowError>>>>),
}

pub(crate) struct LiquidBatchReader {
    liquid_cache: LiquidCachedRowGroupRef,
    current_batch_id: BatchID,
    selection: VecDeque<RowSelector>,
    schema: SchemaRef,
    batch_size: usize,
    row_filter: Option<LiquidRowFilter>,
    predicate_readers: Vec<Box<dyn ArrayReader>>,
    projection_reader: Box<dyn ArrayReader>,
    state: LiquidBatchReaderState,
}

impl LiquidBatchReader {
    pub(crate) fn new(
        batch_size: usize,
        array_reader: Box<dyn ArrayReader>,
        selection: RowSelection,
        filter_readers: Vec<Box<dyn ArrayReader>>,
        row_filter: Option<LiquidRowFilter>,
        liquid_cache: LiquidCachedRowGroupRef,
    ) -> Self {
        let schema = match array_reader.get_data_type() {
            DataType::Struct(fields) => Schema::new(fields.clone()),
            _ => unreachable!("Struct array reader's data type is not struct!"),
        };

        Self {
            liquid_cache,
            current_batch_id: BatchID::from_raw(0),
            selection: selection.into(),
            schema: Arc::new(schema),
            batch_size,
            row_filter,
            predicate_readers: filter_readers,
            projection_reader: array_reader,
            state: LiquidBatchReaderState::Init,
        }
    }

    pub(crate) fn take_filter(&mut self) -> Option<LiquidRowFilter> {
        self.row_filter.take()
    }

    async fn build_predicate_filter(
        &mut self,
        selection: Vec<RowSelector>,
    ) -> Result<RowSelection, ArrowError> {
        match &mut self.row_filter {
            None => Ok(selection.into()),
            Some(filter) => {
                debug_assert_eq!(
                    self.predicate_readers.len(),
                    filter.predicates.len(),
                    "predicate readers and predicates should have the same length"
                );

                let mut input_selection = row_selector_to_boolean_buffer(&selection);
                let mut final_selection = input_selection.clone();
                let selection_size = input_selection.len();

                for (predicate, reader) in filter
                    .predicates
                    .iter_mut()
                    .zip(self.predicate_readers.iter_mut())
                {
                    if input_selection.count_set_bits() == 0 {
                        reader.skip_records(selection_size).unwrap();
                        continue;
                    }

                    let cached_result = self.liquid_cache.evaluate_selection_with_predicate(
                        self.current_batch_id,
                        &input_selection,
                        predicate.as_mut(),
                    ).await;

                    let boolean_mask = if let Some(result) = cached_result {
                        reader.skip_records(selection_size).unwrap();
                        result?
                    } else {
                        // slow case, where the predicate column is not cached
                        // we need to read from parquet file
                        let row_selection = RowSelection::from_filters(&[BooleanArray::new(
                            input_selection.clone(),
                            None,
                        )]);

                        let record_batch =
                            read_record_batch_from_parquet(reader, row_selection.iter())?;
                        let filter_mask = predicate.evaluate(record_batch).unwrap();
                        let filter_mask = match filter_mask.null_count() {
                            0 => filter_mask,
                            _ => prep_null_mask_filter(&filter_mask),
                        };
                        let (buffer, null) = filter_mask.into_parts();
                        assert!(null.is_none());
                        buffer
                    };

                    if ABLATION_STUDY_MODE >= AblationStudyMode::SelectiveWithLateMaterialization {
                        input_selection = boolean_buffer_and_then(&input_selection, &boolean_mask);
                    } else {
                        final_selection = boolean_buffer_and_then(&final_selection, &boolean_mask);
                    }
                }
                if ABLATION_STUDY_MODE >= AblationStudyMode::SelectiveWithLateMaterialization {
                    final_selection = input_selection;
                }
                let filter = BooleanArray::new(final_selection, None);
                Ok(RowSelection::from_filters(&[filter]))
            }
        }
    }

    fn read_selection(
        &mut self,
        selection: RowSelection,
    ) -> Result<Option<RecordBatch>, ArrowError> {
        self.current_batch_id.inc();
        if !selection.selects_any() {
            self.projection_reader.skip_records(self.batch_size)?;
            return Ok(None);
        }

        for selector in selection.iter() {
            if selector.skip {
                self.projection_reader.skip_records(selector.row_count)?;
            } else {
                self.projection_reader.read_records(selector.row_count)?;
            }
        }
        let array = self.projection_reader.consume_batch()?;
        let struct_array = array.as_struct();
        let batch = RecordBatch::from(struct_array);
        Ok(Some(batch))
    }

    fn schema(&self) -> SchemaRef {
        self.schema.clone()
    }
}


impl Stream for LiquidBatchReader {
    type Item = Result<RecordBatch, ArrowError>;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let this = unsafe {self.get_unchecked_mut()};
        loop {
            match this.state {
                LiquidBatchReaderState::Init => {
                    if let Some(selection) = take_next_batch(&mut this.selection, this.batch_size) {
                        let fut = Box::pin(this.build_predicate_filter(selection));
                        this.state = LiquidBatchReaderState::ReadingPredicate(fut);
                    } else {
                        return Poll::Ready(None)
                    }
                },
                LiquidBatchReaderState::ReadingPredicate(fut) => {
                    
                }
            }
        }
        while let Some(selection) = take_next_batch(&mut this.selection, this.batch_size) {
            let row_selection: RowSelection;
            {
                // Spawn the async predicate filter as a future
                let fut = this.build_predicate_filter(selection);
                // Pin the future on the stack
                let mut fut = Box::pin(fut);

                // Poll the future
                match fut.as_mut().poll(cx) {
                    Poll::Ready(Ok(filtered_selection)) => {
                        row_selection = filtered_selection;
                    }
                    Poll::Ready(Err(e)) => return Poll::Ready(Some(Err(e))),
                    Poll::Pending => return Poll::Pending,
                }
            }

            match this.read_selection(row_selection) {
                Ok(Some(record_batch)) => return Poll::Ready(Some(Ok(record_batch))),
                Ok(None) => continue,
                Err(e) => return Poll::Ready(Some(Err(e))),
            }
        }
        Poll::Ready(None)
    }
}

// impl RecordBatchReader for LiquidBatchReader {
//     fn schema(&self) -> SchemaRef {
//         self.schema.clone()
//     }
// }
