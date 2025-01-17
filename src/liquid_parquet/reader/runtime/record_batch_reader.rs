use std::collections::VecDeque;
use std::sync::Arc;

use arrow::array::Array;
use arrow::array::AsArray;
use arrow::array::RecordBatchReader;
use arrow::array::{ArrayRef, RecordBatch};
use arrow::compute::prep_null_mask_filter;
use arrow_schema::{ArrowError, DataType, Schema, SchemaRef};
use parquet::arrow::array_reader::ArrayReader;
use parquet::{
    arrow::arrow_reader::{RowSelection, RowSelector},
    errors::ParquetError,
};

use super::LiquidRowFilter;

fn read_selection(
    reader: &mut dyn ArrayReader,
    selection: &RowSelection,
) -> Result<ArrayRef, ParquetError> {
    for selector in selection.iter() {
        if selector.skip {
            let skipped = reader.skip_records(selector.row_count)?;
            debug_assert_eq!(skipped, selector.row_count, "failed to skip rows");
        } else {
            let read_records = reader.read_records(selector.row_count)?;
            debug_assert_eq!(read_records, selector.row_count, "failed to read rows");
        }
    }
    reader.consume_batch()
}

/// Take the next selection from the selection queue, and return the selection
/// whose selected row count is to_select or less (if input selection is exhausted).
fn take_next_selection(
    selection: &mut VecDeque<RowSelector>,
    to_select: usize,
) -> Option<RowSelection> {
    let mut current_selected = 0;
    let mut rt = Vec::new();
    while let Some(front) = selection.pop_front() {
        if front.skip {
            rt.push(front);
            continue;
        }

        if current_selected + front.row_count <= to_select {
            rt.push(front);
            current_selected += front.row_count;
        } else {
            let select = to_select - current_selected;
            let remaining = front.row_count - select;
            rt.push(RowSelector::select(select));
            selection.push_front(RowSelector::select(remaining));

            return Some(rt.into());
        }
    }
    if !rt.is_empty() {
        return Some(rt.into());
    }
    None
}

pub struct FilteredParquetRecordBatchReader {
    batch_size: usize,
    array_reader: Box<dyn ArrayReader>,
    predicate_readers: Vec<Box<dyn ArrayReader>>,
    schema: SchemaRef,
    selection: VecDeque<RowSelector>,
    row_filter: Option<LiquidRowFilter>,
}

impl FilteredParquetRecordBatchReader {
    #[allow(unused)]
    pub(crate) fn new(
        batch_size: usize,
        array_reader: Box<dyn ArrayReader>,
        selection: RowSelection,
        filter_readers: Vec<Box<dyn ArrayReader>>,
        row_filter: Option<LiquidRowFilter>,
    ) -> Self {
        let schema = match array_reader.get_data_type() {
            DataType::Struct(ref fields) => Schema::new(fields.clone()),
            _ => unreachable!("Struct array reader's data type is not struct!"),
        };

        Self {
            batch_size,
            array_reader,
            predicate_readers: filter_readers,
            schema: Arc::new(schema),
            selection: selection.into(),
            row_filter,
        }
    }

    #[allow(unused)]
    pub(crate) fn take_filter(&mut self) -> Option<LiquidRowFilter> {
        self.row_filter.take()
    }

    /// Take a selection, and return the new selection where the rows are filtered by the predicate.
    fn build_predicate_filter(
        &mut self,
        mut selection: RowSelection,
    ) -> Result<RowSelection, ArrowError> {
        match &mut self.row_filter {
            None => Ok(selection),
            Some(filter) => {
                debug_assert_eq!(
                    self.predicate_readers.len(),
                    filter.predicates.len(),
                    "predicate readers and predicates should have the same length"
                );

                for (predicate, reader) in filter
                    .predicates
                    .iter_mut()
                    .zip(self.predicate_readers.iter_mut())
                {
                    let array = read_selection(reader.as_mut(), &selection)?;
                    let batch = RecordBatch::from(array.as_struct_opt().ok_or_else(|| {
                        ArrowError::ParquetError(
                            "Struct array reader should return struct array".to_string(),
                        )
                    })?);
                    let input_rows = batch.num_rows();
                    let predicate_filter = predicate.evaluate(batch)?;
                    if predicate_filter.len() != input_rows {
                        return Err(ArrowError::ParquetError(format!(
                            "ArrowPredicate predicate returned {} rows, expected {input_rows}",
                            predicate_filter.len()
                        )));
                    }
                    let predicate_filter = match predicate_filter.null_count() {
                        0 => predicate_filter,
                        _ => prep_null_mask_filter(&predicate_filter),
                    };
                    let raw = RowSelection::from_filters(&[predicate_filter]);
                    selection = selection.and_then(&raw);
                }
                Ok(selection)
            }
        }
    }
}

impl Iterator for FilteredParquetRecordBatchReader {
    type Item = Result<RecordBatch, ArrowError>;

    fn next(&mut self) -> Option<Self::Item> {
        // With filter pushdown, it's very hard to predict the number of rows to return -- depends on the selectivity of the filter.
        // We can do one of the following:
        // 1. Add a coalescing step to coalesce the resulting batches.
        // 2. Ask parquet reader to collect more rows before returning.

        // Approach 1 has the drawback of extra overhead of coalesce batch, which can be painful to be efficient.
        // Code below implements approach 2, where we keep consuming the selection until we select at least 3/4 of the batch size.
        // It boils down to leveraging array_reader's ability to collect large batches natively,
        //    rather than concatenating multiple small batches.

        let mut selected = 0;
        while let Some(cur_selection) =
            take_next_selection(&mut self.selection, self.batch_size - selected)
        {
            let filtered_selection = match self.build_predicate_filter(cur_selection) {
                Ok(selection) => selection,
                Err(e) => return Some(Err(e)),
            };

            for selector in filtered_selection.iter() {
                if selector.skip {
                    self.array_reader.skip_records(selector.row_count).ok()?;
                } else {
                    self.array_reader.read_records(selector.row_count).ok()?;
                }
            }
            selected += filtered_selection.row_count();
            if selected >= (self.batch_size / 4 * 3) {
                break;
            }
        }
        if selected == 0 {
            return None;
        }

        let array = self.array_reader.consume_batch().ok()?;
        let struct_array = array
            .as_struct_opt()
            .ok_or_else(|| {
                ArrowError::ParquetError(
                    "Struct array reader should return struct array".to_string(),
                )
            })
            .ok()?;
        Some(Ok(RecordBatch::from(struct_array.clone())))
    }
}

impl RecordBatchReader for FilteredParquetRecordBatchReader {
    fn schema(&self) -> SchemaRef {
        self.schema.clone()
    }
}
