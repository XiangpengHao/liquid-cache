use std::{collections::VecDeque, sync::Arc};

use arrow::array::{Array, RecordBatch};
use arrow::buffer::BooleanBuffer;
use arrow::compute::prep_null_mask_filter;
use arrow::record_batch::{RecordBatchOptions, RecordBatchReader};
use arrow_schema::{ArrowError, Field, Schema, SchemaRef};
use parquet::arrow::arrow_reader::{ArrowPredicate, RowSelection, RowSelector};

use crate::cache::{BatchID, LiquidCachedRowGroupRef};
use crate::reader::plantime::LiquidRowFilter;
use crate::reader::runtime::utils::take_next_batch;
use crate::utils::{boolean_buffer_and_then, row_selector_to_boolean_buffer};

pub(crate) struct LiquidBatchReader {
    liquid_cache: LiquidCachedRowGroupRef,
    current_batch_id: BatchID,
    selection: VecDeque<RowSelector>,
    schema: SchemaRef,
    batch_size: usize,
    row_filter: Option<LiquidRowFilter>,
    projection_columns: Vec<usize>,
}

impl LiquidBatchReader {
    pub(crate) fn new(
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

    pub(crate) fn take_filter(&mut self) -> Option<LiquidRowFilter> {
        self.row_filter.take()
    }

    fn build_predicate_filter(
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

            let column_ids = predicate.predicate_column_ids();

            let boolean_array = if column_ids.is_empty() {
                let options = RecordBatchOptions::new().with_row_count(Some(input_selection.len()));
                let empty_batch = RecordBatch::try_new_with_options(
                    Arc::new(Schema::new(Vec::<Field>::new())),
                    Vec::new(),
                    &options,
                )?;
                predicate.evaluate(empty_batch)?
            } else {
                let mut arrays = Vec::with_capacity(column_ids.len());
                let mut fields = Vec::with_capacity(column_ids.len());

                for column_id in column_ids {
                    let column =
                        self.liquid_cache
                            .get_column(column_id as u64)
                            .ok_or_else(|| {
                                ArrowError::ComputeError(format!(
                                    "predicate column {column_id} not present in liquid cache"
                                ))
                            })?;
                    let array = column
                        .get_arrow_array_with_filter(self.current_batch_id, &input_selection)
                        .ok_or_else(|| {
                            ArrowError::ComputeError(format!(
                                "predicate column {column_id} batch {} not cached",
                                *self.current_batch_id as usize
                            ))
                        })?;
                    arrays.push(array);
                    fields.push(column.field());
                }

                let schema = Arc::new(Schema::new(fields));
                let record_batch = RecordBatch::try_new(schema, arrays)?;
                predicate.evaluate(record_batch)?
            };

            let boolean_mask = if boolean_array.null_count() == 0 {
                boolean_array.into_parts().0
            } else {
                prep_null_mask_filter(&boolean_array).into_parts().0
            };

            input_selection = boolean_buffer_and_then(&input_selection, &boolean_mask);
        }

        Ok(input_selection)
    }

    fn read_from_cache(
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

impl Iterator for LiquidBatchReader {
    type Item = Result<RecordBatch, ArrowError>;

    fn next(&mut self) -> Option<Self::Item> {
        while let Some(selection) = take_next_batch(&mut self.selection, self.batch_size) {
            let filtered_selection = match self.build_predicate_filter(selection) {
                Ok(buffer) => buffer,
                Err(e) => return Some(Err(e)),
            };

            match self.read_from_cache(&filtered_selection) {
                Ok(Some(batch)) => {
                    self.current_batch_id.inc();
                    return Some(Ok(batch));
                }
                Ok(None) => {
                    self.current_batch_id.inc();
                    continue;
                }
                Err(e) => return Some(Err(e)),
            }
        }
        None
    }
}

impl RecordBatchReader for LiquidBatchReader {
    fn schema(&self) -> SchemaRef {
        self.schema.clone()
    }
}
