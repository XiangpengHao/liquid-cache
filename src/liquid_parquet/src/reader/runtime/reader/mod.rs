use std::collections::VecDeque;
use std::sync::Arc;

use arrow::array::{Array, AsArray, BooleanArray, RecordBatch, RecordBatchReader};
use arrow::buffer::BooleanBuffer;
use arrow::compute::prep_null_mask_filter;
use arrow_schema::{ArrowError, DataType, Field, Schema, SchemaRef};
use parquet::arrow::array_reader::ArrayReader;
use parquet::arrow::arrow_reader::{ArrowPredicate, RowSelection, RowSelector};

use crate::cache::LiquidCachedRowGroupRef;
use crate::reader::runtime::parquet_bridge::get_predicate_column_id;
use crate::reader::runtime::utils::{boolean_buffer_and_then, take_next_batch};

use super::LiquidRowFilter;
use super::utils::row_selector_to_boolean_buffer;

mod cached_array_reader;
pub(crate) use cached_array_reader::build_cached_array_reader;
pub(super) mod cached_page;

#[cfg(test)]
mod tests;

fn build_predicate_from_cache(
    cache: &LiquidCachedRowGroupRef,
    row_id: usize,
    input_selection: &BooleanBuffer,
    predicate: &mut Box<dyn ArrowPredicate>,
) -> Option<BooleanBuffer> {
    let projection = predicate.projection();
    let column_id = get_predicate_column_id(projection);
    let cache = cache.get_column(column_id)?;
    let result = cache
        .eval_selection_with_predicate(row_id, input_selection, predicate.as_mut())
        .transpose()
        .unwrap();
    result
}

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
    let record_batch = RecordBatch::try_new(
        Arc::new(Schema::new(vec![Field::new(
            "-",
            array.data_type().clone(),
            array.is_nullable(),
        )])),
        vec![array],
    )?;
    Ok(record_batch)
}

pub(super) struct LiquidBatchReader {
    liquid_cache: LiquidCachedRowGroupRef,
    current_row_id: usize,
    selection: VecDeque<RowSelector>,
    schema: SchemaRef,
    batch_size: usize,
    row_filter: Option<LiquidRowFilter>,
    predicate_readers: Vec<Box<dyn ArrayReader>>,
    projection_reader: Box<dyn ArrayReader>,
}

impl LiquidBatchReader {
    pub(super) fn new(
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
            current_row_id: 0,
            selection: selection.into(),
            schema: Arc::new(schema),
            batch_size,
            row_filter,
            predicate_readers: filter_readers,
            projection_reader: array_reader,
        }
    }

    pub(super) fn take_filter(&mut self) -> Option<LiquidRowFilter> {
        self.row_filter.take()
    }

    fn build_predicate_filter(
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

                let mut cur_selection = row_selector_to_boolean_buffer(&selection);

                for (predicate, reader) in filter
                    .predicates
                    .iter_mut()
                    .zip(self.predicate_readers.iter_mut())
                {
                    if let Some(result) = build_predicate_from_cache(
                        &self.liquid_cache,
                        self.current_row_id,
                        &cur_selection,
                        predicate,
                    ) {
                        cur_selection = boolean_buffer_and_then(&cur_selection, &result);
                        reader.skip_records(self.batch_size).unwrap();
                    } else {
                        // slow case, where the predicate column is not cached
                        // we need to read from parquet file
                        let record_batch =
                            read_record_batch_from_parquet(reader, selection.iter())?;
                        let filter_mask = predicate.evaluate(record_batch).unwrap();
                        let filter_mask = match filter_mask.null_count() {
                            0 => filter_mask,
                            _ => prep_null_mask_filter(&filter_mask),
                        };
                        let (buffer, null) = filter_mask.into_parts();
                        assert!(null.is_none());
                        cur_selection = boolean_buffer_and_then(&cur_selection, &buffer);
                    }
                }
                let filter = BooleanArray::new(cur_selection, None);
                Ok(RowSelection::from_filters(&[filter]))
            }
        }
    }

    fn read_selection(&mut self, selection: RowSelection) -> Result<RecordBatch, ArrowError> {
        self.current_row_id += self.batch_size;
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
        Ok(batch)
    }
}

impl Iterator for LiquidBatchReader {
    type Item = Result<RecordBatch, ArrowError>;

    fn next(&mut self) -> Option<Self::Item> {
        if let Some(selection) = take_next_batch(&mut self.selection, self.batch_size) {
            let filtered_selection = self.build_predicate_filter(selection).unwrap();
            let record_batch = self.read_selection(filtered_selection).unwrap();
            return Some(Ok(record_batch));
        }
        None
    }
}

impl RecordBatchReader for LiquidBatchReader {
    fn schema(&self) -> SchemaRef {
        self.schema.clone()
    }
}
