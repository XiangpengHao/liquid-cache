use std::collections::VecDeque;
use std::sync::Arc;

use arrow::array::{Array, AsArray, BooleanArray, RecordBatch, RecordBatchReader};
use arrow::compute::{filter_record_batch, prep_null_mask_filter};
use arrow_schema::{ArrowError, DataType, Schema, SchemaRef};
use parquet::arrow::ProjectionMask;
use parquet::arrow::array_reader::ArrayReader;
use parquet::arrow::arrow_reader::{ArrowPredicate, RowSelection, RowSelector};

use super::LiquidRowFilter;
use crate::cache::{BatchID, LiquidCachedRowGroupRef};
use crate::reader::LiquidPredicate;
use crate::reader::runtime::liquid_predicate::is_predicate_supported_by_liquid;
use crate::reader::runtime::utils::take_next_batch;
use crate::utils::{boolean_buffer_and_then, row_selector_to_boolean_buffer};

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

pub(crate) struct LiquidBatchReader {
    liquid_cache: LiquidCachedRowGroupRef,
    current_batch_id: BatchID,
    selection: VecDeque<RowSelector>,
    schema: SchemaRef,
    batch_size: usize,
    row_filter: Option<LiquidRowFilter>,
    predicate_readers: Vec<Box<dyn ArrayReader>>,
    projection_reader: Box<dyn ArrayReader>,
    can_optimize_single_column_filter_projection: bool,
}

impl LiquidBatchReader {
    pub(crate) fn new(
        batch_size: usize,
        array_reader: Box<dyn ArrayReader>,
        selection: RowSelection,
        filter_readers: Vec<Box<dyn ArrayReader>>,
        mut row_filter: Option<LiquidRowFilter>,
        liquid_cache: LiquidCachedRowGroupRef,
        projection_mask: Option<ProjectionMask>,
    ) -> Self {
        let schema = match array_reader.get_data_type() {
            DataType::Struct(fields) => Schema::new(fields.clone()),
            _ => unreachable!("Struct array reader's data type is not struct!"),
        };

        let can_optimize_single_column_filter_projection =
            can_optimize_single_column_filter_projection(&mut row_filter, &projection_mask);

        Self {
            liquid_cache,
            current_batch_id: BatchID::from_raw(0),
            selection: selection.into(),
            schema: Arc::new(schema),
            batch_size,
            row_filter,
            predicate_readers: filter_readers,
            projection_reader: array_reader,
            can_optimize_single_column_filter_projection,
        }
    }

    pub(crate) fn take_filter(&mut self) -> Option<LiquidRowFilter> {
        self.row_filter.take()
    }

    fn build_predicate_filter(
        &mut self,
        selection: Vec<RowSelector>,
    ) -> Result<RowSelection, ArrowError> {
        let Some(filter) = &mut self.row_filter else {
            return Ok(selection.into());
        };

        debug_assert_eq!(
            self.predicate_readers.len(),
            filter.predicates.len(),
            "predicate readers and predicates should have the same length"
        );

        let mut input_selection = row_selector_to_boolean_buffer(&selection);
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
                predicate,
            );

            let boolean_mask = if let Some(result) = cached_result {
                reader.skip_records(selection_size).unwrap();
                result?
            } else {
                // slow case, where the predicate column is not cached
                // we need to read from parquet file
                let row_selection =
                    RowSelection::from_filters(&[BooleanArray::new(input_selection.clone(), None)]);

                let record_batch = read_record_batch_from_parquet(reader, row_selection.iter())?;
                let filter_mask = predicate.evaluate(record_batch).unwrap();
                let filter_mask = match filter_mask.null_count() {
                    0 => filter_mask,
                    _ => prep_null_mask_filter(&filter_mask),
                };
                let (buffer, null) = filter_mask.into_parts();
                assert!(null.is_none());
                buffer
            };
            input_selection = boolean_buffer_and_then(&input_selection, &boolean_mask);
        }
        Ok(RowSelection::from_filters(&[BooleanArray::new(
            input_selection,
            None,
        )]))
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
}

impl Iterator for LiquidBatchReader {
    type Item = Result<RecordBatch, ArrowError>;

    fn next(&mut self) -> Option<Self::Item> {
        match self.can_optimize_single_column_filter_projection {
            true => {
                let predicate = &mut self.row_filter.as_mut().unwrap().predicates[0];
                while let Some(selection) = take_next_batch(&mut self.selection, self.batch_size) {
                    match read_and_filter_single_column(
                        selection,
                        &mut self.projection_reader,
                        predicate,
                        &mut self.current_batch_id,
                    ) {
                        Ok(Some(record_batch)) => return Some(Ok(record_batch)),
                        Ok(None) => continue, // No rows passed the filter, try next batch
                        Err(e) => return Some(Err(e)),
                    }
                }
            }
            false => {
                while let Some(selection) = take_next_batch(&mut self.selection, self.batch_size) {
                    match self.build_predicate_filter(selection) {
                        Ok(filtered_selection) => {
                            match self.read_selection(filtered_selection) {
                                Ok(Some(record_batch)) => return Some(Ok(record_batch)),
                                Ok(None) => continue, // No rows to read, try next batch
                                Err(e) => return Some(Err(e)),
                            }
                        }
                        Err(e) => return Some(Err(e)),
                    }
                }
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

fn read_and_filter_single_column(
    selection: Vec<RowSelector>,
    projection_reader: &mut Box<dyn ArrayReader>,
    predicate: &mut LiquidPredicate,
    batch_id: &mut BatchID,
) -> Result<Option<RecordBatch>, ArrowError> {
    // This optimized path reads the single column data and applies the predicate directly
    // instead of the two-step process of building filter then reading selection
    for selector in &selection {
        if selector.skip {
            projection_reader.skip_records(selector.row_count)?;
        } else {
            projection_reader.read_records(selector.row_count)?;
        }
    }

    let array = projection_reader.consume_batch()?;
    let struct_array = array.as_struct();
    let record_batch = RecordBatch::from(struct_array);

    // Now apply the predicate to the record batch to get the boolean mask
    let boolean_mask = predicate.evaluate(record_batch.clone())?;

    let boolean_mask = match boolean_mask.null_count() {
        0 => boolean_mask,
        _ => prep_null_mask_filter(&boolean_mask),
    };

    let filtered_batch = filter_record_batch(&record_batch, &boolean_mask)?;

    batch_id.inc();

    Ok(if filtered_batch.num_rows() > 0 {
        Some(filtered_batch)
    } else {
        None
    })
}

/// Check if we can use the optimized single column filter+projection path
/// for example: SELECT A FROM table WHERE A > 10
///
/// In this case, there's no point to pushdown the filter.
fn can_optimize_single_column_filter_projection(
    row_filter: &mut Option<LiquidRowFilter>,
    projection_mask: &Option<ProjectionMask>,
) -> bool {
    let Some(filter) = row_filter else {
        return false;
    };

    if filter.predicates.len() != 1 {
        return false;
    }

    let predicate = &mut filter.predicates[0];
    let expr = predicate.physical_expr_physical_column_index();

    // Check if this predicate is supported by liquid array predicate evaluation
    // If so, return None to skip this optimization and let liquid predicate handle it
    if is_predicate_supported_by_liquid(expr) {
        return false;
    }

    let predicate_projection = predicate.projection();

    use crate::reader::runtime::get_predicate_column_id;
    let predicate_column_ids = get_predicate_column_id(predicate_projection);

    if predicate_column_ids.len() != 1 {
        return false;
    }
    let Some(projection_mask) = projection_mask else {
        return false;
    };

    if predicate_projection != projection_mask {
        return false;
    }
    true
}
