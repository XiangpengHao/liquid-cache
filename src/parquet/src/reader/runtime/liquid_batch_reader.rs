use std::collections::VecDeque;
use std::sync::Arc;

use arrow::array::{Array, AsArray, BooleanArray, RecordBatch, RecordBatchReader};
use arrow::buffer::BooleanBuffer;
use arrow::compute::{filter_record_batch, prep_null_mask_filter};
use arrow_schema::{ArrowError, DataType, Schema, SchemaRef};
use parquet::arrow::ProjectionMask;
use parquet::arrow::array_reader::ArrayReader;
use parquet::arrow::arrow_reader::{ArrowPredicate, RowSelection, RowSelector};

use crate::cache::{BatchID, LiquidCachedRowGroupRef};
use crate::reader::plantime::{LiquidPredicate, LiquidRowFilter};
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
    projection_mask: Option<ProjectionMask>,
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
            projection_mask,
            can_optimize_single_column_filter_projection,
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

        debug_assert_eq!(
            self.predicate_readers.len(),
            filter.predicates().len(),
            "predicate readers and predicates should have the same length"
        );

        let selection_size = input_selection.len();

        for (predicate, reader) in filter
            .predicates_mut()
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
                let result = result?;
                let filter_mask = match result.null_count() {
                    0 => result,
                    _ => prep_null_mask_filter(&result),
                };
                filter_mask.into_parts().0
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
        Ok(input_selection)
    }

    fn try_read_from_cache(
        &mut self,
        selection: &BooleanBuffer,
    ) -> Result<Option<RecordBatch>, ArrowError> {
        // Try to read from cache first - check if all projected columns are cached
        let mut all_cached = true;
        let mut cached_arrays = Vec::new();

        // Get the column indices that are actually projected
        use crate::reader::runtime::get_predicate_column_id;
        let column_indices: Vec<usize> = if let Some(ref projection_mask) = self.projection_mask {
            get_predicate_column_id(projection_mask)
        } else {
            (0..self.schema.fields().len()).collect()
        };

        for &column_idx in &column_indices {
            if let Some(column) = self.liquid_cache.get_column(column_idx as u64) {
                if let Some(array) =
                    column.get_arrow_array_with_filter(self.current_batch_id, selection)
                {
                    cached_arrays.push(array);
                } else {
                    all_cached = false;
                    break;
                }
            } else {
                all_cached = false;
                break;
            }
        }

        // It's possible that the projection is empty, e.g., SELECT COUNT(*) FROM table.
        if !cached_arrays.is_empty() && all_cached {
            // All columns are cached, skip reading from projection_reader to keep it in sync
            self.projection_reader.skip_records(selection.len())?;

            let batch = RecordBatch::try_new(self.schema.clone(), cached_arrays)?;
            return Ok(Some(batch));
        }

        // Cache miss - return None to indicate fallback needed
        Ok(None)
    }

    fn read_selection(
        &mut self,
        selection: BooleanBuffer,
    ) -> Result<Option<RecordBatch>, ArrowError> {
        // Try to read from cache first, this avoids the expensive read/skip operations.
        if let Some(batch) = self.try_read_from_cache(&selection)? {
            return Ok(Some(batch));
        }

        let selection = RowSelection::from_filters(&[BooleanArray::new(selection, None)]);

        if !selection.selects_any() {
            self.projection_reader.skip_records(self.batch_size)?;
            return Ok(None);
        }

        // Fall back to original approach when not all columns are cached,
        // note that this will still read from cache for columns that are cached.
        let batch = read_record_batch_from_parquet(&mut self.projection_reader, selection.iter())?;
        Ok(Some(batch))
    }
}

impl Iterator for LiquidBatchReader {
    type Item = Result<RecordBatch, ArrowError>;

    fn next(&mut self) -> Option<Self::Item> {
        match self.can_optimize_single_column_filter_projection {
            true => {
                let predicate = &mut self.row_filter.as_mut().unwrap().predicates_mut()[0];
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
                                Ok(Some(record_batch)) => {
                                    self.current_batch_id.inc();
                                    return Some(Ok(record_batch));
                                }
                                Ok(None) => {
                                    self.current_batch_id.inc();
                                    continue; // No rows to read, try next batch
                                }
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

    if filter.predicates().len() != 1 {
        return false;
    }

    let predicate = &mut filter.predicates_mut()[0];
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

#[cfg(test)]
mod tests {
    use crate::reader::plantime::FilterCandidateBuilder;

    use super::*;
    use arrow::array::{Int32Array, RecordBatch, StringArray};
    use arrow::datatypes::{DataType, Field, Schema};
    use datafusion::common::ScalarValue;
    use datafusion::datasource::schema_adapter::DefaultSchemaAdapterFactory;
    use datafusion::logical_expr::Operator;
    use datafusion::physical_expr::PhysicalExpr;
    use datafusion::physical_expr::expressions::{BinaryExpr, Column, Literal};
    use datafusion::physical_plan::metrics;
    use parquet::arrow::ProjectionMask;
    use parquet::arrow::arrow_reader::{ArrowReaderMetadata, ArrowReaderOptions};
    use parquet::arrow::arrow_writer::ArrowWriter;
    use std::sync::Arc;

    #[test]
    fn test_can_optimize_single_column_filter_projection() {
        // Create test schema
        let schema = Arc::new(Schema::new(vec![
            Field::new("col_a", DataType::Int32, false),
            Field::new("col_b", DataType::Utf8, false),
        ]));

        // Create test parquet metadata
        let temp_file = tempfile::NamedTempFile::new().unwrap();
        let mut writer =
            ArrowWriter::try_new(temp_file.reopen().unwrap(), schema.clone(), None).unwrap();

        let col_a = Arc::new(Int32Array::from(vec![1, 2, 3, 4, 5]));
        let col_b = Arc::new(StringArray::from(vec!["a", "b", "c", "d", "e"]));
        let batch = RecordBatch::try_new(schema.clone(), vec![col_a, col_b]).unwrap();

        writer.write(&batch).unwrap();
        writer.close().unwrap();

        let file = std::fs::File::open(temp_file.path()).unwrap();
        let metadata = ArrowReaderMetadata::load(&file, ArrowReaderOptions::new()).unwrap();
        let parquet_metadata = metadata.metadata();

        // Helper function to create predicate
        let create_predicate = |expr: Arc<dyn PhysicalExpr>| -> LiquidPredicate {
            let adapter_factory = Arc::new(DefaultSchemaAdapterFactory);
            let builder =
                FilterCandidateBuilder::new(expr, schema.clone(), schema.clone(), adapter_factory);
            let candidate = builder.build(parquet_metadata).unwrap().unwrap();
            let projection = candidate.projection(parquet_metadata);

            LiquidPredicate::try_new_with_metrics(
                candidate,
                projection,
                metrics::Count::new(),
                metrics::Count::new(),
                metrics::Time::new(),
            )
            .unwrap()
        };

        // Test 1: No filter - should return false
        assert!(!can_optimize_single_column_filter_projection(
            &mut None, &None
        ));

        // Test 2: Multiple predicates - should return false
        let expr1: Arc<dyn PhysicalExpr> = Arc::new(BinaryExpr::new(
            Arc::new(Column::new("col_a", 0)),
            Operator::Eq,
            Arc::new(Literal::new(ScalarValue::Int32(Some(1)))),
        ));
        let expr2: Arc<dyn PhysicalExpr> = Arc::new(BinaryExpr::new(
            Arc::new(Column::new("col_a", 0)),
            Operator::Gt,
            Arc::new(Literal::new(ScalarValue::Int32(Some(2)))),
        ));
        let mut filter = Some(LiquidRowFilter::new(vec![
            create_predicate(expr1),
            create_predicate(expr2),
        ]));
        assert!(!can_optimize_single_column_filter_projection(
            &mut filter,
            &None
        ));

        // Test 3: String literal (supported by liquid) - should return false
        let expr: Arc<dyn PhysicalExpr> = Arc::new(BinaryExpr::new(
            Arc::new(Column::new("col_b", 1)),
            Operator::Eq,
            Arc::new(Literal::new(ScalarValue::Utf8(Some("test".to_string())))),
        ));
        let mut filter = Some(LiquidRowFilter::new(vec![create_predicate(expr)]));
        assert!(!can_optimize_single_column_filter_projection(
            &mut filter,
            &None
        ));

        // Test 4: Primitive literal with no projection mask - should return false
        let expr: Arc<dyn PhysicalExpr> = Arc::new(BinaryExpr::new(
            Arc::new(Column::new("col_a", 0)),
            Operator::Eq,
            Arc::new(Literal::new(ScalarValue::Int32(Some(1)))),
        ));
        let mut filter = Some(LiquidRowFilter::new(vec![create_predicate(expr)]));
        assert!(!can_optimize_single_column_filter_projection(
            &mut filter,
            &None
        ));

        // Test 5: Projection mismatch - should return false
        let expr: Arc<dyn PhysicalExpr> = Arc::new(BinaryExpr::new(
            Arc::new(Column::new("col_a", 0)),
            Operator::Eq,
            Arc::new(Literal::new(ScalarValue::Int32(Some(1)))),
        ));
        let mut filter = Some(LiquidRowFilter::new(vec![create_predicate(expr)]));
        let projection_mask = ProjectionMask::roots(
            parquet_metadata.file_metadata().schema_descr(),
            vec![1], // Column 1 instead of column 0
        );
        assert!(!can_optimize_single_column_filter_projection(
            &mut filter,
            &Some(projection_mask)
        ));

        // Test 6: SUCCESS - primitive literal with matching projection - should return true
        let expr: Arc<dyn PhysicalExpr> = Arc::new(BinaryExpr::new(
            Arc::new(Column::new("col_a", 0)),
            Operator::Eq,
            Arc::new(Literal::new(ScalarValue::Int32(Some(1)))),
        ));
        let mut filter = Some(LiquidRowFilter::new(vec![create_predicate(expr)]));
        let projection_mask = ProjectionMask::roots(
            parquet_metadata.file_metadata().schema_descr(),
            vec![0], // Column 0 matches the predicate
        );
        assert!(can_optimize_single_column_filter_projection(
            &mut filter,
            &Some(projection_mask)
        ));
    }
}
