use std::collections::VecDeque;
use std::sync::Arc;

use arrow::array::{RecordBatch, RecordBatchReader};
use arrow_schema::{ArrowError, DataType, Schema, SchemaRef};
use parquet::arrow::array_reader::ArrayReader;
use parquet::arrow::arrow_reader::{ArrowPredicate, RowSelection, RowSelector};
use parquet_batch_reader::ParquetRecordBatchReader;

use crate::cache::LiquidCachedRowGroupRef;

use super::LiquidRowFilter;
use super::utils::take_next_selection;

pub(super) mod parquet_batch_reader;

/// Assumptions:
/// 1. The selection must be in range of the cache.
/// 2. The predicate must be operate on only one column.
fn build_predicate_from_cache_with_filter(
    cache: &LiquidCachedRowGroupRef,
    input_selection: RowSelection,
    predicate: &mut Box<dyn ArrowPredicate>,
) -> Result<RowSelection, ArrowError> {
    todo!()
}

struct LiquidBatchReader {
    parquet_reader: ParquetRecordBatchReader,
    liquid_cache: LiquidCachedRowGroupRef,
    current_row_id: usize,
    selection: VecDeque<RowSelector>,
    schema: SchemaRef,
    batch_size: usize,
}

impl LiquidBatchReader {
    fn new(
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

        let parquet_reader = ParquetRecordBatchReader::new(
            batch_size,
            array_reader,
            selection.clone(),
            filter_readers,
            row_filter,
        );

        Self {
            parquet_reader,
            liquid_cache,
            current_row_id: 0,
            selection: selection.into(),
            schema: Arc::new(schema),
            batch_size,
        }
    }
}

impl Iterator for LiquidBatchReader {
    type Item = Result<RecordBatch, ArrowError>;

    fn next(&mut self) -> Option<Self::Item> {
        let mut selected = 0;
        while let Some(cur_selection) =
            take_next_selection(&mut self.selection, self.batch_size - selected)
        {
            todo!()
        }

        if selected == 0 {
            return None;
        }

        todo!()
    }
}

impl RecordBatchReader for LiquidBatchReader {
    fn schema(&self) -> SchemaRef {
        self.schema.clone()
    }
}
