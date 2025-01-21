use std::collections::VecDeque;
use std::sync::Arc;

use arrow::array::{ArrayDataBuilder, BooleanArray, RecordBatch, StructArray, UInt32Array};
use arrow::compute::concat_batches;
use arrow_schema::{DataType, Field, Fields, Schema, SchemaRef};
use parquet::arrow::arrow_reader::{ArrowPredicate, RowSelector};

use super::{ArrayIdentifier, BooleanSelection, LiquidCache};

/// Iterator over `RecordBatch` for `get_record_batch_by_slice`.
pub struct SlicedRecordBatchIter<'a> {
    cache: &'a LiquidCache,
    row_group_id: usize,
    selection: VecDeque<RowSelector>,
    schema: SchemaRef,
    parquet_column_ids: Vec<usize>,
    row_id: usize,
    current_selected: usize,
    batch_size: usize,
}

impl<'a> SlicedRecordBatchIter<'a> {
    pub(crate) fn new(
        cache: &'a LiquidCache,
        row_group_id: usize,
        selection: VecDeque<RowSelector>,
        mut schema: SchemaRef,
        parquet_column_ids: Vec<usize>,
        batch_size: usize,
    ) -> Self {
        if parquet_column_ids.is_empty() {
            let data_type = DataType::Struct(Fields::empty());
            let fields = vec![Field::new("empty", data_type, false)];
            schema = Arc::new(Schema::new(fields));
        }
        SlicedRecordBatchIter {
            cache,
            row_group_id,
            selection,
            schema,
            parquet_column_ids,
            row_id: 0,
            current_selected: 0,
            batch_size,
        }
    }

    /// Coalesces the output of the iterator into batches of a specified size.
    pub fn into_coalesced(self, batch_size: usize) -> CoalescedIter<Self> {
        CoalescedIter::new(self, batch_size)
    }
}

impl Iterator for SlicedRecordBatchIter<'_> {
    type Item = RecordBatch;

    fn next(&mut self) -> Option<Self::Item> {
        while let Some(row_selector) = self.selection.pop_front() {
            if row_selector.skip {
                self.row_id += row_selector.row_count;
                continue;
            }

            assert!(self.current_selected < row_selector.row_count);
            let current_row_id = self.row_id + self.current_selected;
            let batch_row_id = (current_row_id / self.batch_size) * self.batch_size;
            let batch_row_count = self.batch_size - (current_row_id % self.batch_size);
            let offset = current_row_id.saturating_sub(batch_row_id);
            let want_to_select = std::cmp::min(
                batch_row_count,
                row_selector.row_count - self.current_selected,
            );

            let record_batch = if self.parquet_column_ids.is_empty() {
                make_dummy_record_batch(&self.schema, want_to_select)
            } else {
                let mut columns = Vec::with_capacity(self.parquet_column_ids.len());
                for &column_id in &self.parquet_column_ids {
                    let id = ArrayIdentifier::new(self.row_group_id, column_id, batch_row_id);
                    let mut array = self.cache.get_arrow_array(&id)?;
                    if offset > 0 || want_to_select < array.len() {
                        array = array.slice(offset, want_to_select);
                    }
                    columns.push(array);
                }

                assert_eq!(columns.len(), self.parquet_column_ids.len());
                RecordBatch::try_new(self.schema.clone(), columns).unwrap()
            };

            assert_eq!(record_batch.num_rows(), want_to_select);

            self.current_selected += want_to_select;
            if self.current_selected < row_selector.row_count {
                self.selection.push_front(row_selector);
            } else {
                self.row_id += row_selector.row_count;
                self.current_selected = 0;
            }
            return Some(record_batch);
        }
        None
    }
}

/// Iterator that yields coalesced `RecordBatch` items.
pub struct TakeRecordBatchIter<'a> {
    cache: &'a LiquidCache,
    row_group_id: usize,
    selection: VecDeque<RowSelector>,
    schema: SchemaRef,
    projected_column_ids: Vec<usize>,
    row_id: usize,
    current_selected: usize,
    batch_size: usize,
    row_idx_of_current_batch: (usize, Vec<u32>), // (batch_id, row_idx)
}

impl<'a> TakeRecordBatchIter<'a> {
    pub(crate) fn new(
        cache: &'a LiquidCache,
        row_group_id: usize,
        selection: VecDeque<RowSelector>,
        mut schema: SchemaRef,
        projected_column_ids: Vec<usize>,
        batch_size: usize,
    ) -> Self {
        let row_idx_of_current_batch = (0, vec![]);

        if projected_column_ids.is_empty() {
            let data_type = DataType::Struct(Fields::empty());
            let fields = vec![Field::new("empty", data_type, false)];
            schema = Arc::new(Schema::new(fields));
        }
        Self {
            cache,
            row_group_id,
            selection,
            schema,
            projected_column_ids,
            row_id: 0,
            current_selected: 0,
            batch_size,
            row_idx_of_current_batch,
        }
    }

    /// Coalesces the output of the iterator into batches of a specified size.
    pub fn into_coalesced(self, batch_size: usize) -> CoalescedIter<Self> {
        CoalescedIter::new(self, batch_size)
    }
}

fn make_dummy_record_batch(schema: &SchemaRef, len: usize) -> RecordBatch {
    let data_type = DataType::Struct(Fields::empty());
    let array = ArrayDataBuilder::new(data_type).len(len).build().unwrap();
    let array = StructArray::from(array);
    RecordBatch::try_new(schema.clone(), vec![Arc::new(array)]).unwrap()
}

impl Iterator for TakeRecordBatchIter<'_> {
    type Item = RecordBatch;

    fn next(&mut self) -> Option<Self::Item> {
        while let Some(row_selector) = self.selection.pop_front() {
            if row_selector.skip {
                self.row_id += row_selector.row_count;
                continue;
            }

            for j in self.current_selected..row_selector.row_count {
                let new_row_idx = self.row_id + j;
                let new_batch_id = (new_row_idx / self.batch_size) * self.batch_size;

                if new_batch_id != self.row_idx_of_current_batch.0 {
                    if !self.row_idx_of_current_batch.1.is_empty() {
                        let batch_id = self.row_idx_of_current_batch.0;
                        let record_batch = if self.projected_column_ids.is_empty() {
                            // means we just make arrays with empty content but keep the row count.
                            make_dummy_record_batch(
                                &self.schema,
                                self.row_idx_of_current_batch.1.len(),
                            )
                        } else {
                            let mut columns = Vec::with_capacity(self.schema.fields().len());
                            let indices =
                                UInt32Array::from(self.row_idx_of_current_batch.1.clone());
                            for &column_id in &self.projected_column_ids {
                                let id =
                                    ArrayIdentifier::new(self.row_group_id, column_id, batch_id);
                                let mut array = self.cache.get_arrow_array(&id).unwrap();
                                array = arrow::compute::kernels::take::take(&array, &indices, None)
                                    .unwrap();
                                columns.push(array);
                            }

                            RecordBatch::try_new(self.schema.clone(), columns).unwrap()
                        };

                        self.row_idx_of_current_batch.1.clear();
                        self.row_idx_of_current_batch.0 = new_batch_id;
                        self.selection.push_front(row_selector);
                        self.current_selected = j;
                        return Some(record_batch);
                    }
                    self.row_idx_of_current_batch.0 = new_batch_id;
                }
                self.row_idx_of_current_batch
                    .1
                    .push((new_row_idx - self.row_idx_of_current_batch.0) as u32);
            }
            self.row_id += row_selector.row_count;
            self.current_selected = 0;
        }

        if !self.row_idx_of_current_batch.1.is_empty() {
            let batch_id = self.row_idx_of_current_batch.0;
            let record_batch = if self.projected_column_ids.is_empty() {
                make_dummy_record_batch(&self.schema, self.row_idx_of_current_batch.1.len())
            } else {
                let mut columns = Vec::with_capacity(self.projected_column_ids.len());
                let indices = UInt32Array::from(self.row_idx_of_current_batch.1.clone());
                for &column_id in &self.projected_column_ids {
                    let id = ArrayIdentifier::new(self.row_group_id, column_id, batch_id);
                    let mut array = self.cache.get_arrow_array(&id).unwrap();
                    array = arrow::compute::kernels::take::take(&array, &indices, None).unwrap();
                    columns.push(array);
                }
                RecordBatch::try_new(self.schema.clone(), columns).unwrap()
            };

            self.row_idx_of_current_batch.1.clear();
            return Some(record_batch);
        }
        None
    }
}

/// CoalescedIter is an iterator that coalesces the output of an inner iterator into batches of a specified size.
pub struct CoalescedIter<T: Iterator<Item = RecordBatch> + Send> {
    inner: T,
    buffer: Vec<RecordBatch>,
    batch_size: usize,
}

impl<T: Iterator<Item = RecordBatch> + Send> CoalescedIter<T> {
    pub(crate) fn new(inner: T, batch_size: usize) -> Self {
        Self {
            inner,
            batch_size,
            buffer: vec![],
        }
    }

    fn add_to_buffer(&mut self, record_batch: RecordBatch) -> Option<RecordBatch> {
        let existing_row_count = self
            .buffer
            .iter()
            .map(|batch| batch.num_rows())
            .sum::<usize>();
        if existing_row_count + record_batch.num_rows() < self.batch_size {
            self.buffer.push(record_batch);
            None
        } else {
            let schema = record_batch.schema();
            self.buffer.push(record_batch);
            let buffer = std::mem::take(&mut self.buffer);
            let coalesced = concat_batches(&schema, &buffer).unwrap();
            Some(coalesced)
        }
    }
}

impl<T: Iterator<Item = RecordBatch> + Send> Iterator for CoalescedIter<T> {
    type Item = RecordBatch;

    fn next(&mut self) -> Option<Self::Item> {
        while let Some(batch) = self.inner.next() {
            if let Some(coalesced) = self.add_to_buffer(batch) {
                return Some(coalesced);
            }
        }
        if !self.buffer.is_empty() {
            let coalesced =
                concat_batches(&self.buffer.first().unwrap().schema(), &self.buffer).unwrap();
            self.buffer.clear();
            return Some(coalesced);
        }
        None
    }
}

/// BooleanSelectionIter is an iterator that yields record batches based on a boolean selection.
pub struct BooleanSelectionIter<'a> {
    cache: &'a LiquidCache,
    selection: BooleanSelection,
    row_group_id: usize,
    schema: SchemaRef,
    parquet_column_ids: Vec<usize>,
    batch_size: usize,
    cur_row_id: usize,
}

impl<'a> BooleanSelectionIter<'a> {
    pub(crate) fn new(
        cache: &'a LiquidCache,
        row_group_id: usize,
        selection: BooleanSelection,
        mut schema: SchemaRef,
        parquet_column_ids: Vec<usize>,
        batch_size: usize,
    ) -> Self {
        if parquet_column_ids.is_empty() {
            let data_type = DataType::Struct(Fields::empty());
            let fields = vec![Field::new("empty", data_type, false)];
            schema = Arc::new(Schema::new(fields));
        }
        Self {
            cache,
            selection,
            row_group_id,
            schema,
            parquet_column_ids,
            batch_size,
            cur_row_id: 0,
        }
    }

    /// Coalesces the output of the iterator into batches of a specified size.
    pub fn into_coalesced(self, batch_size: usize) -> CoalescedIter<Self> {
        CoalescedIter::new(self, batch_size)
    }
}

impl Iterator for BooleanSelectionIter<'_> {
    type Item = RecordBatch;
    fn next(&mut self) -> Option<Self::Item> {
        while self.cur_row_id < self.selection.len() {
            let want_to_select = self.batch_size.min(self.selection.len() - self.cur_row_id);
            let selection = self.selection.slice(self.cur_row_id, want_to_select);
            if selection.true_count() == 0 {
                // no rows are selected, skip this batch
                self.cur_row_id += want_to_select;
                continue;
            }

            let record_batch = if self.parquet_column_ids.is_empty() {
                make_dummy_record_batch(&self.schema, selection.true_count())
            } else {
                let mut columns = Vec::with_capacity(self.schema.fields().len());
                let mut fields = vec![];
                for (i, &column_id) in self.parquet_column_ids.iter().enumerate() {
                    let id = ArrayIdentifier::new(self.row_group_id, column_id, self.cur_row_id);
                    let array = self
                        .cache
                        .get_arrow_array_with_selection(&id, Some(&selection))
                        .unwrap();
                    fields.push(Field::new(
                        self.schema.field(i).name(),
                        array.data_type().clone(),
                        array.is_nullable(),
                    ));
                    columns.push(array);
                }
                let schema = Arc::new(Schema::new(fields));
                RecordBatch::try_new(schema, columns).unwrap()
            };

            self.cur_row_id += want_to_select;
            return Some(record_batch);
        }

        None
    }
}

/// BooleanSelectionIter is an iterator that yields record batches based on a boolean selection.
pub struct BooleanSelectionPredicateIter<'a, 'b: 'a> {
    cache: &'a LiquidCache,
    selection: &'b BooleanSelection,
    row_group_id: usize,
    schema: SchemaRef,
    parquet_column_id: usize,
    batch_size: usize,
    cur_row_id: usize,
    predicate: &'b mut Box<dyn ArrowPredicate>,
}

impl<'a, 'b> BooleanSelectionPredicateIter<'a, 'b> {
    pub(crate) fn new(
        cache: &'a LiquidCache,
        row_group_id: usize,
        selection: &'b BooleanSelection,
        schema: SchemaRef,
        parquet_column_id: usize,
        batch_size: usize,
        predicate: &'b mut Box<dyn ArrowPredicate>,
    ) -> Self {
        Self {
            cache,
            selection,
            row_group_id,
            schema,
            parquet_column_id,
            batch_size,
            cur_row_id: 0,
            predicate,
        }
    }
}

impl Iterator for BooleanSelectionPredicateIter<'_, '_> {
    type Item = BooleanArray;

    fn next(&mut self) -> Option<Self::Item> {
        while self.cur_row_id < self.selection.len() {
            let want_to_select = self.batch_size.min(self.selection.len() - self.cur_row_id);
            let selection = self.selection.slice(self.cur_row_id, want_to_select);
            if selection.true_count() == 0 {
                // no rows are selected, skip this batch
                self.cur_row_id += want_to_select;
                continue;
            }

            let id =
                ArrayIdentifier::new(self.row_group_id, self.parquet_column_id, self.cur_row_id);
            let filter = self
                .cache
                .get_arrow_array_with_selection_and_predicate(
                    &id,
                    Some(&selection),
                    self.predicate,
                    &self.schema,
                )
                .unwrap();
            // let record_batch = RecordBatch::try_new(self.schema.clone(), vec![array]).unwrap();
            // let filter = self.predicate.evaluate(record_batch).unwrap();

            self.cur_row_id += want_to_select;
            return Some(filter);
        }

        None
    }
}

#[cfg(test)]
mod tests {
    use crate::liquid_parquet::cache::CacheStates;

    use super::*;
    use arrow::array::{ArrayRef, AsArray, Int32Array};
    use arrow::datatypes::{DataType, Field, Int32Type, Schema};
    use arrow::record_batch::RecordBatch;
    use parquet::arrow::arrow_reader::RowSelection;
    use std::sync::Arc;

    fn set_up_cache() -> LiquidCache {
        /// Helper function to create a RecordBatch with a single Int32 column.
        fn create_record_batch(name: &str, num_rows: usize, start: i32) -> RecordBatch {
            let array = Int32Array::from_iter_values(start..(start + num_rows as i32));
            let schema = Arc::new(Schema::new(vec![Field::new(name, DataType::Int32, false)]));
            RecordBatch::try_new(schema, vec![Arc::new(array) as ArrayRef]).unwrap()
        }

        let cache = LiquidCache::new_inner(CacheStates::InMemory, 32);

        let row_group_id = 0;
        let column_id = 0;

        // Populate the cache with 42 rows of data split into two batches
        // Batch 1: rows 0-31
        let batch1 = create_record_batch("a", 32, 0);
        let id1 = ArrayIdentifier::new(row_group_id, column_id, 0);
        cache.insert_arrow_array(&id1, batch1.column(0).clone());

        // Batch 2: rows 32-41
        let batch2 = create_record_batch("a", 10, 32);
        let id2 = ArrayIdentifier::new(row_group_id, column_id, 32);
        cache.insert_arrow_array(&id2, batch2.column(0).clone());
        cache
    }

    #[test]
    fn test_get_coalesced_record_batch_iter() {
        let cache = set_up_cache();
        let row_group_id = 0;
        let column_id = 0;
        let parquet_column_ids = vec![column_id];
        let schema = Arc::new(Schema::new(vec![Field::new("a", DataType::Int32, false)]));

        // Define various row selections
        let selections = gen_selections();

        let expected: Vec<Vec<Vec<i32>>> = vec![
            vec![(0..32).collect(), (32..42).collect()],
            vec![(0..32).collect()],
            vec![(33..38).collect()],
            vec![(0..8).chain(10..26).chain(36..40).collect()],
            vec![(8..10).chain(26..32).chain(32..36).collect()],
        ];

        for (selection, expected) in selections.iter().zip(expected) {
            let expected = expected
                .into_iter()
                .map(|range| Int32Array::from(range))
                .collect::<Vec<_>>();
            let selection = RowSelection::from(selection.clone());
            let by_take_record_batches = cache
                .get_record_batch_by_take(row_group_id, &selection, &schema, &parquet_column_ids)
                .into_coalesced(32)
                .collect::<Vec<_>>();
            check_result(by_take_record_batches, &expected);

            let by_slice_record_batches = cache
                .get_record_batch_by_slice(row_group_id, &selection, &schema, &parquet_column_ids)
                .into_coalesced(32)
                .collect::<Vec<_>>();
            check_result(by_slice_record_batches, &expected);

            let by_filter_record_batches = cache
                .get_record_batches_by_filter(
                    row_group_id,
                    BooleanSelection::from(selection),
                    &schema,
                    &parquet_column_ids,
                )
                .into_coalesced(32)
                .collect::<Vec<_>>();
            check_result(by_filter_record_batches, &expected);
        }
    }

    #[test]
    fn test_get_coalesced_record_batch_iter_no_column() {
        let cache = set_up_cache();
        let row_group_id = 0;
        let parquet_column_ids = vec![];
        let schema = Arc::new(Schema::new(vec![Field::new("a", DataType::Int32, false)]));

        // Define various row selections
        let selections = gen_selections();

        let expected = vec![vec![32, 10], vec![32], vec![5], vec![24, 4], vec![8, 4]];

        for (selection, expected) in selections.iter().zip(expected) {
            let selection = RowSelection::from(selection.clone());
            let record_batches = cache
                .get_record_batches_by_filter(
                    row_group_id,
                    BooleanSelection::from(selection),
                    &schema,
                    &parquet_column_ids,
                )
                .collect::<Vec<_>>();
            assert_eq!(record_batches.len(), expected.len());
            for (batch, expected) in record_batches.into_iter().zip(expected) {
                assert_eq!(batch.num_rows(), expected);
            }
        }
    }

    #[test]
    fn test_get_record_batch() {
        let row_group_id = 0;
        let column_id = 0;
        let parquet_column_ids = vec![column_id];
        let schema = Arc::new(Schema::new(vec![Field::new("a", DataType::Int32, false)]));

        let cache = set_up_cache();

        // Define various row selections
        let selections = gen_selections();

        let by_slice_expected: Vec<Vec<Vec<i32>>> = vec![
            vec![(0..32).collect(), (32..42).collect()],
            vec![(0..32).collect()],
            vec![(33..38).collect()],
            vec![(0..8).collect(), (10..26).collect(), (36..40).collect()],
            vec![(8..10).collect(), (26..32).collect(), (32..36).collect()],
        ];

        for (selection, expected) in selections.iter().zip(by_slice_expected) {
            let expected = expected
                .into_iter()
                .map(|range| Int32Array::from(range))
                .collect::<Vec<_>>();
            let selection = RowSelection::from(selection.clone());
            let record_batches = cache
                .get_record_batch_by_slice(row_group_id, &selection, &schema, &parquet_column_ids)
                .collect::<Vec<_>>();
            check_result(record_batches, &expected);
        }

        let by_take_filter_expected: Vec<Vec<Vec<i32>>> = vec![
            vec![(0..32).collect(), (32..42).collect()],
            vec![(0..32).collect()],
            vec![(33..38).collect()],
            vec![(0..8).chain(10..26).collect(), (36..40).collect()],
            vec![(8..10).chain(26..32).collect(), (32..36).collect()],
        ];

        for (selection, expected) in selections.iter().zip(by_take_filter_expected.iter()) {
            let expected = expected
                .into_iter()
                .map(|range| Int32Array::from(range.clone()))
                .collect::<Vec<_>>();

            let selection = RowSelection::from(selection.clone());
            let take_record_batches = cache
                .get_record_batch_by_take(row_group_id, &selection, &schema, &parquet_column_ids)
                .collect::<Vec<_>>();
            check_result(take_record_batches, &expected);

            let selection = BooleanSelection::from(selection);
            let record_batches = cache
                .get_record_batches_by_filter(row_group_id, selection, &schema, &parquet_column_ids)
                .collect::<Vec<_>>();
            check_result(record_batches, &expected);
        }
    }

    fn gen_selections() -> Vec<Vec<RowSelector>> {
        vec![
            vec![RowSelector::select(42)],
            vec![RowSelector::select(32)],
            vec![RowSelector::skip(33), RowSelector::select(5)],
            vec![
                RowSelector::select(8),
                RowSelector::skip(2),
                RowSelector::select(16),
                RowSelector::skip(10),
                RowSelector::select(4),
            ],
            vec![
                RowSelector::skip(8),
                RowSelector::select(2),
                RowSelector::skip(16),
                RowSelector::select(10),
                RowSelector::skip(4),
            ],
        ]
    }

    fn check_result(record_batches: Vec<RecordBatch>, expected: &Vec<Int32Array>) {
        assert_eq!(record_batches.len(), expected.len());
        for (batch, expected) in record_batches.into_iter().zip(expected) {
            let actual = batch.column(0).as_primitive::<Int32Type>();
            assert_eq!(actual, expected);
        }
    }
}
