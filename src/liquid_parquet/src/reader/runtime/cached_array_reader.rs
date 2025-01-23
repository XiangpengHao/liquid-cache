use std::{any::Any, collections::VecDeque};

use arrow::array::ArrayRef;
use arrow_schema::DataType;
use parquet::{
    arrow::{
        array_reader::{ArrayReader, StructArrayReader},
        arrow_reader::{RowGroups, RowSelector},
    },
    errors::ParquetError,
};

use crate::{
    cache::{ArrayIdentifier, LiquidCacheRef},
    reader::runtime::parquet_bridge::StructArrayReaderBridge,
};

use super::parquet_bridge::{ParquetField, ParquetFieldType};

/// A cached array reader will cache the rows in batch_size granularity.
///
/// This reader always writes whatever from inner reader to the cache, regardless of whether the rows are already cached.
/// This is because the read path of the array reader need to decompress parquet pages
/// to know the next page, which means that, when page index is not present,
/// even if all row group is skipped, we need to decompress the entire row group.
/// To efficiently skip the cached rows, we simply don't call the array reader.
///
/// Let's say the batch size is 32, and the cache is initially empty.
/// 1. Read (0..32) rows -> cache (0..32) rows to cache, emit (0..32) rows
/// 2. Read (32..35) skip (35..64) rows -> cache (32..64) rows to cache, emit (32..35) rows
/// 3. Skip (64..96) rows -> don't cache anything
///
/// The cache will be (0..64) rows.
///
/// Typically, user will call read_records and skip_records multiple times,
/// the inner reader will accumulate the records, and return to user when consume_batch is called.
/// The invariant is that, the sum of read_records rows should be equal to the records returned by consume_batch.
struct CachedArrayReader {
    inner: Box<dyn ArrayReader>,
    current_row: usize,
    inner_row_id: usize,
    column_id: usize,
    row_group_id: usize,
    selection: VecDeque<RowSelector>,
    liquid_cache: LiquidCacheRef,
}

impl CachedArrayReader {
    fn new(
        inner: Box<dyn ArrayReader>,
        row_group_id: usize,
        column_id: usize,
        liquid_cache: LiquidCacheRef,
    ) -> Self {
        Self {
            inner,
            current_row: 0,
            inner_row_id: 0,
            row_group_id,
            column_id,
            selection: VecDeque::new(),
            liquid_cache,
        }
    }
}

impl ArrayReader for CachedArrayReader {
    fn as_any(&self) -> &dyn Any {
        self.inner.as_any()
    }

    fn get_data_type(&self) -> &DataType {
        self.inner.get_data_type()
    }

    fn read_records(&mut self, request_size: usize) -> Result<usize, ParquetError> {
        let batch_size = self.liquid_cache.batch_size();
        assert!(request_size <= batch_size);

        self.selection.push_back(RowSelector::select(request_size));

        let starting_batch_id = self.current_row / batch_size * batch_size;
        let ending_batch_id = (self.current_row + request_size - 1) / batch_size * batch_size;

        let start_batch =
            ArrayIdentifier::new(self.row_group_id, self.column_id, starting_batch_id);
        let end_batch = ArrayIdentifier::new(self.row_group_id, self.column_id, ending_batch_id);

        if self.liquid_cache.get_len(&start_batch).is_none() {
            let read = self.inner.read_records(batch_size)?;
            assert!(read >= request_size);
            let batch = self.inner.consume_batch()?;
            self.liquid_cache.insert_arrow_array(&start_batch, batch);
            self.inner_row_id += read;
        }

        if self.liquid_cache.get_len(&end_batch).is_none() {
            let read = self.inner.read_records(batch_size)?;
            assert!(read >= request_size);
            let batch = self.inner.consume_batch()?;
            self.liquid_cache.insert_arrow_array(&end_batch, batch);
            self.inner_row_id += read;
        }

        self.current_row += request_size;
        return Ok(request_size);
    }

    fn consume_batch(&mut self) -> Result<ArrayRef, ParquetError> {
        let row_count: usize = self.selection.iter().map(|s| s.row_count).sum();
        let batch_size = self.liquid_cache.batch_size();
        let mut current_row = self.current_row - row_count;

        let mut rt = vec![];
        for selector in self.selection.iter() {
            if selector.skip {
                current_row += selector.row_count;
                continue;
            }
            let ending_row = current_row + selector.row_count;
            let starting_batch_id = current_row / batch_size * batch_size;
            let ending_batch_id = (ending_row - 1) / batch_size * batch_size;

            if starting_batch_id == ending_batch_id {
                // easy case
                let batch =
                    ArrayIdentifier::new(self.row_group_id, self.column_id, starting_batch_id);
                let full_array = self.liquid_cache.get_arrow_array(&batch).unwrap();
                let offset = current_row - starting_batch_id;
                rt.push(full_array.slice(offset, selector.row_count));
            } else {
                // need to split the select
                let start_batch =
                    ArrayIdentifier::new(self.row_group_id, self.column_id, starting_batch_id);
                let start_full_array = self.liquid_cache.get_arrow_array(&start_batch).unwrap();
                let end_batch =
                    ArrayIdentifier::new(self.row_group_id, self.column_id, ending_batch_id);
                let end_full_array = self.liquid_cache.get_arrow_array(&end_batch).unwrap();

                let start_select = ending_batch_id - current_row;
                let end_select = ending_row - ending_batch_id;
                debug_assert_eq!(
                    current_row - starting_batch_id + start_select,
                    start_full_array.len(),
                    "if we have a next batch, then the previous batch must be fully selected"
                );
                debug_assert_eq!(
                    start_full_array.len(),
                    batch_size,
                    "if we have a next batch, then the previous batch must be size of batch_size"
                );
                rt.push(start_full_array.slice(current_row - starting_batch_id, start_select));
                rt.push(end_full_array.slice(0, end_select));
            }
            current_row += selector.row_count;
        }
        self.selection.clear();

        if rt.is_empty() {
            // return empty array
            return Ok(self.inner.consume_batch()?);
        }
        let concat =
            arrow::compute::concat(&rt.as_slice().iter().map(|a| a.as_ref()).collect::<Vec<_>>())
                .unwrap();
        Ok(concat)
    }

    fn skip_records(&mut self, to_skip: usize) -> Result<usize, ParquetError> {
        let batch_size = self.liquid_cache.batch_size();
        assert!(to_skip <= batch_size);
        self.selection.push_back(RowSelector::skip(to_skip));

        self.current_row += to_skip;

        let current_batch_id = self.current_row / batch_size * batch_size;
        let inner_batch_id = self.inner_row_id / batch_size * batch_size;
        if inner_batch_id < current_batch_id {
            // moved to a new batch
            self.inner.skip_records(batch_size)?;
            self.inner_row_id += batch_size;
            assert_eq!(self.inner_row_id, current_batch_id);
        }

        Ok(to_skip)
    }

    fn get_def_levels(&self) -> Option<&[i16]> {
        self.inner.get_def_levels()
    }

    fn get_rep_levels(&self) -> Option<&[i16]> {
        self.inner.get_rep_levels()
    }
}

fn get_column_ids(
    field: Option<&ParquetField>,
    projection: &parquet::arrow::ProjectionMask,
) -> Vec<usize> {
    let Some(field) = field else {
        return vec![];
    };

    match &field.field_type {
        ParquetFieldType::Group { children, .. } => match &field.arrow_type {
            DataType::Struct(_) => {
                let mut column_ids = vec![];
                for parquet in children.iter() {
                    match parquet.field_type {
                        ParquetFieldType::Primitive { col_idx, .. } => {
                            if projection.leaf_included(col_idx) {
                                column_ids.push(col_idx);
                            }
                        }
                        _ => panic!("We only support primitives and structs"),
                    }
                }
                column_ids
            }
            _ => panic!("We only support primitives and structs"),
        },
        _ => panic!("We only support primitives and structs"),
    }
}

fn instrument_array_reader(
    reader: Box<dyn ArrayReader>,
    row_group_idx: usize,
    column_ids: &[usize],
    liquid_cache: LiquidCacheRef,
) -> Box<dyn ArrayReader> {
    if reader
        .as_any()
        .downcast_ref::<StructArrayReader>()
        .is_none()
    {
        panic!("The reader must be a StructArrayReader");
    }

    let raw = Box::into_raw(reader);
    let mut struct_reader = unsafe { Box::from_raw(raw as *mut StructArrayReader) };

    let bridged_reader = StructArrayReaderBridge::from_parquet(&mut struct_reader);

    let children = std::mem::take(&mut bridged_reader.children);

    assert_eq!(children.len(), column_ids.len());
    let instrumented_readers = column_ids
        .iter()
        .zip(children)
        .map(|(column_id, reader)| {
            let reader = Box::new(CachedArrayReader::new(
                reader,
                row_group_idx,
                *column_id,
                liquid_cache.clone(),
            ));
            reader as _
        })
        .collect();

    bridged_reader.children = instrumented_readers;

    struct_reader
}

pub fn build_cached_array_reader(
    field: Option<&ParquetField>,
    projection: &parquet::arrow::ProjectionMask,
    row_groups: &dyn RowGroups,
    row_group_idx: usize,
    liquid_cache: LiquidCacheRef,
) -> Result<Box<dyn ArrayReader>, ParquetError> {
    let reader = parquet::arrow::array_reader::build_array_reader(
        #[allow(clippy::missing_transmute_annotations)]
        unsafe {
            std::mem::transmute(field)
        },
        projection,
        row_groups,
    )?;

    let column_ids = get_column_ids(field, projection);
    if column_ids.is_empty() {
        return Ok(reader);
    }

    Ok(instrument_array_reader(
        reader,
        row_group_idx,
        &column_ids,
        liquid_cache.clone(),
    ))
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow::array::Int32Array;

    use crate::{LiquidCacheMode, cache::LiquidCache};

    use super::*;

    struct MockArrayReader {
        rows: Vec<i32>,
        output: Vec<i32>,
        current_row: usize,
    }

    impl ArrayReader for MockArrayReader {
        fn as_any(&self) -> &dyn Any {
            self
        }

        fn get_data_type(&self) -> &DataType {
            todo!()
        }

        fn get_def_levels(&self) -> Option<&[i16]> {
            todo!()
        }

        fn get_rep_levels(&self) -> Option<&[i16]> {
            todo!()
        }

        fn read_records(&mut self, request: usize) -> Result<usize, ParquetError> {
            self.output
                .extend_from_slice(&self.rows[self.current_row..self.current_row + request]);
            self.current_row += request;
            Ok(request)
        }

        fn consume_batch(&mut self) -> Result<ArrayRef, ParquetError> {
            let output = std::mem::take(&mut self.output);
            Ok(Arc::new(Int32Array::from(output)))
        }

        fn skip_records(&mut self, num_records: usize) -> Result<usize, ParquetError> {
            self.current_row += num_records;
            Ok(num_records)
        }
    }

    const BATCH_SIZE: usize = 32;
    const TOTAL_ROWS: usize = 96;

    fn set_up_cache() -> (CachedArrayReader, Arc<LiquidCache>) {
        let liquid_cache = Arc::new(LiquidCache::new(LiquidCacheMode::InMemoryArrow, BATCH_SIZE));
        let rows: Vec<i32> = (0..TOTAL_ROWS as i32).collect();
        let mock_reader = MockArrayReader {
            rows: rows.clone(),
            output: vec![],
            current_row: 0,
        };
        let cached_reader =
            CachedArrayReader::new(Box::new(mock_reader), 0, 0, liquid_cache.clone());
        (cached_reader, liquid_cache)
    }

    fn get_expected_cached_value(id: &ArrayIdentifier) -> ArrayRef {
        let row_id = id.row_id() as i32;
        Arc::new(Int32Array::from_iter_values(
            row_id..(row_id + BATCH_SIZE as i32),
        ))
    }

    #[test]
    fn test_read_at_batch_boundary() {
        let (mut reader, cache) = set_up_cache();

        // Read and consume
        for i in (0..96).step_by(32) {
            let read1 = reader.read_records(32).unwrap();
            assert_eq!(read1, 32);
            let expected = reader.consume_batch().unwrap();
            let id1 = ArrayIdentifier::new(0, 0, i);
            let actual = cache.get_arrow_array(&id1).unwrap();
            assert_eq!(&expected, &actual);
            let cache_expected = get_expected_cached_value(&id1);
            assert_eq!(&cache_expected, &actual);
        }

        // Read all and consume
        let (mut reader, cache) = set_up_cache();
        for _ in (0..96).step_by(32) {
            let read1 = reader.read_records(32).unwrap();
            assert_eq!(read1, 32);
        }
        let expected = reader.consume_batch().unwrap();
        assert_eq!(expected.len(), 96);

        let mut cached = vec![];
        for i in (0..96).step_by(32) {
            let id1 = ArrayIdentifier::new(0, 0, i);
            let array = cache.get_arrow_array(&id1).unwrap();
            let expected = get_expected_cached_value(&id1);
            assert_eq!(&array, &expected);
            cached.push(array);
        }
        let actual = arrow::compute::concat(
            &cached
                .as_slice()
                .iter()
                .map(|a| a.as_ref())
                .collect::<Vec<_>>(),
        )
        .unwrap();
        assert_eq!(&expected, &actual);

        // Read and skip
        let (mut reader, cache) = set_up_cache();
        reader.read_records(32).unwrap();
        reader.skip_records(32).unwrap();
        reader.read_records(32).unwrap();
        let expected = reader.consume_batch().unwrap();
        assert_eq!(expected.len(), 64);
        let id1 = ArrayIdentifier::new(0, 0, 0);
        let actual = cache.get_arrow_array(&id1).unwrap();
        assert_eq!(&actual, &get_expected_cached_value(&id1));
        let id2 = ArrayIdentifier::new(0, 0, 32);
        assert!(cache.get_len(&id2).is_none());
        let id3 = ArrayIdentifier::new(0, 0, 64);
        let actual = cache.get_arrow_array(&id3).unwrap();
        assert_eq!(&actual, &get_expected_cached_value(&id3));
    }

    #[test]
    fn test_edge_cases() {
        let (mut reader, _cache) = set_up_cache();
        reader.consume_batch().unwrap();
    }

    #[test]
    fn test_skip_partial() {
        let (mut reader, cache) = set_up_cache();
        reader.read_records(20).unwrap();
        reader.skip_records(20).unwrap();
        reader.read_records(20).unwrap();

        let id0 = ArrayIdentifier::new(0, 0, 0);
        let cached_array0 = cache.get_arrow_array(&id0).unwrap();
        assert_eq!(&cached_array0, &get_expected_cached_value(&id0));
        let id1 = ArrayIdentifier::new(0, 0, 32);
        let cached_array1 = cache.get_arrow_array(&id1).unwrap();
        assert_eq!(&cached_array1, &get_expected_cached_value(&id1));
    }

    #[test]
    fn test_read_partial_batch() {
        let (mut reader, cache) = set_up_cache();

        {
            reader.read_records(20).unwrap();
            let id = ArrayIdentifier::new(0, 0, 0);
            let cached_array = cache.get_arrow_array(&id).unwrap();
            assert_eq!(&cached_array, &get_expected_cached_value(&id));
            let read_array = reader.consume_batch().unwrap();
            assert_eq!(read_array.len(), 20);
        }

        {
            reader.read_records(20).unwrap();
            let id = ArrayIdentifier::new(0, 0, 32);
            let cached_array = cache.get_arrow_array(&id).unwrap();
            assert_eq!(&cached_array, &get_expected_cached_value(&id));
            let read_array = reader.consume_batch().unwrap();
            assert_eq!(read_array.len(), 20);
        }

        {
            reader.read_records(32).unwrap();
            let id = ArrayIdentifier::new(0, 0, 64);
            let cached_array = cache.get_arrow_array(&id).unwrap();
            assert_eq!(&cached_array, &get_expected_cached_value(&id));
            let read_array = reader.consume_batch().unwrap();
            assert_eq!(read_array.len(), 32);
        }
    }
}
