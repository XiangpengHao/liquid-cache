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
    cache::{CachedColumnRef, CachedRowGroupRef},
    reader::runtime::parquet_bridge::StructArrayReaderBridge,
};

use super::parquet_bridge::{ParquetField, ParquetFieldType};

/// A cached array reader will cache the rows in batch_size granularity.
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
///
/// Invariants
/// 1. The sum of read_records rows should be equal to the records returned by consume_batch.
struct CachedArrayReader {
    inner: Box<dyn ArrayReader>,
    current_row: usize,
    inner_row_id: usize,
    selection: VecDeque<RowSelector>,
    liquid_cache: CachedColumnRef,
}

impl CachedArrayReader {
    fn new(inner: Box<dyn ArrayReader>, liquid_cache: CachedColumnRef) -> Self {
        Self {
            inner,
            current_row: 0,
            inner_row_id: 0,
            selection: VecDeque::new(),
            liquid_cache,
        }
    }

    fn batch_size(&self) -> usize {
        self.liquid_cache.batch_size()
    }

    fn fetch_batch(&mut self, batch_id: usize) -> Result<(), ParquetError> {
        // now we need to read from inner reader, first check if the inner id is up to date.
        if self.inner_row_id < batch_id {
            let to_skip = batch_id - self.inner_row_id;
            let skipped = self.inner.skip_records(to_skip)?;
            assert_eq!(skipped, to_skip);
            self.inner_row_id = batch_id;
        }
        let read = self.inner.read_records(self.batch_size())?;
        let batch = self.inner.consume_batch()?;
        self.liquid_cache.insert_arrow_array(batch_id, batch);
        self.inner_row_id += read;
        Ok(())
    }

    fn read_records_inner(&mut self, request_size: usize) -> Result<usize, ParquetError> {
        let batch_size = self.batch_size();
        assert!(request_size <= batch_size);

        self.selection.push_back(RowSelector::select(request_size));

        let starting_batch_id = self.current_row / batch_size * batch_size;
        let ending_batch_id = (self.current_row + request_size - 1) / batch_size * batch_size;

        if !self.liquid_cache.is_cached(starting_batch_id) {
            self.fetch_batch(starting_batch_id)?;
        }

        if ending_batch_id != starting_batch_id && !self.liquid_cache.is_cached(ending_batch_id) {
            self.fetch_batch(ending_batch_id)?;
        }

        self.current_row += request_size;
        Ok(request_size)
    }

    fn skip_records_inner(&mut self, to_skip: usize) -> Result<usize, ParquetError> {
        assert!(to_skip <= self.batch_size());
        self.selection.push_back(RowSelector::skip(to_skip));

        self.current_row += to_skip;

        // we don't skip inner reader until we need to read from inner reader.

        Ok(to_skip)
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
        let batch_size = self.batch_size();
        let mut read = 0;

        while read < request_size {
            let size = std::cmp::min(batch_size, request_size - read);
            read += self.read_records_inner(size)?;
        }
        Ok(read)
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
                let full_array = self
                    .liquid_cache
                    .get_arrow_array(starting_batch_id)
                    .unwrap();
                let offset = current_row - starting_batch_id;
                rt.push(full_array.slice(offset, selector.row_count));
            } else {
                // need to split the select
                let start_full_array = self
                    .liquid_cache
                    .get_arrow_array(starting_batch_id)
                    .unwrap();
                let end_full_array = self.liquid_cache.get_arrow_array(ending_batch_id).unwrap();

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
            return self.inner.consume_batch();
        }
        let concat =
            arrow::compute::concat(&rt.as_slice().iter().map(|a| a.as_ref()).collect::<Vec<_>>())
                .unwrap();
        Ok(concat)
    }

    fn skip_records(&mut self, to_skip: usize) -> Result<usize, ParquetError> {
        let mut skipped = 0;

        let batch_size = self.liquid_cache.batch_size();

        while skipped < to_skip {
            let size = std::cmp::min(batch_size, to_skip - skipped);
            skipped += self.skip_records_inner(size)?;
        }
        Ok(skipped)
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
    column_ids: &[usize],
    liquid_cache: CachedRowGroupRef,
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
            let column_cache = liquid_cache.column(*column_id);
            let reader = Box::new(CachedArrayReader::new(reader, column_cache));
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
    liquid_cache: CachedRowGroupRef,
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
        read_cnt: usize,
        skip_cnt: usize,
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
            self.read_cnt += 1;
            Ok(request)
        }

        fn consume_batch(&mut self) -> Result<ArrayRef, ParquetError> {
            let output = std::mem::take(&mut self.output);
            Ok(Arc::new(Int32Array::from(output)))
        }

        fn skip_records(&mut self, num_records: usize) -> Result<usize, ParquetError> {
            self.current_row += num_records;
            self.skip_cnt += 1;
            Ok(num_records)
        }
    }

    impl CachedArrayReader {
        fn inner(&self) -> &MockArrayReader {
            self.inner
                .as_any()
                .downcast_ref::<MockArrayReader>()
                .unwrap()
        }
    }

    const BATCH_SIZE: usize = 32;
    const TOTAL_ROWS: usize = 96;

    fn set_up_reader() -> (CachedArrayReader, CachedColumnRef) {
        let liquid_cache = Arc::new(LiquidCache::new(LiquidCacheMode::InMemoryArrow, BATCH_SIZE));
        let row_group = liquid_cache.row_group(0);
        let reader = set_up_reader_with_cache(row_group.column(0).clone());
        (reader, row_group.column(0))
    }

    fn set_up_reader_with_cache(cache: CachedColumnRef) -> CachedArrayReader {
        let rows: Vec<i32> = (0..TOTAL_ROWS as i32).collect();
        let mock_reader = MockArrayReader {
            rows: rows.clone(),
            output: vec![],
            current_row: 0,
            read_cnt: 0,
            skip_cnt: 0,
        };
        CachedArrayReader::new(Box::new(mock_reader), cache)
    }

    fn get_expected_cached_value(id: usize) -> ArrayRef {
        let row_id = id as i32;
        Arc::new(Int32Array::from_iter_values(
            row_id..(row_id + BATCH_SIZE as i32),
        ))
    }

    #[test]
    fn test_read_at_batch_boundary() {
        let (mut reader, cache) = set_up_reader();

        // Read and consume
        for i in (0..96).step_by(32) {
            let read1 = reader.read_records(32).unwrap();
            assert_eq!(read1, 32);
            let expected = reader.consume_batch().unwrap();
            let actual = cache.get_arrow_array(i).unwrap();
            assert_eq!(&expected, &actual);
            let cache_expected = get_expected_cached_value(i);
            assert_eq!(&cache_expected, &actual);
        }

        // Read all and consume
        let (mut reader, cache) = set_up_reader();
        for _ in (0..96).step_by(32) {
            let read1 = reader.read_records(32).unwrap();
            assert_eq!(read1, 32);
        }
        let expected = reader.consume_batch().unwrap();
        assert_eq!(expected.len(), 96);

        let mut cached = vec![];
        for i in (0..96).step_by(32) {
            let array = cache.get_arrow_array(i).unwrap();
            let expected = get_expected_cached_value(i);
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
        let (mut reader, cache) = set_up_reader();
        reader.read_records(32).unwrap();
        reader.skip_records(32).unwrap();
        reader.read_records(32).unwrap();
        let expected = reader.consume_batch().unwrap();
        assert_eq!(expected.len(), 64);
        assert_contains(&cache, 0);
        assert_not_contains(&cache, 32);
        assert_contains(&cache, 64);
    }

    #[test]
    fn test_edge_cases() {
        let (mut reader, _cache) = set_up_reader();
        reader.consume_batch().unwrap();
    }

    #[test]
    fn test_large_skip_and_read() {
        {
            let (mut reader, cache) = set_up_reader();
            reader.read_records(90).unwrap();
            let array = reader.consume_batch().unwrap();
            assert_eq!(array.len(), 90);
            assert_contains(&cache, 0);
            assert_contains(&cache, 32);
            assert_contains(&cache, 64);
        }
        {
            let (mut reader, cache) = set_up_reader();
            reader.skip_records(40).unwrap();
            let array = reader.consume_batch().unwrap();
            assert_eq!(array.len(), 0);
            assert_not_contains(&cache, 0);

            reader.read_records(40).unwrap();
            assert_contains(&cache, 32);

            assert_contains(&cache, 64);
            let array = reader.consume_batch().unwrap();
            assert_eq!(array.len(), 40);
        }
    }

    #[test]
    fn test_skip_partial() {
        let (mut reader, cache) = set_up_reader();
        reader.read_records(20).unwrap();
        reader.skip_records(20).unwrap();
        reader.read_records(20).unwrap();

        assert_contains(&cache, 0);
        assert_contains(&cache, 32);
    }

    #[test]
    fn test_read_partial_batch() {
        let (mut reader, cache) = set_up_reader();

        {
            reader.read_records(20).unwrap();
            assert_contains(&cache, 0);
            let read_array = reader.consume_batch().unwrap();
            assert_eq!(read_array.len(), 20);
        }

        {
            reader.read_records(20).unwrap();
            assert_contains(&cache, 32);
            let read_array = reader.consume_batch().unwrap();
            assert_eq!(read_array.len(), 20);
        }

        {
            reader.read_records(32).unwrap();
            assert_contains(&cache, 64);
            let read_array = reader.consume_batch().unwrap();
            assert_eq!(read_array.len(), 32);
        }
    }

    #[test]
    fn test_read_with_all_cached() {
        let (mut reader, cache) = set_up_reader();
        reader.read_records(96).unwrap();
        let array = reader.consume_batch().unwrap();
        assert_eq!(array.len(), 96);
        assert_eq!(reader.inner().read_cnt, 3);
        assert_eq!(reader.inner().skip_cnt, 0);
        for id in [0, 32, 64] {
            assert_contains(&cache, id);
        }

        let mut reader = set_up_reader_with_cache(cache.clone());
        reader.read_records(96).unwrap();
        let array = reader.consume_batch().unwrap();
        assert_eq!(array.len(), 96);
        assert_eq!(reader.inner().read_cnt, 0);
        assert_eq!(reader.inner().skip_cnt, 0);
        for id in [0, 32, 64] {
            assert_contains(&cache, id);
        }

        let mut reader = set_up_reader_with_cache(cache.clone());
        reader.read_records(20).unwrap();
        reader.skip_records(20).unwrap();
        reader.read_records(20).unwrap();
        assert_eq!(reader.inner().read_cnt, 0);
        assert_eq!(reader.inner().skip_cnt, 0);
        let array = reader.consume_batch().unwrap();
        assert_eq!(array.len(), 40);
    }

    fn assert_contains(cache: &CachedColumnRef, id: usize) {
        let actual = cache.get_arrow_array(id).unwrap();
        let expected = get_expected_cached_value(id);
        assert_eq!(&actual, &expected);
    }

    fn assert_not_contains(cache: &CachedColumnRef, id: usize) {
        assert!(cache.get_arrow_array(id).is_none());
    }

    #[test]
    fn test_read_with_partial_cached() {
        fn get_warm_cache() -> CachedColumnRef {
            let (mut reader, cache) = set_up_reader();
            reader.read_records(32).unwrap();
            reader.skip_records(32).unwrap();
            reader.read_records(32).unwrap();
            let array = reader.consume_batch().unwrap();
            assert_eq!(array.len(), 64);
            assert_contains(&cache, 0);
            assert_not_contains(&cache, 32);
            assert_eq!(reader.inner().read_cnt, 2);
            assert_eq!(reader.inner().skip_cnt, 1);
            cache
        }

        let cache = get_warm_cache();
        let mut reader = set_up_reader_with_cache(cache.clone());
        reader.read_records(96).unwrap();
        let array = reader.consume_batch().unwrap();
        assert_eq!(array.len(), 96);
        assert_eq!(reader.inner().read_cnt, 1);
        assert_eq!(reader.inner().skip_cnt, 1);
        for id in [0, 32, 64] {
            assert_contains(&cache, id);
        }

        let cache = get_warm_cache();
        let mut reader = set_up_reader_with_cache(cache.clone());
        reader.read_records(16).unwrap();
        reader.skip_records(48).unwrap();
        reader.read_records(16).unwrap();
        reader.skip_records(16).unwrap();
        let array = reader.consume_batch().unwrap();
        assert_eq!(array.len(), 32);
        assert_eq!(reader.inner().read_cnt, 0);
        assert_eq!(reader.inner().skip_cnt, 0);
        assert_contains(&cache, 0);
        assert_not_contains(&cache, 32);
        assert_contains(&cache, 64);
    }
}
