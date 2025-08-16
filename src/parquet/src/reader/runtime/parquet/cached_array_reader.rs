use std::any::Any;

use ahash::AHashMap;
use arrow::array::{ArrayRef, BooleanArray, BooleanBufferBuilder, new_empty_array};
use arrow_schema::{DataType, Field, Fields};
use liquid_cache_common::{cast_from_parquet_to_liquid_type, coerce_parquet_type_to_liquid_type};
use parquet::{
    arrow::{
        array_reader::{ArrayReader, StructArrayReader},
        arrow_reader::RowGroups,
    },
    errors::ParquetError,
};

use super::super::parquet_bridge::{ParquetField, ParquetFieldType};
use crate::{
    cache::{BatchID, LiquidCachedColumnRef, LiquidCachedRowGroupRef},
    reader::runtime::parquet_bridge::StructArrayReaderBridge,
};

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
    data_type: DataType,
    current_row: usize,
    inner_row_id: usize,
    selection_buffer: BooleanBufferBuilder,
    liquid_cache: LiquidCachedColumnRef,
    reader_local_cache: AHashMap<BatchID, ArrayRef>,
    next_batch_to_check_cached: u16,
}

impl CachedArrayReader {
    fn new(inner: Box<dyn ArrayReader>, liquid_cache: LiquidCachedColumnRef) -> Self {
        let inner_type = inner.get_data_type();
        let data_type = coerce_parquet_type_to_liquid_type(inner_type, liquid_cache.cache_mode());

        Self {
            inner,
            data_type,
            current_row: 0,
            inner_row_id: 0,
            selection_buffer: BooleanBufferBuilder::new(0),
            liquid_cache,
            reader_local_cache: AHashMap::new(),
            next_batch_to_check_cached: 0,
        }
    }

    fn batch_size(&self) -> usize {
        self.liquid_cache.batch_size()
    }

    fn fetch_batch(&mut self, batch_id: BatchID) -> Result<(), ParquetError> {
        // now we need to read from inner reader, first check if the inner id is up to date.
        let row_id = *batch_id as usize * self.batch_size();
        if self.inner_row_id < row_id {
            let to_skip = row_id - self.inner_row_id;
            let skipped = self.inner.skip_records(to_skip)?;
            assert_eq!(skipped, to_skip);
            self.inner_row_id = row_id;
        }
        let read = self.inner.read_records(self.batch_size())?;
        let array = self.inner.consume_batch()?;

        // This is a special case for the Utf8View type, because parquet read string as Utf8View.
        // But the reader reads as Dictionary or Utf8 type, depending on the cache mode.
        let array = cast_from_parquet_to_liquid_type(array, self.liquid_cache.cache_mode());

        _ = self.liquid_cache.insert(batch_id, array.clone());
        self.reader_local_cache.insert(batch_id, array);

        self.inner_row_id += read;
        Ok(())
    }

    fn is_cached(&self, batch_id: BatchID) -> bool {
        self.reader_local_cache.contains_key(&batch_id) || self.liquid_cache.is_cached(batch_id)
    }

    fn ensure_cached(&mut self, batch_id: BatchID) -> Result<(), ParquetError> {
        if *batch_id >= self.next_batch_to_check_cached {
            if !self.is_cached(batch_id) {
                self.fetch_batch(batch_id)?;
            }
            self.next_batch_to_check_cached = *batch_id + 1;
        }
        Ok(())
    }

    fn read_records_inner(&mut self, request_size: usize) -> Result<usize, ParquetError> {
        let batch_size = self.batch_size();
        assert!(request_size <= batch_size);

        self.selection_buffer.append_n(request_size, true);

        let starting_batch = BatchID::from_row_id(self.current_row, batch_size);
        let ending_batch = BatchID::from_row_id(self.current_row + request_size - 1, batch_size);

        self.ensure_cached(starting_batch)?;
        self.ensure_cached(ending_batch)?;

        self.current_row += request_size;
        Ok(request_size)
    }

    fn skip_records_inner(&mut self, to_skip: usize) -> Result<usize, ParquetError> {
        assert!(to_skip <= self.batch_size());
        self.selection_buffer.append_n(to_skip, false);

        self.current_row += to_skip;

        // we don't skip inner reader until we need to read from inner reader.

        Ok(to_skip)
    }

    fn consume_batch_inner(&mut self) -> Result<ArrayRef, ParquetError> {
        let batch_size = self.batch_size();
        let mut selection_builder = std::mem::replace(
            &mut self.selection_buffer,
            BooleanBufferBuilder::new(batch_size),
        );
        let selection_buffer = selection_builder.finish();
        let row_count = selection_buffer.len();
        let batch_size = self.liquid_cache.batch_size();
        let start_row = self.current_row - row_count;

        if row_count == 0 {
            return Ok(new_empty_array(&self.data_type));
        }

        // Calculate batch range involved in this selection
        let start_batch = start_row / batch_size;
        let end_batch = (start_row + row_count - 1) / batch_size;

        let mut rt = vec![];
        for batch_id in start_batch..=end_batch {
            let batch_start = batch_id * batch_size;
            let batch_end = batch_start + batch_size - 1;
            let batch_id = BatchID::from_raw(batch_id.try_into().unwrap());

            // Calculate overlap between selection and this batch
            let overlap_start = start_row.max(batch_start);
            let overlap_end = (start_row + row_count - 1).min(batch_end);

            if overlap_start > overlap_end {
                continue; // No overlap with this batch
            }

            // Get corresponding slice from selection buffer
            let selection_start = overlap_start - start_row;
            let selection_length = overlap_end - overlap_start + 1;
            let mask = selection_buffer.slice(selection_start, selection_length);

            if mask.count_set_bits() == 0 {
                continue;
            }

            // Get cached array and apply filter
            let array = match self
                .liquid_cache
                .get_arrow_array_with_filter(batch_id, &mask)
            {
                Some(array) => array,
                None => {
                    let array = self.reader_local_cache.remove(&batch_id).unwrap();
                    let mask = BooleanArray::from(mask);
                    arrow::compute::filter(&array, &mask).unwrap()
                }
            };

            if !array.data_type().equals_datatype(&self.data_type) {
                panic!(
                    "data type mismatch, input {:?}, expected {:?}",
                    array.data_type(),
                    self.data_type
                );
            }

            rt.push(array);
        }

        match rt.len() {
            0 => Ok(new_empty_array(&self.data_type)),
            1 => Ok(rt.into_iter().next().unwrap()),
            _ => Ok(arrow::compute::concat(
                &rt.iter().map(|a| a.as_ref()).collect::<Vec<_>>(),
            )?),
        }
    }
}

impl ArrayReader for CachedArrayReader {
    fn as_any(&self) -> &dyn Any {
        self.inner.as_any()
    }

    fn get_data_type(&self) -> &DataType {
        &self.data_type
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
        let array = self.consume_batch_inner()?;
        debug_assert_eq!(&self.data_type, array.data_type());
        Ok(array)
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
                        _ => {
                            log::info!(
                                "We only support primitives and structs, got child field:\n{parquet:?}"
                            );
                        }
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
    liquid_cache: LiquidCachedRowGroupRef,
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
    let struct_fields = match &bridged_reader.data_type {
        DataType::Struct(fields) => fields,
        _ => panic!("The previous data type must be a struct"),
    };

    let children = std::mem::take(&mut bridged_reader.children);

    assert_eq!(children.len(), column_ids.len());
    assert_eq!(struct_fields.len(), column_ids.len());
    let instrumented_readers: Vec<Box<dyn ArrayReader>> = column_ids
        .iter()
        .zip(children)
        .zip(struct_fields)
        .map(|((column_id, reader), field)| {
            let column_cache = liquid_cache.create_column(*column_id as u64, field.clone());
            let reader = Box::new(CachedArrayReader::new(reader, column_cache));
            reader as _
        })
        .collect();

    let previous_datatype = &bridged_reader.data_type;
    let previous_fields = match previous_datatype {
        DataType::Struct(fields) => fields,
        _ => panic!("The previous data type must be a struct"),
    };
    let fields = Fields::from(
        instrumented_readers
            .iter()
            .zip(previous_fields)
            .map(|(r, f)| {
                let name = f.name();
                Field::new(name, r.get_data_type().clone(), f.is_nullable())
            })
            .collect::<Vec<Field>>(),
    );
    let new_data_type = DataType::Struct(fields);
    bridged_reader.data_type = new_data_type;
    bridged_reader.children = instrumented_readers;

    struct_reader
}

pub fn build_cached_array_reader(
    field: Option<&ParquetField>,
    projection: &parquet::arrow::ProjectionMask,
    row_groups: &dyn RowGroups,
    liquid_cache: LiquidCachedRowGroupRef,
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
    use liquid_cache_common::LiquidCacheMode;
    use liquid_cache_storage::policies::DiscardPolicy;

    use super::*;
    use crate::cache::LiquidCache;

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
            &DataType::Int32
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

    fn set_up_reader() -> (CachedArrayReader, LiquidCachedColumnRef) {
        let tmp_dir = tempfile::tempdir().unwrap();
        let liquid_cache = Arc::new(LiquidCache::new(
            BATCH_SIZE,
            usize::MAX,
            tmp_dir.path().to_path_buf(),
            LiquidCacheMode::LiquidBlocking,
            Box::new(DiscardPolicy),
        ));
        let file = liquid_cache.register_or_get_file("test".to_string());
        let row_group = file.row_group(0);
        let reader = set_up_reader_with_cache(
            row_group
                .create_column(0, Arc::new(Field::new("test", DataType::Int32, false)))
                .clone(),
        );
        (reader, row_group.get_column(0).unwrap())
    }

    fn set_up_reader_with_cache(cache: LiquidCachedColumnRef) -> CachedArrayReader {
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
            let actual = cache
                .get_arrow_array_test_only(BatchID::from_row_id(i, BATCH_SIZE))
                .unwrap();
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
            let array = cache
                .get_arrow_array_test_only(BatchID::from_row_id(i, BATCH_SIZE))
                .unwrap();
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

    fn assert_contains(cache: &LiquidCachedColumnRef, id: usize) {
        let actual = cache
            .get_arrow_array_test_only(BatchID::from_row_id(id, BATCH_SIZE))
            .unwrap();
        let expected = get_expected_cached_value(id);
        assert_eq!(&actual, &expected);
    }

    fn assert_not_contains(cache: &LiquidCachedColumnRef, id: usize) {
        assert!(
            cache
                .get_arrow_array_test_only(BatchID::from_row_id(id, BATCH_SIZE))
                .is_none()
        );
    }

    #[test]
    fn test_read_with_partial_cached() {
        fn get_warm_cache() -> LiquidCachedColumnRef {
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

    #[test]
    fn test_ensure_cached_behavior() {
        let (mut reader, _cache) = set_up_reader(); // BATCH_SIZE is 32

        assert_eq!(reader.inner().read_cnt, 0);
        assert_eq!(reader.next_batch_to_check_cached, 0);

        reader.read_records_inner(10).unwrap();
        assert_eq!(reader.inner().read_cnt, 1, "Fetch for batch 0");
        assert_eq!(reader.current_row, 10);
        assert_eq!(reader.next_batch_to_check_cached, 1);

        reader.read_records_inner(10).unwrap();
        assert_eq!(reader.inner().read_cnt, 1, "No new fetch, still in batch 0");
        assert_eq!(reader.current_row, 20);
        assert_eq!(reader.next_batch_to_check_cached, 1); // Remains 1

        reader.read_records_inner(20).unwrap(); // Reads 12 from batch 0, 8 from batch 1
        assert_eq!(reader.inner().read_cnt, 2, "Fetch for batch 1");
        assert_eq!(reader.current_row, 40);
        assert_eq!(reader.next_batch_to_check_cached, 2);

        reader.consume_batch_inner().unwrap();

        reader.read_records_inner(10).unwrap();
        assert_eq!(
            reader.inner().read_cnt,
            2,
            "No new fetch, still in batch 1 (already checked)"
        );
        assert_eq!(reader.current_row, 50);
        assert_eq!(reader.next_batch_to_check_cached, 2); // Remains 2
    }
}
