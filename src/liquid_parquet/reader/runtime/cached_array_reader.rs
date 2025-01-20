use std::any::Any;

use arrow::array::ArrayRef;
use arrow_schema::DataType;
use parquet::{
    arrow::{
        array_reader::{ArrayReader, StructArrayReader},
        arrow_reader::RowGroups,
    },
    errors::ParquetError,
};

use crate::liquid_parquet::{
    cache::{ArrayIdentifier, LiquidCache},
    reader::runtime::parquet_bridge::StructArrayReaderBridge,
};

use super::parquet_bridge::{ParquetField, ParquetFieldType};

struct CachedArrayReader {
    inner: Box<dyn ArrayReader>,
    current_row_id: usize,
    column_id: usize,
    row_group_id: usize,
    current_cached: Vec<BufferValueType>,
}

enum BufferValueType {
    Cached(usize),
    Parquet,
}

impl CachedArrayReader {
    fn new(inner: Box<dyn ArrayReader>, row_group_id: usize, column_id: usize) -> Self {
        Self {
            inner,
            current_row_id: 0,
            row_group_id,
            column_id,
            current_cached: vec![],
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
        let row_batch_id = self.current_row_id / 8192 * 8192;
        let batch_id = ArrayIdentifier::new(self.row_group_id, self.column_id, row_batch_id);
        if let Some(cached_size) = LiquidCache::get().get_len(&batch_id) {
            if (self.current_row_id + request_size) <= (row_batch_id + cached_size) {
                let to_skip = request_size;
                self.current_cached
                    .push(BufferValueType::Cached(request_size));

                let skipped = self.inner.skip_records(to_skip).unwrap();
                assert_eq!(skipped, to_skip);
                self.current_row_id += to_skip;
                return Ok(to_skip);
            }
        }

        let records_read = self.inner.read_records(request_size).unwrap();
        self.current_cached.push(BufferValueType::Parquet);
        self.current_row_id += records_read;
        Ok(records_read)
    }

    fn consume_batch(&mut self) -> Result<ArrayRef, ParquetError> {
        let mut parquet_count = 0;
        let mut cached_rows = 0;
        for value in self.current_cached.iter() {
            match value {
                BufferValueType::Cached(rows) => {
                    cached_rows += rows;
                }
                BufferValueType::Parquet => parquet_count += 1,
            }
        }

        let parquet_records = self.inner.consume_batch().unwrap();

        if parquet_records.len() > 0 && cached_rows == 0 && parquet_count == 1 {
            let row_id = self.current_row_id - parquet_records.len();
            let row_batch_id = row_id / 8192 * 8192;

            // no cached records
            // only one parquet read
            let batch_id = ArrayIdentifier::new(self.row_group_id, self.column_id, row_batch_id);
            LiquidCache::get().insert_arrow_array(&batch_id, parquet_records.clone());
        }

        self.current_cached.clear();

        Ok(parquet_records)
    }

    fn skip_records(&mut self, num_records: usize) -> Result<usize, ParquetError> {
        let skipped = self.inner.skip_records(num_records).unwrap();
        self.current_row_id += skipped;
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
    row_group_idx: usize,
    column_ids: &[usize],
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
            let reader = Box::new(CachedArrayReader::new(reader, row_group_idx, *column_id));
            reader as _
        })
        .collect();

    bridged_reader.children = instrumented_readers;

    struct_reader
}

pub fn build_array_reader(
    field: Option<&ParquetField>,
    projection: &parquet::arrow::ProjectionMask,
    row_groups: &dyn RowGroups,
    row_group_idx: usize,
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

    Ok(instrument_array_reader(reader, row_group_idx, &column_ids))
}
