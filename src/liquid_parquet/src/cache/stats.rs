use super::{CachedBatch, LiquidCache};
use arrow::array::{ArrayBuilder, RecordBatch, StringBuilder, UInt64Builder};
use arrow_schema::{DataType, Field, Schema, SchemaRef};
use parquet::{
    arrow::ArrowWriter, basic::Compression, errors::ParquetError,
    file::properties::WriterProperties,
};
use std::{
    fs::File,
    path::Path,
    sync::{Arc, atomic::Ordering},
};

struct StatsWriter {
    writer: ArrowWriter<File>,
    schema: SchemaRef,
    file_path_builder: StringBuilder,
    row_group_id_builder: UInt64Builder,
    column_id_builder: UInt64Builder,
    row_start_id_builder: UInt64Builder,
    row_count_builder: UInt64Builder,
    memory_size_builder: UInt64Builder,
    cache_type_builder: StringBuilder,
    hit_count_builder: UInt64Builder,
}

impl StatsWriter {
    fn new(file_path: impl AsRef<Path>) -> Result<Self, ParquetError> {
        let schema = Arc::new(Schema::new(vec![
            Field::new("row_group_id", DataType::UInt64, false),
            Field::new("column_id", DataType::UInt64, false),
            Field::new("row_start_id", DataType::UInt64, false),
            Field::new("row_count", DataType::UInt64, true),
            Field::new("memory_size", DataType::UInt64, false),
            Field::new("cache_type", DataType::Utf8, false),
            Field::new("hit_count", DataType::UInt64, false),
            Field::new("file_path", DataType::Utf8, false),
        ]));

        let file = File::create(file_path)?;
        let write_props = WriterProperties::builder()
            .set_compression(Compression::LZ4)
            .set_created_by("liquid-cache-stats".to_string())
            .build();
        let writer = ArrowWriter::try_new(file, schema.clone(), Some(write_props))?;
        Ok(Self {
            writer,
            schema,
            file_path_builder: StringBuilder::with_capacity(8192, 8192),
            row_group_id_builder: UInt64Builder::new(),
            column_id_builder: UInt64Builder::new(),
            row_start_id_builder: UInt64Builder::new(),
            row_count_builder: UInt64Builder::new(),
            memory_size_builder: UInt64Builder::new(),
            cache_type_builder: StringBuilder::with_capacity(8192, 8192),
            hit_count_builder: UInt64Builder::new(),
        })
    }

    fn build_batch(&mut self) -> Result<RecordBatch, ParquetError> {
        let row_group_id_array = self.row_group_id_builder.finish();
        let column_id_array = self.column_id_builder.finish();
        let row_start_id_array = self.row_start_id_builder.finish();
        let row_count_array = self.row_count_builder.finish();
        let memory_size_array = self.memory_size_builder.finish();
        let cache_type_array = self.cache_type_builder.finish();
        let hit_count_array = self.hit_count_builder.finish();
        let file_path_array = self.file_path_builder.finish();
        Ok(RecordBatch::try_new(
            self.schema.clone(),
            vec![
                Arc::new(row_group_id_array),
                Arc::new(column_id_array),
                Arc::new(row_start_id_array),
                Arc::new(row_count_array),
                Arc::new(memory_size_array),
                Arc::new(cache_type_array),
                Arc::new(hit_count_array),
                Arc::new(file_path_array),
            ],
        )?)
    }

    #[allow(clippy::too_many_arguments)]
    fn append_entry(
        &mut self,
        file_path: &str,
        row_group_id: u64,
        column_id: u64,
        row_start_id: u64,
        row_count: Option<u64>,
        memory_size: u64,
        cache_type: &str,
        hit_count: u64,
    ) -> Result<(), ParquetError> {
        self.row_group_id_builder.append_value(row_group_id);
        self.column_id_builder.append_value(column_id);
        self.row_start_id_builder.append_value(row_start_id);
        self.row_count_builder.append_option(row_count);
        self.memory_size_builder.append_value(memory_size);
        self.cache_type_builder.append_value(cache_type);
        self.hit_count_builder.append_value(hit_count);
        self.file_path_builder.append_value(file_path);
        if self.row_start_id_builder.len() >= 8192 {
            let batch = self.build_batch()?;
            self.writer.write(&batch)?;
        }
        Ok(())
    }

    fn finish(mut self) -> Result<(), ParquetError> {
        let batch = self.build_batch()?;
        self.writer.write(&batch)?;
        self.writer.close()?;
        Ok(())
    }
}

impl LiquidCache {
    pub fn memory_usage_bytes(&self) -> u64 {
        let files = self.files.lock().unwrap();
        let mut memory_consumption = 0;
        for (_, file_lock) in files.iter() {
            let row_groups = file_lock.row_groups.lock().unwrap();
            for (_, row_group) in row_groups.iter() {
                let columns = row_group.columns.read().unwrap();
                for (_, column) in columns.iter() {
                    let cached_entry = column.rows.read().unwrap();
                    for (_, cached_entry) in cached_entry.iter() {
                        let cached_entry_v = cached_entry.value();
                        memory_consumption += cached_entry_v.memory_usage();
                    }
                }
            }
        }
        memory_consumption as u64
    }

    pub fn write_stats(&self, parquet_file_path: impl AsRef<Path>) -> Result<(), ParquetError> {
        let mut writer = StatsWriter::new(parquet_file_path)?;
        let files = self.files.lock().unwrap();
        for (file_path, file_lock) in files.iter() {
            let row_groups = file_lock.row_groups.lock().unwrap();
            for (row_group_id, row_group) in row_groups.iter() {
                let columns = row_group.columns.read().unwrap();
                for (column_id, row_mapping) in columns.iter() {
                    for (row_start_id, cached_entry) in row_mapping.rows.read().unwrap().iter() {
                        let cached_entry_v = cached_entry.value();
                        let cache_type = match cached_entry_v {
                            CachedBatch::ArrowMemory(_) => "InMemory",
                            CachedBatch::LiquidMemory(_) => "LiquidMemory",
                            CachedBatch::OnDiskLiquid => "OnDiskLiquid",
                        };

                        let memory_size = cached_entry_v.memory_usage();
                        let row_count = match cached_entry_v {
                            CachedBatch::ArrowMemory(array) => Some(array.len() as u64),
                            CachedBatch::LiquidMemory(array) => Some(array.len() as u64),
                            CachedBatch::OnDiskLiquid => None,
                        };

                        writer.append_entry(
                            file_path,
                            *row_group_id as u64,
                            *column_id as u64,
                            *row_start_id as u64,
                            row_count,
                            memory_size as u64,
                            cache_type,
                            cached_entry.hit_count.load(Ordering::Relaxed) as u64,
                        )?;
                    }
                }
            }
        }
        writer.finish()?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::io::Read;

    use crate::LiquidCacheMode;

    use super::*;
    use arrow::{
        array::{Array, AsArray},
        datatypes::UInt64Type,
    };
    use bytes::Bytes;
    use parquet::arrow::arrow_reader::ParquetRecordBatchReader;
    use tempfile::NamedTempFile;

    #[test]
    fn test_stats_writer() -> Result<(), ParquetError> {
        let tmp_dir = tempfile::tempdir().unwrap();
        let cache = LiquidCache::new(1024, usize::MAX, tmp_dir.path().to_path_buf());
        let array = Arc::new(arrow::array::Int32Array::from(vec![1, 2, 3]));
        let num_rows = 8 * 8 * 8 * 8;

        let mut row_group_id_sum = 0;
        let mut column_id_sum = 0;
        let mut row_start_id_sum = 0;
        let mut row_count_sum = 0;
        let mut memory_size_sum = 0;
        let mut hit_count_sum = 0;
        for file_no in 0..8 {
            let file_name = format!("test_{file_no}.parquet");
            let file = cache.register_or_get_file(file_name, LiquidCacheMode::InMemoryArrow);
            for rg in 0..8 {
                for col in 0..8 {
                    for row in 0..8 {
                        let row_group = file.row_group(rg);
                        let column = row_group.get_column_or_create(col);
                        assert!(column.insert_arrow_array(row, array.clone()).is_ok());

                        row_group_id_sum += rg as u64;
                        column_id_sum += col as u64;
                        row_start_id_sum += row as u64;
                        row_count_sum += array.len() as u64;
                        memory_size_sum += array.get_array_memory_size();

                        if row % 2 == 0 {
                            _ = column.get_arrow_array_test_only(row).unwrap();
                            hit_count_sum += 1;
                        }
                    }
                }
            }
        }

        let mut tmp_file = NamedTempFile::new()?;
        cache.write_stats(tmp_file.path())?;

        // Read and verify stats
        let mut bytes = Vec::new();
        tmp_file.read_to_end(&mut bytes)?;
        let bytes = Bytes::from(bytes);
        let reader = ParquetRecordBatchReader::try_new(bytes, 8192)?;

        let batch = reader.into_iter().next().unwrap()?;
        assert_eq!(batch.num_rows(), num_rows);

        macro_rules! uint64_col {
            ($batch:expr, $col_idx:expr) => {
                $batch
                    .column_by_name($col_idx)
                    .unwrap()
                    .as_primitive::<UInt64Type>()
            };
        }

        let row_group_id_array = uint64_col!(batch, "row_group_id");
        let column_id_array = uint64_col!(batch, "column_id");
        let row_start_id_array = uint64_col!(batch, "row_start_id");
        let row_count_array = uint64_col!(batch, "row_count");
        let memory_size_array = uint64_col!(batch, "memory_size");
        let hit_count_array = uint64_col!(batch, "hit_count");

        assert_eq!(
            row_group_id_array.iter().map(|v| v.unwrap()).sum::<u64>(),
            row_group_id_sum
        );
        assert_eq!(
            column_id_array.iter().map(|v| v.unwrap()).sum::<u64>(),
            column_id_sum
        );
        assert_eq!(
            row_start_id_array.iter().map(|v| v.unwrap()).sum::<u64>(),
            row_start_id_sum
        );
        assert_eq!(
            row_count_array.iter().map(|v| v.unwrap()).sum::<u64>(),
            row_count_sum
        );
        assert_eq!(
            memory_size_array.iter().map(|v| v.unwrap()).sum::<u64>(),
            memory_size_sum as u64
        );
        assert_eq!(
            hit_count_array.iter().map(|v| v.unwrap()).sum::<u64>(),
            hit_count_sum
        );

        Ok(())
    }
}
