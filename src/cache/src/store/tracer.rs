use crate::sync::{Arc, Mutex, atomic::AtomicBool};
use std::{
    fs::File,
    path::Path,
    time::{SystemTime, UNIX_EPOCH},
};

use arrow::{
    array::{ArrayRef, RecordBatch, UInt64Array},
    datatypes::{DataType, Field, Schema},
};
use parquet::{
    arrow::arrow_writer::ArrowWriter, basic::Compression, file::properties::WriterProperties,
};

use super::utils::CacheEntryID;

struct TraceEvent {
    entry_id: CacheEntryID,
    cache_memory_bytes: usize,
    entry_size: usize,
    time_stamp_nanos: u128,
}

pub struct CacheTracer {
    enabled: AtomicBool,
    entries: Mutex<Vec<TraceEvent>>,
}

impl std::fmt::Debug for CacheTracer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "CacheTracer")
    }
}

impl CacheTracer {
    pub(super) fn new() -> Self {
        Self {
            enabled: AtomicBool::new(false),
            entries: Mutex::new(Vec::new()),
        }
    }

    pub fn enable(&self) {
        self.enabled
            .store(true, std::sync::atomic::Ordering::Relaxed);
    }

    pub fn disable(&self) {
        self.enabled
            .store(false, std::sync::atomic::Ordering::Relaxed);
    }

    fn enabled(&self) -> bool {
        self.enabled.load(std::sync::atomic::Ordering::Relaxed)
    }

    pub(super) fn trace_get(
        &self,
        entry_id: CacheEntryID,
        cache_memory_bytes: usize,
        entry_size: usize,
    ) {
        if !self.enabled() {
            return;
        }
        let mut entries = self.entries.lock().unwrap();
        let time_stamp_nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        entries.push(TraceEvent {
            entry_id,
            cache_memory_bytes,
            entry_size,
            time_stamp_nanos,
        });
    }

    pub fn flush(&self, to_file: impl AsRef<Path>) {
        let mut entries = self.entries.lock().unwrap();
        if entries.is_empty() {
            return; // Nothing to flush
        }

        let schema = Arc::new(Schema::new(vec![
            Field::new("file_id", DataType::UInt64, false),
            Field::new("row_group_id", DataType::UInt64, false),
            Field::new("column_id", DataType::UInt64, false),
            Field::new("batch_id", DataType::UInt64, false),
            Field::new("entry_size", DataType::UInt64, false),
            Field::new("cache_memory_bytes", DataType::UInt64, false),
            Field::new("time_stamp_nanos", DataType::UInt64, false),
        ]));

        let num_rows = entries.len();
        let mut file_ids = Vec::with_capacity(num_rows);
        let mut row_group_ids = Vec::with_capacity(num_rows);
        let mut column_ids = Vec::with_capacity(num_rows);
        let mut batch_ids = Vec::with_capacity(num_rows);
        let mut entry_sizes = Vec::with_capacity(num_rows);
        let mut cache_memory_bytes_vec = Vec::with_capacity(num_rows);
        let mut time_stamp_nanos_vec = Vec::with_capacity(num_rows);

        for event in entries.iter() {
            file_ids.push(event.entry_id.file_id_inner());
            row_group_ids.push(event.entry_id.row_group_id_inner());
            column_ids.push(event.entry_id.column_id_inner());
            batch_ids.push(event.entry_id.batch_id_inner()); // Assuming batch_id_inner exists or add it
            entry_sizes.push(event.entry_size as u64);
            cache_memory_bytes_vec.push(event.cache_memory_bytes as u64);
            time_stamp_nanos_vec.push(event.time_stamp_nanos as u64);
        }

        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(UInt64Array::from(file_ids)) as ArrayRef,
                Arc::new(UInt64Array::from(row_group_ids)) as ArrayRef,
                Arc::new(UInt64Array::from(column_ids)) as ArrayRef,
                Arc::new(UInt64Array::from(batch_ids)) as ArrayRef,
                Arc::new(UInt64Array::from(entry_sizes)) as ArrayRef,
                Arc::new(UInt64Array::from(cache_memory_bytes_vec)) as ArrayRef,
                Arc::new(UInt64Array::from(time_stamp_nanos_vec)) as ArrayRef,
            ],
        )
        .expect("Failed to create record batch");

        let file = File::create(to_file).expect("Failed to create trace file");
        let props = WriterProperties::builder()
            .set_compression(Compression::SNAPPY)
            .build();
        let mut writer = ArrowWriter::try_new(file, schema, Some(props))
            .expect("Failed to create parquet writer");

        writer
            .write(&batch)
            .expect("Failed to write batch to parquet file");
        writer.close().expect("Failed to close parquet writer");

        entries.clear(); // Clear entries after successful flush
    }
}

#[cfg(test)]
mod tests {
    use crate::store::utils::BatchID;

    use super::*;
    use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
    use std::fs::File;
    use tempfile::tempdir;

    #[test]
    fn test_cache_tracer_enable_disable() {
        let tracer = CacheTracer::new();
        assert!(!tracer.enabled());

        tracer.enable();
        assert!(tracer.enabled());

        tracer.disable();
        assert!(!tracer.enabled());
    }

    #[test]
    fn test_cache_tracer_event_recording() {
        let tracer = CacheTracer::new();

        // Should not record when disabled
        let entry_id = CacheEntryID::new(1, 2, 3, BatchID::from_raw(4));
        tracer.trace_get(entry_id, 1000, 100);
        assert!(tracer.entries.lock().unwrap().is_empty());

        // Should record when enabled
        tracer.enable();
        tracer.trace_get(entry_id, 1000, 100);
        assert_eq!(tracer.entries.lock().unwrap().len(), 1);

        // Multiple events
        tracer.trace_get(entry_id, 2000, 100);
        assert_eq!(tracer.entries.lock().unwrap().len(), 2);

        // Check entry data
        let entries = tracer.entries.lock().unwrap();
        assert_eq!(entries[0].entry_id, entry_id);
        assert_eq!(entries[0].cache_memory_bytes, 1000);
        assert_eq!(entries[0].entry_size, 100);
        assert_eq!(entries[1].entry_id, entry_id);
        assert_eq!(entries[1].cache_memory_bytes, 2000);
        assert_eq!(entries[1].entry_size, 100);
    }

    #[test]
    fn test_cache_tracer_flush_empty() {
        let temp_dir = tempdir().unwrap();
        let file_path = temp_dir.path().join("empty_trace.parquet");

        let tracer = CacheTracer::new();
        tracer.flush(&file_path);

        // File shouldn't exist since there was nothing to flush
        assert!(!file_path.exists());
    }

    #[test]
    fn test_cache_tracer_flush() {
        let temp_dir = tempdir().unwrap();
        let file_path = temp_dir.path().join("trace.parquet");

        let tracer = CacheTracer::new();
        tracer.enable();

        // Add some entries
        let entry_id1 = CacheEntryID::new(1, 2, 3, BatchID::from_raw(4));
        let entry_id2 = CacheEntryID::new(5, 6, 7, BatchID::from_raw(8));

        tracer.trace_get(entry_id1, 1000, 100);
        tracer.trace_get(entry_id2, 2000, 100);

        // Flush to file
        tracer.flush(&file_path);

        // Verify entries were cleared
        assert!(tracer.entries.lock().unwrap().is_empty());

        // Verify file exists
        assert!(file_path.exists());

        // Read the Parquet file and verify contents
        let file = File::open(&file_path).unwrap();
        let reader = ParquetRecordBatchReaderBuilder::try_new(file)
            .unwrap()
            .with_batch_size(1024)
            .build()
            .unwrap();

        let batch = reader.into_iter().next().unwrap().unwrap();
        assert_eq!(batch.num_rows(), 2);

        // Verify the columns exist
        assert_eq!(batch.num_columns(), 7);

        // Check file_id column values
        let file_id_array = batch
            .column(0)
            .as_any()
            .downcast_ref::<UInt64Array>()
            .unwrap();
        assert_eq!(file_id_array.value(0), 1);
        assert_eq!(file_id_array.value(1), 5);

        // Check row_group_id column values
        let row_group_id_array = batch
            .column(1)
            .as_any()
            .downcast_ref::<UInt64Array>()
            .unwrap();
        assert_eq!(row_group_id_array.value(0), 2);
        assert_eq!(row_group_id_array.value(1), 6);

        // Check column_id column values
        let column_id_array = batch
            .column(2)
            .as_any()
            .downcast_ref::<UInt64Array>()
            .unwrap();
        assert_eq!(column_id_array.value(0), 3);
        assert_eq!(column_id_array.value(1), 7);

        // Check batch_id column values
        let batch_id_array = batch
            .column(3)
            .as_any()
            .downcast_ref::<UInt64Array>()
            .unwrap();
        assert_eq!(batch_id_array.value(0), 4);
        assert_eq!(batch_id_array.value(1), 8);

        // Check entry_size column values
        let entry_size_array = batch
            .column(4)
            .as_any()
            .downcast_ref::<UInt64Array>()
            .unwrap();
        assert_eq!(entry_size_array.value(0), 100);
        assert_eq!(entry_size_array.value(1), 100);

        // Check cache_memory_bytes column values
        let cache_memory_bytes_array = batch
            .column(5)
            .as_any()
            .downcast_ref::<UInt64Array>()
            .unwrap();
        assert_eq!(cache_memory_bytes_array.value(0), 1000);
        assert_eq!(cache_memory_bytes_array.value(1), 2000);
    }

    #[test]
    fn test_cache_tracer_multiple_flush() {
        let temp_dir = tempdir().unwrap();
        let file_path1 = temp_dir.path().join("trace1.parquet");
        let file_path2 = temp_dir.path().join("trace2.parquet");

        let tracer = CacheTracer::new();
        tracer.enable();

        // Add first batch of entries
        tracer.trace_get(CacheEntryID::new(1, 2, 3, BatchID::from_raw(4)), 1000, 100);
        tracer.flush(&file_path1);

        // Add second batch of entries
        tracer.trace_get(CacheEntryID::new(5, 6, 7, BatchID::from_raw(8)), 2000, 100);
        tracer.flush(&file_path2);

        // Verify both files exist
        assert!(file_path1.exists());
        assert!(file_path2.exists());

        // Verify first file has one entry
        let file1 = File::open(&file_path1).unwrap();
        let reader1 = ParquetRecordBatchReaderBuilder::try_new(file1)
            .unwrap()
            .with_batch_size(1024)
            .build()
            .unwrap();
        let batch1 = reader1.into_iter().next().unwrap().unwrap();
        assert_eq!(batch1.num_rows(), 1);

        // Verify second file has one entry
        let file2 = File::open(&file_path2).unwrap();
        let reader2 = ParquetRecordBatchReaderBuilder::try_new(file2)
            .unwrap()
            .with_batch_size(1024)
            .build()
            .unwrap();
        let batch2 = reader2.into_iter().next().unwrap().unwrap();
        assert_eq!(batch2.num_rows(), 1);
    }
}
