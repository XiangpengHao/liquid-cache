use std::{
    fs::File,
    path::Path,
    sync::{Arc, Mutex, atomic::AtomicBool},
    time::{SystemTime, UNIX_EPOCH},
};

use arrow::{
    array::{ArrayRef, RecordBatch, UInt64Array},
    datatypes::{DataType, Field, Schema},
};
use parquet::{
    arrow::arrow_writer::ArrowWriter, basic::Compression, file::properties::WriterProperties,
};

use super::CacheEntryID;

struct TraceEvent {
    entry_id: CacheEntryID,
    cache_memory_bytes: usize,
    time_stamp_nanos: u128,
}

pub(super) struct CacheTracer {
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

    pub(super) fn enable(&self) {
        self.enabled
            .store(true, std::sync::atomic::Ordering::Relaxed);
    }

    pub(super) fn disable(&self) {
        self.enabled
            .store(false, std::sync::atomic::Ordering::Relaxed);
    }

    fn enabled(&self) -> bool {
        self.enabled.load(std::sync::atomic::Ordering::Relaxed)
    }

    pub(super) fn trace_get(&self, entry_id: CacheEntryID, cache_memory_bytes: usize) {
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
            time_stamp_nanos,
        });
    }

    pub(super) fn flush(&self, to_file: impl AsRef<Path>) {
        let mut entries = self.entries.lock().unwrap();
        if entries.is_empty() {
            return; // Nothing to flush
        }

        let schema = Arc::new(Schema::new(vec![
            Field::new("file_id", DataType::UInt64, false),
            Field::new("row_group_id", DataType::UInt64, false),
            Field::new("column_id", DataType::UInt64, false),
            Field::new("batch_id", DataType::UInt64, false),
            Field::new("cache_memory_bytes", DataType::UInt64, false),
            Field::new("time_stamp_nanos", DataType::UInt64, false),
        ]));

        let num_rows = entries.len();
        let mut file_ids = Vec::with_capacity(num_rows);
        let mut row_group_ids = Vec::with_capacity(num_rows);
        let mut column_ids = Vec::with_capacity(num_rows);
        let mut batch_ids = Vec::with_capacity(num_rows);
        let mut cache_memory_bytes_vec = Vec::with_capacity(num_rows);
        let mut time_stamp_nanos_vec = Vec::with_capacity(num_rows);

        for event in entries.iter() {
            file_ids.push(event.entry_id.file_id_inner());
            row_group_ids.push(event.entry_id.row_group_id_inner());
            column_ids.push(event.entry_id.column_id_inner());
            batch_ids.push(event.entry_id.batch_id_inner()); // Assuming batch_id_inner exists or add it
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
