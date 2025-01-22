use std::sync::{atomic::Ordering, Arc};

use arrow::array::{DictionaryArray, RecordBatch, StringArray, UInt32Array, UInt64Array};
use arrow_schema::{DataType, Field, Schema};

use super::{CacheType, CachedColumnBatch, LiquidCache, LockCtx};

/// ArrowCacheStatistics is used to collect statistics about the arrow array cache.
#[derive(Debug, serde::Serialize, Default)]
pub struct LiquidCacheStatistics {
    /// Row group ids
    pub row_group_ids: Vec<u64>,
    /// Column ids
    pub column_ids: Vec<u64>,
    /// Row start ids
    pub row_start_ids: Vec<u64>,
    /// Row counts
    pub row_counts: Vec<u64>,
    /// Memory sizes
    pub memory_sizes: Vec<u64>,
    /// Cache types
    pub cache_types: Vec<CacheType>,
    /// Hit counts
    pub hit_counts: Vec<u64>,
}

impl LiquidCacheStatistics {
    /// Create a new ArrowCacheStatistics.
    pub fn new() -> Self {
        LiquidCacheStatistics {
            row_group_ids: Vec::new(),
            column_ids: Vec::new(),
            row_start_ids: Vec::new(),
            row_counts: Vec::new(),
            memory_sizes: Vec::new(),
            cache_types: Vec::new(),
            hit_counts: Vec::new(),
        }
    }

    /// Add an entry to the statistics.
    #[allow(clippy::too_many_arguments)]
    pub fn add_entry(
        &mut self,
        row_group_id: u64,
        column_id: u64,
        row_start_id: u64,
        row_count: u64,
        memory_size: u64,
        cache_type: CacheType,
        hit_count: u64,
    ) {
        self.row_group_ids.push(row_group_id);
        self.column_ids.push(column_id);
        self.row_start_ids.push(row_start_id);
        self.row_counts.push(row_count);
        self.memory_sizes.push(memory_size);
        self.cache_types.push(cache_type);
        self.hit_counts.push(hit_count);
    }

    /// Get the total memory usage of the cache.
    pub fn memory_usage(&self) -> usize {
        self.memory_sizes.iter().sum::<u64>() as usize
    }

    /// Convert the statistics to a record batch.
    pub fn into_record_batch(self) -> RecordBatch {
        let row_group_ids = Arc::new(UInt64Array::from_iter(self.row_group_ids));
        let column_ids = Arc::new(UInt64Array::from(self.column_ids));
        let row_start_ids = Arc::new(UInt64Array::from(self.row_start_ids));
        let row_counts = Arc::new(UInt64Array::from(self.row_counts));
        let memory_sizes = Arc::new(UInt64Array::from(self.memory_sizes));

        let cache_type_values: Vec<&str> = self
            .cache_types
            .iter()
            .map(|ct| match ct {
                CacheType::InMemory => "InMemory",
                CacheType::OnDisk => "OnDisk",
                CacheType::Etc => "Etc",
            })
            .collect();
        let cache_types = Arc::new(DictionaryArray::new(
            UInt32Array::from((0..self.cache_types.len() as u32).collect::<Vec<u32>>()),
            Arc::new(StringArray::from(cache_type_values)),
        ));

        let hit_counts = Arc::new(UInt64Array::from(self.hit_counts.clone()));

        RecordBatch::try_new(
            Arc::new(Schema::new(vec![
                Field::new("row_group_id", DataType::UInt64, false),
                Field::new("column_id", DataType::UInt64, false),
                Field::new("row_start_id", DataType::UInt64, false),
                Field::new("row_count", DataType::UInt64, false),
                Field::new("memory_size", DataType::UInt64, false),
                Field::new(
                    "cache_type",
                    DataType::Dictionary(Box::new(DataType::UInt32), Box::new(DataType::Utf8)),
                    false,
                ),
                Field::new("hit_count", DataType::UInt64, false),
            ])),
            vec![
                row_group_ids,
                column_ids,
                row_start_ids,
                row_counts,
                memory_sizes,
                cache_types,
                hit_counts,
            ],
        )
        .unwrap()
    }
}

impl LiquidCache {
    /// Collect statistics about the cache.
    pub fn stats(&self) -> LiquidCacheStatistics {
        let mut stats = LiquidCacheStatistics::new();
        let mut ctx = LockCtx::UNLOCKED;

        for (row_group_id, row_group_lock) in self.value.iter().enumerate() {
            let (row_group, mut ctx) = row_group_lock.read(&mut ctx);

            for (column_id, row_mapping) in row_group.iter() {
                for (row_start_id, cached_entry) in row_mapping {
                    let cached_entry = cached_entry.value(&mut ctx);
                    let cache_type = match &cached_entry.0.value {
                        CachedColumnBatch::ArrowMemory(_) => CacheType::InMemory,
                        CachedColumnBatch::ArrowDisk(_) => CacheType::OnDisk,
                        CachedColumnBatch::LiquidMemory(_) => CacheType::Etc,
                    };

                    let memory_size = cached_entry.0.value.memory_usage();
                    let row_count = match &cached_entry.0.value {
                        CachedColumnBatch::ArrowMemory(array) => array.len(),
                        CachedColumnBatch::ArrowDisk(_) => 0, // We don't know the row count for on-disk entries
                        CachedColumnBatch::LiquidMemory(array) => array.len(),
                    };

                    stats.add_entry(
                        row_group_id as u64,
                        *column_id as u64,
                        *row_start_id as u64,
                        row_count as u64,
                        memory_size as u64,
                        cache_type,
                        cached_entry.0.hit_count.load(Ordering::Relaxed) as u64,
                    );
                }
            }
        }

        stats
    }
}
