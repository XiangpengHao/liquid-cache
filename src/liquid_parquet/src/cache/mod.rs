use ahash::AHashMap;
use arrow::array::AsArray;
use arrow::array::{ArrayRef, RecordBatch, RecordBatchWriter};
use arrow::ipc::reader::FileReader;
use arrow::ipc::writer::FileWriter;
use arrow_schema::{DataType, Field, Schema};
use std::fmt::Display;
use std::io::Seek;
use std::ops::{DerefMut, Range};
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Arc, Mutex, RwLock, RwLockReadGuard};
use utils::RangedFile;
mod stats;

use super::liquid_array::{AsLiquidArray, LiquidArrayRef, LiquidPrimitiveArray, LiquidStringArray};
mod utils;
use arrow::array::types::{
    Int8Type as ArrowInt8Type, Int16Type as ArrowInt16Type, Int32Type as ArrowInt32Type,
    Int64Type as ArrowInt64Type, UInt8Type as ArrowUInt8Type, UInt16Type as ArrowUInt16Type,
    UInt32Type as ArrowUInt32Type, UInt64Type as ArrowUInt64Type,
};

static ARROW_DISK_CACHE_PATH: &str = "target/arrow_disk_cache.etc";

#[derive(Debug)]
enum CacheStates {
    InMemory,
    OnDisk(Mutex<std::fs::File>),
    NoCache,
    Liquid(LiquidCompressorStates),
}

struct LiquidCompressorStates {
    fsst_compressor: RwLock<AHashMap<usize, Arc<fsst::Compressor>>>, // column_id -> compressor
}

impl std::fmt::Debug for LiquidCompressorStates {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "EtcCompressorStates")
    }
}

impl LiquidCompressorStates {
    fn new() -> Self {
        Self {
            fsst_compressor: RwLock::new(AHashMap::new()),
        }
    }
}

#[derive(Debug)]
struct CachedEntryInner {
    value: CachedColumnBatch,
    hit_count: AtomicU32,
}

impl CachedEntryInner {
    fn new(value: CachedColumnBatch) -> Self {
        Self {
            value,
            hit_count: AtomicU32::new(0),
        }
    }
}

#[derive(Debug)]
struct CachedEntry {
    inner: RwLock<CachedEntryInner>,
}

impl CachedEntry {
    fn increment_hit_count(&self) {
        self.inner
            .read()
            .unwrap()
            .hit_count
            .fetch_add(1, Ordering::Relaxed);
    }

    fn value(&self) -> RwLockReadGuard<'_, CachedEntryInner> {
        self.inner.read().unwrap()
    }

    fn new_in_memory(array: ArrayRef) -> Self {
        let val = CachedColumnBatch::ArrowMemory(array);
        CachedEntry {
            inner: RwLock::new(CachedEntryInner::new(val)),
        }
    }

    fn new(value: CachedColumnBatch) -> Self {
        CachedEntry {
            inner: RwLock::new(CachedEntryInner::new(value)),
        }
    }
}

#[derive(Debug)]
enum CachedColumnBatch {
    ArrowMemory(ArrayRef),
    ArrowDisk(Range<u64>),
    LiquidMemory(LiquidArrayRef),
}

impl CachedColumnBatch {
    fn memory_usage(&self) -> usize {
        match self {
            Self::ArrowMemory(array) => array.get_array_memory_size(),
            Self::ArrowDisk(_) => 0,
            Self::LiquidMemory(array) => array.get_array_memory_size(),
        }
    }

    fn convert_to(&mut self, to: &CacheStates) {
        match (&self, to) {
            (Self::ArrowMemory(v), CacheStates::OnDisk(file)) => {
                let mut file = file.lock().unwrap();

                // Align start_pos to next 512 boundary for better disk I/O
                let start_pos = file.metadata().unwrap().len();
                let start_pos = (start_pos + 511) & !511;
                let start_pos = file.seek(std::io::SeekFrom::Start(start_pos)).unwrap();

                let mut writer = std::io::BufWriter::new(file.deref_mut());
                let schema = Arc::new(Schema::new(vec![Field::new(
                    "_",
                    v.data_type().clone(),
                    v.is_nullable(),
                )]));
                let mut arrow_writer = FileWriter::try_new(&mut writer, &schema).unwrap();
                let record_batch = RecordBatch::try_new(schema, vec![v.clone()]).unwrap();
                arrow_writer.write(&record_batch).unwrap();
                arrow_writer.close().unwrap();

                let file = writer.into_inner().unwrap();
                let end_pos = file.stream_position().unwrap();
                *self = CachedColumnBatch::ArrowDisk(start_pos..end_pos);
            }
            _ => unimplemented!("convert {} to {:?} not implemented", self, to),
        }
    }
}

impl Display for CachedColumnBatch {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ArrowMemory(_) => write!(f, "ArrowMemory"),
            Self::ArrowDisk(_) => write!(f, "ArrowDisk"),
            Self::LiquidMemory(_) => write!(f, "LiquidMemory"),
        }
    }
}

/// CacheType is used to identify the type of cache.
#[derive(Debug, serde::Serialize)]
pub enum CacheType {
    InMemory,
    OnDisk,
    Etc,
}

#[derive(Debug)]
struct CachedRows {
    #[allow(unused)]
    row_group_id: usize,
    #[allow(unused)]
    column_id: usize,
    rows: AHashMap<usize, CachedEntry>,
}

impl CachedRows {
    fn new(row_group_id: usize, column_id: usize) -> Self {
        Self {
            row_group_id,
            column_id,
            rows: AHashMap::new(),
        }
    }
}

#[derive(Debug)]
pub struct CachedRowGroup {
    row_group_id: usize,
    cache_mode: CacheStates,
    batch_size: usize,
    columns: RwLock<AHashMap<usize, CachedRows>>,
}

impl CachedRowGroup {
    fn new(row_group_id: usize, cache_mode: CacheStates, batch_size: usize) -> Self {
        Self {
            row_group_id,
            cache_mode,
            batch_size,
            columns: RwLock::new(AHashMap::new()),
        }
    }

    pub(crate) fn batch_size(&self) -> usize {
        self.batch_size
    }

    pub(crate) fn is_cached(&self, column_id: usize, row_id: usize) -> bool {
        let column = self.columns.read().unwrap();
        if let Some(row_cache) = column.get(&column_id) {
            row_cache.rows.contains_key(&row_id)
        } else {
            false
        }
    }

    /// Get an arrow array from the cache with a selection.
    pub fn get_arrow_array(&self, column_id: usize, row_id: usize) -> Option<ArrayRef> {
        if matches!(self.cache_mode, CacheStates::NoCache) {
            return None;
        }

        let cache = self.columns.read().unwrap();

        let column_cache = cache.get(&column_id)?;
        let cached_entry = column_cache.rows.get(&row_id)?;
        cached_entry.increment_hit_count();
        let cached_entry = cached_entry.value();
        match &cached_entry.value {
            CachedColumnBatch::ArrowMemory(array) => Some(array.clone()),
            CachedColumnBatch::ArrowDisk(range) => {
                let file = std::fs::File::open(ARROW_DISK_CACHE_PATH).ok()?;
                let ranged_file = RangedFile::new(file, range.clone()).ok()?;

                let reader = std::io::BufReader::new(ranged_file);
                let mut arrow_reader = FileReader::try_new(reader, None).ok()?;
                let batch = arrow_reader.next().unwrap().unwrap();
                let array = batch.column(0);
                Some(array.clone())
            }
            CachedColumnBatch::LiquidMemory(array) => {
                if let Some(string_array) = array.as_string_array_opt() {
                    let arrow_array = string_array.to_dict_string();
                    Some(Arc::new(arrow_array))
                } else {
                    let arrow_array = array.to_arrow_array();
                    Some(arrow_array)
                }
            }
        }
    }

    pub(crate) fn insert_arrow_array(&self, column_id: usize, row_id: usize, array: ArrayRef) {
        if matches!(self.cache_mode, CacheStates::NoCache) {
            return;
        }

        if self.is_cached(column_id, row_id) {
            return;
        }

        let mut cache = self.columns.write().unwrap();

        let column_cache = cache
            .entry(column_id)
            .or_insert_with(|| CachedRows::new(self.row_group_id, column_id));

        match &self.cache_mode {
            CacheStates::InMemory => {
                let old = column_cache
                    .rows
                    .insert(row_id, CachedEntry::new_in_memory(array));
                assert!(old.is_none());
            }
            CacheStates::OnDisk(_file) => {
                let mut cached_value = CachedColumnBatch::ArrowMemory(array);
                cached_value.convert_to(&self.cache_mode);

                column_cache
                    .rows
                    .insert(row_id, CachedEntry::new(cached_value));
            }

            CacheStates::NoCache => {
                unreachable!()
            }

            CacheStates::Liquid(states) => {
                let data_type = array.data_type();
                let array = array.as_ref();
                if data_type.is_primitive() {
                    let primitive: LiquidArrayRef = match data_type {
                        DataType::Int8 => {
                            Arc::new(LiquidPrimitiveArray::<ArrowInt8Type>::from_arrow_array(
                                array.as_primitive::<ArrowInt8Type>().clone(),
                            ))
                        }
                        DataType::Int16 => {
                            Arc::new(LiquidPrimitiveArray::<ArrowInt16Type>::from_arrow_array(
                                array.as_primitive::<ArrowInt16Type>().clone(),
                            ))
                        }
                        DataType::Int32 => {
                            Arc::new(LiquidPrimitiveArray::<ArrowInt32Type>::from_arrow_array(
                                array.as_primitive::<ArrowInt32Type>().clone(),
                            ))
                        }
                        DataType::Int64 => {
                            Arc::new(LiquidPrimitiveArray::<ArrowInt64Type>::from_arrow_array(
                                array.as_primitive::<ArrowInt64Type>().clone(),
                            ))
                        }
                        DataType::UInt8 => {
                            Arc::new(LiquidPrimitiveArray::<ArrowUInt8Type>::from_arrow_array(
                                array.as_primitive::<ArrowUInt8Type>().clone(),
                            ))
                        }
                        DataType::UInt16 => {
                            Arc::new(LiquidPrimitiveArray::<ArrowUInt16Type>::from_arrow_array(
                                array.as_primitive::<ArrowUInt16Type>().clone(),
                            ))
                        }
                        DataType::UInt32 => {
                            Arc::new(LiquidPrimitiveArray::<ArrowUInt32Type>::from_arrow_array(
                                array.as_primitive::<ArrowUInt32Type>().clone(),
                            ))
                        }
                        DataType::UInt64 => {
                            Arc::new(LiquidPrimitiveArray::<ArrowUInt64Type>::from_arrow_array(
                                array.as_primitive::<ArrowUInt64Type>().clone(),
                            ))
                        }
                        _ => panic!("unsupported data type {:?}", data_type),
                    };
                    column_cache.rows.insert(
                        row_id,
                        CachedEntry::new(CachedColumnBatch::LiquidMemory(primitive)),
                    );
                    return;
                }
                // other types
                match array.data_type() {
                    DataType::Utf8View => {
                        let compressor = states.fsst_compressor.read().unwrap();
                        if let Some(compressor) = compressor.get(&column_id) {
                            let compressed = LiquidStringArray::from_string_view_array(
                                array.as_string_view(),
                                Some(compressor.clone()),
                            );
                            column_cache.rows.insert(
                                row_id,
                                CachedEntry::new(CachedColumnBatch::LiquidMemory(Arc::new(
                                    compressed,
                                ))),
                            );
                            return;
                        }

                        drop(compressor);
                        let mut compressors = states.fsst_compressor.write().unwrap();
                        let compressed =
                            LiquidStringArray::from_string_view_array(array.as_string_view(), None);
                        let compressor = compressed.compressor();
                        compressors.insert(column_id, compressor);
                        column_cache.rows.insert(
                            row_id,
                            CachedEntry::new(CachedColumnBatch::LiquidMemory(Arc::new(compressed))),
                        );
                    }
                    DataType::Utf8 => {
                        let compressor = states.fsst_compressor.read().unwrap();
                        if let Some(compressor) = compressor.get(&column_id) {
                            let compressed = LiquidStringArray::from_string_array(
                                array.as_string::<i32>(),
                                Some(compressor.clone()),
                            );
                            column_cache.rows.insert(
                                row_id,
                                CachedEntry::new(CachedColumnBatch::LiquidMemory(Arc::new(
                                    compressed,
                                ))),
                            );
                            return;
                        }

                        drop(compressor);
                        let mut compressors = states.fsst_compressor.write().unwrap();
                        let compressed =
                            LiquidStringArray::from_string_array(array.as_string::<i32>(), None);
                        let compressor = compressed.compressor();
                        compressors.insert(column_id, compressor);
                        column_cache.rows.insert(
                            row_id,
                            CachedEntry::new(CachedColumnBatch::LiquidMemory(Arc::new(compressed))),
                        );
                    }
                    DataType::Dictionary(_, _) => {
                        if let Some(dict_array) = array.as_dictionary_opt::<ArrowUInt16Type>() {
                            let values = dict_array.values();
                            if let Some(_string_array) = values.as_string_opt::<i32>() {
                                let etc_array = LiquidStringArray::from_dict_array(dict_array);
                                column_cache.rows.insert(
                                    row_id,
                                    CachedEntry::new(CachedColumnBatch::LiquidMemory(Arc::new(
                                        etc_array,
                                    ))),
                                );
                                return;
                            }
                        }

                        panic!("unsupported data type {:?}", array.data_type());
                    }
                    _ => panic!("unsupported data type {:?}", array.data_type()),
                }
            }
        }
    }
}

pub type CachedRowGroupRef = Arc<CachedRowGroup>;

/// LiquidCache.
/// Caching granularity: column batch.
/// To identify a column batch, we need:
/// 1. Parquet file path
/// 2. Row group index
/// 3. Column index
/// 4. Row offset, assuming each batch is at most 8192 rows.
#[derive(Debug)]
pub struct LiquidCache {
    row_groups: Mutex<AHashMap<usize, Arc<CachedRowGroup>>>,
    cache_mode: LiquidCacheMode,
    // cache granularity
    batch_size: usize,
}

pub type LiquidCacheRef = Arc<LiquidCache>;

#[derive(Debug, Copy, Clone)]
pub enum LiquidCacheMode {
    InMemoryArrow,
    OnDiskArrow,
    NoCache,
    InMemoryLiquid,
}

impl From<&CacheStates> for LiquidCacheMode {
    fn from(cache_states: &CacheStates) -> Self {
        match cache_states {
            CacheStates::InMemory => LiquidCacheMode::InMemoryArrow,
            CacheStates::OnDisk(_) => LiquidCacheMode::OnDiskArrow,
            CacheStates::NoCache => LiquidCacheMode::NoCache,
            CacheStates::Liquid(_) => LiquidCacheMode::InMemoryLiquid,
        }
    }
}

impl LiquidCache {
    pub fn new(cache_mode: LiquidCacheMode, batch_size: usize) -> Self {
        assert!(batch_size.is_power_of_two());

        LiquidCache {
            row_groups: Mutex::new(AHashMap::new()),
            cache_mode,
            batch_size,
        }
    }

    fn make_states(cache_mode: LiquidCacheMode) -> CacheStates {
        match cache_mode {
            LiquidCacheMode::InMemoryArrow => CacheStates::InMemory,
            LiquidCacheMode::OnDiskArrow => CacheStates::OnDisk(Mutex::new(
                std::fs::File::create(ARROW_DISK_CACHE_PATH).unwrap(),
            )),
            LiquidCacheMode::NoCache => CacheStates::NoCache,
            LiquidCacheMode::InMemoryLiquid => CacheStates::Liquid(LiquidCompressorStates::new()),
        }
    }

    pub fn batch_size(&self) -> usize {
        self.batch_size
    }

    /// Check if the cache is enabled.
    pub fn cache_enabled(&self) -> bool {
        !matches!(self.cache_mode, LiquidCacheMode::NoCache)
    }

    pub fn cache_mode(&self) -> LiquidCacheMode {
        self.cache_mode
    }

    pub fn row_group(&self, row_group_id: usize) -> CachedRowGroupRef {
        let mut row_groups = self.row_groups.lock().unwrap();
        let row_group = row_groups.entry(row_group_id).or_insert_with(|| {
            Arc::new(CachedRowGroup::new(
                row_group_id,
                Self::make_states(self.cache_mode),
                self.batch_size,
            ))
        });
        row_group.clone()
    }

    /// Reset the cache.
    pub fn reset(&self) {
        let mut row_groups = self.row_groups.lock().unwrap();
        row_groups.clear();
    }
}
