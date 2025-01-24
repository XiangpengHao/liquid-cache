use ahash::AHashMap;
use arrow::array::AsArray;
use arrow::array::{ArrayRef, RecordBatch, RecordBatchWriter};
use arrow::ipc::reader::FileReader;
use arrow::ipc::writer::FileWriter;
use arrow_schema::{DataType, Field, Schema};
use lock_spec::*;
use std::fmt::Display;
use std::io::Seek;
use std::ops::{DerefMut, Range};
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Arc, RwLockReadGuard};
use utils::RangedFile;
mod stats;

use super::liquid_array::{AsLiquidArray, LiquidArrayRef, LiquidPrimitiveArray, LiquidStringArray};
mod lock_spec;
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
    OnDisk(OrderedMutex<LockDiskFile, std::fs::File>),
    NoCache,
    Liquid(LiquidCompressorStates),
}

struct LiquidCompressorStates {
    fsst_compressor: OrderedRwLock<LockEtcFsstCompressor, AHashMap<usize, Arc<fsst::Compressor>>>, // column_id -> compressor
}

impl std::fmt::Debug for LiquidCompressorStates {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "EtcCompressorStates")
    }
}

impl LiquidCompressorStates {
    fn new() -> Self {
        Self {
            fsst_compressor: OrderedRwLock::new(AHashMap::new()),
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
    inner: OrderedRwLock<LockedEntry, CachedEntryInner>,
}

impl CachedEntry {
    fn increment_hit_count<L>(&self, ctx: &mut LockCtx<L>)
    where
        L: LockBefore<LockedEntry>,
    {
        self.inner
            .read(ctx)
            .0
            .hit_count
            .fetch_add(1, Ordering::Relaxed);
    }

    fn value<L>(
        &self,
        ctx: &mut LockCtx<L>,
    ) -> (RwLockReadGuard<'_, CachedEntryInner>, LockCtx<LockedEntry>)
    where
        L: LockBefore<LockedEntry>,
    {
        self.inner.read(ctx)
    }

    fn new_in_memory(array: ArrayRef) -> Self {
        let val = CachedColumnBatch::ArrowMemory(array);
        CachedEntry {
            inner: OrderedRwLock::new(CachedEntryInner::new(val)),
        }
    }

    fn new(value: CachedColumnBatch) -> Self {
        CachedEntry {
            inner: OrderedRwLock::new(CachedEntryInner::new(value)),
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

    fn convert_to(&mut self, to: &CacheStates, ctx: &mut LockCtx<LockColumnMapping>) {
        match (&self, to) {
            (Self::ArrowMemory(v), CacheStates::OnDisk(file)) => {
                let (mut file, _) = file.lock(ctx);

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
    columns: OrderedRwLock<LockColumnMapping, AHashMap<usize, CachedRows>>,
}

impl CachedRowGroup {
    fn new(row_group_id: usize, cache_mode: CacheStates, batch_size: usize) -> Self {
        Self {
            row_group_id,
            cache_mode,
            batch_size,
            columns: OrderedRwLock::new(AHashMap::new()),
        }
    }

    pub(crate) fn batch_size(&self) -> usize {
        self.batch_size
    }

    pub(crate) fn is_cached(&self, column_id: usize, row_id: usize) -> bool {
        let mut ctx = LockCtx::UNLOCKED;
        let (column, _) = self.columns.read(&mut ctx);
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

        let mut ctx = LockCtx::UNLOCKED;

        let (cache, mut ctx_column) = self.columns.read(&mut ctx);

        let column_cache = cache.get(&column_id)?;
        let cached_entry = column_cache.rows.get(&row_id)?;
        cached_entry.increment_hit_count(&mut ctx_column);
        let cached_entry = cached_entry.value(&mut ctx_column);
        match &cached_entry.0.value {
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

        let mut ctx = LockCtx::UNLOCKED;
        if self.is_cached(column_id, row_id) {
            return;
        }

        let (mut cache, mut column_ctx) = self.columns.write(&mut ctx);

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
                cached_value.convert_to(&self.cache_mode, &mut column_ctx);

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
                        let (compressor, _) = states.fsst_compressor.read(&mut column_ctx);
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
                        let (mut compressors, _) = states.fsst_compressor.write(&mut column_ctx);
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
                        let (compressor, _) = states.fsst_compressor.read(&mut column_ctx);
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
                        let (mut compressors, _) = states.fsst_compressor.write(&mut column_ctx);
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
    /// Vec of RwLocks, where index is the row group index and value is the ColumnMapping
    value: Vec<Arc<CachedRowGroup>>,
    cache_mode: CacheStates,
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
        const MAX_ROW_GROUPS: usize = 512;

        LiquidCache {
            value: (0..MAX_ROW_GROUPS)
                .map(|i| {
                    Arc::new(CachedRowGroup::new(
                        i,
                        Self::make_states(cache_mode),
                        batch_size,
                    ))
                })
                .collect(),
            cache_mode: Self::make_states(cache_mode),
            batch_size,
        }
    }

    fn make_states(cache_mode: LiquidCacheMode) -> CacheStates {
        match cache_mode {
            LiquidCacheMode::InMemoryArrow => CacheStates::InMemory,
            LiquidCacheMode::OnDiskArrow => CacheStates::OnDisk(OrderedMutex::new(
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
        !matches!(self.cache_mode, CacheStates::NoCache)
    }

    pub fn cache_mode(&self) -> LiquidCacheMode {
        (&self.cache_mode).into()
    }

    pub fn row_group(&self, row_group_id: usize) -> CachedRowGroupRef {
        self.value[row_group_id].clone()
    }

    /// Reset the cache.
    pub fn reset(&self) {
        let mut ctx = LockCtx::UNLOCKED;
        for row_group in self.value.iter() {
            let (mut row_group, _) = row_group.columns.write(&mut ctx);
            row_group.clear();
        }
    }
}
