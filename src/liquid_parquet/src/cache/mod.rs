use super::liquid_array::LiquidArrayRef;
use crate::liquid_array::ipc::{self, LiquidIPCContext};
use crate::{ABLATION_STUDY_MODE, AblationStudyMode, LiquidPredicate};
use ahash::AHashMap;
use arrow::array::{Array, ArrayRef, BooleanArray, RecordBatch};
use arrow::buffer::BooleanBuffer;
use arrow::compute::prep_null_mask_filter;
use arrow_schema::{ArrowError, DataType, Field, Schema};
use bytes::Bytes;
use liquid_cache_common::CacheMode;
use std::fmt::Display;
use std::fs::File;
use std::fs::OpenOptions;
use std::io::{Read, Write};
use std::path::PathBuf;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicU16, AtomicU64, Ordering};
use std::sync::{Arc, LazyLock, Mutex, RwLock};
pub(crate) use store::BatchID;
use store::{BudgetAccounting, CacheAdvice, CacheEntryID, CacheStore};
use tokio::runtime::Runtime;
use transcode::transcode_liquid_inner;

mod stats;
mod store;
mod transcode;

#[cfg(debug_assertions)]
static LOGGER_FILE: OnceLock<Arc<Mutex<File>>> = OnceLock::new();

#[cfg(debug_assertions)]
fn get_file_handle() -> Arc<Mutex<File>> {
    LOGGER_FILE
        .get_or_init(|| {
            let mut file = OpenOptions::new()
                .write(true)
                .create(true)
                .truncate(true)
                .open("./cache_trace.csv")
                .expect("Failed to open log file");
            _ = file.write(b"file,row_group,col,row,size,type\n").unwrap();
            Arc::new(Mutex::new(file))
        })
        .clone()
}

/// A dedicated Tokio thread pool for background transcoding tasks.
/// This pool is built with 4 worker threads.
static TRANSCODE_THREAD_POOL: LazyLock<Runtime> = LazyLock::new(|| {
    tokio::runtime::Builder::new_multi_thread()
        .worker_threads(4)
        .thread_name("transcode-worker")
        .enable_all()
        .build()
        .unwrap()
});

struct LiquidCompressorStates {
    fsst_compressor: RwLock<Option<Arc<fsst::Compressor>>>,
}

impl std::fmt::Debug for LiquidCompressorStates {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "EtcCompressorStates")
    }
}

impl LiquidCompressorStates {
    fn new() -> Self {
        Self {
            fsst_compressor: RwLock::new(None),
        }
    }
}

#[derive(Debug, Clone)]
enum CachedBatch {
    ArrowMemory(ArrayRef),
    LiquidMemory(LiquidArrayRef),
    OnDiskLiquid,
}

impl CachedBatch {
    fn memory_usage_bytes(&self) -> usize {
        match self {
            Self::ArrowMemory(array) => array.get_array_memory_size(),
            Self::LiquidMemory(array) => array.get_array_memory_size(),
            Self::OnDiskLiquid => 0,
        }
    }
}

impl Display for CachedBatch {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ArrowMemory(_) => write!(f, "ArrowMemory"),
            Self::LiquidMemory(_) => write!(f, "LiquidMemory"),
            Self::OnDiskLiquid => write!(f, "OnDiskLiquid"),
        }
    }
}

#[derive(Debug)]
pub struct LiquidCachedColumn {
    cache_mode: LiquidCacheMode,
    liquid_compressor_states: Arc<LiquidCompressorStates>,
    cache_store: Arc<CacheStore>,
    cache_dir: PathBuf,
    field: Arc<Field>,
    column_id: u64,
    row_group_id: u64,
    file_id: u64,
    inserted_batch_count: AtomicU16,
}

pub type LiquidCachedColumnRef = Arc<LiquidCachedColumn>;

pub enum InsertArrowArrayError {
    CacheFull,
    AlreadyCached,
}

impl LiquidCachedColumn {
    fn new(
        cache_mode: LiquidCacheMode,
        field: Arc<Field>,
        cache_store: Arc<CacheStore>,
        column_id: u64,
        row_group_id: u64,
        file_id: u64,
    ) -> Self {
        let cache_dir = cache_store
            .config()
            .cache_root_dir()
            .join(format!("file_{}", file_id))
            .join(format!("rg_{}", row_group_id))
            .join(format!("col_{}", column_id));
        std::fs::create_dir_all(&cache_dir).expect("Failed to create cache directory");
        Self {
            cache_mode,
            cache_dir,
            liquid_compressor_states: Arc::new(LiquidCompressorStates::new()),
            field,
            cache_store,
            column_id,
            row_group_id,
            file_id,
            inserted_batch_count: AtomicU16::new(0),
        }
    }

    fn budget(&self) -> &BudgetAccounting {
        self.cache_store.budget()
    }

    /// row_id must be on a batch boundary.
    fn entry_id(&self, batch_id: BatchID) -> CacheEntryID {
        // debug_assert!(row_id % self.cache_store.config().batch_size() == 0);
        // let batch_id = BatchID::from_row_id(row_id, self.cache_store.config().batch_size());
        CacheEntryID::new(self.file_id, self.row_group_id, self.column_id, batch_id)
    }

    pub(crate) fn cache_mode(&self) -> LiquidCacheMode {
        self.cache_mode
    }

    fn fsst_compressor(&self) -> &RwLock<Option<Arc<fsst::Compressor>>> {
        &self.liquid_compressor_states.fsst_compressor
    }

    pub(crate) fn batch_size(&self) -> usize {
        self.cache_store.config().batch_size()
    }

    pub(crate) fn is_cached(&self, batch_id: BatchID) -> bool {
        self.cache_store.is_cached(&self.entry_id(batch_id))
    }

    /// Reads a liquid array from disk.
    /// Panics if the file does not exist.
    fn read_liquid_from_disk(&self, batch_id: BatchID) -> LiquidArrayRef {
        // TODO: maybe use async here?
        // But async in tokio is way slower than sync.
        let path = self.cache_dir.join(format!("row_{}.bin", *batch_id));
        let mut file = File::open(path).unwrap();
        let mut bytes = Vec::new();
        file.read_to_end(&mut bytes).unwrap();
        let bytes = Bytes::from(bytes);
        let context = &self.fsst_compressor().read().unwrap();
        let compressor = context.as_ref().cloned();
        ipc::read_from_bytes(bytes, &LiquidIPCContext::new(compressor))
    }

    /// Evaluates a predicate on a liquid array.
    /// It optimistically tries to evaluate on liquid array, and if that fails,
    /// it falls back to evaluating on an arrow array.
    fn eval_selection_with_predicate_inner(
        &self,
        predicate: &mut dyn LiquidPredicate,
        array: &LiquidArrayRef,
    ) -> BooleanBuffer {
        match predicate.evaluate_liquid(array).unwrap() {
            Some(v) => {
                let (buffer, _) = v.into_parts();
                buffer
            }
            None => {
                let arrow_batch = array.to_arrow_array();
                let schema = Schema::new(vec![self.field.clone()]);
                let record_batch =
                    RecordBatch::try_new(Arc::new(schema), vec![arrow_batch]).unwrap();
                let boolean_array = predicate.evaluate(record_batch).unwrap();
                let (buffer, _) = boolean_array.into_parts();
                buffer
            }
        }
    }

    fn arrow_array_to_record_batch(&self, array: ArrayRef) -> RecordBatch {
        let schema = Arc::new(Schema::new(vec![self.field.clone()]));
        RecordBatch::try_new(schema, vec![array]).unwrap()
    }

    fn eval_selection_with_predicate(
        &self,
        batch_id: BatchID,
        selection: &BooleanBuffer,
        predicate: &mut dyn LiquidPredicate,
    ) -> Option<Result<BooleanBuffer, ArrowError>> {
        #[cfg(debug_assertions)]
        {
            let handle = get_file_handle();
            let mut file = handle.lock().unwrap();
            writeln!(
                &mut file,
                "{},{},{},{},{},predicate",
                self.file_id,
                self.row_group_id,
                self.column_id,
                *batch_id,
                self.memory_usage()
            )
            .unwrap();
        }

        let cached_entry = self.cache_store.get(&self.entry_id(batch_id))?;
        match &cached_entry {
            CachedBatch::ArrowMemory(array) => {
                let boolean_array = BooleanArray::new(selection.clone(), None);
                let selected = arrow::compute::filter(array, &boolean_array).unwrap();
                let record_batch = self.arrow_array_to_record_batch(selected);
                let boolean_array = predicate.evaluate(record_batch).unwrap();
                let predicate_filter = match boolean_array.null_count() {
                    0 => boolean_array,
                    _ => prep_null_mask_filter(&boolean_array),
                };
                let (buffer, _) = predicate_filter.into_parts();
                Some(Ok(buffer))
            }
            CachedBatch::OnDiskLiquid => {
                let array = self.read_liquid_from_disk(batch_id);
                let boolean_array = BooleanArray::new(selection.clone(), None);
                let filtered = array.filter(&boolean_array);
                let buffer = self.eval_selection_with_predicate_inner(predicate, &filtered);
                Some(Ok(buffer))
            }
            CachedBatch::LiquidMemory(array) => match ABLATION_STUDY_MODE {
                AblationStudyMode::FullDecoding => {
                    let boolean_array = BooleanArray::new(selection.clone(), None);
                    let arrow = array.to_arrow_array();
                    let filtered = arrow::compute::filter(&arrow, &boolean_array).unwrap();
                    let record_batch = self.arrow_array_to_record_batch(filtered);
                    let boolean_array = predicate.evaluate(record_batch).unwrap();
                    let (buffer, _) = boolean_array.into_parts();
                    Some(Ok(buffer))
                }
                AblationStudyMode::SelectiveDecoding
                | AblationStudyMode::SelectiveWithLateMaterialization => {
                    let boolean_array = BooleanArray::new(selection.clone(), None);
                    let filtered = array.filter(&boolean_array);
                    let arrow = filtered.to_arrow_array();
                    let record_batch = self.arrow_array_to_record_batch(arrow);
                    let boolean_array = predicate.evaluate(record_batch).unwrap();
                    let (buffer, _) = boolean_array.into_parts();
                    Some(Ok(buffer))
                }
                AblationStudyMode::EvaluateOnEncodedData
                | AblationStudyMode::EvaluateOnPartialEncodedData => {
                    let boolean_array = BooleanArray::new(selection.clone(), None);
                    let filtered = array.filter(&boolean_array);
                    let buffer = self.eval_selection_with_predicate_inner(predicate, &filtered);
                    Some(Ok(buffer))
                }
            },
        }
    }

    pub(crate) fn get_arrow_array_with_filter(
        &self,
        batch_id: BatchID,
        filter: &BooleanArray,
    ) -> Option<ArrayRef> {
        #[cfg(debug_assertions)]
        {
            let handle = get_file_handle();
            let mut file = handle.lock().unwrap();
            writeln!(
                &mut file,
                "{},{},{},{},{},filter",
                self.file_id,
                self.row_group_id,
                self.column_id,
                *batch_id,
                self.memory_usage()
            )
            .unwrap();
        }

        let inner_value = self.cache_store.get(&self.entry_id(batch_id))?;
        match &inner_value {
            CachedBatch::ArrowMemory(array) => {
                let filtered = arrow::compute::filter(array, filter).unwrap();
                Some(filtered)
            }
            CachedBatch::LiquidMemory(array) => match ABLATION_STUDY_MODE {
                AblationStudyMode::FullDecoding => {
                    let arrow = array.to_arrow_array();
                    let filtered = arrow::compute::filter(&arrow, filter).unwrap();
                    Some(filtered)
                }
                _ => {
                    let filtered = array.filter(filter);
                    Some(filtered.to_best_arrow_array())
                }
            },
            CachedBatch::OnDiskLiquid => {
                let array = self.read_liquid_from_disk(batch_id);
                let filtered = array.filter(filter);
                Some(filtered.to_best_arrow_array())
            }
        }
    }

    #[cfg(test)]
    pub(crate) fn get_arrow_array_test_only(&self, batch_id: BatchID) -> Option<ArrayRef> {
        let cached_entry = self.cache_store.get(&self.entry_id(batch_id))?;
        match cached_entry {
            CachedBatch::ArrowMemory(array) => Some(array),
            CachedBatch::LiquidMemory(array) => Some(array.to_best_arrow_array()),
            CachedBatch::OnDiskLiquid => {
                let array = self.read_liquid_from_disk(batch_id);
                Some(array.to_best_arrow_array())
            }
        }
    }

    fn insert_as_liquid_foreground(
        self: &Arc<Self>,
        batch_id: BatchID,
        array: ArrayRef,
    ) -> Result<(), InsertArrowArrayError> {
        let transcoded = match transcode_liquid_inner(&array, &self.liquid_compressor_states) {
            Ok(transcoded) => transcoded,
            Err(_) => {
                log::warn!(
                    "unsupported data type {:?}, inserting as arrow array",
                    array.data_type()
                );
                self.cache_store
                    .insert(
                        self.entry_id(batch_id),
                        CachedBatch::ArrowMemory(array.clone()),
                    )
                    .expect("failed to insert arrow memory");
                return Ok(());
            }
        };

        match self.cache_store.insert(
            self.entry_id(batch_id),
            CachedBatch::LiquidMemory(transcoded.clone()),
        ) {
            Ok(_) => {}
            Err(CacheAdvice::InsertToDisk) => {
                self.write_to_disk(transcoded, batch_id);
            }
            Err(CacheAdvice::EvictAndRetry(to_evict)) => {
                let Some(removed) = self.cache_store.remove(&to_evict) else {
                    return self.insert_as_liquid_foreground(batch_id, array);
                };
                self.evict_batch(batch_id, removed);
                return self.insert_as_liquid_foreground(batch_id, array);
            }
        }

        Ok(())
    }

    fn evict_batch(self: &Arc<Self>, batch_id: BatchID, entry: CachedBatch) {
        match entry {
            CachedBatch::ArrowMemory(array) => {
                self.transcode_and_write_to_disk(array, batch_id);
            }
            CachedBatch::LiquidMemory(array) => {
                self.write_to_disk(array, batch_id);
            }
            CachedBatch::OnDiskLiquid => {
                unreachable!("on disk liquid should not be evicted");
            }
        }
    }

    fn insert_as_liquid_background(
        self: &Arc<Self>,
        batch_id: BatchID,
        array: ArrayRef,
    ) -> Result<(), InsertArrowArrayError> {
        match self.cache_store.insert(
            self.entry_id(batch_id),
            CachedBatch::ArrowMemory(array.clone()),
        ) {
            Ok(_) => {
                let column_arc = Arc::clone(self);
                TRANSCODE_THREAD_POOL.spawn(async move {
                    column_arc.transcode_to_liquid(batch_id, array);
                });
                Ok(())
            }
            Err(CacheAdvice::InsertToDisk) => {
                let column_arc = Arc::clone(self);
                let array_to_write = array;
                TRANSCODE_THREAD_POOL.spawn(async move {
                    column_arc.transcode_and_write_to_disk(array_to_write, batch_id);
                });
                Err(InsertArrowArrayError::CacheFull)
            }
            Err(CacheAdvice::EvictAndRetry(to_evict)) => {
                let column_arc = Arc::clone(self);
                TRANSCODE_THREAD_POOL.spawn(async move {
                    let Some(removed) = column_arc.cache_store.remove(&to_evict) else {
                        return;
                    };
                    column_arc.evict_batch(batch_id, removed);
                    _ = column_arc.insert_as_liquid_background(batch_id, array);
                });
                Err(InsertArrowArrayError::CacheFull)
            }
        }
    }

    /// Insert an arrow array into the cache.
    pub(crate) fn insert_arrow_array(
        self: &Arc<Self>,
        batch_id: BatchID,
        array: ArrayRef,
    ) -> Result<(), InsertArrowArrayError> {
        if self.is_cached(batch_id) {
            return Err(InsertArrowArrayError::AlreadyCached);
        }

        self.inserted_batch_count.fetch_add(1, Ordering::Relaxed);

        // This is a special case for the Utf8View type, because the rest of the system expects a Dictionary type,
        // But the reader reads as Utf8View types.
        // So to be consistent, we cast to a Dictionary type here.
        let array = if array.data_type() == &DataType::Utf8View {
            arrow::compute::kernels::cast(
                &array,
                &DataType::Dictionary(Box::new(DataType::UInt16), Box::new(DataType::Utf8)),
            )
            .unwrap()
        } else {
            array
        };

        match &self.cache_mode {
            LiquidCacheMode::InMemoryArrow => {
                let entry_id = self.entry_id(batch_id);
                match self
                    .cache_store
                    .insert(entry_id, CachedBatch::ArrowMemory(array.clone()))
                {
                    Ok(_) => Ok(()),
                    Err(CacheAdvice::InsertToDisk) => {
                        unreachable!("in memory arrow should not need to write to disk");
                    }
                    Err(CacheAdvice::EvictAndRetry(to_evict)) => {
                        if let Some(removed) = self.cache_store.remove(&to_evict) {
                            self.evict_batch(batch_id, removed);
                        }
                        self.insert_arrow_array(batch_id, array)
                    }
                }
            }
            LiquidCacheMode::OnDiskArrow => {
                unimplemented!()
            }
            LiquidCacheMode::InMemoryLiquid {
                transcode_in_background,
            } => {
                if *transcode_in_background {
                    self.insert_as_liquid_background(batch_id, array)
                } else {
                    self.insert_as_liquid_foreground(batch_id, array)
                }
            }
        }
    }

    fn transcode_and_write_to_disk(self: &Arc<Self>, array: ArrayRef, batch_id: BatchID) {
        let transcoded = transcode_liquid_inner(&array, &self.liquid_compressor_states)
            .expect("failed to transcode");
        // Here we have no thing but to panic, because we are in a background thread, and we run out of memory, and we don't support the data type.

        self.write_to_disk(transcoded, batch_id);
    }

    fn write_to_disk(self: &Arc<Self>, array: LiquidArrayRef, batch_id: BatchID) {
        let bytes = array.to_bytes();
        let disk_size = bytes.len();
        let file_path = self.cache_dir.join(format!("row_{}.bin", *batch_id));
        let mut file = File::create(file_path).unwrap();
        file.write_all(&bytes).unwrap();
        self.budget().add_used_disk_bytes(disk_size);
        self.cache_store
            .insert(self.entry_id(batch_id), CachedBatch::OnDiskLiquid)
            .expect("failed to insert on disk liquid");
    }

    fn transcode_to_liquid(self: &Arc<Self>, batch_id: BatchID, array: ArrayRef) {
        match transcode_liquid_inner(&array, &self.liquid_compressor_states) {
            Ok(transcoded) => {
                let updated = self.cache_store.insert(
                    self.entry_id(batch_id),
                    CachedBatch::LiquidMemory(transcoded),
                );
                if updated.is_err() {
                    log::warn!("failed to insert liquid memory");
                }
            }
            Err(array) => {
                // if the array data type is not supported yet, we just leave it as is.
                log::warn!("unsupported data type {:?}", array.data_type());
            }
        }
    }

    fn memory_usage(&self) -> u64 {
        let mut total_memory = 0;
        let batch_count = self.inserted_batch_count.load(Ordering::Relaxed);

        for batch in 0..batch_count {
            let batch_id = BatchID::from_raw(batch);
            if let Some(cached_entry) = self.cache_store.get(&self.entry_id(batch_id)) {
                total_memory += cached_entry.memory_usage_bytes() as u64;
            }
        }

        total_memory
    }
}

#[derive(Debug)]
pub struct LiquidCachedRowGroup {
    cache_mode: LiquidCacheMode,
    columns: RwLock<AHashMap<u64, Arc<LiquidCachedColumn>>>,
    cache_store: Arc<CacheStore>,
    row_group_id: u64,
    file_id: u64,
}

impl LiquidCachedRowGroup {
    fn new(
        cache_mode: LiquidCacheMode,
        cache_store: Arc<CacheStore>,
        row_group_id: u64,
        file_id: u64,
    ) -> Self {
        let cache_dir = cache_store
            .config()
            .cache_root_dir()
            .join(format!("file_{}", file_id))
            .join(format!("rg_{}", row_group_id));
        std::fs::create_dir_all(&cache_dir).expect("Failed to create cache directory");
        Self {
            cache_mode,
            columns: RwLock::new(AHashMap::new()),
            cache_store,
            row_group_id,
            file_id,
        }
    }

    pub fn create_column(&self, column_id: u64, field: Arc<Field>) -> LiquidCachedColumnRef {
        use std::collections::hash_map::Entry;
        let mut columns = self.columns.write().unwrap();

        let field = match field.data_type() {
            DataType::Utf8View => {
                let field: Field = Field::clone(&field);
                Arc::new(field.with_data_type(DataType::Dictionary(
                    Box::new(DataType::UInt16),
                    Box::new(DataType::Utf8),
                )))
            }
            DataType::Utf8 | DataType::LargeUtf8 => unreachable!(),
            _ => field,
        };

        let column = columns.entry(column_id);

        match column {
            Entry::Occupied(entry) => {
                let v = entry.get().clone();
                assert_eq!(v.field, field);
                v
            }
            Entry::Vacant(entry) => {
                let column = Arc::new(LiquidCachedColumn::new(
                    self.cache_mode,
                    field,
                    self.cache_store.clone(),
                    column_id,
                    self.row_group_id,
                    self.file_id,
                ));
                entry.insert(column.clone());
                column
            }
        }
    }

    pub fn get_column(&self, column_id: u64) -> Option<LiquidCachedColumnRef> {
        self.columns.read().unwrap().get(&column_id).cloned()
    }

    pub fn evaluate_selection_with_predicate(
        &self,
        batch_id: BatchID,
        selection: &BooleanBuffer,
        predicate: &mut dyn LiquidPredicate,
    ) -> Option<Result<BooleanBuffer, ArrowError>> {
        let column_ids = predicate.predicate_column_ids();

        if column_ids.len() == 1 {
            // If we only have one column, we can short-circuit and try to evaluate the predicate on encoded data.
            let column_id = column_ids[0];
            let cache = self.get_column(column_id as u64)?;
            cache.eval_selection_with_predicate(batch_id, selection, predicate)
        } else {
            // Otherwise, we need to first convert the data into arrow arrays.
            let mask = BooleanArray::from(selection.clone());
            let mut arrays = Vec::new();
            let mut fields = Vec::new();
            for column_id in column_ids {
                let column = self.get_column(column_id as u64)?;
                let array = column.get_arrow_array_with_filter(batch_id, &mask)?;
                arrays.push(array);
                fields.push(column.field.clone());
            }
            let schema = Arc::new(Schema::new(fields));
            let record_batch = RecordBatch::try_new(schema, arrays).unwrap();
            let boolean_array = predicate.evaluate(record_batch).unwrap();
            let (buffer, _) = boolean_array.into_parts();
            Some(Ok(buffer))
        }
    }
}

pub type LiquidCachedRowGroupRef = Arc<LiquidCachedRowGroup>;

#[derive(Debug)]
pub struct LiquidCachedFile {
    row_groups: Mutex<AHashMap<u64, Arc<LiquidCachedRowGroup>>>,
    cache_mode: LiquidCacheMode,
    cache_store: Arc<CacheStore>,
    file_id: u64,
}

impl LiquidCachedFile {
    fn new(cache_mode: LiquidCacheMode, cache_store: Arc<CacheStore>, file_id: u64) -> Self {
        Self {
            cache_mode,
            row_groups: Mutex::new(AHashMap::new()),
            cache_store,
            file_id,
        }
    }

    pub fn row_group(&self, row_group_id: u64) -> LiquidCachedRowGroupRef {
        let mut row_groups = self.row_groups.lock().unwrap();
        let row_group = row_groups.entry(row_group_id).or_insert_with(|| {
            Arc::new(LiquidCachedRowGroup::new(
                self.cache_mode,
                self.cache_store.clone(),
                row_group_id,
                self.file_id,
            ))
        });
        row_group.clone()
    }

    fn reset(&self) {
        self.cache_store.reset();
    }

    pub fn cache_mode(&self) -> LiquidCacheMode {
        self.cache_mode
    }

    #[cfg(test)]
    pub fn memory_usage(&self) -> usize {
        self.cache_store.budget().memory_usage_bytes()
    }
}

/// A reference to a cached file.
pub type LiquidCachedFileRef = Arc<LiquidCachedFile>;

/// The main cache structure.
#[derive(Debug)]
pub struct LiquidCache {
    /// Files -> RowGroups -> Columns -> Batches
    files: Mutex<AHashMap<String, Arc<LiquidCachedFile>>>,

    cache_store: Arc<CacheStore>,

    current_file_id: AtomicU64,
}

/// A reference to the main cache structure.
pub type LiquidCacheRef = Arc<LiquidCache>;

/// The mode of the cache.
#[derive(Debug, Copy, Clone)]
pub enum LiquidCacheMode {
    /// The baseline that reads the arrays as is.
    InMemoryArrow,
    /// The baseline that reads the arrays as is, but stores the data on disk.
    OnDiskArrow,
    /// The baseline that reads the arrays as is, but transcode the data into liquid arrays in the background.
    InMemoryLiquid {
        /// Whether to transcode the data into liquid arrays in the background.
        transcode_in_background: bool,
    },
}

impl From<CacheMode> for LiquidCacheMode {
    fn from(value: CacheMode) -> Self {
        match value {
            CacheMode::Liquid => LiquidCacheMode::InMemoryLiquid {
                transcode_in_background: true,
            },
            CacheMode::Arrow => LiquidCacheMode::InMemoryArrow,
            CacheMode::LiquidEagerTranscode => LiquidCacheMode::InMemoryLiquid {
                transcode_in_background: false,
            },
            CacheMode::Parquet => unreachable!(),
        }
    }
}

impl LiquidCache {
    /// Create a new cache
    pub fn new(batch_size: usize, max_cache_bytes: usize, cache_dir: PathBuf) -> Self {
        assert!(batch_size.is_power_of_two());

        LiquidCache {
            files: Mutex::new(AHashMap::new()),
            cache_store: Arc::new(CacheStore::new(batch_size, max_cache_bytes, cache_dir)),
            current_file_id: AtomicU64::new(0),
        }
    }

    /// Register a file in the cache.
    pub fn register_or_get_file(
        &self,
        file_path: String,
        cache_mode: LiquidCacheMode,
    ) -> LiquidCachedFileRef {
        let mut files = self.files.lock().unwrap();
        let value = files.entry(file_path.clone()).or_insert_with(|| {
            let file_id = self.current_file_id.fetch_add(1, Ordering::Relaxed);
            Arc::new(LiquidCachedFile::new(
                cache_mode,
                self.cache_store.clone(),
                file_id,
            ))
        });
        value.clone()
    }

    /// Get a file from the cache.
    pub fn get_file(&self, file_path: String) -> Option<LiquidCachedFileRef> {
        let files = self.files.lock().unwrap();
        files.get(&file_path).cloned()
    }

    /// Get the batch size of the cache.
    pub fn batch_size(&self) -> usize {
        self.cache_store.config().batch_size()
    }

    /// Get the max cache bytes of the cache.
    pub fn max_cache_bytes(&self) -> usize {
        self.cache_store.config().max_cache_bytes()
    }

    /// Get the memory usage of the cache in bytes.
    pub fn memory_usage_bytes(&self) -> usize {
        self.cache_store.budget().memory_usage_bytes()
    }

    /// Get the disk usage of the cache in bytes.
    pub fn disk_usage_bytes(&self) -> usize {
        self.cache_store.budget().disk_usage_bytes()
    }

    /// Reset the cache.
    pub fn reset(&self) {
        let mut files = self.files.lock().unwrap();
        for file in files.values_mut() {
            file.reset();
        }
        self.cache_store.reset();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::{Array, Int32Array};

    /// Test that verifies background transcoding updates both the stored value type
    /// and the memory accounting.
    #[test]
    fn test_background_transcoding() {
        let batch_size = 64;
        let max_cache_bytes = 1024 * 1024; // a generous limit for the test
        let tmp_dir = tempfile::tempdir().unwrap();
        let cache = LiquidCache::new(batch_size, max_cache_bytes, tmp_dir.path().to_path_buf());
        let file = cache.register_or_get_file(
            "test_file2".to_string(),
            LiquidCacheMode::InMemoryLiquid {
                transcode_in_background: true,
            },
        );
        let row_group = file.row_group(0);
        let column =
            row_group.create_column(0, Arc::new(Field::new("test", DataType::Int32, false)));

        let arrow_array = Arc::new(Int32Array::from(vec![10; 100_000])) as ArrayRef;
        let arrow_size = arrow_array.get_array_memory_size();

        let batch_id = BatchID::from_raw(0);
        assert!(
            column
                .insert_arrow_array(batch_id, arrow_array.clone())
                .is_ok()
        );

        let size_before = column.budget().memory_usage_bytes();
        loop {
            // Now check that the entry has been transcoded.
            let entry = column.cache_store.get(&column.entry_id(batch_id)).unwrap();
            match entry {
                CachedBatch::LiquidMemory(ref liquid) => {
                    // The memory usage after transcoding should be less or equal.
                    let new_size = liquid.get_array_memory_size();
                    assert!(
                        new_size <= arrow_size,
                        "Memory usage did not decrease after transcoding"
                    );
                    // Also, verify that the type conversion has happened.
                    // (For example, calling to_arrow_array() should work.)
                    let _ = liquid.to_arrow_array();
                    break;
                }
                CachedBatch::ArrowMemory(_) | CachedBatch::OnDiskLiquid => {
                    continue;
                }
            }
        }

        let size_after = column.budget().memory_usage_bytes();
        assert!(
            size_after <= size_before,
            "Cache memory increased after transcoding, size_before={} size_after={}",
            size_before,
            size_after
        );

        println!(
            "Test background transcoding: size_before={} size_after={}",
            size_before, size_after
        );
    }

    #[test]
    fn test_concurrent_arrow_insert_race_condition_over_allocation() {
        use arrow::array::{ArrayRef, Int32Array};
        use std::sync::{Arc, Barrier};
        use std::thread;

        let batch_size = 64;
        let dummy_array: ArrayRef =
            Arc::new(Int32Array::from((0..100).map(|_| 0).collect::<Vec<i32>>()));
        let dummy_size = dummy_array.get_array_memory_size();

        let max_cache_bytes = dummy_size;

        let tmp_dir = tempfile::tempdir().unwrap();
        let cache = LiquidCache::new(batch_size, max_cache_bytes, tmp_dir.path().to_path_buf());
        let file = cache.register_or_get_file(
            "race_condition_test_file".to_string(),
            LiquidCacheMode::InMemoryLiquid {
                transcode_in_background: false,
            },
        );
        let row_group = file.row_group(0);
        let column =
            row_group.create_column(0, Arc::new(Field::new("test", DataType::Int32, false)));

        // Spawn many threads so that they all attempt to insert concurrently.
        let num_threads = 4;
        let barrier = Arc::new(Barrier::new(num_threads));

        let handles: Vec<_> = (0..num_threads)
            .map(|i| {
                let barrier = barrier.clone();
                let column = Arc::clone(&column);
                thread::spawn(move || {
                    // Wait until all threads are ready.
                    let array: ArrayRef = Arc::new(Int32Array::from(
                        (0..100).map(|_| i as i32).collect::<Vec<_>>(),
                    ));
                    barrier.wait();
                    let batch_id = BatchID::from_raw(i as u16);
                    let _ = column.insert_arrow_array(batch_id, array);
                })
            })
            .collect();

        for handle in handles {
            handle.join().unwrap();
        }

        let final_mem = column.budget().memory_usage_bytes();

        assert!(
            final_mem <= max_cache_bytes,
            "Final memory usage {} exceeds configured max {}",
            final_mem,
            max_cache_bytes
        );
    }

    #[test]
    fn test_on_disk_cache() {
        // Create a cache with a small memory limit to force on-disk storage
        let batch_size = 64;
        let max_cache_bytes = 1024; // Small enough to force disk storage
        let tmp_dir = tempfile::tempdir().unwrap();
        let cache_path = tmp_dir.path().to_path_buf();

        let cache = LiquidCache::new(batch_size, max_cache_bytes, cache_path.clone());

        // Register a file with non-background transcode mode
        let file_name = "test_on_disk_file".to_string();
        let file = cache.register_or_get_file(
            file_name.clone(),
            LiquidCacheMode::InMemoryLiquid {
                transcode_in_background: false,
            },
        );

        // Create row group and column
        let row_group_id = 5;
        let column_id = 10;
        let row_group = file.row_group(row_group_id);
        let column = row_group.create_column(
            column_id,
            Arc::new(Field::new("test", DataType::Int32, false)),
        );

        // Create a large array that won't fit in memory
        let large_array = Arc::new(Int32Array::from(vec![42; 10000])) as ArrayRef;
        let array_size = large_array.get_array_memory_size();

        // Verify array is larger than our memory limit
        assert!(
            array_size > max_cache_bytes,
            "Test array should be larger than memory limit"
        );

        // Insert the array
        let row_id = 20;
        let batch_id = BatchID::from_raw(row_id as u16);
        assert!(
            column
                .insert_arrow_array(batch_id, large_array.clone())
                .is_ok()
        );

        let expected_file_path = cache_path
            .join("file_0")
            .join(format!("rg_{}", row_group_id))
            .join(format!("col_{}", column_id))
            .join(format!("row_{}.bin", row_id));

        assert!(
            expected_file_path.exists(),
            "On-disk cache file not found at expected path: {:?}",
            expected_file_path
        );

        // 2. Verify memory accounting
        let memory_usage = column.budget().memory_usage_bytes();
        let disk_usage = column.budget().disk_usage_bytes();

        // Memory usage should be minimal since data is on disk
        assert!(
            memory_usage < array_size,
            "Memory usage ({}) should be less than original array size ({})",
            memory_usage,
            array_size
        );

        // Disk usage should be non-zero
        assert!(
            disk_usage > 0,
            "Disk usage should be greater than zero, got {}",
            disk_usage
        );

        // 3. Check that the data is stored as OnDiskLiquid
        {
            let cached_entry = column.cache_store.get(&column.entry_id(batch_id)).unwrap();
            match cached_entry {
                CachedBatch::OnDiskLiquid => {
                    // This is what we expect
                }
                other => panic!("Expected OnDiskLiquid, got: {}", other),
            }
        }

        // 4. Read the data back and verify its content
        let retrieved_array = column
            .get_arrow_array_test_only(batch_id)
            .expect("Failed to read array");

        assert_eq!(&retrieved_array, &large_array);
    }
}
