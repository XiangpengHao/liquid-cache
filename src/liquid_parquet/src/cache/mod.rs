use super::liquid_array::LiquidArrayRef;
use crate::liquid_array::ipc::{self, LiquidIPCContext};
use crate::sync::{
    Mutex, RwLock,
    atomic::{AtomicU64, Ordering},
};
use crate::{ABLATION_STUDY_MODE, AblationStudyMode, LiquidPredicate};
use ahash::AHashMap;
use arrow::array::{Array, ArrayRef, BooleanArray, RecordBatch};
use arrow::buffer::BooleanBuffer;
use arrow::compute::prep_null_mask_filter;
use arrow_schema::{ArrowError, DataType, Field, Schema};
use bytes::Bytes;
use liquid_cache_common::{LiquidCacheMode, coerce_from_parquet_to_liquid_type};
use std::fmt::Display;
use std::fs::File;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::{Arc, LazyLock};
use store::{CacheAdvice, CacheStore};
use tokio::runtime::Runtime;
use transcode::transcode_liquid_inner;
pub(crate) use utils::BatchID;
use utils::{CacheEntryID, ColumnAccessPath};

mod budget;
/// Module containing cache eviction policies like FIFO
pub mod policies;
mod stats;
mod store;
mod tracer;
mod transcode;
mod utils;

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
pub enum CachedBatch {
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

    fn reference_count(&self) -> usize {
        match self {
            Self::ArrowMemory(array) => Arc::strong_count(array),
            Self::LiquidMemory(array) => Arc::strong_count(array),
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
    cache_store: Arc<CacheStore>,
    field: Arc<Field>,
    column_path: ColumnAccessPath,
}

pub type LiquidCachedColumnRef = Arc<LiquidCachedColumn>;

pub enum InsertArrowArrayError {
    AlreadyCached,
}

impl LiquidCachedColumn {
    fn new(
        field: Arc<Field>,
        cache_store: Arc<CacheStore>,
        column_id: u64,
        row_group_id: u64,
        file_id: u64,
    ) -> Self {
        let column_path = ColumnAccessPath::new(file_id, row_group_id, column_id);
        column_path.initialize_dir(cache_store.config().cache_root_dir());
        Self {
            field,
            cache_store,
            column_path,
        }
    }

    /// row_id must be on a batch boundary.
    fn entry_id(&self, batch_id: BatchID) -> CacheEntryID {
        self.column_path.entry_id(batch_id)
    }

    pub(crate) fn cache_mode(&self) -> &LiquidCacheMode {
        self.cache_store.config().cache_mode()
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
        let entry_id = self.entry_id(batch_id);
        let path = entry_id.on_disk_path(self.cache_store.config().cache_root_dir());
        let compressor = self.cache_store.compressor_states(&entry_id);
        let compressor = compressor.fsst_compressor.read().unwrap().clone();
        let mut file = File::open(path).unwrap();
        let mut bytes = Vec::new();
        file.read_to_end(&mut bytes).unwrap();
        let bytes = Bytes::from(bytes);
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
        let compressor = self.cache_store.compressor_states(&self.entry_id(batch_id));
        let transcoded = match transcode_liquid_inner(&array, &compressor) {
            Ok(transcoded) => transcoded,
            Err(_) => {
                log::warn!(
                    "unsupported data type {:?}, inserting as arrow array",
                    array.data_type()
                );
                self.cache_store.insert(
                    self.entry_id(batch_id),
                    CachedBatch::ArrowMemory(array.clone()),
                );
                return Ok(());
            }
        };

        self.cache_store.insert(
            self.entry_id(batch_id),
            CachedBatch::LiquidMemory(transcoded.clone()),
        );

        Ok(())
    }

    fn insert_as_liquid_background(
        self: &Arc<Self>,
        batch_id: BatchID,
        array: ArrayRef,
    ) -> Result<(), InsertArrowArrayError> {
        self.cache_store.insert(
            self.entry_id(batch_id),
            CachedBatch::ArrowMemory(array.clone()),
        );
        let column_arc = Arc::clone(self);
        TRANSCODE_THREAD_POOL.spawn(async move {
            column_arc.transcode_to_liquid(batch_id, array);
        });
        Ok(())
    }

    /// Insert an array into the cache.
    pub(crate) fn insert(
        self: &Arc<Self>,
        batch_id: BatchID,
        array: ArrayRef,
    ) -> Result<(), InsertArrowArrayError> {
        if self.is_cached(batch_id) {
            return Err(InsertArrowArrayError::AlreadyCached);
        }

        match self.cache_mode() {
            LiquidCacheMode::InMemoryArrow => {
                let entry_id = self.entry_id(batch_id);
                self.cache_store
                    .insert(entry_id, CachedBatch::ArrowMemory(array.clone()));
                Ok(())
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

    fn transcode_to_liquid(self: &Arc<Self>, batch_id: BatchID, array: ArrayRef) {
        let compressor = self.cache_store.compressor_states(&self.entry_id(batch_id));
        match transcode_liquid_inner(&array, &compressor) {
            Ok(transcoded) => {
                self.cache_store.insert(
                    self.entry_id(batch_id),
                    CachedBatch::LiquidMemory(transcoded),
                );
            }
            Err(array) => {
                // if the array data type is not supported yet, we just leave it as is.
                log::warn!("unsupported data type {:?}", array.data_type());
            }
        }
    }
}

#[derive(Debug)]
pub struct LiquidCachedRowGroup {
    columns: RwLock<AHashMap<u64, Arc<LiquidCachedColumn>>>,
    cache_store: Arc<CacheStore>,
    row_group_id: u64,
    file_id: u64,
}

impl LiquidCachedRowGroup {
    fn new(cache_store: Arc<CacheStore>, row_group_id: u64, file_id: u64) -> Self {
        let cache_dir = cache_store
            .config()
            .cache_root_dir()
            .join(format!("file_{file_id}"))
            .join(format!("rg_{row_group_id}"));
        std::fs::create_dir_all(&cache_dir).expect("Failed to create cache directory");
        Self {
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
                let new_data_type = coerce_from_parquet_to_liquid_type(
                    field.data_type(),
                    self.cache_store.config().cache_mode(),
                );
                Arc::new(field.with_data_type(new_data_type))
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
    cache_store: Arc<CacheStore>,
    file_id: u64,
}

impl LiquidCachedFile {
    fn new(cache_store: Arc<CacheStore>, file_id: u64) -> Self {
        Self {
            row_groups: Mutex::new(AHashMap::new()),
            cache_store,
            file_id,
        }
    }

    pub fn row_group(&self, row_group_id: u64) -> LiquidCachedRowGroupRef {
        let mut row_groups = self.row_groups.lock().unwrap();
        let row_group = row_groups.entry(row_group_id).or_insert_with(|| {
            Arc::new(LiquidCachedRowGroup::new(
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

    pub fn cache_mode(&self) -> &LiquidCacheMode {
        self.cache_store.config().cache_mode()
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

impl LiquidCache {
    /// Create a new cache
    pub fn new(
        batch_size: usize,
        max_cache_bytes: usize,
        cache_dir: PathBuf,
        cache_mode: LiquidCacheMode,
    ) -> Self {
        assert!(batch_size.is_power_of_two());
        let cache_policy = Box::new(policies::DiscardPolicy);

        LiquidCache {
            files: Mutex::new(AHashMap::new()),
            cache_store: Arc::new(CacheStore::new(
                batch_size,
                max_cache_bytes,
                cache_dir,
                cache_mode,
                cache_policy,
            )),
            current_file_id: AtomicU64::new(0),
        }
    }

    /// Register a file in the cache.
    pub fn register_or_get_file(&self, file_path: String) -> LiquidCachedFileRef {
        let mut files = self.files.lock().unwrap();
        let value = files.entry(file_path.clone()).or_insert_with(|| {
            let file_id = self.current_file_id.fetch_add(1, Ordering::Relaxed);
            Arc::new(LiquidCachedFile::new(self.cache_store.clone(), file_id))
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

    /// Flush the cache trace to a file.
    pub fn flush_trace(&self, to_file: impl AsRef<Path>) {
        self.cache_store.tracer().flush(to_file);
    }

    /// Enable the cache trace.
    pub fn enable_trace(&self) {
        self.cache_store.tracer().enable();
    }

    /// Disable the cache trace.
    pub fn disable_trace(&self) {
        self.cache_store.tracer().disable();
    }

    /// Reset the cache.
    pub fn reset(&self) {
        let mut files = self.files.lock().unwrap();
        for file in files.values_mut() {
            file.reset();
        }
        self.cache_store.reset();
    }

    /// Get the cache mode of the cache.
    pub fn cache_mode(&self) -> &LiquidCacheMode {
        self.cache_store.config().cache_mode()
    }
}
