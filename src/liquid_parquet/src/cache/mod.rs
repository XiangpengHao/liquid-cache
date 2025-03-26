use super::liquid_array::{LiquidArrayRef, LiquidByteArray, LiquidPrimitiveArray};
use crate::liquid_array::ipc::{self, LiquidIPCContext};
use crate::liquid_array::LiquidFloatArray;
use crate::{ABLATION_STUDY_MODE, AblationStudyMode, LiquidPredicate};
use ahash::AHashMap;
use arrow::array::types::{
    Int8Type as ArrowInt8Type, Int16Type as ArrowInt16Type, Int32Type as ArrowInt32Type,
    Int64Type as ArrowInt64Type, UInt8Type as ArrowUInt8Type, UInt16Type as ArrowUInt16Type,
    UInt32Type as ArrowUInt32Type, UInt64Type as ArrowUInt64Type,
    Float32Type as ArrowFloat32Type, Float64Type as ArrowFloat64Type
};
use arrow::array::{Array, ArrayRef, AsArray, BooleanArray, RecordBatch};
use arrow::buffer::BooleanBuffer;
use arrow::compute::prep_null_mask_filter;
use arrow_schema::{ArrowError, DataType, Field, Schema};
use bytes::Bytes;
use std::fmt::Display;
use std::fs::File;
use std::io::{Read, Write};
use std::path::PathBuf;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Arc, LazyLock, Mutex, RwLock};
use tokio::runtime::Runtime;

mod config;
mod stats;
pub(crate) use config::CacheConfig;

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

#[derive(Debug)]
struct CachedEntry {
    value: CachedBatch,
    hit_count: AtomicU32,
}

impl CachedEntry {
    fn increment_hit_count(&self) {
        self.hit_count.fetch_add(1, Ordering::Relaxed);
    }

    fn value(&self) -> &CachedBatch {
        &self.value
    }

    fn new_in_memory(array: ArrayRef) -> Self {
        let val = CachedBatch::ArrowMemory(array);
        CachedEntry {
            value: val,
            hit_count: AtomicU32::new(0),
        }
    }
}

#[derive(Debug)]
enum CachedBatch {
    ArrowMemory(ArrayRef),
    LiquidMemory(LiquidArrayRef),
    OnDiskLiquid,
}

impl CachedBatch {
    fn memory_usage(&self) -> usize {
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
    config: Arc<CacheConfig>,
    liquid_compressor_states: LiquidCompressorStates,
    rows: RwLock<AHashMap<usize, CachedEntry>>,
    cache_dir: PathBuf,
}

pub type LiquidCachedColumnRef = Arc<LiquidCachedColumn>;

fn arrays_to_record_batch(arrays: &[ArrayRef]) -> RecordBatch {
    let fields = arrays
        .iter()
        .map(|array| Field::new("_", array.data_type().clone(), array.is_nullable()))
        .collect::<Vec<_>>();
    let schema = Arc::new(Schema::new(fields));
    RecordBatch::try_new(schema, arrays.to_vec()).unwrap()
}

pub enum InsertArrowArrayError {
    CacheFull(ArrayRef),
    AlreadyCached,
}

impl LiquidCachedColumn {
    fn new(cache_mode: LiquidCacheMode, config: Arc<CacheConfig>, cache_dir: PathBuf) -> Self {
        Self {
            cache_mode,
            config,
            cache_dir,
            rows: RwLock::new(AHashMap::new()),
            liquid_compressor_states: LiquidCompressorStates::new(),
        }
    }

    pub(crate) fn cache_mode(&self) -> LiquidCacheMode {
        self.cache_mode
    }

    fn fsst_compressor(&self) -> &RwLock<Option<Arc<fsst::Compressor>>> {
        &self.liquid_compressor_states.fsst_compressor
    }

    pub(crate) fn batch_size(&self) -> usize {
        self.config.batch_size()
    }

    pub(crate) fn is_cached(&self, row_id: usize) -> bool {
        let rows = self.rows.read().unwrap();
        rows.contains_key(&row_id)
    }

    /// Reads a liquid array from disk.
    /// Panics if the file does not exist.
    fn read_liquid_from_disk(&self, row_id: usize) -> LiquidArrayRef {
        // TODO: maybe use async here?
        // But async in tokio is way slower than sync.
        let path = self.cache_dir.join(format!("row_{}.bin", row_id));
        let mut file = File::open(path).unwrap();
        let mut bytes = Vec::new();
        file.read_to_end(&mut bytes).unwrap();
        let bytes = Bytes::from(bytes);
        let context = &self.fsst_compressor().read().unwrap();
        let compressor = context.as_ref().cloned();
        ipc::read_from_bytes(bytes, &LiquidIPCContext::new(compressor))
    }

    pub(crate) fn eval_selection_with_predicate(
        &self,
        row_id: usize,
        selection: &BooleanBuffer,
        predicate: &mut dyn LiquidPredicate,
    ) -> Option<Result<BooleanBuffer, ArrowError>> {
        let cached_entry = self.rows.read().unwrap();
        let entry = cached_entry.get(&row_id)?;
        let inner_value = entry.value();
        match inner_value {
            CachedBatch::ArrowMemory(array) => {
                let boolean_array = BooleanArray::new(selection.clone(), None);
                let selected = arrow::compute::filter(array, &boolean_array).unwrap();
                let record_batch = arrays_to_record_batch(&[selected]);
                let boolean_array = predicate.evaluate(record_batch).unwrap();
                let predicate_filter = match boolean_array.null_count() {
                    0 => boolean_array,
                    _ => prep_null_mask_filter(&boolean_array),
                };
                let (buffer, _) = predicate_filter.into_parts();
                Some(Ok(buffer))
            }
            CachedBatch::OnDiskLiquid => {
                let array = self.read_liquid_from_disk(row_id);
                let boolean_array = BooleanArray::new(selection.clone(), None);
                let filtered = array.filter(&boolean_array);
                let boolean_array = predicate.evaluate_liquid(&filtered).unwrap();
                let (buffer, _) = boolean_array.into_parts();
                Some(Ok(buffer))
            }
            CachedBatch::LiquidMemory(array) => match ABLATION_STUDY_MODE {
                AblationStudyMode::FullDecoding => {
                    let boolean_array = BooleanArray::new(selection.clone(), None);
                    let arrow = array.to_arrow_array();
                    let filtered = arrow::compute::filter(&arrow, &boolean_array).unwrap();
                    let record_batch = arrays_to_record_batch(&[filtered]);
                    let boolean_array = predicate.evaluate(record_batch).unwrap();
                    let (buffer, _) = boolean_array.into_parts();
                    Some(Ok(buffer))
                }
                AblationStudyMode::SelectiveDecoding
                | AblationStudyMode::SelectiveWithLateMaterialization => {
                    let boolean_array = BooleanArray::new(selection.clone(), None);
                    let filtered = array.filter(&boolean_array);
                    let arrow = filtered.to_arrow_array();
                    let record_batch = arrays_to_record_batch(&[arrow]);
                    let boolean_array = predicate.evaluate(record_batch).unwrap();
                    let (buffer, _) = boolean_array.into_parts();
                    Some(Ok(buffer))
                }
                AblationStudyMode::EvaluateOnEncodedData
                | AblationStudyMode::EvaluateOnPartialEncodedData => {
                    let boolean_array = BooleanArray::new(selection.clone(), None);
                    let filtered = array.filter(&boolean_array);
                    let boolean_array = predicate.evaluate_liquid(&filtered).unwrap();
                    let (buffer, _) = boolean_array.into_parts();
                    Some(Ok(buffer))
                }
            },
        }
    }

    pub(crate) fn get_arrow_array_with_filter(
        &self,
        row_id: usize,
        filter: &BooleanArray,
    ) -> Option<ArrayRef> {
        let rows = self.rows.read().unwrap();

        let cached_entry = rows.get(&row_id)?;
        cached_entry.increment_hit_count();
        let inner_value = cached_entry.value();
        match inner_value {
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
                let array = self.read_liquid_from_disk(row_id);
                let filtered = array.filter(filter);
                Some(filtered.to_best_arrow_array())
            }
        }
    }

    #[cfg(test)]
    pub(crate) fn get_arrow_array_test_only(&self, row_id: usize) -> Option<ArrayRef> {
        let rows = self.rows.read().unwrap();

        let cached_entry = rows.get(&row_id)?;
        cached_entry.increment_hit_count();
        let cached_entry = cached_entry.value();
        match cached_entry {
            CachedBatch::ArrowMemory(array) => Some(array.clone()),
            CachedBatch::LiquidMemory(array) => Some(array.to_best_arrow_array()),
            CachedBatch::OnDiskLiquid => {
                let array = self.read_liquid_from_disk(row_id);
                Some(array.to_best_arrow_array())
            }
        }
    }

    fn insert_as_arrow_array(
        &self,
        row_id: usize,
        array: ArrayRef,
    ) -> Result<(), InsertArrowArrayError> {
        let mut rows = self.rows.write().unwrap();
        let array_size = array.get_array_memory_size();
        match self.config.try_reserve_memory(array_size) {
            Ok(_) => {
                let old = rows.insert(row_id, CachedEntry::new_in_memory(array));
                assert!(old.is_none());
                Ok(())
            }
            Err(_) => Err(InsertArrowArrayError::CacheFull(array)),
        }
    }

    fn insert_as_liquid_foreground(
        self: &Arc<Self>,
        row_id: usize,
        array: ArrayRef,
    ) -> Result<(), InsertArrowArrayError> {
        let transcoded = match self.transcode_liquid_inner(&array) {
            Ok(transcoded) => transcoded,
            Err(_) => {
                log::warn!(
                    "unsupported data type {:?}, inserting as arrow array",
                    array.data_type()
                );
                return self.insert_as_arrow_array(row_id, array);
            }
        };

        let memory_required = transcoded.get_array_memory_size();

        if self.config.try_reserve_memory(memory_required).is_err() {
            self.write_to_disk(transcoded, row_id);
            return Ok(());
        }

        let mut rows = self.rows.write().unwrap();
        rows.insert(
            row_id,
            CachedEntry {
                value: CachedBatch::LiquidMemory(transcoded),
                hit_count: AtomicU32::new(0),
            },
        );
        Ok(())
    }

    fn insert_as_liquid_background(
        self: &Arc<Self>,
        row_id: usize,
        array: ArrayRef,
    ) -> Result<(), InsertArrowArrayError> {
        let arrow_size = array.get_array_memory_size();
        if self.config.try_reserve_memory(arrow_size).is_err() {
            let column_arc = Arc::clone(self);
            let array_to_write = array.clone();
            TRANSCODE_THREAD_POOL.spawn(async move {
                column_arc.background_transcode_to_disk(array_to_write, row_id);
            });
            return Err(InsertArrowArrayError::CacheFull(array));
        }

        let mut rows = self.rows.write().unwrap();
        rows.insert(row_id, CachedEntry::new_in_memory(array.clone()));

        // Submit a background transcoding task to our dedicated thread pool.
        let column_arc = Arc::clone(self);
        TRANSCODE_THREAD_POOL.spawn(async move {
            column_arc.background_transcode_to_liquid(row_id);
        });
        Ok(())
    }

    /// Insert an arrow array into the cache.
    pub(crate) fn insert_arrow_array(
        self: &Arc<Self>,
        row_id: usize,
        array: ArrayRef,
    ) -> Result<(), InsertArrowArrayError> {
        if self.is_cached(row_id) {
            return Err(InsertArrowArrayError::AlreadyCached);
        }

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
            LiquidCacheMode::InMemoryArrow => self.insert_as_arrow_array(row_id, array),
            LiquidCacheMode::OnDiskArrow => {
                unimplemented!()
            }
            LiquidCacheMode::InMemoryLiquid {
                transcode_in_background,
            } => {
                if *transcode_in_background {
                    self.insert_as_liquid_background(row_id, array)
                } else {
                    self.insert_as_liquid_foreground(row_id, array)
                }
            }
        }
    }

    fn background_transcode_to_disk(self: &Arc<Self>, array: ArrayRef, row_id: usize) {
        let transcoded = self
            .transcode_liquid_inner(&array)
            .expect("failed to transcode");
        // Here we have no thing but to panic, because we are in a background thread, and we run out of memory, and we don't support the data type.

        self.write_to_disk(transcoded, row_id);
    }

    fn write_to_disk(self: &Arc<Self>, array: LiquidArrayRef, row_id: usize) {
        let bytes = array.to_bytes();
        let disk_size = bytes.len();
        let file_path = self.cache_dir.join(format!("row_{}.bin", row_id));
        let mut file = File::create(file_path).unwrap();
        file.write_all(&bytes).unwrap();
        self.config.add_used_disk_bytes(disk_size);
        let mut rows = self.rows.write().unwrap();
        rows.insert(
            row_id,
            CachedEntry {
                value: CachedBatch::OnDiskLiquid,
                hit_count: AtomicU32::new(0),
            },
        );
        self.config.add_used_disk_bytes(disk_size);
    }

    /// This method is run in the background. It acquires a write lock on the cache,
    /// checks that the stored value is still an ArrowMemory batch, and then runs the
    /// expensive transcoding, replacing the entry with a LiquidMemory batch.
    fn background_transcode_to_liquid(self: &Arc<Self>, row_id: usize) {
        let mut rows = self.rows.write().unwrap();
        if let Some(entry) = rows.get_mut(&row_id) {
            match &entry.value {
                CachedBatch::ArrowMemory(array) => {
                    let previous_size = entry.value.memory_usage();
                    match self.transcode_liquid_inner(array) {
                        Ok(transcoded) => {
                            let new_size = transcoded.get_array_memory_size();
                            if self
                                .config
                                .try_update_memory_usage_after_transcoding(previous_size, new_size)
                                .is_err()
                            {
                                return;
                            }
                            entry.value = CachedBatch::LiquidMemory(transcoded);
                        }
                        Err(array) => {
                            // if the array data type is not supported yet, we just leave it as is.
                            log::warn!("unsupported data type {:?}", array.data_type());
                        }
                    }
                }
                CachedBatch::LiquidMemory(_) => {
                    // Already transcoded.
                }
                CachedBatch::OnDiskLiquid => {
                    // Not arrow memory, so we don't need to transcode.
                }
            }
        }
    }

    /// This method is used to transcode an arrow array into a liquid array.
    ///
    /// Returns the transcoded liquid array if successful, otherwise returns the original arrow array.
    fn transcode_liquid_inner<'a>(
        &self,
        array: &'a ArrayRef,
    ) -> Result<LiquidArrayRef, &'a ArrayRef> {
        let data_type = array.data_type();
        if data_type.is_primitive() {
            // For primitive types, perform the transcoding.
            let liquid_array: LiquidArrayRef = match data_type {
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
                DataType::Float32 => {
                    Arc::new(LiquidFloatArray::<ArrowFloat32Type>::from_arrow_array(
                        array.as_primitive::<ArrowFloat32Type>().clone(),
                    ))
                }
                DataType::Float64 => {
                    Arc::new(LiquidFloatArray::<ArrowFloat64Type>::from_arrow_array(
                        array.as_primitive::<ArrowFloat64Type>().clone(),
                    ))
                }
                _ => {
                    // For unsupported primitive types, leave the value unchanged.
                    log::warn!("unsupported primitive type {:?}", data_type);
                    return Err(array);
                }
            };
            return Ok(liquid_array);
        }

        // Handle string/dictionary types.
        match array.data_type() {
            DataType::Utf8View => {
                let compressor = self.fsst_compressor().read().unwrap();
                if let Some(compressor) = compressor.as_ref() {
                    let compressed = LiquidByteArray::from_string_view_array(
                        array.as_string_view(),
                        compressor.clone(),
                    );
                    return Ok(Arc::new(compressed));
                }
                drop(compressor);
                let mut compressors = self.fsst_compressor().write().unwrap();
                let (compressor, compressed) =
                    LiquidByteArray::train_from_arrow_view(array.as_string_view());
                *compressors = Some(compressor);
                Ok(Arc::new(compressed))
            }
            DataType::Utf8 => {
                let compressor = self.fsst_compressor().read().unwrap();
                if let Some(compressor) = compressor.as_ref() {
                    let compressed = LiquidByteArray::from_string_array(
                        array.as_string::<i32>(),
                        compressor.clone(),
                    );
                    return Ok(Arc::new(compressed));
                }
                drop(compressor);
                let mut compressors = self.fsst_compressor().write().unwrap();
                let (compressor, compressed) =
                    LiquidByteArray::train_from_arrow(array.as_string::<i32>());
                *compressors = Some(compressor);
                Ok(Arc::new(compressed))
            }
            DataType::Dictionary(_, _) => {
                if let Some(dict_array) = array.as_dictionary_opt::<ArrowUInt16Type>() {
                    let compressor = self.fsst_compressor().read().unwrap();
                    if let Some(compressor) = compressor.as_ref() {
                        let liquid_array = unsafe {
                            LiquidByteArray::from_unique_dict_array(dict_array, compressor.clone())
                        };
                        return Ok(Arc::new(liquid_array));
                    }
                    drop(compressor);
                    let mut compressors = self.fsst_compressor().write().unwrap();
                    let (compressor, liquid_array) =
                        LiquidByteArray::train_from_arrow_dict(dict_array);
                    *compressors = Some(compressor);
                    return Ok(Arc::new(liquid_array));
                }
                log::warn!("unsupported data type {:?}", array.data_type());
                Err(array)
            }
            _ => {
                log::warn!("unsupported data type {:?}", array.data_type());
                Err(array)
            }
        }
    }
}

#[derive(Debug)]
pub struct LiquidCachedRowGroup {
    cache_mode: LiquidCacheMode,
    config: Arc<CacheConfig>,
    columns: RwLock<AHashMap<usize, Arc<LiquidCachedColumn>>>,
    cache_dir: PathBuf,
}

impl LiquidCachedRowGroup {
    fn new(cache_mode: LiquidCacheMode, config: Arc<CacheConfig>, cache_dir: PathBuf) -> Self {
        std::fs::create_dir_all(&cache_dir).expect("Failed to create cache directory");
        Self {
            cache_mode,
            config,
            columns: RwLock::new(AHashMap::new()),
            cache_dir,
        }
    }

    pub fn get_column_or_create(&self, column_id: usize) -> LiquidCachedColumnRef {
        self.columns
            .write()
            .unwrap()
            .entry(column_id)
            .or_insert_with(|| {
                let cache_dir = self.cache_dir.join(format!("col_{}", column_id));
                std::fs::create_dir_all(&cache_dir).expect("Failed to create cache directory");
                Arc::new(LiquidCachedColumn::new(
                    self.cache_mode,
                    self.config.clone(),
                    cache_dir,
                ))
            })
            .clone()
    }

    pub fn get_column(&self, column_id: usize) -> Option<LiquidCachedColumnRef> {
        self.columns.read().unwrap().get(&column_id).cloned()
    }

    pub fn evaluate_selection_with_predicate(
        &self,
        row_id: usize,
        selection: &BooleanBuffer,
        predicate: &mut dyn LiquidPredicate,
    ) -> Option<Result<BooleanBuffer, ArrowError>> {
        let column_ids = predicate.predicate_column_ids();

        if column_ids.len() == 1 {
            // If we only have one column, we can short-circuit and try to evaluate the predicate on encoded data.
            let column_id = column_ids[0];
            let cache = self.get_column(column_id)?;
            cache.eval_selection_with_predicate(row_id, selection, predicate)
        } else {
            // Otherwise, we need to first convert the data into arrow arrays.
            let mask = BooleanArray::from(selection.clone());
            let mut arrays = Vec::new();
            for column_id in column_ids {
                let column = self.get_column(column_id)?;
                let array = column.get_arrow_array_with_filter(row_id, &mask)?;
                arrays.push(array);
            }
            let record_batch = arrays_to_record_batch(&arrays);
            let boolean_array = predicate.evaluate(record_batch).unwrap();
            let (buffer, _) = boolean_array.into_parts();
            Some(Ok(buffer))
        }
    }
}

pub type LiquidCachedRowGroupRef = Arc<LiquidCachedRowGroup>;

#[derive(Debug)]
pub struct LiquidCachedFile {
    row_groups: Mutex<AHashMap<usize, Arc<LiquidCachedRowGroup>>>,
    cache_mode: LiquidCacheMode,
    config: Arc<CacheConfig>,
    cache_dir: PathBuf,
}

impl LiquidCachedFile {
    pub(crate) fn new(
        cache_mode: LiquidCacheMode,
        config: Arc<CacheConfig>,
        cache_dir: PathBuf,
    ) -> Self {
        std::fs::create_dir_all(&cache_dir).expect("Failed to create cache directory");
        Self {
            cache_mode,
            config,
            cache_dir,
            row_groups: Mutex::new(AHashMap::new()),
        }
    }

    pub fn row_group(&self, row_group_id: usize) -> LiquidCachedRowGroupRef {
        let mut row_groups = self.row_groups.lock().unwrap();
        let row_group = row_groups.entry(row_group_id).or_insert_with(|| {
            let cache_dir = self.cache_dir.join(format!("rg_{}", row_group_id));
            Arc::new(LiquidCachedRowGroup::new(
                self.cache_mode,
                self.config.clone(),
                cache_dir,
            ))
        });
        row_group.clone()
    }

    fn reset(&self) {
        let mut row_groups = self.row_groups.lock().unwrap();
        row_groups.clear();
    }

    pub fn cache_mode(&self) -> LiquidCacheMode {
        self.cache_mode
    }

    #[cfg(test)]
    pub fn memory_usage(&self) -> usize {
        self.config.memory_usage_bytes()
    }
}

/// A reference to a cached file.
pub type LiquidCachedFileRef = Arc<LiquidCachedFile>;

/// The main cache structure.
#[derive(Debug)]
pub struct LiquidCache {
    /// Files -> RowGroups -> Columns -> Batches
    files: Mutex<AHashMap<String, Arc<LiquidCachedFile>>>,

    config: Arc<CacheConfig>,

    cache_dir: PathBuf,
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

impl LiquidCache {
    /// Create a new cache
    pub fn new(batch_size: usize, max_cache_bytes: usize, cache_dir: PathBuf) -> Self {
        assert!(batch_size.is_power_of_two());

        LiquidCache {
            files: Mutex::new(AHashMap::new()),
            config: Arc::new(CacheConfig::new(batch_size, max_cache_bytes)),
            cache_dir,
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
            let sanitized_file_path =
                liquid_cache_common::utils::sanitize_path_for_dirname(&file_path);
            let cache_dir = self.cache_dir.join(&sanitized_file_path);
            Arc::new(LiquidCachedFile::new(
                cache_mode,
                self.config.clone(),
                cache_dir,
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
        self.config.batch_size()
    }

    /// Get the max cache bytes of the cache.
    pub fn max_cache_bytes(&self) -> usize {
        self.config.max_cache_bytes()
    }

    /// Get the memory usage of the cache in bytes.
    pub fn memory_usage_bytes(&self) -> usize {
        self.config.memory_usage_bytes()
    }

    /// Get the disk usage of the cache in bytes.
    pub fn disk_usage_bytes(&self) -> usize {
        self.config.disk_usage_bytes()
    }

    /// Reset the cache.
    pub fn reset(&self) {
        let mut files = self.files.lock().unwrap();
        for file in files.values_mut() {
            file.reset();
        }
        self.config.reset_usage();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::{Array, Int32Array};

    /// Helper function to get the memory size from an Arrow array.
    /// (Assuming your Arrow arrays implement `get_array_memory_size`.)
    fn array_mem_size(array: &ArrayRef) -> usize {
        array.get_array_memory_size()
    }

    /// Test that verifies the memory accounting logic.
    #[test]
    fn test_memory_accounting() {
        let batch_size = 64;
        let max_cache_bytes = 4096;
        let tmp_dir = tempfile::tempdir().unwrap();
        let cache = LiquidCache::new(batch_size, max_cache_bytes, tmp_dir.path().to_path_buf());
        let file =
            cache.register_or_get_file("test_file".to_string(), LiquidCacheMode::InMemoryArrow);
        let row_group = file.row_group(0);
        let column = row_group.get_column_or_create(0);

        // Test 1: Basic insertion and size tracking
        let array1 = Arc::new(Int32Array::from(vec![1; 100])) as ArrayRef;
        let array1_size = array_mem_size(&array1);
        assert!(column.insert_arrow_array(0, array1.clone()).is_ok());
        assert!(column.is_cached(0));
        assert_eq!(column.config.memory_usage_bytes(), array1_size);

        // Test 2: Multiple insertions within limit
        let array2 = Arc::new(Int32Array::from(vec![2; 200])) as ArrayRef;
        let array2_size = array_mem_size(&array2);
        assert!(column.insert_arrow_array(1, array2.clone()).is_ok());
        assert!(column.is_cached(1));
        assert_eq!(
            column.config.memory_usage_bytes(),
            array1_size + array2_size
        );

        let remaining_space = max_cache_bytes - (array1_size + array2_size);
        let exact_fit_array = Arc::new(Int32Array::from(vec![3; remaining_space / 4])) as ArrayRef;
        assert!(
            column
                .insert_arrow_array(2, exact_fit_array.clone())
                .is_err()
        );
        assert!(!column.is_cached(2));
        assert_eq!(
            column.config.memory_usage_bytes(),
            array1_size + array2_size
        );

        assert!(column.insert_arrow_array(0, array1.clone()).is_err());
        assert_eq!(
            column.config.memory_usage_bytes(),
            array1_size + array2_size
        );
    }

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
        let column = row_group.get_column_or_create(0);

        let arrow_array = Arc::new(Int32Array::from(vec![10; 100_000])) as ArrayRef;
        let arrow_size = arrow_array.get_array_memory_size();

        assert!(column.insert_arrow_array(0, arrow_array.clone()).is_ok());

        let size_before = column.config.memory_usage_bytes();
        loop {
            // Now check that the entry has been transcoded.
            let rows = column.rows.read().unwrap();
            let entry = rows.get(&0).expect("Expected row 0 to be cached");
            match entry.value {
                CachedBatch::LiquidMemory(ref liquid) => {
                    // The memory usage after transcoding should be less or equal.
                    let new_size = entry.value.memory_usage();
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

        let size_after = column.config.memory_usage_bytes();
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
            LiquidCacheMode::InMemoryArrow,
        );
        let row_group = file.row_group(0);
        let column = row_group.get_column_or_create(0);

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
                    let _ = column.insert_arrow_array(i, array);
                })
            })
            .collect();

        for handle in handles {
            handle.join().unwrap();
        }

        let final_mem = column.config.memory_usage_bytes();

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
        let column = row_group.get_column_or_create(column_id);

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
        assert!(
            column
                .insert_arrow_array(row_id, large_array.clone())
                .is_ok()
        );

        // 1. Verify the file path structure
        let expected_file_path = cache_path
            .join("test_on_disk_file")
            .join(format!("rg_{}", row_group_id))
            .join(format!("col_{}", column_id))
            .join(format!("row_{}.bin", row_id));

        assert!(
            expected_file_path.exists(),
            "On-disk cache file not found at expected path: {:?}",
            expected_file_path
        );

        // 2. Verify memory accounting
        let memory_usage = column.config.memory_usage_bytes();
        let disk_usage = column.config.disk_usage_bytes();

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
            let rows = column.rows.read().unwrap();
            let cached_entry = rows.get(&row_id).expect("Row should be cached");
            match cached_entry.value() {
                CachedBatch::OnDiskLiquid => {
                    // This is what we expect
                }
                other => panic!("Expected OnDiskLiquid, got: {}", other),
            }
        }

        // 4. Read the data back and verify its content
        let retrieved_array = column
            .get_arrow_array_test_only(row_id)
            .expect("Failed to read array");

        assert_eq!(&retrieved_array, &large_array);
    }
}
