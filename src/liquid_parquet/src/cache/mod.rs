use ahash::AHashMap;
use arrow::array::{Array, AsArray, BooleanArray};
use arrow::array::{ArrayRef, RecordBatch};
use arrow::buffer::BooleanBuffer;
use arrow::compute::{cast, prep_null_mask_filter};
use arrow_schema::{ArrowError, DataType, Field, Schema};
use log::warn;
use std::fmt::Display;
use std::sync::atomic::{AtomicU32, AtomicUsize, Ordering};
use std::sync::{Arc, LazyLock, Mutex, RwLock};
use tokio::runtime::Runtime;
mod stats;
use super::liquid_array::{LiquidArrayRef, LiquidByteArray, LiquidPrimitiveArray};
use crate::{ABLATION_STUDY_MODE, AblationStudyMode, LiquidPredicate};
use arrow::array::types::{
    Int8Type as ArrowInt8Type, Int16Type as ArrowInt16Type, Int32Type as ArrowInt32Type,
    Int64Type as ArrowInt64Type, UInt8Type as ArrowUInt8Type, UInt16Type as ArrowUInt16Type,
    UInt32Type as ArrowUInt32Type, UInt64Type as ArrowUInt64Type,
};

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
}

impl CachedBatch {
    fn memory_usage(&self) -> usize {
        match self {
            Self::ArrowMemory(array) => array.get_array_memory_size(),
            Self::LiquidMemory(array) => array.get_array_memory_size(),
        }
    }
}

impl Display for CachedBatch {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ArrowMemory(_) => write!(f, "ArrowMemory"),
            Self::LiquidMemory(_) => write!(f, "LiquidMemory"),
        }
    }
}

#[derive(Debug)]
pub struct LiquidCachedColumn {
    cache_mode: LiquidCacheMode,
    config: CacheConfig,
    liquid_compressor_states: LiquidCompressorStates,
    rows: RwLock<AHashMap<usize, CachedEntry>>,
}

pub type LiquidCachedColumnRef = Arc<LiquidCachedColumn>;

fn array_to_record_batch(array: ArrayRef) -> RecordBatch {
    let schema = Arc::new(Schema::new(vec![Field::new(
        "_",
        array.data_type().clone(),
        array.is_nullable(),
    )]));
    RecordBatch::try_new(schema, vec![array]).unwrap()
}

pub enum InsertArrowArrayError {
    CacheFull(ArrayRef),
    AlreadyCached,
}

impl LiquidCachedColumn {
    fn new(cache_mode: LiquidCacheMode, config: CacheConfig) -> Self {
        Self {
            cache_mode,
            config,
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
        self.config.batch_size
    }

    pub(crate) fn is_cached(&self, row_id: usize) -> bool {
        let rows = self.rows.read().unwrap();
        rows.contains_key(&row_id)
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
                let record_batch = array_to_record_batch(selected);
                let boolean_array = predicate.evaluate(record_batch).unwrap();
                let predicate_filter = match boolean_array.null_count() {
                    0 => boolean_array,
                    _ => prep_null_mask_filter(&boolean_array),
                };
                let (buffer, _) = predicate_filter.into_parts();
                Some(Ok(buffer))
            }
            CachedBatch::LiquidMemory(array) => match ABLATION_STUDY_MODE {
                AblationStudyMode::FullDecoding => {
                    let boolean_array = BooleanArray::new(selection.clone(), None);
                    let arrow = array.to_arrow_array();
                    let filtered = arrow::compute::filter(&arrow, &boolean_array).unwrap();
                    let record_batch = array_to_record_batch(filtered);
                    let boolean_array = predicate.evaluate(record_batch).unwrap();
                    let (buffer, _) = boolean_array.into_parts();
                    Some(Ok(buffer))
                }
                AblationStudyMode::SelectiveDecoding
                | AblationStudyMode::SelectiveWithLateMaterialization => {
                    let boolean_array = BooleanArray::new(selection.clone(), None);
                    let filtered = array.filter(&boolean_array);
                    let arrow = filtered.to_arrow_array();
                    let record_batch = array_to_record_batch(arrow);
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
        }
    }

    /// Insert an arrow array into the cache.
    // TODO: return an error if the array is too large to fit in the cache.
    pub(crate) fn insert_arrow_array(
        self: &Arc<Self>,
        row_id: usize,
        array: ArrayRef,
    ) -> Result<(), InsertArrowArrayError> {
        if self.is_cached(row_id) {
            return Err(InsertArrowArrayError::AlreadyCached);
        }

        let current_cache_size = self.config.current_cache_size.load(Ordering::Relaxed);
        let array_size = array.get_array_memory_size();
        if current_cache_size + array_size > self.config.max_cache_bytes {
            return Err(InsertArrowArrayError::CacheFull(array));
        }

        let mut rows = self.rows.write().unwrap();

        match &self.cache_mode {
            LiquidCacheMode::InMemoryArrow => {
                let old = rows.insert(row_id, CachedEntry::new_in_memory(array));
                assert!(old.is_none());
                self.config
                    .current_cache_size
                    .fetch_add(array_size, Ordering::Relaxed);
                Ok(())
            }
            LiquidCacheMode::OnDiskArrow => {
                unimplemented!()
            }
            LiquidCacheMode::InMemoryLiquid {
                transcode_in_background,
            } => {
                let array = if array.data_type() == &DataType::Utf8View {
                    cast(
                        &array,
                        &DataType::Dictionary(Box::new(DataType::UInt16), Box::new(DataType::Utf8)),
                    )
                    .unwrap()
                } else {
                    array
                };

                if *transcode_in_background {
                    self.config
                        .current_cache_size
                        .fetch_add(array.get_array_memory_size(), Ordering::Relaxed);

                    // Insert the arrow array first, without doing the expensive transcoding.
                    rows.insert(row_id, CachedEntry::new_in_memory(array.clone()));

                    // Submit a background transcoding task to our dedicated thread pool.
                    let column_arc = Arc::clone(self);
                    TRANSCODE_THREAD_POOL.spawn(async move {
                        column_arc.background_transcode(row_id);
                    });
                    Ok(())
                } else {
                    let transcoded = self.transcode_inner(&array);
                    let new_size = transcoded.memory_usage();
                    self.config
                        .current_cache_size
                        .fetch_add(new_size, Ordering::Relaxed);
                    rows.insert(row_id, CachedEntry {
                        value: transcoded,
                        hit_count: AtomicU32::new(0),
                    });
                    Ok(())
                }
            }
        }
    }

    /// This method is run in the background. It acquires a write lock on the cache,
    /// checks that the stored value is still an ArrowMemory batch, and then runs the
    /// expensive transcoding, replacing the entry with a LiquidMemory batch.
    fn background_transcode(self: &Arc<Self>, row_id: usize) {
        let mut rows = self.rows.write().unwrap();
        if let Some(entry) = rows.get_mut(&row_id) {
            match &entry.value {
                CachedBatch::ArrowMemory(array) => {
                    let previous_size = entry.value.memory_usage();
                    let transcoded = self.transcode_inner(array);
                    let new_size = transcoded.memory_usage();
                    if previous_size < new_size {
                        warn!(
                            "Transcoding increased the size of the array, previous size: {}, new size: {}, double check this is correct",
                            previous_size, new_size
                        );
                        return;
                    }
                    self.config
                        .current_cache_size
                        .fetch_sub(previous_size - new_size, Ordering::Relaxed);
                    entry.value = transcoded;
                }
                CachedBatch::LiquidMemory(_) => {
                    // Already transcoded.
                }
            }
        }
    }

    /// This method is used to transcode an arrow array into a liquid array.
    fn transcode_inner(&self, array: &ArrayRef) -> CachedBatch {
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
                _ => {
                    // For unsupported primitive types, leave the value unchanged.
                    return CachedBatch::ArrowMemory(array.clone());
                }
            };
            return CachedBatch::LiquidMemory(liquid_array);
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
                    return CachedBatch::LiquidMemory(Arc::new(compressed));
                }
                drop(compressor);
                let mut compressors = self.fsst_compressor().write().unwrap();
                let (compressor, compressed) =
                    LiquidByteArray::train_from_arrow_view(array.as_string_view());
                *compressors = Some(compressor);
                CachedBatch::LiquidMemory(Arc::new(compressed))
            }
            DataType::Utf8 => {
                let compressor = self.fsst_compressor().read().unwrap();
                if let Some(compressor) = compressor.as_ref() {
                    let compressed = LiquidByteArray::from_string_array(
                        array.as_string::<i32>(),
                        compressor.clone(),
                    );
                    return CachedBatch::LiquidMemory(Arc::new(compressed));
                }
                drop(compressor);
                let mut compressors = self.fsst_compressor().write().unwrap();
                let (compressor, compressed) =
                    LiquidByteArray::train_from_arrow(array.as_string::<i32>());
                *compressors = Some(compressor);
                CachedBatch::LiquidMemory(Arc::new(compressed))
            }
            DataType::Dictionary(_, _) => {
                if let Some(dict_array) = array.as_dictionary_opt::<ArrowUInt16Type>() {
                    let compressor = self.fsst_compressor().read().unwrap();
                    if let Some(compressor) = compressor.as_ref() {
                        let liquid_array = unsafe {
                            LiquidByteArray::from_unique_dict_array(dict_array, compressor.clone())
                        };
                        return CachedBatch::LiquidMemory(Arc::new(liquid_array));
                    }
                    drop(compressor);
                    let mut compressors = self.fsst_compressor().write().unwrap();
                    let (compressor, liquid_array) =
                        LiquidByteArray::train_from_arrow_dict(dict_array);
                    *compressors = Some(compressor);
                    return CachedBatch::LiquidMemory(Arc::new(liquid_array));
                }
                panic!("unsupported data type {:?}", array.data_type());
            }
            _ => panic!("unsupported data type {:?}", array.data_type()),
        }
    }
}

#[derive(Debug)]
pub struct LiquidCachedRowGroup {
    cache_mode: LiquidCacheMode,
    config: CacheConfig,
    columns: RwLock<AHashMap<usize, Arc<LiquidCachedColumn>>>,
}

impl LiquidCachedRowGroup {
    fn new(cache_mode: LiquidCacheMode, config: CacheConfig) -> Self {
        Self {
            cache_mode,
            config,
            columns: RwLock::new(AHashMap::new()),
        }
    }

    pub fn get_column_or_create(&self, column_id: usize) -> LiquidCachedColumnRef {
        self.columns
            .write()
            .unwrap()
            .entry(column_id)
            .or_insert_with(|| {
                Arc::new(LiquidCachedColumn::new(
                    self.cache_mode,
                    self.config.clone(),
                ))
            })
            .clone()
    }

    pub fn get_column(&self, column_id: usize) -> Option<LiquidCachedColumnRef> {
        self.columns.read().unwrap().get(&column_id).cloned()
    }
}

pub type LiquidCachedRowGroupRef = Arc<LiquidCachedRowGroup>;

#[derive(Debug)]
pub struct LiquidCachedFile {
    row_groups: Mutex<AHashMap<usize, Arc<LiquidCachedRowGroup>>>,
    cache_mode: LiquidCacheMode,
    config: CacheConfig,
}

impl LiquidCachedFile {
    pub(crate) fn new(cache_mode: LiquidCacheMode, config: CacheConfig) -> Self {
        Self {
            cache_mode,
            config,
            row_groups: Mutex::new(AHashMap::new()),
        }
    }

    pub fn row_group(&self, row_group_id: usize) -> LiquidCachedRowGroupRef {
        let mut row_groups = self.row_groups.lock().unwrap();
        let row_group = row_groups.entry(row_group_id).or_insert_with(|| {
            Arc::new(LiquidCachedRowGroup::new(
                self.cache_mode,
                self.config.clone(),
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
        self.config.current_cache_size.load(Ordering::Relaxed)
    }
}

pub type LiquidCachedFileRef = Arc<LiquidCachedFile>;

#[derive(Debug)]
pub struct LiquidCache {
    /// Files -> RowGroups -> Columns -> Batches
    files: Mutex<AHashMap<String, Arc<LiquidCachedFile>>>,

    config: CacheConfig,
}

pub type LiquidCacheRef = Arc<LiquidCache>;

#[derive(Debug, Copy, Clone)]
pub enum LiquidCacheMode {
    /// The baseline that reads the schema as is.
    InMemoryArrow,
    OnDiskArrow,
    InMemoryLiquid {
        transcode_in_background: bool,
    },
}

#[derive(Debug, Clone)]
pub(crate) struct CacheConfig {
    batch_size: usize,
    max_cache_bytes: usize,
    current_cache_size: Arc<AtomicUsize>,
}

impl CacheConfig {
    #[cfg(test)]
    pub fn new(batch_size: usize, max_cache_bytes: usize) -> Self {
        Self {
            batch_size,
            max_cache_bytes,
            current_cache_size: Arc::new(AtomicUsize::new(0)),
        }
    }
}

impl LiquidCache {
    pub fn new(batch_size: usize, max_cache_bytes: usize) -> Self {
        assert!(batch_size.is_power_of_two());

        LiquidCache {
            files: Mutex::new(AHashMap::new()),
            config: CacheConfig {
                batch_size,
                max_cache_bytes,
                current_cache_size: Arc::new(AtomicUsize::new(0)),
            },
        }
    }

    /// Register a file in the cache.
    pub fn register_or_get_file(
        &self,
        file_path: String,
        cache_mode: LiquidCacheMode,
    ) -> LiquidCachedFileRef {
        let mut files = self.files.lock().unwrap();
        let value = files
            .entry(file_path.clone())
            .or_insert_with(|| Arc::new(LiquidCachedFile::new(cache_mode, self.config.clone())));
        value.clone()
    }

    /// Get a file from the cache.
    pub fn get_file(&self, file_path: String) -> Option<LiquidCachedFileRef> {
        let files = self.files.lock().unwrap();
        files.get(&file_path).cloned()
    }

    pub fn batch_size(&self) -> usize {
        self.config.batch_size
    }

    /// Reset the cache.
    pub fn reset(&self) {
        let mut files = self.files.lock().unwrap();
        for file in files.values_mut() {
            file.reset();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::{Array, Int32Array};
    use std::sync::atomic::Ordering;
    use std::{thread, time::Duration};

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
        let cache = LiquidCache::new(batch_size, max_cache_bytes);
        let file =
            cache.register_or_get_file("test_file".to_string(), LiquidCacheMode::InMemoryArrow);
        let row_group = file.row_group(0);
        let column = row_group.get_column_or_create(0);

        // Test 1: Basic insertion and size tracking
        let array1 = Arc::new(Int32Array::from(vec![1; 100])) as ArrayRef;
        let array1_size = array_mem_size(&array1);
        assert!(column.insert_arrow_array(0, array1.clone()).is_ok());
        assert!(column.is_cached(0));
        assert_eq!(
            column.config.current_cache_size.load(Ordering::Relaxed),
            array1_size
        );

        // Test 2: Multiple insertions within limit
        let array2 = Arc::new(Int32Array::from(vec![2; 200])) as ArrayRef;
        let array2_size = array_mem_size(&array2);
        assert!(column.insert_arrow_array(1, array2.clone()).is_ok());
        assert!(column.is_cached(1));
        assert_eq!(
            column.config.current_cache_size.load(Ordering::Relaxed),
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
            column.config.current_cache_size.load(Ordering::Relaxed),
            array1_size + array2_size
        );

        assert!(column.insert_arrow_array(0, array1.clone()).is_err());
        assert_eq!(
            column.config.current_cache_size.load(Ordering::Relaxed),
            array1_size + array2_size
        );
    }

    /// Test that verifies background transcoding updates both the stored value type
    /// and the memory accounting.
    #[test]
    fn test_background_transcoding() {
        // Use InMemoryLiquid mode with background transcoding enabled.
        let batch_size = 64;
        let max_cache_bytes = 1024 * 1024; // a generous limit for the test
        let cache = LiquidCache::new(batch_size, max_cache_bytes);
        let file =
            cache.register_or_get_file("test_file2".to_string(), LiquidCacheMode::InMemoryLiquid {
                transcode_in_background: true,
            });
        let row_group = file.row_group(0);
        let column = row_group.get_column_or_create(0);

        let arrow_array = Arc::new(Int32Array::from(vec![10; 100_000])) as ArrayRef;
        let arrow_size = arrow_array.get_array_memory_size();

        assert!(column.insert_arrow_array(0, arrow_array.clone()).is_ok());

        // Immediately after insertion, the batch should be of type ArrowMemory.
        {
            let rows = column.rows.read().unwrap();
            let entry = rows.get(&0).expect("Expected row 0 to be cached");
            match entry.value {
                CachedBatch::ArrowMemory(_) => {} // as expected
                CachedBatch::LiquidMemory(_) => panic!("Should not have transcoded immediately"),
            }
        }
        let size_before = column.config.current_cache_size.load(Ordering::Relaxed);
        assert_eq!(size_before, arrow_size);

        // Wait briefly to let the background transcoding complete.
        thread::sleep(Duration::from_millis(200));

        // Now check that the entry has been transcoded.
        {
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
                }
                CachedBatch::ArrowMemory(_) => {
                    panic!("Expected background transcoding to have occurred")
                }
            }
        }
        let size_after = column.config.current_cache_size.load(Ordering::Relaxed);
        assert!(
            size_after <= size_before,
            "Cache memory counter should have decreased after transcoding"
        );

        println!(
            "Test background transcoding: size_before={} size_after={}",
            size_before, size_after
        );
    }
}
