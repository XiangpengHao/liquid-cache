use ahash::AHashMap;
use arrow::array::{Array, AsArray, BooleanArray};
use arrow::array::{ArrayRef, RecordBatch};
use arrow::buffer::BooleanBuffer;
use arrow::compute::prep_null_mask_filter;
use arrow_schema::{ArrowError, DataType, Field, Schema};
use std::fmt::Display;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Arc, Mutex, RwLock};
mod stats;
use super::liquid_array::{LiquidArrayRef, LiquidByteArray, LiquidPrimitiveArray};
use crate::LiquidPredicate;
use arrow::array::types::{
    Int8Type as ArrowInt8Type, Int16Type as ArrowInt16Type, Int32Type as ArrowInt32Type,
    Int64Type as ArrowInt64Type, UInt8Type as ArrowUInt8Type, UInt16Type as ArrowUInt16Type,
    UInt32Type as ArrowUInt32Type, UInt64Type as ArrowUInt64Type,
};

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

    fn new(value: CachedBatch) -> Self {
        CachedEntry {
            value,
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
    #[allow(unused)]
    row_group_id: usize,
    #[allow(unused)]
    column_id: usize,
    cache_mode: LiquidCacheMode,
    liquid_compressor_states: LiquidCompressorStates,
    batch_size: usize,
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

impl LiquidCachedColumn {
    fn new(
        row_group_id: usize,
        column_id: usize,
        cache_mode: LiquidCacheMode,
        batch_size: usize,
    ) -> Self {
        Self {
            row_group_id,
            column_id,
            cache_mode,
            batch_size,
            rows: RwLock::new(AHashMap::new()),
            liquid_compressor_states: LiquidCompressorStates::new(),
        }
    }

    fn fsst_compressor(&self) -> &RwLock<Option<Arc<fsst::Compressor>>> {
        &self.liquid_compressor_states.fsst_compressor
    }

    pub(crate) fn batch_size(&self) -> usize {
        self.batch_size
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
            CachedBatch::LiquidMemory(array) => {
                let boolean_array = BooleanArray::new(selection.clone(), None);
                let filtered = array.filter(&boolean_array);
                let boolean_array = predicate.evaluate_liquid(&filtered).unwrap();
                let (buffer, _) = boolean_array.into_parts();
                Some(Ok(buffer))
            }
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
            CachedBatch::LiquidMemory(array) => {
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
        }
    }

    pub(crate) fn insert_arrow_array(&self, row_id: usize, array: ArrayRef) {
        if self.is_cached(row_id) {
            return;
        }

        let mut rows = self.rows.write().unwrap();

        match &self.cache_mode {
            LiquidCacheMode::InMemoryArrow => {
                let old = rows.insert(row_id, CachedEntry::new_in_memory(array));
                assert!(old.is_none());
            }
            LiquidCacheMode::OnDiskArrow => {
                unimplemented!()
            }
            LiquidCacheMode::InMemoryLiquid => {
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
                    rows.insert(
                        row_id,
                        CachedEntry::new(CachedBatch::LiquidMemory(primitive)),
                    );
                    return;
                }

                // string types
                match array.data_type() {
                    DataType::Utf8View => {
                        let compressor = self.fsst_compressor().read().unwrap();
                        if let Some(compressor) = compressor.as_ref() {
                            let compressed = LiquidByteArray::from_string_view_array(
                                array.as_string_view(),
                                compressor.clone(),
                            );
                            rows.insert(
                                row_id,
                                CachedEntry::new(CachedBatch::LiquidMemory(Arc::new(compressed))),
                            );
                            return;
                        }

                        drop(compressor);
                        let mut compressors = self.fsst_compressor().write().unwrap();
                        let (compressor, compressed) =
                            LiquidByteArray::train_from_arrow_view(array.as_string_view());
                        *compressors = Some(compressor);
                        rows.insert(
                            row_id,
                            CachedEntry::new(CachedBatch::LiquidMemory(Arc::new(compressed))),
                        );
                    }
                    DataType::Utf8 => {
                        let compressor = self.fsst_compressor().read().unwrap();
                        if let Some(compressor) = compressor.as_ref() {
                            let compressed = LiquidByteArray::from_string_array(
                                array.as_string::<i32>(),
                                compressor.clone(),
                            );
                            rows.insert(
                                row_id,
                                CachedEntry::new(CachedBatch::LiquidMemory(Arc::new(compressed))),
                            );
                            return;
                        }

                        drop(compressor);
                        let mut compressors = self.fsst_compressor().write().unwrap();
                        let (compressor, compressed) =
                            LiquidByteArray::train_from_arrow(array.as_string::<i32>());
                        *compressors = Some(compressor);
                        rows.insert(
                            row_id,
                            CachedEntry::new(CachedBatch::LiquidMemory(Arc::new(compressed))),
                        );
                    }
                    DataType::Dictionary(_, _) => {
                        if let Some(dict_array) = array.as_dictionary_opt::<ArrowUInt16Type>() {
                            let compressor = self.fsst_compressor().read().unwrap();
                            if let Some(compressor) = compressor.as_ref() {
                                let liquid_array = LiquidByteArray::from_dict_array(
                                    dict_array,
                                    compressor.clone(),
                                );
                                rows.insert(
                                    row_id,
                                    CachedEntry::new(CachedBatch::LiquidMemory(Arc::new(
                                        liquid_array,
                                    ))),
                                );
                                return;
                            }

                            drop(compressor);
                            let mut compressors = self.fsst_compressor().write().unwrap();
                            let (compressor, liquid_array) =
                                LiquidByteArray::train_from_arrow_dict(dict_array);
                            *compressors = Some(compressor);
                            rows.insert(
                                row_id,
                                CachedEntry::new(CachedBatch::LiquidMemory(Arc::new(liquid_array))),
                            );
                            return;
                        }
                        panic!("unsupported data type {:?}", array.data_type());
                    }
                    _ => panic!("unsupported data type {:?}", array.data_type()),
                }
            }
        }
    }
}

#[derive(Debug)]
pub struct LiquidCachedRowGroup {
    row_group_id: usize,
    cache_mode: LiquidCacheMode,
    batch_size: usize,
    columns: RwLock<AHashMap<usize, Arc<LiquidCachedColumn>>>,
}

impl LiquidCachedRowGroup {
    fn new(row_group_id: usize, cache_mode: LiquidCacheMode, batch_size: usize) -> Self {
        Self {
            row_group_id,
            cache_mode,
            batch_size,
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
                    self.row_group_id,
                    column_id,
                    self.cache_mode,
                    self.batch_size,
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
    batch_size: usize,
}

impl LiquidCachedFile {
    pub(crate) fn new(cache_mode: LiquidCacheMode, batch_size: usize) -> Self {
        Self {
            cache_mode,
            batch_size,
            row_groups: Mutex::new(AHashMap::new()),
        }
    }

    pub fn row_group(&self, row_group_id: usize) -> LiquidCachedRowGroupRef {
        let mut row_groups = self.row_groups.lock().unwrap();
        let row_group = row_groups.entry(row_group_id).or_insert_with(|| {
            Arc::new(LiquidCachedRowGroup::new(
                row_group_id,
                self.cache_mode,
                self.batch_size,
            ))
        });
        row_group.clone()
    }

    fn reset(&self) {
        let mut row_groups = self.row_groups.lock().unwrap();
        row_groups.clear();
    }
}

pub type LiquidCachedFileRef = Arc<LiquidCachedFile>;

#[derive(Debug)]
pub struct LiquidCache {
    /// Files -> RowGroups -> Columns -> Batches
    files: Mutex<AHashMap<String, Arc<LiquidCachedFile>>>,

    /// cache granularity
    batch_size: usize,
}

pub type LiquidCacheRef = Arc<LiquidCache>;

#[derive(Debug, Copy, Clone)]
pub enum LiquidCacheMode {
    InMemoryArrow,
    OnDiskArrow,
    InMemoryLiquid,
}

impl LiquidCache {
    pub fn new(batch_size: usize) -> Self {
        assert!(batch_size.is_power_of_two());

        LiquidCache {
            files: Mutex::new(AHashMap::new()),
            batch_size,
        }
    }

    /// Register a file in the cache. Panic if the file is already registered.
    pub fn register_file(
        &self,
        file_path: String,
        cache_mode: LiquidCacheMode,
    ) -> LiquidCachedFileRef {
        let file = Arc::new(LiquidCachedFile::new(cache_mode, self.batch_size));
        let mut files = self.files.lock().unwrap();
        let old = files.insert(file_path.clone(), file.clone());
        assert!(old.is_none(), "file already registered");
        file
    }

    /// Get a file from the cache.
    pub fn get_file(&self, file_path: String) -> Option<LiquidCachedFileRef> {
        let files = self.files.lock().unwrap();
        files.get(&file_path).cloned()
    }

    pub fn batch_size(&self) -> usize {
        self.batch_size
    }

    /// Reset the cache.
    pub fn reset(&self) {
        let mut files = self.files.lock().unwrap();
        for file in files.values_mut() {
            file.reset();
        }
    }
}
