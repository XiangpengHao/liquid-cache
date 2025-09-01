//! This module contains the cache implementation for the Parquet reader.
//!

use crate::cache::io::{ColumnAccessPath, ParquetIoContext, blocking_reading_io, blocking_sans_io};
use crate::reader::{LiquidPredicate, extract_multi_column_or};
use crate::sync::{Mutex, RwLock};
use ahash::AHashMap;
use arrow::array::{Array, ArrayRef, BooleanArray, RecordBatch};
use arrow::buffer::BooleanBuffer;
use arrow::compute::prep_null_mask_filter;
use arrow_schema::{ArrowError, DataType, Field, Schema};
use liquid_cache_common::{LiquidCacheMode, coerce_parquet_type_to_liquid_type};
use liquid_cache_storage::cache::cached_data::GetWithPredicateResult;
use liquid_cache_storage::cache::io_state::{IoStateMachine, SansIo, TryGet};
use liquid_cache_storage::cache::{CachePolicy, CacheStorage, CacheStorageBuilder};
use parquet::arrow::arrow_reader::ArrowPredicate;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

mod id;
mod io;
mod stats;

pub use id::{BatchID, ParquetArrayID};

/// A column in the cache.
#[derive(Debug)]
pub struct LiquidCachedColumn {
    cache_store: Arc<CacheStorage>,
    field: Arc<Field>,
    column_path: ColumnAccessPath,
}

pub(crate) type LiquidCachedColumnRef = Arc<LiquidCachedColumn>;

/// Error type for inserting an arrow array into the cache.
#[derive(Debug)]
pub enum InsertArrowArrayError {
    /// The array is already cached.
    AlreadyCached,
}

impl LiquidCachedColumn {
    fn new(
        field: Arc<Field>,
        cache_store: Arc<CacheStorage>,
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
    fn entry_id(&self, batch_id: BatchID) -> ParquetArrayID {
        self.column_path.entry_id(batch_id)
    }

    pub(crate) fn cache_mode(&self) -> &LiquidCacheMode {
        self.cache_store.config().cache_mode()
    }

    pub(crate) fn batch_size(&self) -> usize {
        self.cache_store.config().batch_size()
    }

    pub(crate) fn is_cached(&self, batch_id: BatchID) -> bool {
        self.cache_store.is_cached(&self.entry_id(batch_id).into())
    }

    fn arrow_array_to_record_batch(&self, array: ArrayRef, field: &Arc<Field>) -> RecordBatch {
        let schema = Arc::new(Schema::new(vec![field.clone()]));
        RecordBatch::try_new(schema, vec![array]).unwrap()
    }

    /// Evaluates a predicate on a cached column.
    pub fn eval_predicate_with_filter(
        &self,
        batch_id: BatchID,
        filter: &BooleanBuffer,
        predicate: &mut LiquidPredicate,
    ) -> Option<Result<BooleanArray, ArrowError>> {
        let cached_entry = self.cache_store.get(&self.entry_id(batch_id).into())?;

        let result = cached_entry
            .get_with_predicate(filter, predicate.physical_expr_physical_column_index());
        let result = blocking_sans_io(result);
        match result {
            GetWithPredicateResult::Evaluated(buffer) => Some(Ok(buffer)),
            GetWithPredicateResult::Filtered(array) => {
                let record_batch = self.arrow_array_to_record_batch(array, &self.field);
                let boolean_array = predicate.evaluate(record_batch).unwrap();
                let predicate_filter = match boolean_array.null_count() {
                    0 => boolean_array,
                    _ => prep_null_mask_filter(&boolean_array),
                };
                Some(Ok(predicate_filter))
            }
        }
    }

    /// Get an arrow array with a filter applied.
    pub fn get_arrow_array_with_filter(
        &self,
        batch_id: BatchID,
        filter: &BooleanBuffer,
    ) -> Option<ArrayRef> {
        let inner_value = self.cache_store.get(&self.entry_id(batch_id).into())?;
        let result = blocking_sans_io(inner_value.get_with_selection(filter));
        result.ok()
    }

    #[cfg(test)]
    pub(crate) fn get_arrow_array_test_only(&self, batch_id: BatchID) -> Option<ArrayRef> {
        let cached_entry = self.cache_store.get(&self.entry_id(batch_id).into())?;
        let result = blocking_sans_io(cached_entry.get_arrow_array());
        Some(result)
    }

    /// Insert an array into the cache.
    pub fn insert(
        self: &Arc<Self>,
        batch_id: BatchID,
        array: ArrayRef,
    ) -> Result<(), InsertArrowArrayError> {
        if self.is_cached(batch_id) {
            return Err(InsertArrowArrayError::AlreadyCached);
        }
        self.cache_store
            .insert(self.entry_id(batch_id).into(), array);
        Ok(())
    }
}

/// A row group in the cache.
#[derive(Debug)]
pub struct LiquidCachedRowGroup {
    columns: RwLock<AHashMap<u64, Arc<LiquidCachedColumn>>>,
    cache_store: Arc<CacheStorage>,
    row_group_id: u64,
    file_id: u64,
}

impl LiquidCachedRowGroup {
    fn new(cache_store: Arc<CacheStorage>, row_group_id: u64, file_id: u64) -> Self {
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

    /// Create a column in the row group.
    pub fn create_column(&self, column_id: u64, field: Arc<Field>) -> LiquidCachedColumnRef {
        use std::collections::hash_map::Entry;
        let mut columns = self.columns.write().unwrap();

        let field = match field.data_type() {
            DataType::Utf8View | DataType::BinaryView => {
                let field: Field = Field::clone(&field);
                let new_data_type = coerce_parquet_type_to_liquid_type(
                    field.data_type(),
                    self.cache_store.config().cache_mode(),
                );
                Arc::new(field.with_data_type(new_data_type))
            }
            DataType::Utf8 | DataType::LargeUtf8 | DataType::Binary | DataType::LargeBinary => {
                unreachable!()
            }
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

    /// Get a column from the row group.
    pub fn get_column(&self, column_id: u64) -> Option<LiquidCachedColumnRef> {
        self.columns.read().unwrap().get(&column_id).cloned()
    }

    /// Evaluate a predicate on a row group.
    pub fn evaluate_selection_with_predicate(
        &self,
        batch_id: BatchID,
        selection: &BooleanBuffer,
        predicate: &mut LiquidPredicate,
    ) -> Option<Result<BooleanArray, ArrowError>> {
        let column_ids = predicate.predicate_column_ids();

        if column_ids.len() == 1 {
            // If we only have one column, we can short-circuit and try to evaluate the predicate on encoded data.
            let column_id = column_ids[0];
            let cache = self.get_column(column_id as u64)?;
            return cache.eval_predicate_with_filter(batch_id, selection, predicate);
        } else if column_ids.len() >= 2 {
            // Try to extract multiple column-literal expressions from OR structure
            if let Some(column_exprs) =
                extract_multi_column_or(predicate.physical_expr_physical_column_index())
            {
                let mut combined_buffer: Option<BooleanArray> = None;

                for (col_idx, expr) in column_exprs {
                    let column = self.get_column(col_idx as u64)?;
                    let entry = self.cache_store.get(&column.entry_id(batch_id).into())?;
                    let io_state = entry.try_read_liquid();
                    let liquid_array = match io_state {
                        SansIo::Ready(None) => {
                            combined_buffer = None;
                            break;
                        }
                        SansIo::Ready(Some(array)) => array,
                        SansIo::Pending((mut state, mut io_req)) => loop {
                            let bytes = blocking_reading_io(&io_req).ok()?;
                            state.feed(bytes);
                            match state.try_get() {
                                TryGet::Ready(array) => break array,
                                TryGet::NeedData((s, req)) => {
                                    state = s;
                                    io_req = req;
                                }
                            }
                        },
                    };
                    let buffer =
                        if let Some(buffer) = liquid_array.try_eval_predicate(&expr, selection) {
                            buffer
                        } else {
                            combined_buffer = None;
                            break;
                        };

                    combined_buffer = Some(match combined_buffer {
                        None => buffer,
                        Some(existing) => {
                            arrow::compute::kernels::boolean::or_kleene(&existing, &buffer).ok()?
                        }
                    });
                }

                if let Some(result) = combined_buffer {
                    return Some(Ok(result));
                }
            }
        }
        // Otherwise, we need to first convert the data into arrow arrays.
        let mut arrays = Vec::new();
        let mut fields = Vec::new();
        for column_id in column_ids {
            let column = self.get_column(column_id as u64)?;
            let array = column.get_arrow_array_with_filter(batch_id, selection)?;
            arrays.push(array);
            fields.push(column.field.clone());
        }
        let schema = Arc::new(Schema::new(fields));
        let record_batch = RecordBatch::try_new(schema, arrays).unwrap();
        let boolean_array = predicate.evaluate(record_batch).unwrap();
        Some(Ok(boolean_array))
    }
}

pub(crate) type LiquidCachedRowGroupRef = Arc<LiquidCachedRowGroup>;

/// A file in the cache.
#[derive(Debug)]
pub struct LiquidCachedFile {
    row_groups: Mutex<AHashMap<u64, Arc<LiquidCachedRowGroup>>>,
    cache_store: Arc<CacheStorage>,
    file_id: u64,
}

impl LiquidCachedFile {
    fn new(cache_store: Arc<CacheStorage>, file_id: u64) -> Self {
        Self {
            row_groups: Mutex::new(AHashMap::new()),
            cache_store,
            file_id,
        }
    }

    /// Get a row group from the cache.
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

    /// Get the cache mode of the cache.
    pub fn cache_mode(&self) -> &LiquidCacheMode {
        self.cache_store.config().cache_mode()
    }
}

/// A reference to a cached file.
pub(crate) type LiquidCachedFileRef = Arc<LiquidCachedFile>;

/// The main cache structure.
#[derive(Debug)]
pub struct LiquidCache {
    /// Files -> RowGroups -> Columns -> Batches
    files: Mutex<AHashMap<String, Arc<LiquidCachedFile>>>,

    cache_store: Arc<CacheStorage>,

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
        cache_policy: Box<dyn CachePolicy>,
    ) -> Self {
        assert!(batch_size.is_power_of_two());
        let cache_storage_builder = CacheStorageBuilder::new()
            .with_batch_size(batch_size)
            .with_max_cache_bytes(max_cache_bytes)
            .with_cache_dir(cache_dir.clone())
            .with_cache_mode(cache_mode)
            .with_policy(cache_policy)
            .with_io_worker(Arc::new(ParquetIoContext::new(cache_dir)));
        let cache_storage = cache_storage_builder.build();

        LiquidCache {
            files: Mutex::new(AHashMap::new()),
            cache_store: cache_storage,
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
    ///
    /// # Safety
    /// This is unsafe because resetting the cache while other threads are using the cache may cause undefined behavior.
    /// You should only call this when no one else is using the cache.
    pub unsafe fn reset(&self) {
        let mut files = self.files.lock().unwrap();
        for file in files.values_mut() {
            file.reset();
        }
        self.cache_store.reset();
    }

    /// Flush all memory-based entries to disk while preserving their format.
    /// Arrow entries become DiskArrow, Liquid entries become DiskLiquid.
    /// Entries already on disk are left unchanged.
    ///
    /// This is for admin use only.
    /// This has no guarantees that some new entry will not be inserted in the meantime, or some entries are promoted to memory again.
    /// You mostly want to use this when no one else is using the cache.
    pub fn flush_data(&self) {
        self.cache_store.flush_all_to_disk();
    }

    /// Get the cache mode of the cache.
    pub fn cache_mode(&self) -> &LiquidCacheMode {
        self.cache_store.config().cache_mode()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cache::{LiquidCache, LiquidCachedRowGroupRef};
    use crate::reader::FilterCandidateBuilder;
    use arrow::array::Int32Array;
    use arrow::buffer::BooleanBuffer;
    use arrow::datatypes::{DataType, Field, Schema};
    use arrow::record_batch::RecordBatch;
    use datafusion::common::ScalarValue;
    use datafusion::datasource::schema_adapter::DefaultSchemaAdapterFactory;
    use datafusion::logical_expr::Operator;
    use datafusion::physical_expr::PhysicalExpr;
    use datafusion::physical_expr::expressions::{BinaryExpr, Literal};
    use datafusion::physical_plan::expressions::Column;
    use liquid_cache_common::LiquidCacheMode;
    use liquid_cache_storage::cache_policies::FiloPolicy;
    use parquet::arrow::ArrowWriter;
    use parquet::arrow::arrow_reader::{ArrowReaderMetadata, ArrowReaderOptions};
    use std::sync::Arc;

    fn setup_cache(batch_size: usize) -> LiquidCachedRowGroupRef {
        let tmp_dir = tempfile::tempdir().unwrap();
        let cache = LiquidCache::new(
            batch_size,
            usize::MAX,
            tmp_dir.path().to_path_buf(),
            LiquidCacheMode::LiquidBlocking,
            Box::new(FiloPolicy::new()),
        );
        let file = cache.register_or_get_file("test".to_string());
        file.row_group(0)
    }

    #[test]
    fn evaluate_or_on_cached_columns() {
        let batch_size = 4;
        let row_group = setup_cache(batch_size);

        let schema = Arc::new(Schema::new(vec![
            Field::new("a", DataType::Int32, false),
            Field::new("b", DataType::Int32, false),
        ]));

        let col_a = row_group.create_column(0, Arc::new(Field::new("a", DataType::Int32, false)));
        let col_b = row_group.create_column(1, Arc::new(Field::new("b", DataType::Int32, false)));

        let batch_id = BatchID::from_row_id(0, batch_size);

        let array_a = Arc::new(Int32Array::from(vec![1, 2, 3, 4]));
        let array_b = Arc::new(Int32Array::from(vec![10, 20, 30, 40]));

        assert!(col_a.insert(batch_id, array_a.clone()).is_ok());
        assert!(col_b.insert(batch_id, array_b.clone()).is_ok());

        // build parquet metadata for predicate construction
        let tmp_meta = tempfile::NamedTempFile::new().unwrap();
        let mut writer =
            ArrowWriter::try_new(tmp_meta.reopen().unwrap(), Arc::clone(&schema), None).unwrap();
        let batch = RecordBatch::try_new(Arc::clone(&schema), vec![array_a, array_b]).unwrap();
        writer.write(&batch).unwrap();
        writer.close().unwrap();
        let file_reader = std::fs::File::open(tmp_meta.path()).unwrap();
        let metadata = ArrowReaderMetadata::load(&file_reader, ArrowReaderOptions::new()).unwrap();

        // expression a = 3 OR b = 20
        let expr_a: Arc<dyn PhysicalExpr> = Arc::new(BinaryExpr::new(
            Arc::new(Column::new("a", 0)),
            Operator::Eq,
            Arc::new(Literal::new(ScalarValue::Int32(Some(3)))),
        ));
        let expr_b: Arc<dyn PhysicalExpr> = Arc::new(BinaryExpr::new(
            Arc::new(Column::new("b", 1)),
            Operator::Eq,
            Arc::new(Literal::new(ScalarValue::Int32(Some(20)))),
        ));
        let expr: Arc<dyn PhysicalExpr> = Arc::new(BinaryExpr::new(expr_a, Operator::Or, expr_b));

        let adapter_factory = Arc::new(DefaultSchemaAdapterFactory);
        let builder = FilterCandidateBuilder::new(
            expr,
            Arc::clone(&schema),
            Arc::clone(&schema),
            adapter_factory,
        );
        let candidate = builder.build(metadata.metadata()).unwrap().unwrap();
        let projection = candidate.projection(metadata.metadata());
        let mut predicate = LiquidPredicate::try_new(candidate, projection).unwrap();

        let selection = BooleanBuffer::new_set(batch_size);
        let result = row_group
            .evaluate_selection_with_predicate(batch_id, &selection, &mut predicate)
            .unwrap()
            .unwrap();

        let expected = BooleanBuffer::collect_bool(batch_size, |i| i == 1 || i == 2).into();
        assert_eq!(result, expected);
    }

    #[test]
    fn evaluate_three_column_or() {
        let batch_size = 8;
        let row_group = setup_cache(batch_size);

        let schema = Arc::new(Schema::new(vec![
            Field::new("a", DataType::Int32, false),
            Field::new("b", DataType::Int32, false),
            Field::new("c", DataType::Int32, false),
        ]));

        let col_a = row_group.create_column(0, Arc::new(Field::new("a", DataType::Int32, false)));
        let col_b = row_group.create_column(1, Arc::new(Field::new("b", DataType::Int32, false)));
        let col_c = row_group.create_column(2, Arc::new(Field::new("c", DataType::Int32, false)));

        let batch_id = BatchID::from_row_id(0, batch_size);

        let array_a = Arc::new(Int32Array::from(vec![1, 2, 3, 4, 5, 6, 7, 8]));
        let array_b = Arc::new(Int32Array::from(vec![10, 20, 30, 40, 50, 60, 70, 80]));
        let array_c = Arc::new(Int32Array::from(vec![
            100, 200, 300, 400, 500, 600, 700, 800,
        ]));

        assert!(col_a.insert(batch_id, array_a.clone()).is_ok());
        assert!(col_b.insert(batch_id, array_b.clone()).is_ok());
        assert!(col_c.insert(batch_id, array_c.clone()).is_ok());

        // build parquet metadata for predicate construction
        let tmp_meta = tempfile::NamedTempFile::new().unwrap();
        let mut writer =
            ArrowWriter::try_new(tmp_meta.reopen().unwrap(), Arc::clone(&schema), None).unwrap();
        let batch =
            RecordBatch::try_new(Arc::clone(&schema), vec![array_a, array_b, array_c]).unwrap();
        writer.write(&batch).unwrap();
        writer.close().unwrap();
        let file_reader = std::fs::File::open(tmp_meta.path()).unwrap();
        let metadata = ArrowReaderMetadata::load(&file_reader, ArrowReaderOptions::new()).unwrap();

        // expression: a = 2 OR b = 40 OR c = 600
        let expr_a: Arc<dyn PhysicalExpr> = Arc::new(BinaryExpr::new(
            Arc::new(Column::new("a", 0)),
            Operator::Eq,
            Arc::new(Literal::new(ScalarValue::Int32(Some(2)))),
        ));
        let expr_b: Arc<dyn PhysicalExpr> = Arc::new(BinaryExpr::new(
            Arc::new(Column::new("b", 1)),
            Operator::Eq,
            Arc::new(Literal::new(ScalarValue::Int32(Some(40)))),
        ));
        let expr_c: Arc<dyn PhysicalExpr> = Arc::new(BinaryExpr::new(
            Arc::new(Column::new("c", 2)),
            Operator::Eq,
            Arc::new(Literal::new(ScalarValue::Int32(Some(600)))),
        ));

        // Build nested OR: (a = 2 OR b = 40) OR c = 600
        let expr_ab = Arc::new(BinaryExpr::new(expr_a, Operator::Or, expr_b));
        let expr: Arc<dyn PhysicalExpr> = Arc::new(BinaryExpr::new(expr_ab, Operator::Or, expr_c));

        let adapter_factory = Arc::new(DefaultSchemaAdapterFactory);
        let builder = FilterCandidateBuilder::new(
            expr,
            Arc::clone(&schema),
            Arc::clone(&schema),
            adapter_factory,
        );
        let candidate = builder.build(metadata.metadata()).unwrap().unwrap();
        let projection = candidate.projection(metadata.metadata());
        let mut predicate = LiquidPredicate::try_new(candidate, projection).unwrap();

        let selection = BooleanBuffer::new_set(batch_size);
        let result = row_group
            .evaluate_selection_with_predicate(batch_id, &selection, &mut predicate)
            .unwrap()
            .unwrap();

        // Expected: row 1 (a=2), row 3 (b=40), row 5 (c=600) -> indices 1, 3, 5
        let expected =
            BooleanBuffer::collect_bool(batch_size, |i| i == 1 || i == 3 || i == 5).into();
        assert_eq!(result, expected);
    }

    #[test]
    fn evaluate_string_column_or() {
        let batch_size = 8;
        let row_group = setup_cache(batch_size);

        let schema = Arc::new(Schema::new(vec![
            Field::new("name", DataType::Utf8View, false),
            Field::new("city", DataType::Utf8View, false),
        ]));

        let col_name =
            row_group.create_column(0, Arc::new(Field::new("name", DataType::Utf8View, false)));
        let col_city =
            row_group.create_column(1, Arc::new(Field::new("city", DataType::Utf8View, false)));

        let batch_id = BatchID::from_row_id(0, batch_size);

        let array_name = Arc::new(arrow::array::StringViewArray::from(vec![
            "Alice", "Bob", "Charlie", "David", "Eve", "Frank", "Grace", "Henry",
        ]));
        let array_city = Arc::new(arrow::array::StringViewArray::from(vec![
            "New York", "London", "Paris", "Tokyo", "Berlin", "Sydney", "Madrid", "Rome",
        ]));

        assert!(col_name.insert(batch_id, array_name.clone()).is_ok());
        assert!(col_city.insert(batch_id, array_city.clone()).is_ok());

        // build parquet metadata for predicate construction
        let tmp_meta = tempfile::NamedTempFile::new().unwrap();
        let mut writer =
            ArrowWriter::try_new(tmp_meta.reopen().unwrap(), Arc::clone(&schema), None).unwrap();
        let batch =
            RecordBatch::try_new(Arc::clone(&schema), vec![array_name, array_city]).unwrap();
        writer.write(&batch).unwrap();
        writer.close().unwrap();
        let file_reader = std::fs::File::open(tmp_meta.path()).unwrap();
        let metadata = ArrowReaderMetadata::load(&file_reader, ArrowReaderOptions::new()).unwrap();

        // expression: name = "Bob" OR city = "Tokyo"
        let expr_name: Arc<dyn PhysicalExpr> = Arc::new(BinaryExpr::new(
            Arc::new(Column::new("name", 0)),
            Operator::Eq,
            Arc::new(Literal::new(ScalarValue::Utf8(Some("Bob".to_string())))),
        ));
        let expr_city: Arc<dyn PhysicalExpr> = Arc::new(BinaryExpr::new(
            Arc::new(Column::new("city", 1)),
            Operator::Eq,
            Arc::new(Literal::new(ScalarValue::Utf8(Some("Tokyo".to_string())))),
        ));
        let expr: Arc<dyn PhysicalExpr> =
            Arc::new(BinaryExpr::new(expr_name, Operator::Or, expr_city));

        let adapter_factory = Arc::new(DefaultSchemaAdapterFactory);
        let builder = FilterCandidateBuilder::new(
            expr,
            Arc::clone(&schema),
            Arc::clone(&schema),
            adapter_factory,
        );
        let candidate = builder.build(metadata.metadata()).unwrap().unwrap();
        let projection = candidate.projection(metadata.metadata());
        let mut predicate = LiquidPredicate::try_new(candidate, projection).unwrap();

        let selection = BooleanBuffer::new_set(batch_size);
        let result = row_group
            .evaluate_selection_with_predicate(batch_id, &selection, &mut predicate)
            .unwrap()
            .unwrap();

        // Expected: row 1 (name="Bob"), row 3 (city="Tokyo") -> indices 1, 3
        let expected = BooleanBuffer::collect_bool(batch_size, |i| i == 1 || i == 3).into();
        assert_eq!(result, expected);
    }
}
