//! Cached data in the cache.

use std::{fmt::Display, sync::Arc};

use arrow::{
    array::{ArrayRef, BooleanArray},
    buffer::BooleanBuffer,
};
use arrow_schema::ArrowError;
use datafusion::physical_plan::PhysicalExpr;

use crate::{cache::io_state::GetLiquidArrayState, liquid_array::LiquidHybridArrayRef};
use crate::{
    cache::{
        EntryID,
        io_state::{GetArrowArrayState, GetWithPredicateState, GetWithSelectionState, SansIo},
    },
    liquid_array::LiquidArrayRef,
};

/// A wrapper around the actual data in the cache.
#[derive(Debug)]
pub struct CachedData<'a> {
    data: CachedBatch,
    id: EntryID,
    io_context: &'a dyn super::core::IoContext,
}

/// The result of predicate pushdown.
#[derive(Debug, PartialEq)]
pub enum GetWithPredicateResult {
    /// The predicate is evaluated on the filtered data and the result is a boolean buffer.
    Evaluated(BooleanArray),

    /// The predicate is not evaluated but data is filtered.
    Filtered(ArrayRef),
}

impl<'a> CachedData<'a> {
    pub(crate) fn new(
        data: CachedBatch,
        id: EntryID,
        io_context: &'a dyn super::core::IoContext,
    ) -> Self {
        Self {
            data,
            id,
            io_context,
        }
    }

    /// Build a sans-IO state machine to obtain an Arrow `ArrayRef` with selection pushdown.
    pub fn get_with_selection<'selection>(
        &self,
        selection: &'selection BooleanBuffer,
    ) -> SansIo<Result<ArrayRef, ArrowError>, GetWithSelectionState<'selection>> {
        match &self.data {
            CachedBatch::MemoryArrow(array) => {
                let selection_array = BooleanArray::new(selection.clone(), None);
                let result = arrow::compute::filter(array, &selection_array);
                SansIo::Ready(result)
            }
            CachedBatch::MemoryLiquid(array) => {
                let filtered = array.filter_to_arrow(selection);
                SansIo::Ready(Ok(filtered))
            }
            CachedBatch::DiskLiquid => GetWithSelectionState::pending_liquid(
                self.io_context.entry_liquid_path(&self.id),
                selection,
                self.io_context.get_compressor_for_entry(&self.id),
            ),
            CachedBatch::MemoryHybridLiquid(array) => {
                let filtered = array.filter_to_arrow(selection);
                match filtered {
                    Ok(array) => SansIo::Ready(Ok(array)),
                    Err(io_request) => GetWithSelectionState::pending_hybrid_liquid(
                        io_request,
                        selection,
                        array.clone(),
                    ),
                }
            }
            CachedBatch::DiskArrow => GetWithSelectionState::pending_arrow(
                self.io_context.entry_arrow_path(&self.id),
                selection,
            ),
        }
    }

    /// Build a sans-IO state machine to obtain an Arrow `ArrayRef`.
    ///
    /// This does not perform IO itself. Instead, it may request IO via [`IoStateMachine::try_get`].
    /// The caller should fulfill the IO request and call [`IoStateMachine::feed`] with the bytes,
    /// then consume the state machine with [`IoStateMachine::try_get`].
    pub fn get_arrow_array(&self) -> SansIo<ArrayRef, GetArrowArrayState> {
        match &self.data {
            CachedBatch::MemoryArrow(array) => SansIo::Ready(array.clone()),
            CachedBatch::MemoryLiquid(array) => SansIo::Ready(array.to_best_arrow_array()),
            CachedBatch::DiskLiquid => {
                let path = self.io_context.entry_liquid_path(&self.id);
                let compressor_states = self.io_context.get_compressor_for_entry(&self.id);
                GetArrowArrayState::pending_liquid(path, compressor_states)
            }
            CachedBatch::MemoryHybridLiquid(array) => {
                let arrow_array = array.to_arrow_array();
                match arrow_array {
                    Ok(array) => SansIo::Ready(array),
                    Err(io_request) => {
                        GetArrowArrayState::pending_hybrid_liquid(io_request, array.clone())
                    }
                }
            }
            CachedBatch::DiskArrow => {
                let path = self.io_context.entry_arrow_path(&self.id);
                GetArrowArrayState::pending_arrow(path)
            }
        }
    }

    #[cfg(test)]
    pub(crate) fn raw_data(&self) -> &CachedBatch {
        &self.data
    }

    /// Try to read the liquid array from the cached data.
    /// Return None if the cached data is not a liquid array.
    ///
    /// Build a sans-IO state machine to obtain a `LiquidArrayRef` if the cached data is liquid.
    pub fn try_read_liquid(&self) -> SansIo<Option<LiquidArrayRef>, GetLiquidArrayState> {
        match &self.data {
            CachedBatch::MemoryLiquid(array) => SansIo::Ready(Some(array.clone())),
            CachedBatch::DiskLiquid => GetLiquidArrayState::pending_liquid(
                self.io_context.entry_liquid_path(&self.id),
                self.io_context.get_compressor_for_entry(&self.id),
            ),
            CachedBatch::MemoryHybridLiquid(array) => {
                GetLiquidArrayState::pending_hybrid_liquid(array.to_liquid(), array.clone())
            }
            CachedBatch::DiskArrow | CachedBatch::MemoryArrow(_) => SansIo::Ready(None),
        }
    }

    /// Get the arrow array with predicate pushdown.
    ///
    /// The `selection` is applied **before** predicate evaluation.
    /// For example, if the selection is `[true, true, false, true, false]`,
    /// The return boolean buffer will be length of 3, each corresponding to the selected rows.
    ///
    /// Returns:
    /// - `PredicatePushdownResult::Evaluated(buffer)`: the predicate is evaluated on the filtered data and the result is a boolean buffer. This only occurs for Liquid-backed string arrays that support predicate evaluation.
    /// - `PredicatePushdownResult::Filtered(array)`: the predicate is not evaluated (e.g., when data is Arrow-backed, predicate is not supported, or an error happens) but data is filtered.
    ///
    /// Build a sans-IO state machine to evaluate a predicate with selection pushdown.
    pub fn get_with_predicate<'predicate, 'selection>(
        &self,
        selection: &'selection BooleanBuffer,
        predicate: &'predicate Arc<dyn PhysicalExpr>,
    ) -> SansIo<GetWithPredicateResult, GetWithPredicateState<'predicate, 'selection>> {
        match &self.data {
            CachedBatch::MemoryArrow(array) => {
                let selection_array = BooleanArray::new(selection.clone(), None);
                let selected = arrow::compute::filter(array, &selection_array).unwrap();
                SansIo::Ready(GetWithPredicateResult::Filtered(selected))
            }
            CachedBatch::DiskArrow => GetWithPredicateState::pending_arrow(
                self.io_context.entry_arrow_path(&self.id),
                selection,
                predicate,
            ),
            CachedBatch::MemoryLiquid(array) => {
                let result = match array.try_eval_predicate(predicate, selection) {
                    Some(buf) => GetWithPredicateResult::Evaluated(buf),
                    None => {
                        let filtered = array.filter_to_arrow(selection);
                        GetWithPredicateResult::Filtered(filtered)
                    }
                };
                SansIo::Ready(result)
            }
            CachedBatch::DiskLiquid => {
                let compressor_states = self.io_context.get_compressor_for_entry(&self.id);
                GetWithPredicateState::pending_liquid(
                    self.io_context.entry_liquid_path(&self.id),
                    selection,
                    predicate,
                    compressor_states,
                )
            }
            CachedBatch::MemoryHybridLiquid(array) => {
                match array.try_eval_predicate(predicate, selection) {
                    Ok(Some(buf)) => SansIo::Ready(GetWithPredicateResult::Evaluated(buf)),
                    Ok(None) => {
                        let filtered = array.filter_to_arrow(selection);
                        match filtered {
                            Ok(array) => SansIo::Ready(GetWithPredicateResult::Filtered(array)),
                            Err(io_request) => GetWithPredicateState::pending_hybrid_liquid(
                                io_request,
                                selection,
                                predicate,
                                array.clone(),
                            ),
                        }
                    }
                    Err(io_request) => GetWithPredicateState::pending_hybrid_liquid(
                        io_request,
                        selection,
                        predicate,
                        array.clone(),
                    ),
                }
            }
        }
    }
}

/// Cached batch.
#[derive(Debug, Clone)]
pub enum CachedBatch {
    /// Cached batch in memory as Arrow array.
    MemoryArrow(ArrayRef),
    /// Cached batch in memory as liquid array.
    MemoryLiquid(LiquidArrayRef),
    /// Cached batch in memory as hybrid liquid array.
    MemoryHybridLiquid(LiquidHybridArrayRef),
    /// Cached batch on disk as liquid array.
    DiskLiquid,
    /// Cached batch on disk as Arrow array.
    DiskArrow,
}

impl CachedBatch {
    /// Get the memory usage of the cached batch.
    pub fn memory_usage_bytes(&self) -> usize {
        match self {
            Self::MemoryArrow(array) => array.get_array_memory_size(),
            Self::MemoryLiquid(array) => array.get_array_memory_size(),
            Self::MemoryHybridLiquid(array) => array.get_array_memory_size(),
            Self::DiskLiquid => 0,
            Self::DiskArrow => 0,
        }
    }

    /// Get the reference count of the cached batch.
    pub fn reference_count(&self) -> usize {
        match self {
            Self::MemoryArrow(array) => Arc::strong_count(array),
            Self::MemoryLiquid(array) => Arc::strong_count(array),
            Self::MemoryHybridLiquid(array) => Arc::strong_count(array),
            Self::DiskLiquid => 0,
            Self::DiskArrow => 0,
        }
    }
}

impl Display for CachedBatch {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::MemoryArrow(_) => write!(f, "MemoryArrow"),
            Self::MemoryLiquid(_) => write!(f, "MemoryLiquid"),
            Self::MemoryHybridLiquid(_) => write!(f, "MemoryHybridLiquid"),
            Self::DiskLiquid => write!(f, "DiskLiquid"),
            Self::DiskArrow => write!(f, "DiskArrow"),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::path::{Path, PathBuf};
    use std::sync::RwLock;

    use crate::cache::utils::arrow_to_bytes;
    use crate::cache::{IoContext, LiquidCompressorStates, transcode_liquid_inner};
    use crate::liquid_array::LiquidByteArray;

    use super::*;
    use crate::cache::io_state::{IoRequest, IoStateMachine, TryGet};
    use ahash::HashMap;
    use arrow::array::{Array, AsArray, Int64Array, RecordBatch, StringArray};
    use arrow::compute as compute_kernels;
    use arrow_schema::{DataType, Field, Schema};
    use bytes::Bytes;
    use datafusion::logical_expr::{ColumnarValue, Operator};
    use datafusion::physical_plan::expressions::{BinaryExpr, Column, Literal};
    use datafusion::scalar::ScalarValue;
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    #[derive(Debug)]
    struct MockIoWorker {
        compressor_states: Arc<LiquidCompressorStates>,
        base_dir: PathBuf,
        store: RwLock<HashMap<PathBuf, Bytes>>,
    }

    impl MockIoWorker {
        fn read_entry(&self, request: &IoRequest) -> Result<Bytes, std::io::Error> {
            let bytes = self
                .store
                .read()
                .unwrap()
                .get(&request.path)
                .unwrap()
                .clone();
            Ok(bytes.clone())
        }

        fn blocking_evict_arrow_to_disk(
            &self,
            entry_id: &EntryID,
            array: &ArrayRef,
        ) -> Result<usize, ArrowError> {
            let bytes = arrow_to_bytes(array)?;
            let path = self.entry_arrow_path(entry_id);
            let len = bytes.len();
            self.store.write().unwrap().insert(path, bytes.clone());
            Ok(len)
        }

        fn blocking_evict_liquid_to_disk(
            &self,
            entry_id: &EntryID,
            liquid_array: &LiquidArrayRef,
        ) -> Result<usize, std::io::Error> {
            let path = self.entry_liquid_path(entry_id);
            let bytes = liquid_array.to_bytes();
            self.store
                .write()
                .unwrap()
                .insert(path, Bytes::from(bytes.to_vec()));
            Ok(bytes.len())
        }
    }

    impl IoContext for MockIoWorker {
        fn base_dir(&self) -> &Path {
            &self.base_dir
        }

        fn get_compressor_for_entry(&self, _entry_id: &EntryID) -> Arc<LiquidCompressorStates> {
            self.compressor_states.clone()
        }

        fn entry_arrow_path(&self, entry_id: &EntryID) -> PathBuf {
            self.base_dir
                .join(format!("{:016x}.arrow", usize::from(*entry_id)))
        }

        fn entry_liquid_path(&self, entry_id: &EntryID) -> PathBuf {
            self.base_dir
                .join(format!("{:016x}.liquid", usize::from(*entry_id)))
        }
    }

    fn io_worker(compressor: Option<Arc<fsst::Compressor>>) -> (tempfile::TempDir, MockIoWorker) {
        let tmp = tempfile::tempdir().unwrap();
        let compressor_state = match compressor {
            Some(compressor) => {
                Arc::new(LiquidCompressorStates::new_with_fsst_compressor(compressor))
            }
            None => Arc::new(LiquidCompressorStates::default()),
        };
        let io = MockIoWorker {
            compressor_states: compressor_state,
            base_dir: tmp.path().to_path_buf(),
            store: RwLock::new(HashMap::default()),
        };
        (tmp, io)
    }

    #[test]
    fn test_get_arrow_array_memory_arrow() {
        let array: ArrayRef = Arc::new(Int64Array::from_iter_values(0..8));
        let id = EntryID::from(1usize);
        let (_tmp, io) = io_worker(None);
        let cached = CachedData::new(CachedBatch::MemoryArrow(array.clone()), id, &io);

        let SansIo::Ready(out) = cached.get_arrow_array() else {
            panic!("should be ready");
        };
        assert_eq!(out.as_ref(), array.as_ref());
    }

    #[test]
    fn test_try_read_liquid() {
        // Build a small liquid string array
        let input = StringArray::from(vec!["a", "b", "a", "c"]);
        let (compressor, etc) = crate::liquid_array::LiquidByteArray::train_from_arrow(&input);
        let liquid_ref: LiquidArrayRef = Arc::new(etc);

        let id = EntryID::from(2usize);
        let (_tmp, io) = io_worker(Some(compressor));
        let in_memory = CachedData::new(CachedBatch::MemoryLiquid(liquid_ref.clone()), id, &io);
        let on_disk = CachedData::new(CachedBatch::DiskLiquid, id, &io);
        io.blocking_evict_liquid_to_disk(&id, &liquid_ref).unwrap();
        let arrow_input: ArrayRef = Arc::new(input);

        {
            let SansIo::Ready(Some(liquid)) = in_memory.try_read_liquid() else {
                panic!("should be liquid");
            };

            assert_eq!(liquid.to_arrow_array().as_ref(), arrow_input.as_ref());
        }

        {
            let SansIo::Pending((mut state, io_request)) = on_disk.try_read_liquid() else {
                panic!("should be pending");
            };

            state.feed(io.read_entry(&io_request).unwrap());
            let TryGet::Ready(liquid) = state.try_get() else {
                panic!("should be ready");
            };
            assert_eq!(liquid.to_arrow_array().as_ref(), arrow_input.as_ref());
        }

        // Try if non-liquid batch can be read as liquid
        let on_disk_non_liquid = CachedData::new(CachedBatch::DiskArrow, id, &io);
        let SansIo::Ready(None) = on_disk_non_liquid.try_read_liquid() else {
            panic!("should be none");
        };
    }

    #[test]
    fn test_get_with_selection_memory() {
        let array: ArrayRef = Arc::new(Int64Array::from_iter_values(0..10));
        let selection = BooleanBuffer::from((0..10).map(|i| i % 2 == 0).collect::<Vec<_>>());

        let id = EntryID::from(3usize);
        let (_tmp, io) = io_worker(None);
        let cached = CachedData::new(CachedBatch::MemoryArrow(array.clone()), id, &io);

        let SansIo::Ready(Ok(filtered)) = cached.get_with_selection(&selection) else {
            panic!("should be ready");
        };
        let selection_array = BooleanArray::new(selection.clone(), None);
        let expected = compute_kernels::filter(&array, &selection_array).unwrap();
        assert_eq!(filtered.as_ref(), expected.as_ref());

        let liquid_array =
            transcode_liquid_inner(&array, &LiquidCompressorStates::default()).unwrap();
        let cached = CachedData::new(CachedBatch::MemoryLiquid(liquid_array), id, &io);

        let SansIo::Ready(Ok(filtered)) = cached.get_with_selection(&selection) else {
            panic!("should be ready");
        };
        assert_eq!(filtered.as_ref(), expected.as_ref());
    }

    #[test]
    fn test_get_with_selection_disk() {
        let array: ArrayRef = Arc::new(Int64Array::from_iter_values(0..10));
        let selection = BooleanBuffer::from((0..10).map(|i| i % 2 == 0).collect::<Vec<_>>());

        let id = EntryID::from(3usize);
        let (_tmp, io) = io_worker(None);
        let cached = CachedData::new(CachedBatch::DiskArrow, id, &io);
        io.blocking_evict_arrow_to_disk(&id, &array).unwrap();

        let test_get = |cached: &CachedData| {
            let SansIo::Pending((mut state, io_request)) = cached.get_with_selection(&selection)
            else {
                panic!("should be pending");
            };

            state.feed(io.read_entry(&io_request).unwrap());
            let TryGet::Ready(Ok(filtered)) = state.try_get() else {
                panic!("should be ready");
            };
            let selection_array = BooleanArray::new(selection.clone(), None);
            let expected = compute_kernels::filter(&array, &selection_array).unwrap();
            assert_eq!(filtered.as_ref(), expected.as_ref());
        };
        test_get(&cached);

        let liquid_array =
            transcode_liquid_inner(&array, &LiquidCompressorStates::default()).unwrap();
        let cached = CachedData::new(CachedBatch::DiskLiquid, id, &io);
        io.blocking_evict_liquid_to_disk(&id, &liquid_array)
            .unwrap();
        test_get(&cached);
    }

    #[test]
    fn test_get_with_predicate_arrow_returns_filtered() {
        // Build a simple Arrow string array
        let data = StringArray::from(vec![Some("a"), Some(""), None, Some("b")]);
        let arrow_array: ArrayRef = Arc::new(data);

        // Build predicate: col == ""
        let expr: Arc<dyn PhysicalExpr> = Arc::new(BinaryExpr::new(
            Arc::new(Column::new("col", 0)),
            Operator::Eq,
            Arc::new(Literal::new(ScalarValue::Utf8(Some("".to_string())))),
        ));

        // Cached as Arrow memory
        let id = EntryID::from(42usize);
        let (_tmp, io) = io_worker(None);
        let cached = CachedData::new(CachedBatch::MemoryArrow(arrow_array.clone()), id, &io);

        // Selection: all-true
        let selection = BooleanBuffer::from(vec![true; arrow_array.len()]);

        // Expect Filtered result for Arrow-backed data, not Evaluated
        let SansIo::Ready(result) = cached.get_with_predicate(&selection, &expr) else {
            panic!("expected immediate result for in-memory arrow");
        };
        match result {
            GetWithPredicateResult::Filtered(arr) => {
                assert_eq!(arr.len(), arrow_array.len());
            }
            GetWithPredicateResult::Evaluated(_) => {
                panic!("Arrow-backed get_with_predicate must not return Evaluated")
            }
        }
    }

    fn test_string_predicate(string_array: &StringArray, expr: &Arc<dyn PhysicalExpr>) {
        let (compressor, liquid) = LiquidByteArray::train_from_arrow(string_array);
        let liquid_ref: LiquidArrayRef = Arc::new(liquid);

        let id = EntryID::from(9usize);
        let (_tmp, io) = io_worker(Some(compressor));
        let arrow_array: ArrayRef = Arc::new(string_array.clone());
        let memory_liquid = CachedData::new(CachedBatch::MemoryLiquid(liquid_ref.clone()), id, &io);
        let disk_liquid = CachedData::new(CachedBatch::DiskLiquid, id, &io);
        let memory_arrow = CachedData::new(CachedBatch::MemoryArrow(arrow_array.clone()), id, &io);
        let disk_arrow = CachedData::new(CachedBatch::DiskArrow, id, &io);
        io.blocking_evict_liquid_to_disk(&id, &liquid_ref).unwrap();
        io.blocking_evict_arrow_to_disk(&id, &arrow_array).unwrap();

        let mut seed_rng = StdRng::seed_from_u64(42);
        for _i in 0..100 {
            let selection = BooleanBuffer::from_iter(
                (0..string_array.len()).map(|_| seed_rng.random_bool(0.5)),
            );

            let (eval_expected, filter_expected) = {
                let selection = BooleanArray::new(selection.clone(), None);
                let filtered = arrow::compute::filter(&string_array, &selection).unwrap();
                let record_batch = RecordBatch::try_new(
                    Arc::new(Schema::new(vec![Field::new("col", DataType::Utf8, true)])),
                    vec![filtered.clone()],
                )
                .unwrap();
                let evaluated = expr.evaluate(&record_batch).unwrap();
                let filter_eval = match evaluated {
                    ColumnarValue::Array(array) => array,
                    ColumnarValue::Scalar(_) => panic!("expected array, got scalar"),
                };
                (filter_eval.as_boolean().clone(), filtered)
            };

            let expect_filtered = |result: GetWithPredicateResult| match result {
                GetWithPredicateResult::Filtered(array) => {
                    assert_eq!(array.as_ref(), filter_expected.as_ref());
                }
                other => panic!("expected filtered, got {other:?}"),
            };
            let expect_evaluated = |result: GetWithPredicateResult| match result {
                GetWithPredicateResult::Evaluated(array) => {
                    assert_eq!(array, eval_expected);
                }
                other => panic!("expected evaluated, got {other:?}"),
            };

            // memory arrow
            {
                let SansIo::Ready(result) = memory_arrow.get_with_predicate(&selection, expr)
                else {
                    panic!("should be ready");
                };
                expect_filtered(result);
            }

            // memory liquid
            {
                let SansIo::Ready(result) = memory_liquid.get_with_predicate(&selection, expr)
                else {
                    panic!("should be ready");
                };
                expect_evaluated(result);
            }

            // disk arrow
            {
                let SansIo::Pending((mut state, io_request)) =
                    disk_arrow.get_with_predicate(&selection, expr)
                else {
                    panic!("should be ready");
                };

                state.feed(io.read_entry(&io_request).unwrap());
                let TryGet::Ready(result) = state.try_get() else {
                    panic!("should be ready");
                };
                expect_filtered(result);
            }

            // disk liquid
            {
                let SansIo::Pending((mut state, io_request)) =
                    disk_liquid.get_with_predicate(&selection, expr)
                else {
                    panic!("should be ready");
                };

                state.feed(io.read_entry(&io_request).unwrap());
                let TryGet::Ready(result) = state.try_get() else {
                    panic!("should be ready");
                };
                expect_evaluated(result);
            }
        }
    }

    #[test]
    fn test_get_with_predicate_evaluated_for_strings() {
        let data = StringArray::from(vec![
            Some("apple"),
            Some("banana"),
            None,
            Some("apple"),
            None,
            Some("cherry"),
        ]);

        let expr: Arc<dyn PhysicalExpr> = Arc::new(BinaryExpr::new(
            Arc::new(Column::new("col", 0)),
            Operator::Eq,
            Arc::new(Literal::new(ScalarValue::Utf8(Some("apple".to_string())))),
        ));

        test_string_predicate(&data, &expr);
    }

    #[test]
    fn test_cached_batch_memory_usage_and_refcount_arrow() {
        let array: ArrayRef = Arc::new(Int64Array::from_iter_values(0..16));
        let batch = CachedBatch::MemoryArrow(array.clone());

        assert_eq!(batch.memory_usage_bytes(), array.get_array_memory_size());
        assert_eq!(batch.reference_count(), Arc::strong_count(&array));
    }
}
