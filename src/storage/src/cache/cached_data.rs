//! Cached data in the cache.

use std::{fmt::Display, sync::Arc};

use arrow::{
    array::{ArrayRef, BooleanArray},
    buffer::BooleanBuffer,
};
use arrow_schema::ArrowError;
use datafusion::physical_plan::PhysicalExpr;

use crate::cache::LiquidCompressorStates;
use crate::{cache::EntryID, liquid_array::LiquidArrayRef};
use bytes::Bytes;
use std::path::PathBuf;

/// A wrapper around the actual data in the cache.
#[derive(Debug)]
pub struct CachedData<'a> {
    data: CachedBatch,
    id: EntryID,
    io_context: &'a dyn super::core::IoContext,
}

/// The result of predicate pushdown.
#[derive(Debug, PartialEq)]
pub enum PredicatePushdownResult {
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
    ) -> SansIo<Result<ArrayRef, ArrowError>, GetWithSelectionSansIo<'selection>> {
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
            CachedBatch::DiskLiquid => {
                let pending = PendingIo::Liquid {
                    path: self.io_context.entry_liquid_path(&self.id),
                    compressor_states: self.io_context.get_compressor_for_entry(&self.id),
                };
                let io_request = pending.as_io_request();
                SansIo::Pending((
                    GetWithSelectionSansIo {
                        state: GetWithSelectionState::NeedBytes { selection, pending },
                    },
                    io_request,
                ))
            }
            CachedBatch::DiskArrow => {
                let pending = PendingIo::Arrow {
                    path: self.io_context.entry_arrow_path(&self.id),
                };
                let io_request = pending.as_io_request();
                SansIo::Pending((
                    GetWithSelectionSansIo {
                        state: GetWithSelectionState::NeedBytes { selection, pending },
                    },
                    io_request,
                ))
            }
        }
    }

    /// Build a sans-IO state machine to obtain an Arrow `ArrayRef`.
    ///
    /// This does not perform IO itself. Instead, it may request IO via [`GetArrowArraySansIo::need_io`].
    /// The caller should fulfill the IO request and call [`GetArrowArraySansIo::feed`] with the bytes,
    /// then consume the state machine with [`GetArrowArraySansIo::into_output`].
    pub fn get_arrow_array_sans_io(&self) -> SansIo<ArrayRef, GetArrowArrayState> {
        match &self.data {
            CachedBatch::MemoryArrow(array) => SansIo::Ready(array.clone()),
            CachedBatch::MemoryLiquid(array) => SansIo::Ready(array.to_best_arrow_array()),
            CachedBatch::DiskLiquid => {
                let path = self.io_context.entry_liquid_path(&self.id);
                let compressor_states = self.io_context.get_compressor_for_entry(&self.id);
                SansIo::Pending(GetArrowArrayState::new_liquid(path, compressor_states))
            }
            CachedBatch::DiskArrow => {
                let path = self.io_context.entry_arrow_path(&self.id);
                SansIo::Pending(GetArrowArrayState::new_arrow(path))
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
            CachedBatch::DiskLiquid => SansIo::Pending(GetLiquidArrayState::new(
                self.io_context.entry_liquid_path(&self.id),
                self.io_context.get_compressor_for_entry(&self.id),
            )),
            _ => SansIo::Ready(None),
        }
    }

    /// Get the arrow array with predicate pushdown.
    ///
    /// The `selection` is applied **before** predicate evaluation.
    /// For example, if the selection is `[true, true, false, true, false]`,
    /// The return boolean buffer will be length of 3, each corresponding to the selected rows.
    ///
    /// Returns:
    /// - `PredicatePushdownResult::Evaluated(buffer)`: the predicate is evaluated on the filtered data and the result is a boolean buffer.
    /// - `PredicatePushdownResult::Filtered(array)`: the predicate is not evaluated (e.g., predicate is not supported or error happens) but data is filtered.
    ///
    /// Build a sans-IO state machine to evaluate a predicate with selection pushdown.
    pub fn get_with_predicate<'predicate, 'selection>(
        &self,
        selection: &'predicate BooleanBuffer,
        predicate: &'predicate Arc<dyn PhysicalExpr>,
    ) -> SansIo<Result<PredicatePushdownResult, ArrowError>, GetWithPredicateState<'predicate>>
    {
        match &self.data {
            CachedBatch::MemoryArrow(array) => {
                let selection_array = BooleanArray::new(selection.clone(), None);
                let selected = arrow::compute::filter(array, &selection_array);
                let result = selected.map(PredicatePushdownResult::Filtered);
                SansIo::Ready(result)
            }
            CachedBatch::DiskArrow => {
                let pending = PendingIo::Arrow {
                    path: self.io_context.entry_arrow_path(&self.id),
                };
                let io_request = pending.as_io_request();
                SansIo::Pending((
                    GetWithPredicateState {
                        state: GetWithPredicateStateInner::NeedBytes {
                            selection,
                            predicate,
                            pending,
                        },
                    },
                    io_request,
                ))
            }
            CachedBatch::MemoryLiquid(array) => {
                let result = match array.try_eval_predicate(predicate, selection) {
                    Ok(Some(buf)) => Ok(PredicatePushdownResult::Evaluated(buf)),
                    Ok(None) => {
                        let filtered = array.filter_to_arrow(selection);
                        Ok(PredicatePushdownResult::Filtered(filtered))
                    }
                    Err(e) => Err(e),
                };
                SansIo::Ready(result)
            }
            CachedBatch::DiskLiquid => {
                let pending = PendingIo::Liquid {
                    path: self.io_context.entry_liquid_path(&self.id),
                    compressor_states: self.io_context.get_compressor_for_entry(&self.id),
                };
                let io_request = pending.as_io_request();
                SansIo::Pending((
                    GetWithPredicateState {
                        state: GetWithPredicateStateInner::NeedBytes {
                            selection,
                            predicate,
                            pending,
                        },
                    },
                    io_request,
                ))
            }
        }
    }
}

/// The result of a sans-IO operation.
#[derive(Debug, Clone)]
pub enum TryGet<T, M> {
    /// The output is ready.
    Ready(T),
    /// The output is not ready. The second element is the state machine and the third element is the IO request.
    NeedData((M, IoRequest)),
}

/// The result of a sans-IO operation.
#[derive(Debug)]
pub enum SansIo<T, M> {
    /// The output is ready.
    Ready(T),
    /// The output needs IO.
    Pending((M, IoRequest)),
}

/// A state machine that can be used to perform a sans-IO operation.
pub trait IoStateMachine: Sized {
    /// The output type of the state machine.
    type Output;

    /// Attempt to get the output. If IO is needed, returns an IO request.
    fn try_get(self) -> TryGet<Self::Output, Self>;

    /// Feed the state machine with IO bytes previously requested by `try_get`.
    fn feed(&mut self, data: Bytes);
}

/// Description of an IO request to satisfy a sans-IO operation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IoRequest {
    /// The path to read.
    pub path: PathBuf,
}

/// Reusable description of pending IO for Arrow or Liquid on-disk formats.
#[derive(Debug, Clone)]
enum PendingIo {
    Arrow {
        path: PathBuf,
    },
    Liquid {
        path: PathBuf,
        compressor_states: Arc<LiquidCompressorStates>,
    },
}

impl PendingIo {
    fn as_io_request(&self) -> IoRequest {
        match self {
            PendingIo::Arrow { path } => IoRequest { path: path.clone() },
            PendingIo::Liquid { path, .. } => IoRequest { path: path.clone() },
        }
    }

    fn decode_arrow(&self, data: Bytes) -> ArrayRef {
        match self {
            PendingIo::Arrow { .. } => {
                let cursor = std::io::Cursor::new(data.to_vec());
                let mut reader = arrow::ipc::reader::StreamReader::try_new(cursor, None)
                    .expect("invalid arrow stream bytes");
                let batch = reader
                    .next()
                    .expect("empty arrow stream")
                    .expect("failed to read arrow stream");
                batch.column(0).clone()
            }
            PendingIo::Liquid {
                compressor_states, ..
            } => {
                let compressor = compressor_states.fsst_compressor();
                let liquid = crate::liquid_array::ipc::read_from_bytes(
                    data,
                    &crate::liquid_array::ipc::LiquidIPCContext::new(compressor),
                );
                liquid.to_arrow_array()
            }
        }
    }

    fn decode_liquid(&self, data: Bytes) -> LiquidArrayRef {
        match self {
            PendingIo::Liquid {
                compressor_states, ..
            } => {
                let compressor = compressor_states.fsst_compressor();
                crate::liquid_array::ipc::read_from_bytes(
                    data,
                    &crate::liquid_array::ipc::LiquidIPCContext::new(compressor),
                )
            }
            PendingIo::Arrow { .. } => panic!("decode_liquid called on Arrow pending io"),
        }
    }
}

/// State machine to obtain an Arrow `ArrayRef` without performing IO internally.
#[derive(Debug)]
pub struct GetArrowArrayState {
    state: GetArrowArrayStateInner,
}

impl GetArrowArrayState {
    fn new_arrow(path: PathBuf) -> (Self, IoRequest) {
        let pending = PendingIo::Arrow { path };
        let io_request = pending.as_io_request();
        let state = Self {
            state: GetArrowArrayStateInner::NeedBytes(pending),
        };
        (state, io_request)
    }

    fn new_liquid(
        path: PathBuf,
        compressor_states: Arc<LiquidCompressorStates>,
    ) -> (Self, IoRequest) {
        let pending = PendingIo::Liquid {
            path,
            compressor_states,
        };
        let io_request = pending.as_io_request();
        let state = Self {
            state: GetArrowArrayStateInner::NeedBytes(pending),
        };
        (state, io_request)
    }
}

#[derive(Debug)]
enum GetArrowArrayStateInner {
    Ready(ArrayRef),
    NeedBytes(PendingIo),
}

impl IoStateMachine for GetArrowArrayState {
    type Output = ArrayRef;

    fn try_get(self) -> TryGet<ArrayRef, Self> {
        match self.state {
            GetArrowArrayStateInner::Ready(array) => TryGet::Ready(array),
            GetArrowArrayStateInner::NeedBytes(ref p) => {
                let io_request = p.as_io_request();
                TryGet::NeedData((self, io_request))
            }
        }
    }

    fn feed(&mut self, data: Bytes) {
        match &mut self.state {
            GetArrowArrayStateInner::Ready(_) => {}
            GetArrowArrayStateInner::NeedBytes(pending_state) => {
                let array = pending_state.decode_arrow(data);
                self.state = GetArrowArrayStateInner::Ready(array);
            }
        }
    }
}

/// State machine: selection pushdown sans-IO
#[derive(Debug)]
pub struct GetWithSelectionSansIo<'selection> {
    state: GetWithSelectionState<'selection>,
}

#[derive(Debug)]
enum GetWithSelectionState<'selection> {
    NeedBytes {
        selection: &'selection BooleanBuffer,
        pending: PendingIo,
    },
    Done(Result<ArrayRef, ArrowError>),
}

impl<'a> IoStateMachine for GetWithSelectionSansIo<'a> {
    type Output = Result<ArrayRef, ArrowError>;

    fn try_get(self) -> TryGet<Result<ArrayRef, ArrowError>, Self> {
        match self.state {
            GetWithSelectionState::Done(r) => TryGet::Ready(r),
            GetWithSelectionState::NeedBytes { ref pending, .. } => {
                let io_request = pending.as_io_request();
                TryGet::NeedData((self, io_request))
            }
        }
    }

    fn feed(&mut self, data: Bytes) {
        match &mut self.state {
            GetWithSelectionState::Done(_) => {}
            GetWithSelectionState::NeedBytes { pending, selection } => match pending {
                PendingIo::Arrow { .. } => {
                    let array = pending.decode_arrow(data);
                    let selection_array = BooleanArray::new(selection.clone(), None);
                    let filtered = arrow::compute::filter(&array, &selection_array);
                    self.state = GetWithSelectionState::Done(filtered);
                }
                PendingIo::Liquid { .. } => {
                    let liquid = pending.decode_liquid(data);
                    let filtered = liquid.filter_to_arrow(selection);
                    self.state = GetWithSelectionState::Done(Ok(filtered));
                }
            },
        }
    }
}

/// State machine: read `LiquidArrayRef` sans-IO
#[derive(Debug)]
pub struct GetLiquidArrayState {
    state: GetLiquidArrayStateInner,
}

impl GetLiquidArrayState {
    fn new(path: PathBuf, compressor_states: Arc<LiquidCompressorStates>) -> (Self, IoRequest) {
        let pending = PendingIo::Liquid {
            path,
            compressor_states,
        };
        let io_request = pending.as_io_request();
        let state = Self {
            state: GetLiquidArrayStateInner::NeedBytes(pending),
        };
        (state, io_request)
    }
}

#[derive(Debug)]
enum GetLiquidArrayStateInner {
    NeedBytes(PendingIo),
    Done(LiquidArrayRef),
}

impl IoStateMachine for GetLiquidArrayState {
    type Output = LiquidArrayRef;

    fn try_get(self) -> TryGet<LiquidArrayRef, Self> {
        match &self.state {
            GetLiquidArrayStateInner::Done(liq) => TryGet::Ready(liq.clone()),
            GetLiquidArrayStateInner::NeedBytes(p) => {
                let io_request = p.as_io_request();
                TryGet::NeedData((self, io_request))
            }
        }
    }

    fn feed(&mut self, data: Bytes) {
        match &mut self.state {
            GetLiquidArrayStateInner::Done(_) => {}
            GetLiquidArrayStateInner::NeedBytes(pending) => {
                let liquid = pending.decode_liquid(data);
                self.state = GetLiquidArrayStateInner::Done(liquid);
            }
        }
    }
}

/// State machine: predicate pushdown sans-IO
#[derive(Debug)]
pub struct GetWithPredicateState<'a> {
    state: GetWithPredicateStateInner<'a>,
}

#[derive(Debug)]
enum GetWithPredicateStateInner<'a> {
    NeedBytes {
        selection: &'a BooleanBuffer,
        predicate: &'a Arc<dyn PhysicalExpr>,
        pending: PendingIo,
    },
    Done(Result<PredicatePushdownResult, ArrowError>),
}

impl<'a> IoStateMachine for GetWithPredicateState<'a> {
    type Output = Result<PredicatePushdownResult, ArrowError>;

    fn try_get(self) -> TryGet<Result<PredicatePushdownResult, ArrowError>, Self> {
        match self.state {
            GetWithPredicateStateInner::Done(r) => TryGet::Ready(r),
            GetWithPredicateStateInner::NeedBytes { ref pending, .. } => {
                let io_request = pending.as_io_request();
                TryGet::NeedData((self, io_request))
            }
        }
    }

    fn feed(&mut self, data: Bytes) {
        match &mut self.state {
            GetWithPredicateStateInner::Done(_) => {}
            GetWithPredicateStateInner::NeedBytes {
                pending,
                selection,
                predicate,
            } => match pending {
                PendingIo::Arrow { .. } => {
                    let array = pending.decode_arrow(data);
                    let selection_array = BooleanArray::new(selection.clone(), None);
                    let filtered = arrow::compute::filter(&array, &selection_array);
                    let result = filtered.map(PredicatePushdownResult::Filtered);
                    self.state = GetWithPredicateStateInner::Done(result);
                }
                PendingIo::Liquid { .. } => {
                    let liquid = pending.decode_liquid(data);
                    let result = liquid
                        .try_eval_predicate(&predicate, &selection)
                        .map(|opt| match opt {
                            Some(buf) => PredicatePushdownResult::Evaluated(buf),
                            None => {
                                let filtered = liquid.filter_to_arrow(&selection);
                                PredicatePushdownResult::Filtered(filtered)
                            }
                        });
                    self.state = GetWithPredicateStateInner::Done(result);
                }
            },
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
            Self::DiskLiquid => 0,
            Self::DiskArrow => 0,
        }
    }

    /// Get the reference count of the cached batch.
    pub fn reference_count(&self) -> usize {
        match self {
            Self::MemoryArrow(array) => Arc::strong_count(array),
            Self::MemoryLiquid(array) => Arc::strong_count(array),
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
            Self::DiskLiquid => write!(f, "DiskLiquid"),
            Self::DiskArrow => write!(f, "DiskArrow"),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::path::Path;
    use std::sync::RwLock;

    use crate::cache::utils::arrow_to_bytes;
    use crate::cache::{IoContext, transcode_liquid_inner};
    use crate::liquid_array::LiquidByteArray;

    use super::*;
    use crate::cache::cached_data::IoStateMachine;
    use ahash::HashMap;
    use arrow::array::{Array, AsArray, Int64Array, RecordBatch, StringArray};
    use arrow::compute as compute_kernels;
    use arrow_schema::{DataType, Field, Schema};
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

        let SansIo::Ready(out) = cached.get_arrow_array_sans_io() else {
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
        io.blocking_evict_liquid_to_disk(&id, &liquid_array).unwrap();
        test_get(&cached);
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

            let expect_filtered = |result: PredicatePushdownResult| match result {
                PredicatePushdownResult::Filtered(array) => {
                    assert_eq!(array.as_ref(), filter_expected.as_ref());
                }
                other => panic!("expected filtered, got {other:?}"),
            };
            let expect_evaluated = |result: PredicatePushdownResult| match result {
                PredicatePushdownResult::Evaluated(array) => {
                    assert_eq!(array, eval_expected);
                }
                other => panic!("expected evaluated, got {other:?}"),
            };

            // memory arrow
            {
                let SansIo::Ready(Ok(result)) = memory_arrow.get_with_predicate(&selection, expr)
                else {
                    panic!("should be ready");
                };
                expect_filtered(result);
            }

            // memory liquid
            {
                let SansIo::Ready(Ok(result)) = memory_liquid.get_with_predicate(&selection, expr)
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
                let TryGet::Ready(Ok(result)) = state.try_get() else {
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
                let TryGet::Ready(Ok(result)) = state.try_get() else {
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
