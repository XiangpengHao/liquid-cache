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
    io_worker: &'a dyn super::core::IoWorker,
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
        io_worker: &'a dyn super::core::IoWorker,
    ) -> Self {
        Self {
            data,
            id,
            io_worker,
        }
    }

    /// Build a sans-IO state machine to obtain an Arrow `ArrayRef` with selection pushdown.
    pub fn get_with_selection_sans_io<'selection>(
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
                    path: self.io_worker.entry_liquid_path(&self.id),
                    compressor_states: self.io_worker.get_compressor_for_entry(&self.id),
                };
                SansIo::Pending(GetWithSelectionSansIo {
                    state: GetWithSelectionState::NeedBytes { selection, pending },
                })
            }
            CachedBatch::DiskArrow => {
                let pending = PendingIo::Arrow {
                    path: self.io_worker.entry_arrow_path(&self.id),
                };
                SansIo::Pending(GetWithSelectionSansIo {
                    state: GetWithSelectionState::NeedBytes { selection, pending },
                })
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
                let path = self.io_worker.entry_liquid_path(&self.id);
                let compressor_states = self.io_worker.get_compressor_for_entry(&self.id);
                SansIo::Pending(GetArrowArrayState::new_liquid(path, compressor_states))
            }
            CachedBatch::DiskArrow => {
                let path = self.io_worker.entry_arrow_path(&self.id);
                SansIo::Pending(GetArrowArrayState::new_arrow(path))
            }
        }
    }

    #[cfg(test)]
    pub(crate) fn raw_data(&self) -> &CachedBatch {
        &self.data
    }

    /// Get the arrow array from the cached data.
    pub fn get_arrow_array(&self) -> ArrayRef {
        match &self.data {
            CachedBatch::MemoryArrow(array) => array.clone(),
            CachedBatch::MemoryLiquid(array) => array.to_best_arrow_array(),
            CachedBatch::DiskLiquid => self
                .io_worker
                .read_liquid_from_disk(&self.id)
                .unwrap()
                .to_best_arrow_array(),
            CachedBatch::DiskArrow => self.io_worker.read_arrow_from_disk(&self.id).unwrap(),
        }
    }

    /// Try to read the liquid array from the cached data.
    /// Return None if the cached data is not a liquid array.
    pub fn try_read_liquid(&self) -> Option<LiquidArrayRef> {
        match &self.data {
            CachedBatch::MemoryLiquid(array) => Some(array.clone()),
            CachedBatch::DiskLiquid => {
                Some(self.io_worker.read_liquid_from_disk(&self.id).unwrap())
            }
            _ => None,
        }
    }

    /// Build a sans-IO state machine to obtain a `LiquidArrayRef` if the cached data is liquid.
    pub fn try_read_liquid_sans_io(&self) -> Option<SansIo<LiquidArrayRef, GetLiquidArrayState>> {
        match &self.data {
            CachedBatch::MemoryLiquid(array) => Some(SansIo::Ready(array.clone())),
            CachedBatch::DiskLiquid => Some(SansIo::Pending(GetLiquidArrayState::new(
                self.io_worker.entry_liquid_path(&self.id),
                self.io_worker.get_compressor_for_entry(&self.id),
            ))),
            _ => None,
        }
    }

    /// Get the arrow array with selection pushdown.
    pub fn get_with_selection(&self, selection: &BooleanBuffer) -> Result<ArrayRef, ArrowError> {
        match &self.data {
            CachedBatch::MemoryArrow(array) => {
                let selection = BooleanArray::new(selection.clone(), None);
                let filtered = arrow::compute::filter(array, &selection)?;
                Ok(filtered)
            }
            CachedBatch::MemoryLiquid(array) => {
                let filtered = array.filter_to_arrow(selection);
                Ok(filtered)
            }
            CachedBatch::DiskLiquid => {
                let array = self.io_worker.read_liquid_from_disk(&self.id)?;
                let filtered = array.filter_to_arrow(selection);
                Ok(filtered)
            }
            CachedBatch::DiskArrow => {
                let array = self.io_worker.read_arrow_from_disk(&self.id)?;
                let selection = BooleanArray::new(selection.clone(), None);
                let filtered = arrow::compute::filter(&array, &selection)?;
                Ok(filtered)
            }
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
    /// ```rust
    /// use liquid_cache_storage::cache::{CacheStorageBuilder, EntryID, cached_data::PredicatePushdownResult};
    /// use liquid_cache_storage::common::LiquidCacheMode;
    /// use arrow::array::{StringArray, BooleanArray};
    /// use arrow::buffer::BooleanBuffer;
    /// use datafusion::logical_expr::Operator;
    /// use datafusion::physical_plan::expressions::{BinaryExpr, Column, Literal};
    /// use datafusion::physical_plan::PhysicalExpr;
    /// use datafusion::scalar::ScalarValue;
    /// use std::sync::Arc;
    ///
    /// let storage = CacheStorageBuilder::new()
    ///     .with_cache_mode(LiquidCacheMode::LiquidBlocking)
    ///     .build();
    ///
    /// let entry_id = EntryID::from(9);
    /// let data = Arc::new(StringArray::from(vec![
    ///     Some("apple"), Some("banana"), None, Some("apple"), Some("cherry"),
    /// ]));
    /// storage.insert(entry_id, data.clone());
    ///
    /// let selection = BooleanBuffer::from(vec![true, true, false, true, true]);
    ///
    /// let expr: Arc<dyn PhysicalExpr> = Arc::new(BinaryExpr::new(
    ///     Arc::new(Column::new("col", 0)),
    ///     Operator::Eq,
    ///     Arc::new(Literal::new(ScalarValue::Utf8(Some("apple".to_string())))),
    /// ));
    ///
    /// let cached = storage.get(&entry_id).unwrap();
    /// let result = cached
    ///     .get_with_predicate(&selection, &expr)
    ///     .unwrap();
    /// let expected = BooleanArray::from(vec![true, false, true, false]);
    /// assert_eq!(result, PredicatePushdownResult::Evaluated(expected));
    /// ```
    pub fn get_with_predicate(
        &self,
        selection: &BooleanBuffer,
        predicate: &Arc<dyn PhysicalExpr>,
    ) -> Result<PredicatePushdownResult, ArrowError> {
        match &self.data {
            CachedBatch::MemoryArrow(array) => {
                let selection = BooleanArray::new(selection.clone(), None);
                let selected = arrow::compute::filter(array, &selection)?;
                Ok(PredicatePushdownResult::Filtered(selected))
            }
            CachedBatch::DiskArrow => {
                let array = self.io_worker.read_arrow_from_disk(&self.id)?;
                let selection = BooleanArray::new(selection.clone(), None);
                let selected = arrow::compute::filter(&array, &selection)?;
                Ok(PredicatePushdownResult::Filtered(selected))
            }
            CachedBatch::MemoryLiquid(array) => {
                self.eval_predicate_with_filter_inner(predicate, array, selection)
            }
            CachedBatch::DiskLiquid => {
                let array = self.io_worker.read_liquid_from_disk(&self.id).unwrap();
                self.eval_predicate_with_filter_inner(predicate, &array, selection)
            }
        }
    }

    /// Build a sans-IO state machine to evaluate a predicate with selection pushdown.
    pub fn get_with_predicate_sans_io<'predicate, 'selection>(
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
                    path: self.io_worker.entry_arrow_path(&self.id),
                };
                SansIo::Pending(GetWithPredicateState {
                    state: GetWithPredicateStateInner::NeedBytes {
                        selection,
                        predicate,
                        pending,
                    },
                })
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
                    path: self.io_worker.entry_liquid_path(&self.id),
                    compressor_states: self.io_worker.get_compressor_for_entry(&self.id),
                };
                SansIo::Pending(GetWithPredicateState {
                    state: GetWithPredicateStateInner::NeedBytes {
                        selection,
                        predicate,
                        pending,
                    },
                })
            }
        }
    }

    fn eval_predicate_with_filter_inner(
        &self,
        predicate: &Arc<dyn PhysicalExpr>,
        array: &LiquidArrayRef,
        selection: &BooleanBuffer,
    ) -> Result<PredicatePushdownResult, ArrowError> {
        match array.try_eval_predicate(predicate, selection)? {
            Some(new_filter) => Ok(PredicatePushdownResult::Evaluated(new_filter)),
            None => {
                let filtered = array.filter_to_arrow(selection);
                Ok(PredicatePushdownResult::Filtered(filtered))
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
    NeedIo((M, IoRequest)),
}

/// The result of a sans-IO operation.
#[derive(Debug)]
pub enum SansIo<T, M> {
    /// The output is ready.
    Ready(T),
    /// The output needs IO.
    Pending(M),
}

/// A state machine that can be used to perform a sans-IO operation.
pub trait SansIoStateMachine: Sized {
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
    fn new_arrow(path: PathBuf) -> Self {
        Self {
            state: GetArrowArrayStateInner::NeedBytes(PendingIo::Arrow { path }),
        }
    }

    fn new_liquid(path: PathBuf, compressor_states: Arc<LiquidCompressorStates>) -> Self {
        Self {
            state: GetArrowArrayStateInner::NeedBytes(PendingIo::Liquid {
                path,
                compressor_states,
            }),
        }
    }
}

#[derive(Debug)]
enum GetArrowArrayStateInner {
    Ready(ArrayRef),
    NeedBytes(PendingIo),
}

impl SansIoStateMachine for GetArrowArrayState {
    type Output = ArrayRef;

    fn try_get(self) -> TryGet<ArrayRef, Self> {
        match self.state {
            GetArrowArrayStateInner::Ready(array) => TryGet::Ready(array),
            GetArrowArrayStateInner::NeedBytes(ref p) => {
                let io_request = p.as_io_request();
                TryGet::NeedIo((self, io_request))
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

impl<'a> SansIoStateMachine for GetWithSelectionSansIo<'a> {
    type Output = Result<ArrayRef, ArrowError>;

    fn try_get(self) -> TryGet<Result<ArrayRef, ArrowError>, Self> {
        match self.state {
            GetWithSelectionState::Done(r) => TryGet::Ready(r),
            GetWithSelectionState::NeedBytes { ref pending, .. } => {
                let io_request = pending.as_io_request();
                TryGet::NeedIo((self, io_request))
            }
        }
    }

    fn feed(&mut self, data: Bytes) {
        match &mut self.state {
            GetWithSelectionState::Done(_) => {}
            GetWithSelectionState::NeedBytes { pending, selection } => {
                let array = pending.decode_arrow(data);
                let selection_array = BooleanArray::new(selection.clone(), None);
                let filtered = arrow::compute::filter(&array, &selection_array);
                self.state = GetWithSelectionState::Done(filtered);
            }
        }
    }
}

/// State machine: read `LiquidArrayRef` sans-IO
#[derive(Debug)]
pub struct GetLiquidArrayState {
    state: GetLiquidArrayStateInner,
}

impl GetLiquidArrayState {
    fn new(path: PathBuf, compressor_states: Arc<LiquidCompressorStates>) -> Self {
        Self {
            state: GetLiquidArrayStateInner::NeedBytes(PendingIo::Liquid {
                path,
                compressor_states,
            }),
        }
    }
}

#[derive(Debug)]
enum GetLiquidArrayStateInner {
    NeedBytes(PendingIo),
    Done(LiquidArrayRef),
}

impl SansIoStateMachine for GetLiquidArrayState {
    type Output = LiquidArrayRef;

    fn try_get(self) -> TryGet<LiquidArrayRef, Self> {
        match &self.state {
            GetLiquidArrayStateInner::Done(liq) => TryGet::Ready(liq.clone()),
            GetLiquidArrayStateInner::NeedBytes(p) => {
                let io_request = p.as_io_request();
                TryGet::NeedIo((self, io_request))
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

impl<'a> SansIoStateMachine for GetWithPredicateState<'a> {
    type Output = Result<PredicatePushdownResult, ArrowError>;

    fn try_get(self) -> TryGet<Result<PredicatePushdownResult, ArrowError>, Self> {
        match self.state {
            GetWithPredicateStateInner::Done(r) => TryGet::Ready(r),
            GetWithPredicateStateInner::NeedBytes { ref pending, .. } => {
                let io_request = pending.as_io_request();
                TryGet::NeedIo((self, io_request))
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
    use crate::liquid_array::LiquidByteArray;

    use super::*;
    use arrow::array::{Array, AsArray, Int64Array, RecordBatch, StringArray};
    use arrow::compute as compute_kernels;
    use arrow_schema::{DataType, Field, Schema};
    use datafusion::logical_expr::{ColumnarValue, Operator};
    use datafusion::physical_plan::expressions::{BinaryExpr, Column, Literal};
    use datafusion::scalar::ScalarValue;
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    fn io_worker() -> (tempfile::TempDir, super::super::core::DefaultIoWorker) {
        let tmp = tempfile::tempdir().unwrap();
        let io = super::super::core::DefaultIoWorker::new(tmp.path().to_path_buf());
        (tmp, io)
    }

    #[test]
    fn test_get_arrow_array_memory_arrow() {
        let array: ArrayRef = Arc::new(Int64Array::from_iter_values(0..8));
        let id = EntryID::from(1usize);
        let (_tmp, io) = io_worker();
        let cached = CachedData::new(CachedBatch::MemoryArrow(array.clone()), id, &io);

        let out = cached.get_arrow_array();
        assert_eq!(out.as_ref(), array.as_ref());
    }

    #[test]
    fn test_try_read_liquid_memory_liquid() {
        // Build a small liquid string array
        let input = StringArray::from(vec!["a", "b", "a", "c"]);
        let (_compressor, etc) = crate::liquid_array::LiquidByteArray::train_from_arrow(&input);
        let liquid_ref: LiquidArrayRef = Arc::new(etc);

        let id = EntryID::from(2usize);
        let (_tmp, io) = io_worker();
        let cached = CachedData::new(CachedBatch::MemoryLiquid(liquid_ref.clone()), id, &io);

        let got = cached.try_read_liquid().expect("should be liquid");
        assert_eq!(got.to_best_arrow_array().len(), input.len());
    }

    #[test]
    fn test_get_with_selection_memory_arrow() {
        let array: ArrayRef = Arc::new(Int64Array::from_iter_values(0..10));
        let selection = BooleanBuffer::from((0..10).map(|i| i % 2 == 0).collect::<Vec<_>>());

        let id = EntryID::from(3usize);
        let (_tmp, io) = io_worker();
        let cached = CachedData::new(CachedBatch::MemoryArrow(array.clone()), id, &io);

        let filtered = cached.get_with_selection(&selection).unwrap();
        let selection = BooleanArray::new(selection.clone(), None);
        let expected = compute_kernels::filter(&array, &selection).unwrap();
        assert_eq!(filtered.as_ref(), expected.as_ref());
    }

    fn test_string_predicate(string_array: &StringArray, expr: &Arc<dyn PhysicalExpr>) {
        let (_compressor, liquid) = LiquidByteArray::train_from_arrow(string_array);
        let liquid_ref: LiquidArrayRef = Arc::new(liquid);

        let id = EntryID::from(9usize);
        let (_tmp, io) = io_worker();
        let cached = CachedData::new(CachedBatch::MemoryLiquid(liquid_ref), id, &io);

        let mut seed_rng = StdRng::seed_from_u64(42);
        for _i in 0..100 {
            let selection = BooleanBuffer::from_iter(
                (0..string_array.len()).map(|_| seed_rng.random_bool(0.5)),
            );

            let expected = {
                let selection = BooleanArray::new(selection.clone(), None);
                let filtered = arrow::compute::filter(&string_array, &selection).unwrap();
                let record_batch = RecordBatch::try_new(
                    Arc::new(Schema::new(vec![Field::new("col", DataType::Utf8, true)])),
                    vec![filtered],
                )
                .unwrap();
                let evaluated = expr.evaluate(&record_batch).unwrap();
                let filtered = match evaluated {
                    ColumnarValue::Array(array) => array,
                    ColumnarValue::Scalar(_) => panic!("expected array, got scalar"),
                };
                filtered.as_boolean().clone()
            };

            let result = cached
                .get_with_predicate(&selection, expr)
                .expect("predicate should succeed");

            match result {
                PredicatePushdownResult::Evaluated(buf) => {
                    assert_eq!(buf, expected);
                }
                other => panic!("expected Evaluated, got {other:?}"),
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
