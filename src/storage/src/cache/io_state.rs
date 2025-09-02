//! Sans-IO state machines for IO operations.

use std::{
    path::{Path, PathBuf},
    sync::Arc,
};

use arrow::{
    array::{ArrayRef, BooleanArray},
    buffer::BooleanBuffer,
};
use arrow_schema::ArrowError;
use bytes::Bytes;
use datafusion::physical_plan::PhysicalExpr;

use crate::{
    cache::{LiquidCompressorStates, cached_data::GetWithPredicateResult},
    liquid_array::{IoRange, LiquidArrayRef, LiquidHybridArrayRef},
};

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
    path: PathBuf,
    range: Option<IoRange>,
}

impl IoRequest {
    /// Get the path of the IO request.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Get the range of the IO request.
    pub fn range(&self) -> Option<&IoRange> {
        self.range.as_ref()
    }

    pub(crate) fn from_path_and_range(path: PathBuf, range: IoRange) -> Self {
        Self {
            path,
            range: Some(range),
        }
    }
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
    LiquidHybrid {
        io_request: IoRequest,
        old: LiquidHybridArrayRef,
    },
}

impl PendingIo {
    fn as_io_request(&self) -> IoRequest {
        match self {
            PendingIo::Arrow { path } => IoRequest {
                path: path.clone(),
                range: None,
            },
            PendingIo::Liquid { path, .. } => IoRequest {
                path: path.clone(),
                range: None,
            },
            PendingIo::LiquidHybrid { io_request, .. } => io_request.clone(),
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
            PendingIo::LiquidHybrid { old, .. } => {
                let new = old.soak(data);
                new.to_arrow_array()
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
            PendingIo::LiquidHybrid { old, .. } => old.soak(data),
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
    pub(crate) fn pending_arrow(path: PathBuf) -> SansIo<ArrayRef, Self> {
        let pending = PendingIo::Arrow { path };
        let io_request = pending.as_io_request();
        let state = Self {
            state: GetArrowArrayStateInner::NeedBytes(pending),
        };
        SansIo::Pending((state, io_request))
    }

    pub(crate) fn pending_liquid(
        path: PathBuf,
        compressor_states: Arc<LiquidCompressorStates>,
    ) -> SansIo<ArrayRef, Self> {
        let pending = PendingIo::Liquid {
            path,
            compressor_states,
        };
        let io_request = pending.as_io_request();
        let state = Self {
            state: GetArrowArrayStateInner::NeedBytes(pending),
        };
        SansIo::Pending((state, io_request))
    }

    pub(crate) fn pending_hybrid_liquid(
        io_request: IoRequest,
        old: LiquidHybridArrayRef,
    ) -> SansIo<ArrayRef, Self> {
        let pending = PendingIo::LiquidHybrid { io_request, old };
        let io_request = pending.as_io_request();
        let state = Self {
            state: GetArrowArrayStateInner::NeedBytes(pending),
        };
        SansIo::Pending((state, io_request))
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
pub struct GetWithSelectionState<'selection> {
    state: GetWithSelectionInner<'selection>,
}

#[derive(Debug)]
enum GetWithSelectionInner<'selection> {
    NeedBytes {
        selection: &'selection BooleanBuffer,
        pending: PendingIo,
    },
    Done(Result<ArrayRef, ArrowError>),
}

impl<'selection> GetWithSelectionState<'selection> {
    pub(crate) fn pending_arrow(
        path: PathBuf,
        selection: &'selection BooleanBuffer,
    ) -> SansIo<Result<ArrayRef, ArrowError>, Self> {
        let pending = PendingIo::Arrow { path };
        let io_request = pending.as_io_request();
        let state = Self {
            state: GetWithSelectionInner::NeedBytes { selection, pending },
        };
        SansIo::Pending((state, io_request))
    }

    pub(crate) fn pending_liquid(
        path: PathBuf,
        selection: &'selection BooleanBuffer,
        compressor_states: Arc<LiquidCompressorStates>,
    ) -> SansIo<Result<ArrayRef, ArrowError>, Self> {
        let pending = PendingIo::Liquid {
            path,
            compressor_states,
        };
        let io_request = pending.as_io_request();
        let state = Self {
            state: GetWithSelectionInner::NeedBytes { selection, pending },
        };
        SansIo::Pending((state, io_request))
    }

    pub(crate) fn pending_hybrid_liquid(
        io_request: IoRequest,
        selection: &'selection BooleanBuffer,
        old: LiquidHybridArrayRef,
    ) -> SansIo<Result<ArrayRef, ArrowError>, Self> {
        let pending = PendingIo::LiquidHybrid { io_request, old };

        let io_request = pending.as_io_request();
        let state = Self {
            state: GetWithSelectionInner::NeedBytes { selection, pending },
        };
        SansIo::Pending((state, io_request))
    }
}

impl<'a> IoStateMachine for GetWithSelectionState<'a> {
    type Output = Result<ArrayRef, ArrowError>;

    fn try_get(self) -> TryGet<Result<ArrayRef, ArrowError>, Self> {
        match self.state {
            GetWithSelectionInner::Done(r) => TryGet::Ready(r),
            GetWithSelectionInner::NeedBytes { ref pending, .. } => {
                let io_request = pending.as_io_request();
                TryGet::NeedData((self, io_request))
            }
        }
    }

    fn feed(&mut self, data: Bytes) {
        match &mut self.state {
            GetWithSelectionInner::Done(_) => {}
            GetWithSelectionInner::NeedBytes { pending, selection } => match pending {
                PendingIo::Arrow { .. } => {
                    let array = pending.decode_arrow(data);
                    let selection_array = BooleanArray::new(selection.clone(), None);
                    let filtered = arrow::compute::filter(&array, &selection_array);
                    self.state = GetWithSelectionInner::Done(filtered);
                }
                PendingIo::Liquid { .. } => {
                    let liquid = pending.decode_liquid(data);
                    let filtered = liquid.filter_to_arrow(selection);
                    self.state = GetWithSelectionInner::Done(Ok(filtered));
                }
                PendingIo::LiquidHybrid { .. } => {
                    let liquid = pending.decode_liquid(data);
                    let filtered = liquid.filter_to_arrow(selection);
                    self.state = GetWithSelectionInner::Done(Ok(filtered));
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
    pub(crate) fn pending_liquid(
        path: PathBuf,
        compressor_states: Arc<LiquidCompressorStates>,
    ) -> SansIo<Option<LiquidArrayRef>, Self> {
        let pending = PendingIo::Liquid {
            path,
            compressor_states,
        };
        let io_request = pending.as_io_request();
        let state = Self {
            state: GetLiquidArrayStateInner::NeedBytes(pending),
        };
        SansIo::Pending((state, io_request))
    }

    pub(crate) fn pending_hybrid_liquid(
        io_request: IoRequest,
        old: LiquidHybridArrayRef,
    ) -> SansIo<Option<LiquidArrayRef>, Self> {
        let pending = PendingIo::LiquidHybrid { io_request, old };
        let io_request = pending.as_io_request();
        let state = Self {
            state: GetLiquidArrayStateInner::NeedBytes(pending),
        };
        SansIo::Pending((state, io_request))
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
pub struct GetWithPredicateState<'predicate, 'selection> {
    state: GetWithPredicateStateInner<'predicate, 'selection>,
}

impl<'predicate, 'selection> GetWithPredicateState<'predicate, 'selection> {
    pub(crate) fn pending_arrow(
        path: PathBuf,
        selection: &'selection BooleanBuffer,
        predicate: &'predicate Arc<dyn PhysicalExpr>,
    ) -> SansIo<GetWithPredicateResult, Self> {
        let pending = PendingIo::Arrow { path };
        let io_request = pending.as_io_request();
        let state = Self {
            state: GetWithPredicateStateInner::NeedBytes {
                selection,
                predicate,
                pending,
            },
        };
        SansIo::Pending((state, io_request))
    }

    pub(crate) fn pending_liquid(
        path: PathBuf,
        selection: &'selection BooleanBuffer,
        predicate: &'predicate Arc<dyn PhysicalExpr>,
        compressor_states: Arc<LiquidCompressorStates>,
    ) -> SansIo<GetWithPredicateResult, Self> {
        let pending = PendingIo::Liquid {
            path,
            compressor_states,
        };
        let io_request = pending.as_io_request();
        let state = Self {
            state: GetWithPredicateStateInner::NeedBytes {
                selection,
                predicate,
                pending,
            },
        };
        SansIo::Pending((state, io_request))
    }

    pub(crate) fn pending_hybrid_liquid(
        io_request: IoRequest,
        selection: &'selection BooleanBuffer,
        predicate: &'predicate Arc<dyn PhysicalExpr>,
        old: LiquidHybridArrayRef,
    ) -> SansIo<GetWithPredicateResult, Self> {
        let pending = PendingIo::LiquidHybrid { io_request, old };
        let io_request = pending.as_io_request();
        let state = Self {
            state: GetWithPredicateStateInner::NeedBytes {
                selection,
                predicate,
                pending,
            },
        };
        SansIo::Pending((state, io_request))
    }
}

#[derive(Debug)]
enum GetWithPredicateStateInner<'a, 'b> {
    NeedBytes {
        selection: &'b BooleanBuffer,
        predicate: &'a Arc<dyn PhysicalExpr>,
        pending: PendingIo,
    },
    Done(GetWithPredicateResult),
}

impl<'a, 'b> IoStateMachine for GetWithPredicateState<'a, 'b> {
    type Output = GetWithPredicateResult;

    fn try_get(self) -> TryGet<GetWithPredicateResult, Self> {
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
                    let filtered = arrow::compute::filter(&array, &selection_array).unwrap();
                    self.state = GetWithPredicateStateInner::Done(
                        GetWithPredicateResult::Filtered(filtered),
                    );
                }
                PendingIo::Liquid { .. } | PendingIo::LiquidHybrid { .. } => {
                    let liquid = pending.decode_liquid(data);
                    let result = liquid.try_eval_predicate(predicate, selection);
                    let result = match result {
                        Some(buf) => GetWithPredicateResult::Evaluated(buf),
                        None => {
                            let filtered = liquid.filter_to_arrow(selection);
                            GetWithPredicateResult::Filtered(filtered)
                        }
                    };
                    self.state = GetWithPredicateStateInner::Done(result);
                }
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cache::transcode::transcode_liquid_inner;
    use crate::cache::utils::{LiquidCompressorStates, arrow_to_bytes};
    use crate::liquid_array::AsLiquidArray;
    use crate::liquid_array::LiquidArray;
    use arrow::array::{Array, ArrayRef, BooleanArray, Int32Array, StringArray};
    use arrow::buffer::BooleanBuffer;
    use bytes::Bytes;
    use datafusion::logical_expr::Operator;
    use datafusion::physical_plan::expressions::{BinaryExpr, Column, Literal};
    use datafusion::scalar::ScalarValue;
    use tempfile::tempdir;

    fn int_array(n: i32) -> ArrayRef {
        Arc::new(Int32Array::from_iter_values(0..n))
    }

    fn arrow_ipc_bytes(arr: &ArrayRef) -> Bytes {
        arrow_to_bytes(arr).expect("arrow to bytes")
    }

    fn liquid_bytes(arr: &ArrayRef, states: &LiquidCompressorStates) -> Bytes {
        let liquid = transcode_liquid_inner(arr, states).unwrap();
        Bytes::from(liquid.to_bytes())
    }

    #[test]
    fn get_arrow_array_state_integer_arrow_and_liquid() {
        let arr = int_array(16);
        let tmp = tempdir().unwrap();
        let path = tmp.path().join("col.arrow");

        // Arrow backend
        let SansIo::Pending((mut s1, _)) = GetArrowArrayState::pending_arrow(path.clone()) else {
            panic!("expected pending");
        };
        s1.feed(arrow_ipc_bytes(&arr));
        let TryGet::Ready(out1) = s1.try_get() else {
            panic!("ready");
        };
        assert_eq!(out1.as_ref(), arr.as_ref());

        // Liquid backend
        let states = LiquidCompressorStates::new();
        let SansIo::Pending((mut s2, _)) =
            GetArrowArrayState::pending_liquid(path, Arc::new(states))
        else {
            panic!("expected pending");
        };
        let states2 = LiquidCompressorStates::new();
        s2.feed(liquid_bytes(&arr, &states2));
        let TryGet::Ready(out2) = s2.try_get() else {
            panic!("ready");
        };
        assert_eq!(out2.as_ref(), arr.as_ref());
    }

    #[test]
    fn get_with_selection_state_integer_arrow_and_liquid() {
        let arr = int_array(10);
        let selection = BooleanBuffer::from((0..10).map(|i| i % 2 == 0).collect::<Vec<_>>());
        let expected = {
            let array = BooleanArray::new(selection.clone(), None);
            arrow::compute::filter(&arr, &array).unwrap()
        };

        let tmp = tempdir().unwrap();
        let path = tmp.path().join("col");

        // Arrow
        let SansIo::Pending((mut s1, _)) =
            GetWithSelectionState::pending_arrow(path.clone(), &selection)
        else {
            panic!("expected pending");
        };
        s1.feed(arrow_ipc_bytes(&arr));
        let TryGet::Ready(Ok(out1)) = s1.try_get() else {
            panic!("ready");
        };
        assert_eq!(out1.as_ref(), expected.as_ref());

        // Liquid
        let states = LiquidCompressorStates::new();
        let SansIo::Pending((mut s2, _)) =
            GetWithSelectionState::pending_liquid(path, &selection, Arc::new(states))
        else {
            panic!("expected pending");
        };
        let states2 = LiquidCompressorStates::new();
        s2.feed(liquid_bytes(&arr, &states2));
        let TryGet::Ready(Ok(out2)) = s2.try_get() else {
            panic!("ready");
        };
        assert_eq!(out2.as_ref(), expected.as_ref());
    }

    #[test]
    fn get_liquid_array_state_liquid_and_hybrid() {
        // Use strings for hybrid coverage
        let input = Arc::new(StringArray::from(vec!["a", "b", "c", "a"])) as ArrayRef;
        let states = LiquidCompressorStates::new();
        let liquid = transcode_liquid_inner(&input, &states).unwrap();

        // Liquid bytes path
        let SansIo::Pending((mut s1, _)) =
            GetLiquidArrayState::pending_liquid("mem".into(), Arc::new(states))
        else {
            panic!("pending");
        };
        s1.feed(Bytes::from(liquid.to_bytes()));
        let TryGet::Ready(out_liquid) = s1.try_get() else {
            panic!("ready");
        };
        assert_eq!(out_liquid.to_arrow_array().as_ref(), input.as_ref());

        // Hybrid path: squeeze and feed FSST slice
        let (hybrid, full_bytes) = liquid.as_byte_view().squeeze().expect("squeeze");
        let io_range = hybrid.to_liquid();
        let fsst_bytes =
            full_bytes.slice(io_range.range().start as usize..io_range.range().end as usize);
        let io_req = IoRequest::from_path_and_range("disk".into(), io_range);
        let SansIo::Pending((mut s2, _)) =
            GetLiquidArrayState::pending_hybrid_liquid(io_req, hybrid)
        else {
            panic!("pending");
        };
        s2.feed(fsst_bytes);
        let TryGet::Ready(out_liquid2) = s2.try_get() else {
            panic!("ready");
        };
        assert_eq!(out_liquid2.to_arrow_array().as_ref(), input.as_ref());
    }

    #[test]
    fn get_with_predicate_state_arrow_liquid_hybrid() {
        // Small string data with an empty string and a null
        let input = Arc::new(StringArray::from(vec![
            Some("a"),
            Some(""),
            None,
            Some("b"),
        ])) as ArrayRef;
        // selection keeps first three
        let selection = BooleanBuffer::from(vec![true, true, true, false]);

        // predicate: col == ""
        let predicate: Arc<dyn PhysicalExpr> = Arc::new(BinaryExpr::new(
            Arc::new(Column::new("col", 0)),
            Operator::Eq,
            Arc::new(Literal::new(ScalarValue::Utf8(Some("".to_string())))),
        ));

        let tmp = tempdir().unwrap();
        let path = tmp.path().join("col");

        // Arrow backend -> Filtered
        let expected_filtered = {
            let array = BooleanArray::new(selection.clone(), None);
            arrow::compute::filter(&input, &array).unwrap()
        };
        let SansIo::Pending((mut s1, _)) =
            GetWithPredicateState::pending_arrow(path.clone(), &selection, &predicate)
        else {
            panic!("pending");
        };
        s1.feed(arrow_ipc_bytes(&input));
        let TryGet::Ready(out1) = s1.try_get() else {
            panic!("ready");
        };
        match out1 {
            GetWithPredicateResult::Filtered(a) => {
                assert_eq!(a.as_ref(), expected_filtered.as_ref())
            }
            _ => panic!("expected filtered"),
        }

        // Liquid backend -> Evaluated (byte-view supports predicate)
        let states = LiquidCompressorStates::new();
        let bytes = liquid_bytes(&input, &states);
        let SansIo::Pending((mut s2, _)) = GetWithPredicateState::pending_liquid(
            path.clone(),
            &selection,
            &predicate,
            Arc::new(states),
        ) else {
            panic!("pending");
        };
        s2.feed(bytes);
        let TryGet::Ready(out2) = s2.try_get() else {
            panic!("ready");
        };
        match out2 {
            GetWithPredicateResult::Evaluated(buf) => {
                assert_eq!(buf.len(), 3);
                assert!(!buf.value(0));
                assert!(buf.value(1));
                assert!(!buf.is_valid(2));
            }
            other => panic!("expected evaluated, got {other:?}"),
        }

        // Hybrid backend
        let liquid = transcode_liquid_inner(&input, &LiquidCompressorStates::new()).unwrap();
        let (hybrid, full_bytes) = liquid.as_byte_view().squeeze().unwrap();
        let io_range = hybrid.to_liquid();
        let fsst_bytes =
            full_bytes.slice(io_range.range().start as usize..io_range.range().end as usize);
        let io_req = IoRequest::from_path_and_range(path, io_range);
        let SansIo::Pending((mut s3, _)) =
            GetWithPredicateState::pending_hybrid_liquid(io_req, &selection, &predicate, hybrid)
        else {
            panic!("pending");
        };
        s3.feed(fsst_bytes);
        let TryGet::Ready(out3) = s3.try_get() else {
            panic!("ready");
        };
        match out3 {
            GetWithPredicateResult::Evaluated(buf) => {
                assert_eq!(buf.len(), 3);
                assert!(!buf.value(0));
                assert!(buf.value(1));
                assert!(!buf.is_valid(2));
            }
            other => panic!("expected evaluated, got {other:?}"),
        }
    }
}
