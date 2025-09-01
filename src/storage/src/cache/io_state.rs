//! Sans-IO state machines for IO operations.

use std::{path::PathBuf, sync::Arc};

use arrow::{
    array::{ArrayRef, BooleanArray},
    buffer::BooleanBuffer,
};
use arrow_schema::ArrowError;
use bytes::Bytes;
use datafusion::physical_plan::PhysicalExpr;

use crate::{
    cache::{LiquidCompressorStates, cached_data::PredicatePushdownResult},
    liquid_array::{LiquidArrayRef, LiquidHybridArrayRef},
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
    LiquidHybrid {
        path: PathBuf,
        old: LiquidHybridArrayRef,
    },
}

impl PendingIo {
    fn as_io_request(&self) -> IoRequest {
        match self {
            PendingIo::Arrow { path } => IoRequest { path: path.clone() },
            PendingIo::Liquid { path, .. } => IoRequest { path: path.clone() },
            PendingIo::LiquidHybrid { path, .. } => IoRequest { path: path.clone() },
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
        path: PathBuf,
        old: LiquidHybridArrayRef,
    ) -> SansIo<ArrayRef, Self> {
        let pending = PendingIo::LiquidHybrid { path, old };
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

impl<'selection> GetWithSelectionSansIo<'selection> {
    pub(crate) fn pending_arrow(
        path: PathBuf,
        selection: &'selection BooleanBuffer,
    ) -> SansIo<Result<ArrayRef, ArrowError>, Self> {
        let pending = PendingIo::Arrow { path };
        let io_request = pending.as_io_request();
        let state = Self {
            state: GetWithSelectionState::NeedBytes { selection, pending },
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
            state: GetWithSelectionState::NeedBytes { selection, pending },
        };
        SansIo::Pending((state, io_request))
    }

    pub(crate) fn pending_hybrid_liquid(
        path: PathBuf,
        selection: &'selection BooleanBuffer,
        old: LiquidHybridArrayRef,
    ) -> SansIo<Result<ArrayRef, ArrowError>, Self> {
        let pending = PendingIo::LiquidHybrid { path, old };

        let io_request = pending.as_io_request();
        let state = Self {
            state: GetWithSelectionState::NeedBytes { selection, pending },
        };
        SansIo::Pending((state, io_request))
    }
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
                PendingIo::LiquidHybrid { .. } => {
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
    pub(crate) fn new(
        path: PathBuf,
        compressor_states: Arc<LiquidCompressorStates>,
    ) -> (Self, IoRequest) {
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
pub struct GetWithPredicateState<'predicate, 'selection> {
    state: GetWithPredicateStateInner<'predicate, 'selection>,
}

impl<'predicate, 'selection> GetWithPredicateState<'predicate, 'selection> {
    pub(crate) fn pending_arrow(
        path: PathBuf,
        selection: &'selection BooleanBuffer,
        predicate: &'predicate Arc<dyn PhysicalExpr>,
    ) -> SansIo<PredicatePushdownResult, Self> {
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
    ) -> SansIo<PredicatePushdownResult, Self> {
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
        path: PathBuf,
        selection: &'selection BooleanBuffer,
        predicate: &'predicate Arc<dyn PhysicalExpr>,
        old: LiquidHybridArrayRef,
    ) -> SansIo<PredicatePushdownResult, Self> {
        let pending = PendingIo::LiquidHybrid { path, old };
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
    Done(PredicatePushdownResult),
}

impl<'a, 'b> IoStateMachine for GetWithPredicateState<'a, 'b> {
    type Output = PredicatePushdownResult;

    fn try_get(self) -> TryGet<PredicatePushdownResult, Self> {
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
                        PredicatePushdownResult::Filtered(filtered),
                    );
                }
                PendingIo::Liquid { .. } | PendingIo::LiquidHybrid { .. } => {
                    let liquid = pending.decode_liquid(data);
                    let result = liquid.try_eval_predicate(predicate, selection);
                    let result = match result {
                        Some(buf) => PredicatePushdownResult::Evaluated(buf),
                        None => {
                            let filtered = liquid.filter_to_arrow(selection);
                            PredicatePushdownResult::Filtered(filtered)
                        }
                    };
                    self.state = GetWithPredicateStateInner::Done(result);
                }
            },
        }
    }
}
