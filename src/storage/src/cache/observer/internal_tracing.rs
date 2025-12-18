use std::sync::Mutex;
use std::{fmt, fmt::Write};

use crate::cache::{CacheExpression, CachedBatchType, EntryID};

#[derive(Clone, PartialEq, Eq, serde::Serialize)]
pub(crate) enum InternalEvent {
    InsertFailed {
        entry: EntryID,
        kind: CachedBatchType,
    },
    InsertSuccess {
        entry: EntryID,
        kind: CachedBatchType,
    },
    SqueezeBegin {
        victims: Vec<EntryID>,
    },
    SqueezeVictim {
        entry: EntryID,
    },
    IoWrite {
        entry: EntryID,
        kind: CachedBatchType,
        bytes: usize,
    },
    IoReadArrow {
        entry: EntryID,
        bytes: usize,
    },
    IoReadLiquid {
        entry: EntryID,
        bytes: usize,
    },
    Read {
        entry: EntryID,
        selection: bool,
        expr: Option<CacheExpression>,
        cached: CachedBatchType,
    },
    Hydrate {
        entry: EntryID,
        cached: CachedBatchType,
        new: CachedBatchType,
    },
    EvalPredicate {
        entry: EntryID,
        selection: bool,
        cached: CachedBatchType,
    },
    ReadSqueezedData {
        entry: EntryID,
        expression: CacheExpression,
    },
    TryReadLiquid {
        entry: EntryID,
    },
}

#[derive(Debug)]
pub(crate) struct EventTracer {
    events: Mutex<Vec<InternalEvent>>,
}

fn fmt_entry_list(buf: &mut String, victims: &[EntryID]) -> fmt::Result {
    buf.push('[');
    for (idx, v) in victims.iter().enumerate() {
        if idx > 0 {
            buf.push(',');
        }
        write!(buf, "{}", usize::from(*v))?;
    }
    buf.push(']');
    Ok(())
}

impl fmt::Display for InternalEvent {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            InternalEvent::InsertFailed { entry, kind } => {
                write!(
                    f,
                    "event=insert_failed entry={} kind={:?}",
                    usize::from(*entry),
                    kind
                )
            }
            InternalEvent::InsertSuccess { entry, kind } => {
                write!(
                    f,
                    "event=insert_success entry={} kind={:?}",
                    usize::from(*entry),
                    kind
                )
            }
            InternalEvent::SqueezeBegin { victims } => {
                let mut buf = String::new();
                fmt_entry_list(&mut buf, victims)?;
                write!(f, "event=squeeze_begin victims={}", buf)
            }
            InternalEvent::SqueezeVictim { entry } => {
                write!(f, "event=squeeze_victim entry={}", usize::from(*entry))
            }
            InternalEvent::IoWrite { entry, kind, bytes } => {
                write!(
                    f,
                    "event=io_write entry={} kind={:?} bytes={}",
                    usize::from(*entry),
                    kind,
                    bytes
                )
            }
            InternalEvent::IoReadArrow { entry, bytes } => {
                write!(
                    f,
                    "event=io_read_arrow entry={} bytes={}",
                    usize::from(*entry),
                    bytes
                )
            }
            InternalEvent::IoReadLiquid { entry, bytes } => {
                write!(
                    f,
                    "event=io_read_liquid entry={} bytes={}",
                    usize::from(*entry),
                    bytes
                )
            }
            InternalEvent::Read {
                entry,
                selection,
                expr,
                cached,
            } => write!(
                f,
                "event=read entry={} selection={} expr={} cached={:?}",
                usize::from(*entry),
                selection,
                expr.as_ref()
                    .map(|e| e.to_string())
                    .unwrap_or_else(|| "None".to_string()),
                cached
            ),
            InternalEvent::Hydrate { entry, cached, new } => write!(
                f,
                "event=hydrate entry={} cached={:?} new={:?}",
                usize::from(*entry),
                cached,
                new
            ),
            InternalEvent::EvalPredicate {
                entry,
                selection,
                cached,
            } => write!(
                f,
                "event=eval_predicate entry={} selection={} cached={:?}",
                usize::from(*entry),
                selection,
                cached
            ),
            InternalEvent::TryReadLiquid { entry } => {
                write!(f, "event=try_read_liquid entry={}", usize::from(*entry))
            }
            InternalEvent::ReadSqueezedData { entry, expression } => {
                write!(
                    f,
                    "event=read_squeezed_data entry={} expression={}",
                    usize::from(*entry),
                    expression
                )
            }
        }
    }
}

impl fmt::Debug for InternalEvent {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self, f)
    }
}

impl EventTracer {
    pub fn new() -> Self {
        Self {
            events: Mutex::new(Vec::new()),
        }
    }

    pub fn record(&self, event: InternalEvent) {
        self.events.lock().unwrap().push(event);
    }

    pub fn drain(&self) -> EventTrace {
        EventTrace {
            events: std::mem::take(&mut *self.events.lock().unwrap()),
        }
    }
}

/// A trace of events that occurred in the cache.
/// This is used for testing only.
#[derive(PartialEq, Eq, serde::Serialize)]
pub struct EventTrace {
    events: Vec<InternalEvent>,
}

impl From<Vec<InternalEvent>> for EventTrace {
    fn from(events: Vec<InternalEvent>) -> Self {
        Self { events }
    }
}

impl fmt::Display for EventTrace {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "{:?}", self)
    }
}

impl fmt::Debug for EventTrace {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "EventTrace: [")?;
        for event in &self.events {
            writeln!(f, "{}", event)?;
        }
        writeln!(f, "]")?;
        Ok(())
    }
}
