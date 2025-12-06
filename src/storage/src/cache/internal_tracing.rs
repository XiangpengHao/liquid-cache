use std::sync::Mutex;
use std::{fmt, fmt::Write};

use crate::cache::{CachedBatchType, EntryID};

#[derive(Clone, PartialEq, Eq)]
pub(crate) enum InternalEvent {
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
    },
    IoReadLiquid {
        entry: EntryID,
    },
    Read {
        entry: EntryID,
        selection: bool,
        expr: bool,
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
            InternalEvent::InsertSuccess { entry, kind } => {
                write!(
                    f,
                    "insert_success entry={} kind={:?}",
                    usize::from(*entry),
                    kind
                )
            }
            InternalEvent::SqueezeBegin { victims } => {
                let mut buf = String::new();
                fmt_entry_list(&mut buf, victims)?;
                write!(f, "squeeze_begin victims={}", buf)
            }
            InternalEvent::SqueezeVictim { entry } => {
                write!(f, "squeeze_victim entry={}", usize::from(*entry))
            }
            InternalEvent::IoWrite { entry, kind, bytes } => {
                write!(
                    f,
                    "io_write entry={} kind={:?} bytes={}",
                    usize::from(*entry),
                    kind,
                    bytes
                )
            }
            InternalEvent::IoReadArrow { entry } => {
                write!(f, "io_read_arrow entry={}", usize::from(*entry))
            }
            InternalEvent::IoReadLiquid { entry } => {
                write!(f, "io_read_liquid entry={}", usize::from(*entry))
            }
            InternalEvent::Read {
                entry,
                selection,
                expr,
                cached,
            } => write!(
                f,
                "read entry={} selection={} expr={} cached={:?}",
                usize::from(*entry),
                selection,
                expr,
                cached
            ),
            InternalEvent::Hydrate { entry, cached, new } => write!(
                f,
                "hydrate entry={} cached={:?} new={:?}",
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
                "eval_predicate entry={} selection={} cached={:?}",
                usize::from(*entry),
                selection,
                cached
            ),
            InternalEvent::TryReadLiquid { entry } => {
                write!(f, "try_read_liquid entry={}", usize::from(*entry))
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

    #[cfg(test)]
    pub fn drain(&self) -> EventTrace {
        EventTrace {
            events: std::mem::take(&mut *self.events.lock().unwrap()),
        }
    }
}

#[cfg(test)]
#[derive(PartialEq, Eq)]
pub(crate) struct EventTrace {
    events: Vec<InternalEvent>,
}

#[cfg(test)]
impl From<Vec<InternalEvent>> for EventTrace {
    fn from(events: Vec<InternalEvent>) -> Self {
        Self { events }
    }
}

#[cfg(test)]
impl fmt::Display for EventTrace {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "{:?}", self)
    }
}

#[cfg(test)]
impl fmt::Debug for EventTrace {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "EventTrace: [")?;
        for event in &self.events {
            writeln!(f, "\t{}", event)?;
        }
        writeln!(f, "]")?;
        Ok(())
    }
}
