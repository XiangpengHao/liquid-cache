use super::parser::{CacheKind, TraceEvent};
use std::collections::BTreeMap;

/// Represents a cache entry with its current state
#[derive(Debug, Clone)]
pub struct CacheEntry {
    pub entry_id: u64,
    /// History of transformations - last element is the current kind
    pub history: Vec<CacheKind>,
}

impl CacheEntry {
    /// Get the current kind (last element in history)
    pub fn current_kind(&self) -> &CacheKind {
        self.history
            .last()
            .expect("Entry must have at least one kind in history")
    }
}

/// Disk I/O statistics
#[derive(Debug, Clone, Default)]
pub struct IoStats {
    pub read_requests: usize,
    pub write_requests: usize,
    pub bytes_read: u64,
    pub bytes_written: u64,
}

/// Tracks the delta changes in I/O stats for the current event
#[derive(Debug, Clone, Default)]
pub struct IoStatsDelta {
    pub read_requests_delta: usize,
    pub write_requests_delta: usize,
    pub bytes_read_delta: u64,
    pub bytes_written_delta: u64,
}

/// Operation happening on an entry
#[derive(Debug, Clone, PartialEq)]
pub enum EntryOperation {
    Reading { expr: Option<String> },
    ReadingSqueezed { expr: String },
    Hydrating { to: CacheKind },
    IoRead,
    IoWrite,
}

/// Victim status for entries
#[derive(Debug, Clone, PartialEq)]
pub enum VictimStatus {
    /// Entry is selected as victim but not yet processed
    Selected,
    /// Entry has been processed as victim (squeezed)
    Squeezed,
}

/// Represents the complete cache state at a point in time
#[derive(Debug, Clone)]
pub struct CacheState {
    /// Map of entry_id -> CacheEntry
    pub entries: BTreeMap<u64, CacheEntry>,
    /// Current squeeze victims being processed
    pub squeeze_victims: Vec<u64>,
    /// Whether we're currently in a squeeze operation
    pub in_squeeze: bool,
    /// Disk I/O statistics
    pub io_stats: IoStats,
    /// Delta changes in I/O stats for the current event
    pub io_stats_delta: IoStatsDelta,
    /// Current operations on entries (entry_id -> operation)
    pub current_operations: BTreeMap<u64, EntryOperation>,
    /// Victim status for entries (entry_id -> status)
    pub victim_status: BTreeMap<u64, VictimStatus>,
    /// Failed insert attempts (entry_id -> attempted kind) - shown as ghost entries
    pub failed_inserts: BTreeMap<u64, CacheKind>,
}

impl CacheState {
    pub fn new() -> Self {
        Self {
            entries: BTreeMap::new(),
            squeeze_victims: Vec::new(),
            in_squeeze: false,
            io_stats: IoStats::default(),
            io_stats_delta: IoStatsDelta::default(),
            current_operations: BTreeMap::new(),
            victim_status: BTreeMap::new(),
            failed_inserts: BTreeMap::new(),
        }
    }

    /// Get all entries as a sorted vector
    pub fn get_entries(&self) -> Vec<&CacheEntry> {
        self.entries.values().collect()
    }

    /// Count entries by kind
    pub fn count_by_kind(&self, kind: &CacheKind) -> usize {
        self.entries
            .values()
            .filter(|e| e.current_kind() == kind)
            .count()
    }

    /// Get total number of entries
    pub fn total_entries(&self) -> usize {
        self.entries.len()
    }
}

/// The cache simulator that processes events and maintains state
pub struct CacheSimulator {
    events: Vec<TraceEvent>,
    current_index: usize,
    state: CacheState,
    history: Vec<CacheState>,
}

impl CacheSimulator {
    pub fn new(events: Vec<TraceEvent>) -> Self {
        let initial_state = CacheState::new();
        Self {
            events,
            current_index: 0,
            state: initial_state.clone(),
            history: vec![initial_state],
        }
    }

    /// Get the current state
    pub fn current_state(&self) -> &CacheState {
        &self.state
    }

    /// Get the current event index
    pub fn current_index(&self) -> usize {
        self.current_index
    }

    /// Get the total number of events
    pub fn total_events(&self) -> usize {
        self.events.len()
    }

    /// Check if we can step forward
    pub fn can_step_forward(&self) -> bool {
        self.current_index < self.events.len()
    }

    /// Step forward one event
    pub fn step_forward(&mut self) {
        if !self.can_step_forward() {
            return;
        }

        let event = self.events[self.current_index].clone();
        self.apply_event(&event);
        self.current_index += 1;
        self.history.push(self.state.clone());
    }

    /// Reset to the beginning
    pub fn reset(&mut self) {
        self.current_index = 0;
        self.state = CacheState::new();
        self.history = vec![self.state.clone()];
    }

    /// Jump to a specific index by rebuilding state from the beginning
    pub fn jump_to(&mut self, index: usize) {
        if index > self.events.len() {
            return;
        }

        // Always rebuild from the beginning - simple and clean
        self.reset();
        while self.current_index < index {
            self.step_forward();
        }
    }

    /// Apply an event to the current state
    fn apply_event(&mut self, event: &TraceEvent) {
        // Clear previous operations and reset I/O deltas
        self.state.current_operations.clear();
        self.state.io_stats_delta = IoStatsDelta::default();

        match event {
            TraceEvent::InsertSuccess { entry, kind } => {
                if let Some(existing) = self.state.entries.get_mut(entry) {
                    // Entry already exists, check if kind changed
                    if existing.current_kind() != kind {
                        existing.history.push(kind.clone());
                    }
                } else {
                    // New entry - start history with this kind
                    self.state.entries.insert(
                        *entry,
                        CacheEntry {
                            entry_id: *entry,
                            history: vec![kind.clone()],
                        },
                    );
                }
                // Clear victim status and failed insert on successful insert
                self.state.victim_status.remove(entry);
                self.state.failed_inserts.remove(entry);
            }
            TraceEvent::InsertFailed { entry, kind } => {
                // Track failed insert as ghost entry
                self.state.failed_inserts.insert(*entry, kind.clone());
            }
            TraceEvent::SqueezeBegin { victims } => {
                self.state.in_squeeze = true;
                self.state.squeeze_victims = victims.clone();
                // Mark all victims as selected
                for victim in victims {
                    self.state.victim_status.insert(*victim, VictimStatus::Selected);
                }
            }
            TraceEvent::SqueezeVictim { entry } => {
                // Remove from squeeze victims list
                self.state.squeeze_victims.retain(|v| v != entry);
                // Mark as squeezed
                self.state.victim_status.insert(*entry, VictimStatus::Squeezed);

                // Check if squeeze is complete
                if self.state.squeeze_victims.is_empty() {
                    self.state.in_squeeze = false;
                }
            }
            TraceEvent::IoWrite { entry, bytes, .. } => {
                self.state.io_stats.write_requests += 1;
                self.state.io_stats.bytes_written += bytes;
                self.state.io_stats_delta.write_requests_delta = 1;
                self.state.io_stats_delta.bytes_written_delta = *bytes;
                self.state
                    .current_operations
                    .insert(*entry, EntryOperation::IoWrite);
            }
            TraceEvent::IoReadArrow { entry, bytes, .. } => {
                self.state.io_stats.read_requests += 1;
                self.state.io_stats.bytes_read += bytes;
                self.state.io_stats_delta.read_requests_delta = 1;
                self.state.io_stats_delta.bytes_read_delta = *bytes;
                self.state
                    .current_operations
                    .insert(*entry, EntryOperation::IoRead);
            }
            TraceEvent::Hydrate { entry, new, .. } => {
                // Hydrate is just an indication, doesn't change state
                self.state
                    .current_operations
                    .insert(*entry, EntryOperation::Hydrating { to: new.clone() });
            }
            TraceEvent::Read { entry, expr, .. } => {
                self.state
                    .current_operations
                    .insert(*entry, EntryOperation::Reading { expr: expr.clone() });
            }
            TraceEvent::ReadSqueezedDate { entry, expression } => {
                self.state
                    .current_operations
                    .insert(*entry, EntryOperation::ReadingSqueezed { expr: expression.clone() });
            }
            TraceEvent::Unknown { .. } => {
                // Unknown events don't change state
            }
        }
    }

    /// Get all events
    pub fn events(&self) -> &[TraceEvent] {
        &self.events
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::trace::parser::parse_trace;

    #[test]
    fn test_simulator_basic() {
        let trace = "event=insert_success entry=0 kind=MemoryArrow\nevent=insert_success entry=1 kind=DiskArrow";
        let events = parse_trace(trace);
        let mut sim = CacheSimulator::new(events);

        assert_eq!(sim.current_index(), 0);
        assert_eq!(sim.total_events(), 2);
        assert_eq!(sim.current_state().total_entries(), 0);

        sim.step_forward();
        assert_eq!(sim.current_state().total_entries(), 1);

        sim.step_forward();
        assert_eq!(sim.current_state().total_entries(), 2);
    }
}
