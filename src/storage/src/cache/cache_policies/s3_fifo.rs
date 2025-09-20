//! S3 FIFO cache policy implementation

use std::collections::{HashMap, HashSet, VecDeque};
use std::fmt;
use std::sync::{Arc, Mutex};

use crate::cache::EntryID;
use crate::cache_policies::CachePolicy;

type EntryFreq = u8;

#[derive(Debug, Default)]
struct S3FifoInternalState {
    small: VecDeque<EntryID>,
    main: VecDeque<EntryID>,
    ghost: VecDeque<EntryID>,
    ghost_set: HashSet<EntryID>,

    frequency: HashMap<EntryID, EntryFreq>,

    small_queue_size: usize,
    main_queue_size: usize,
    total_size: usize,
}

impl S3FifoInternalState {
    fn cap_frequency(freq: u8) -> u8 {
        std::cmp::min(freq, 3)
    }

    fn inc_frequency(&mut self, entry_id: &EntryID) {
        if let Some(freq) = self.frequency.get_mut(entry_id) {
            *freq = Self::cap_frequency(*freq + 1);
        }
    }

    fn dec_frequency(&mut self, entry_id: &EntryID) {
        if let Some(freq) = self.frequency.get_mut(entry_id) {
            *freq = Self::cap_frequency(*freq - 1);
        }
    }

    fn inc_small_queue_size(&mut self, size: usize) {
        self.small_queue_size += size;
        self.total_size += size;
    }

    fn dec_small_queue_size(&mut self, size: usize) {
        self.small_queue_size -= size;
        self.total_size -= size;
    }

    fn inc_main_queue_size(&mut self, size: usize) {
        self.main_queue_size += size;
        self.total_size += size;
    }

    fn dec_main_queue_size(&mut self, size: usize) {
        self.main_queue_size -= size;
        self.total_size -= size;
    }

    fn small_queue_fraction(&self) -> f32 {
        if self.total_size == 0 {
            0.0
        } else {
            self.small_queue_size as f32 / self.total_size as f32
        }
    }

    fn check_if_entry_exists_in_small_or_main(&self, entry_id: &EntryID) -> bool {
        self.frequency.contains_key(entry_id) && !self.ghost_set.contains(entry_id)
    }
}

type S3FifoEntrySizeFn = Option<Arc<dyn Fn(&EntryID) -> usize + Send + Sync>>;

/// The policy that implements object size aware S3Fifo algorithm using Deque.
#[derive(Default)]
pub struct S3FifoPolicy {
    state: Mutex<S3FifoInternalState>,
    size_of: S3FifoEntrySizeFn,
}

impl fmt::Debug for S3FifoPolicy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("S3FifoPolicy")
            .field("state", &self.state)
            .finish()
    }
}

unsafe impl Send for S3FifoPolicy {}
unsafe impl Sync for S3FifoPolicy {}

impl S3FifoPolicy {
    /// Create a new [`S3FifoPolicy`].
    pub fn new(size_of: S3FifoEntrySizeFn) -> Self {
        Self {
            state: Mutex::new(S3FifoInternalState::default()),
            size_of,
        }
    }

    fn entry_size(&self, entry_id: &EntryID) -> usize {
        self.size_of.as_ref().map(|f| f(entry_id)).unwrap_or(1)
    }

    fn evict_from_small(&self, state: &mut S3FifoInternalState) -> Option<EntryID> {
        if let Some(element) = state.small.pop_back() {
            let freq = *state.frequency.get_mut(&element).unwrap_or(&mut 0);
            let entry_size = self.entry_size(&element);
            state.dec_small_queue_size(entry_size);

            // If frequency is greater than one, give it another chance, by moving to main queue
            if freq > 1 {
                state.main.push_front(element);
                state.inc_main_queue_size(entry_size);
                None
            } else {
                // Move it to ghost queue
                state.ghost.push_front(element);
                state.ghost_set.insert(element);
                state.frequency.remove(&element);
                Some(element)
            }
        } else {
            None
        }
    }

    fn evict_from_main(&self, state: &mut S3FifoInternalState) -> Option<EntryID> {
        if let Some(element) = state.small.pop_back() {
            let freq = *state.frequency.get_mut(&element).unwrap_or(&mut 0);
            let entry_size = self.entry_size(&element);
            state.dec_main_queue_size(entry_size);
            if freq > 0 {
                state.main.push_front(element);
                state.inc_main_queue_size(entry_size);
                state.dec_frequency(&element);
                None
            } else {
                state.frequency.remove(&element);
                Some(element)
            }
        } else {
            None
        }
    }
}

impl CachePolicy for S3FifoPolicy {
    fn advise(&self, cnt: usize) -> Vec<EntryID> {
        let mut state = self.state.lock().unwrap();
        let mut advices = Vec::with_capacity(cnt);
        let threshold_for_small_eviction = 0.1;
        while advices.len() < cnt && state.total_size > 0 {
            let victim = if !state.small.is_empty()
                && state.small_queue_fraction() >= threshold_for_small_eviction
            {
                self.evict_from_small(&mut state)
            } else {
                self.evict_from_main(&mut state)
            };

            if let Some(v) = victim {
                advices.push(v);
            } else {
                break;
            }
        }
        advices
    }

    fn notify_insert(&self, entry_id: &EntryID) {
        let mut state = self.state.lock().unwrap();
        let entry_size = self.entry_size(entry_id);

        if state.check_if_entry_exists_in_small_or_main(entry_id) {
            state.inc_frequency(entry_id);
        } else if state.ghost_set.contains(entry_id) {
            state.ghost_set.remove(entry_id);
            state.ghost.retain(|x| *x != *entry_id);
            state.main.push_front(*entry_id);
            state.inc_main_queue_size(entry_size);
        } else {
            state.small.push_front(*entry_id);
            state.inc_small_queue_size(entry_size);
            state.frequency.insert(*entry_id, 0);
        }
    }

    fn notify_access(&self, entry_id: &EntryID) {
        let mut state = self.state.lock().unwrap();
        if state.check_if_entry_exists_in_small_or_main(entry_id) {
            state.inc_frequency(entry_id);
        }
    }
}

impl Drop for S3FifoPolicy {
    fn drop(&mut self) {
        let mut state = self.state.lock().unwrap();
        state.small.clear();
        state.main.clear();
        state.ghost.clear();
        state.ghost_set.clear();
        state.frequency.clear();
        state.total_size = 0;
        state.small_queue_size = 0;
        state.main_queue_size = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cache::utils::EntryID;

    fn entry(id: usize) -> EntryID {
        id.into()
    }

    #[test]
    fn test_s3fifo_basic_insert_eviction() {
        let policy = S3FifoPolicy::new(None);
        let e1 = entry(1);
        let e2 = entry(2);
        let e3 = entry(3);

        policy.notify_insert(&e1);
        policy.notify_insert(&e2);
        policy.notify_insert(&e3);

        let evicted = policy.advise(1);
        assert_eq!(evicted.len(), 1);
    }

    #[test]
    fn test_s3fifo_frequency_increase() {
        let policy = S3FifoPolicy::new(None);
        let e1 = entry(1);
        policy.notify_insert(&e1);
        policy.notify_access(&e1);

        let state = policy.state.lock().unwrap();
        assert_eq!(*state.frequency.get(&e1).unwrap(), 1);
    }

    #[test]
    fn test_s3fifo_eviction_order() {
        let policy = S3FifoPolicy::new(None);
        let e1 = entry(1);
        let e2 = entry(2);

        policy.notify_insert(&e1);
        policy.notify_insert(&e2);

        let evicted = policy.advise(1);
        assert_eq!(evicted[0], e1);
    }

    #[test]
    fn test_s3fifo_ghost_promote() {
        let policy = S3FifoPolicy::new(None);
        let e1 = entry(1);

        policy.notify_insert(&e1);
        let evicted = policy.advise(1);
        assert_eq!(evicted[0], e1);

        // Re-insert evicted entry from ghost
        policy.notify_insert(&e1);
        let state = policy.state.lock().unwrap();
        assert!(state.main.contains(&e1));
        assert!(!state.ghost_set.contains(&e1));
    }

    #[test]
    fn test_s3fifo_size_aware_fraction() {
        let policy = S3FifoPolicy::new(Some(Arc::new(
            |id: &EntryID| {
                if id.gt(&entry(10)) { 100 } else { 1 }
            },
        )));
        let e1 = entry(20);
        let e2 = entry(30);

        policy.notify_insert(&e1);
        policy.notify_insert(&e2);

        let state = policy.state.lock().unwrap();
        assert_eq!(state.small_queue_size, 200);
        assert_eq!(state.small_queue_fraction(), 1.0);
    }

    #[test]
    fn test_insert_and_access_updates_freq() {
        let policy = S3FifoPolicy::new(None);
        let e1 = entry(1);

        policy.notify_insert(&e1);
        policy.notify_access(&e1);
        policy.notify_access(&e1);

        let state = policy.state.lock().unwrap();
        assert_eq!(*state.frequency.get(&e1).unwrap(), 2); // capped at 3
    }

    #[test]
    fn test_freq_cap_at_three() {
        let policy = S3FifoPolicy::new(None);
        let e1 = entry(1);

        policy.notify_insert(&e1);
        for _ in 0..10 {
            policy.notify_access(&e1);
        }

        let state = policy.state.lock().unwrap();
        assert_eq!(*state.frequency.get(&e1).unwrap(), 3);
    }

    #[test]
    fn test_eviction_from_s_to_ghost() {
        let policy = S3FifoPolicy::new(None);
        let e1 = entry(1);

        policy.notify_insert(&e1);
        let evicted = policy.advise(1);

        assert_eq!(evicted[0], e1);
        let state = policy.state.lock().unwrap();
        assert!(state.ghost_set.contains(&e1));
        assert!(state.ghost.contains(&e1));
    }

    #[test]
    fn test_eviction_promotes_to_m_if_freq_gt_one() {
        let policy = S3FifoPolicy::new(None);
        let e1 = entry(1);

        policy.notify_insert(&e1);
        policy.notify_access(&e1);
        policy.notify_access(&e1);

        let _ = policy.advise(1);
        let state = policy.state.lock().unwrap();
        assert!(state.main.contains(&e1));
        assert!(!state.ghost_set.contains(&e1));
    }
}
