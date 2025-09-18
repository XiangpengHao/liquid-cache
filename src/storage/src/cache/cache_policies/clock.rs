//! CLOCK (second-chance) cache policy implementation with optional size awareness.

use std::{collections::HashMap, fmt, ptr::NonNull, sync::Arc};

use crate::{cache::utils::EntryID, sync::Mutex};

use super::CachePolicy;

type ClockEntrySizeFn = Option<Arc<dyn Fn(&EntryID) -> usize + Send + Sync>>;

/// The CLOCK (second-chance) eviction policy with optional size awareness.
#[derive(Default)]
pub struct ClockPolicy {
    state: Mutex<ClockInternalState>,
    size_of: ClockEntrySizeFn,
}

#[derive(Debug)]
struct ClockNode {
    entry_id: EntryID,
    referenced: bool,
    prev: Option<NonNull<ClockNode>>,
    next: Option<NonNull<ClockNode>>,
}

#[derive(Debug, Default)]
struct ClockInternalState {
    map: HashMap<EntryID, NonNull<ClockNode>>,
    head: Option<NonNull<ClockNode>>,
    tail: Option<NonNull<ClockNode>>,
    hand: Option<NonNull<ClockNode>>,
    total_size: usize,
}

impl fmt::Debug for ClockPolicy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ClockPolicy")
            .field("state", &self.state)
            .finish()
    }
}

impl ClockPolicy {
    /// Create a new CLOCK policy.
    pub fn new() -> Self {
        Self::new_with_size_fn(None)
    }

    /// Create a new CLOCK policy with size awareness.
    pub fn new_with_size_fn(size_of: ClockEntrySizeFn) -> Self {
        ClockPolicy {
            state: Mutex::new(ClockInternalState::default()),
            size_of,
        }
    }

    fn entry_size(&self, entry_id: &EntryID) -> usize {
        self.size_of.as_ref().map(|f| f(entry_id)).unwrap_or(1)
    }

    unsafe fn unlink_node(&self, state: &mut ClockInternalState, mut node_ptr: NonNull<ClockNode>) {
        unsafe {
            let node = node_ptr.as_mut();

            if let Some(mut p) = node.prev {
                p.as_mut().next = node.next;
            } else {
                state.head = node.next;
            }

            if let Some(mut n) = node.next {
                n.as_mut().prev = node.prev;
            } else {
                state.tail = node.prev;
            }

            if state.hand == Some(node_ptr) {
                state.hand = node.next.or(node.prev);
            }

            node.prev = None;
            node.next = None;
        }
    }

    unsafe fn insert_after(
        &self,
        state: &mut ClockInternalState,
        mut new_ptr: NonNull<ClockNode>,
        mut existing_ptr: NonNull<ClockNode>,
    ) {
        unsafe {
            let new_node = new_ptr.as_mut();
            let existing_node = existing_ptr.as_mut();

            new_node.prev = Some(existing_ptr);
            new_node.next = existing_node.next;

            if let Some(mut next) = existing_node.next {
                next.as_mut().prev = Some(new_ptr);
            } else {
                state.tail = Some(new_ptr);
            }

            existing_node.next = Some(new_ptr);
        }
    }
}

unsafe impl Send for ClockPolicy {}
unsafe impl Sync for ClockPolicy {}

impl CachePolicy for ClockPolicy {
    fn advise(&self, cnt: usize) -> Vec<EntryID> {
        let mut state = self.state.lock().unwrap();
        if cnt == 0 {
            return Vec::new();
        }

        let mut evicted = Vec::with_capacity(cnt);
        let mut cursor = state.hand.or(state.head);
        for _ in 0..cnt {
            loop {
                let Some(ptr) = cursor else {
                    state.hand = None;
                    break;
                };

                unsafe {
                    let node = ptr.as_ref();
                    if node.referenced {
                        let mut_mut = ptr.as_ptr();
                        (*mut_mut).referenced = false;
                        cursor = node.next.or(state.head);
                        state.hand = cursor;
                    } else {
                        let victim_id = (*ptr.as_ptr()).entry_id;
                        let succ = (*ptr.as_ptr()).next;
                        self.unlink_node(&mut state, ptr);
                        state.map.remove(&victim_id);
                        state.total_size -= self.entry_size(&victim_id);
                        state.hand = succ.or(state.head);
                        evicted.push(victim_id);
                        drop(Box::from_raw(ptr.as_ptr()));
                        cursor = state.hand;
                        break;
                    }
                }
            }

            if state.hand.is_none() {
                break;
            }
        }

        evicted
    }

    fn notify_insert(&self, entry_id: &EntryID) {
        let mut state = self.state.lock().unwrap();

        if let Some(mut existing) = state.map.get(entry_id).copied() {
            unsafe {
                existing.as_mut().referenced = true;
            }
            return;
        }

        let node = ClockNode {
            entry_id: *entry_id,
            referenced: true,
            prev: None,
            next: None,
        };
        let new_ptr = match NonNull::new(Box::into_raw(Box::new(node))) {
            Some(ptr) => ptr,
            None => panic!("Failed to allocate memory for CLOCK node"),
        };

        if let Some(tail_ptr) = state.tail {
            unsafe { self.insert_after(&mut state, new_ptr, tail_ptr) };
            if state.hand.is_none() {
                state.hand = Some(new_ptr);
            }
        } else {
            state.head = Some(new_ptr);
            state.tail = Some(new_ptr);
            state.hand = Some(new_ptr);
        }

        state.map.insert(*entry_id, new_ptr);
        state.total_size += self.entry_size(entry_id);
    }

    fn notify_access(&self, entry_id: &EntryID) {
        let state = self.state.lock().unwrap();
        if let Some(&ptr) = state.map.get(entry_id) {
            unsafe {
                (*ptr.as_ptr()).referenced = true;
            }
        }
    }
}

impl Drop for ClockPolicy {
    fn drop(&mut self) {
        if let Ok(mut state) = self.state.lock() {
            for (_, ptr) in state.map.drain() {
                unsafe {
                    drop(Box::from_raw(ptr.as_ptr()));
                }
            }
            state.head = None;
            state.tail = None;
            state.hand = None;
            state.total_size = 0;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cache::utils::{EntryID, create_cache_store, create_test_arrow_array};

    fn entry(id: usize) -> EntryID {
        id.into()
    }

    #[test]
    fn test_clock_policy_insertion_order() {
        let advisor = ClockPolicy::new();

        let entry_id1 = EntryID::from(1);
        let entry_id2 = EntryID::from(2);
        let entry_id3 = EntryID::from(3);

        advisor.notify_insert(&entry_id1);
        advisor.notify_insert(&entry_id2);
        advisor.notify_insert(&entry_id3);

        assert_eq!(advisor.advise(1), vec![entry_id1]);
    }

    #[test]
    fn test_clock_policy_sequential_evictions() {
        let advisor = ClockPolicy::new();

        let entry_id1 = EntryID::from(1);
        let entry_id2 = EntryID::from(2);
        let entry_id3 = EntryID::from(3);

        advisor.notify_insert(&entry_id1);
        advisor.notify_insert(&entry_id2);
        advisor.notify_insert(&entry_id3);

        assert_eq!(advisor.advise(1), vec![entry_id1]);
        assert_eq!(advisor.advise(1), vec![entry_id2]);
        assert_eq!(advisor.advise(1), vec![entry_id3]);
    }

    #[test]
    fn test_clock_policy_single_item() {
        let advisor = ClockPolicy::new();

        let entry_id1 = EntryID::from(1);
        advisor.notify_insert(&entry_id1);

        assert_eq!(advisor.advise(1), vec![entry_id1]);
    }

    #[test]
    fn test_clock_policy_advise_empty() {
        let advisor = ClockPolicy::new();

        assert_eq!(advisor.advise(1), vec![]);
    }

    #[test]
    fn test_clock_policy_integration_with_store() {
        let advisor = ClockPolicy::new();
        let store = create_cache_store(3000, Box::new(advisor));

        let entry_id1 = EntryID::from(1);
        let entry_id2 = EntryID::from(2);
        let entry_id3 = EntryID::from(3);

        store.insert(entry_id1, create_test_arrow_array(100));
        store.insert(entry_id2, create_test_arrow_array(100));
        store.insert(entry_id3, create_test_arrow_array(100));

        let entry_id4 = EntryID::from(4);
        store.insert(entry_id4, create_test_arrow_array(100));

        if let Some(data) = store.get(&entry_id1) {
            match data.raw_data() {
                crate::cache::cached_data::CachedBatch::DiskLiquid => {}
                _ => panic!("Expected OnDiskLiquid, got {:?}", data.raw_data()),
            }
        }
        assert!(store.get(&entry_id2).is_some());
        assert!(store.get(&entry_id3).is_some());
        assert!(store.get(&entry_id4).is_some());
    }

    #[test]
    fn test_clock_policy_size_awareness_with_closure() {
        let policy =
            ClockPolicy::new_with_size_fn(Some(Arc::new(
                |id: &EntryID| {
                    if id.gt(&entry(10)) { 100 } else { 1 }
                },
            )));

        let e1 = entry(1);
        let e2 = entry(2);
        let e3 = entry(11);

        policy.notify_insert(&e1);
        policy.notify_insert(&e2);
        policy.notify_insert(&e3);

        let state = policy.state.lock().unwrap();
        assert_eq!(state.total_size, 102);
    }

    #[test]
    fn test_clock_policy_size_awareness_without_closure() {
        let policy = ClockPolicy::new();

        let e1 = entry(1);
        let e2 = entry(2);
        let e3 = entry(11);

        policy.notify_insert(&e1);
        policy.notify_insert(&e2);
        policy.notify_insert(&e3);

        let state = policy.state.lock().unwrap();
        assert_eq!(state.total_size, 3);
    }

    #[test]
    fn test_clock_policy_size_tracking_on_eviction() {
        let policy =
            ClockPolicy::new_with_size_fn(Some(Arc::new(
                |id: &EntryID| {
                    if id.gt(&entry(10)) { 100 } else { 1 }
                },
            )));

        let e1 = entry(1);
        let e2 = entry(2);
        let e3 = entry(11);

        policy.notify_insert(&e1);
        policy.notify_insert(&e2);
        policy.notify_insert(&e3);

        {
            let state = policy.state.lock().unwrap();
            assert_eq!(state.total_size, 102);
        }

        let evicted = policy.advise(1);
        assert_eq!(evicted, vec![e1]);

        {
            let state = policy.state.lock().unwrap();
            assert_eq!(state.total_size, 101);
        }

        let evicted = policy.advise(1);
        assert_eq!(evicted, vec![e2]);

        {
            let state = policy.state.lock().unwrap();
            assert_eq!(state.total_size, 100);
        }
    }
}
