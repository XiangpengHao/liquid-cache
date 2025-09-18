//! Size-aware SIEVE cache policy implementation.

use std::{collections::HashMap, fmt, ptr::NonNull, sync::Arc};

use crate::{cache::utils::EntryID, sync::Mutex};

use super::CachePolicy;

#[derive(Debug)]
struct SieveNode {
    entry_id: EntryID,
    visited: bool,
    prev: Option<NonNull<SieveNode>>,
    next: Option<NonNull<SieveNode>>,
}

#[derive(Debug, Default)]
struct SieveInternalState {
    map: HashMap<EntryID, NonNull<SieveNode>>,
    head: Option<NonNull<SieveNode>>,
    tail: Option<NonNull<SieveNode>>,
    hand: Option<NonNull<SieveNode>>,
    total_size: usize,
}

type SieveEntrySizeFn = Option<Arc<dyn Fn(&EntryID) -> usize + Send + Sync>>;

/// The policy that implements object size aware SIEVE algorithm using a HashMap and a doubly linked list.
#[derive(Default)]
pub struct SievePolicy {
    state: Mutex<SieveInternalState>,
    size_of: SieveEntrySizeFn,
}

impl fmt::Debug for SievePolicy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SievePolicy")
            .field("state", &self.state)
            .finish()
    }
}

impl SievePolicy {
    /// Create a new [`SievePolicy`].
    pub fn new(size_of: SieveEntrySizeFn) -> Self {
        Self {
            state: Mutex::new(SieveInternalState::default()),
            size_of,
        }
    }

    fn entry_size(&self, entry_id: &EntryID) -> usize {
        self.size_of.as_ref().map(|f| f(entry_id)).unwrap_or(1)
    }

    unsafe fn push_front(&self, state: &mut SieveInternalState, mut node_ptr: NonNull<SieveNode>) {
        let node = unsafe { node_ptr.as_mut() };
        node.prev = None;
        node.next = state.head;
        if let Some(mut head) = state.head {
            unsafe { head.as_mut().prev = Some(node_ptr) };
        } else {
            state.tail = Some(node_ptr);
            state.hand = Some(node_ptr);
        }
        state.head = Some(node_ptr);
    }

    unsafe fn unlink_node(&self, state: &mut SieveInternalState, mut node_ptr: NonNull<SieveNode>) {
        unsafe {
            let node = node_ptr.as_mut();
            match node.prev {
                Some(mut prev) => prev.as_mut().next = node.next,
                None => state.head = node.next,
            }
            match node.next {
                Some(mut next) => next.as_mut().prev = node.prev,
                None => state.tail = node.prev,
            }
            if state.hand == Some(node_ptr) {
                state.hand = node.prev;
            }
            node.prev = None;
            node.next = None;
        }
    }
}

unsafe impl Send for SievePolicy {}
unsafe impl Sync for SievePolicy {}

impl CachePolicy for SievePolicy {
    fn advise(&self, cnt: usize) -> Vec<EntryID> {
        let mut state = self.state.lock().unwrap();
        let mut advices = Vec::with_capacity(cnt);
        for _ in 0..cnt {
            let mut hand_ptr = match state.hand.or(state.tail) {
                Some(p) => p,
                None => break,
            };
            loop {
                let node = unsafe { hand_ptr.as_mut() };
                if node.visited {
                    node.visited = false;
                    hand_ptr = node.prev.unwrap_or(state.tail.unwrap());
                    state.hand = Some(hand_ptr);
                } else {
                    let victim_id = node.entry_id;
                    let node_ptr = state.map.remove(&victim_id).unwrap();
                    unsafe {
                        self.unlink_node(&mut state, node_ptr);
                        drop(Box::from_raw(node_ptr.as_ptr()));
                    }
                    state.total_size -= self.entry_size(&victim_id);
                    advices.push(victim_id);
                    state.hand = node.prev.or(state.tail);
                    break;
                }
            }
        }
        advices
    }

    fn notify_insert(&self, entry_id: &EntryID) {
        let mut state = self.state.lock().unwrap();
        if state.map.contains_key(entry_id) {
            if let Some(mut node_ptr) = state.map.get(entry_id).copied() {
                unsafe {
                    node_ptr.as_mut().visited = true;
                }
            }
            return;
        }

        let node = SieveNode {
            entry_id: *entry_id,
            prev: None,
            next: None,
            visited: false,
        };
        let node_ptr = NonNull::new(Box::into_raw(Box::new(node))).unwrap();
        state.map.insert(*entry_id, node_ptr);
        unsafe {
            self.push_front(&mut state, node_ptr);
        };
        state.total_size += self.entry_size(entry_id);
    }

    fn notify_access(&self, entry_id: &EntryID) {
        let state = self.state.lock().unwrap();
        if let Some(mut node_ptr) = state.map.get(entry_id).copied() {
            unsafe {
                node_ptr.as_mut().visited = true;
            };
        }
    }
}

impl Drop for SievePolicy {
    fn drop(&mut self) {
        let mut state = self.state.lock().unwrap();
        for (_, node_ptr) in state.map.drain() {
            unsafe {
                drop(Box::from_raw(node_ptr.as_ptr()));
            }
        }
        state.head = None;
        state.tail = None;
        state.hand = None;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cache::utils::{EntryID, create_cache_store, create_test_arrow_array};

    fn entry(id: usize) -> EntryID {
        id.into()
    }

    fn assert_evict_advice(policy: &SievePolicy, expect_evict: EntryID) {
        let advice = policy.advise(1);
        assert_eq!(advice, vec![expect_evict]);
    }

    #[test]
    fn test_sieve_insert_order() {
        let policy = SievePolicy::new(None);
        let e1 = entry(1);
        let e2 = entry(2);
        let e3 = entry(3);

        policy.notify_insert(&e1);
        policy.notify_insert(&e2);
        policy.notify_insert(&e3);

        assert_evict_advice(&policy, e1);
    }

    #[test]
    fn test_sieve_access_sets_visited() {
        let policy = SievePolicy::new(None);
        let e1 = entry(1);
        let e2 = entry(2);
        let e3 = entry(3);

        policy.notify_insert(&e1);
        policy.notify_insert(&e2);
        policy.notify_insert(&e3);

        policy.notify_access(&e1);
        assert_evict_advice(&policy, e2);
    }

    #[test]
    fn test_sieve_reinsert_marks_visited() {
        let policy = SievePolicy::new(None);
        let e1 = entry(1);
        let e2 = entry(2);

        policy.notify_insert(&e1);
        policy.notify_insert(&e2);

        policy.notify_insert(&e1);

        assert_evict_advice(&policy, e2);
    }

    #[test]
    fn test_sieve_advise_empty() {
        let policy = SievePolicy::new(None);
        assert_eq!(policy.advise(1), vec![]);
    }

    #[test]
    fn test_sieve_with_sizeof_closure_defined() {
        let policy = SievePolicy::new(Some(Arc::new(
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
    fn test_sieve_sizeof_without_closure() {
        let policy = SievePolicy::new(None);

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
    fn test_sieve_integration() {
        let advisor = SievePolicy::new(None);
        let store = create_cache_store(3000, Box::new(advisor));

        let entry_id1 = EntryID::from(1);
        let entry_id2 = EntryID::from(2);
        let entry_id3 = EntryID::from(3);

        store.insert(entry_id1, create_test_arrow_array(100));
        store.insert(entry_id2, create_test_arrow_array(100));
        store.insert(entry_id3, create_test_arrow_array(100));

        assert!(store.get(&entry_id1).is_some());
        assert!(store.get(&entry_id2).is_some());
        assert!(store.get(&entry_id3).is_some());
    }
}
