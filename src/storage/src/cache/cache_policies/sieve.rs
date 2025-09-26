//! Size-aware SIEVE cache policy implementation.

use std::{collections::HashMap, fmt, ptr::NonNull, sync::Arc};

use crate::{cache::utils::EntryID, sync::Mutex};

use super::{
    CachePolicy,
    doubly_linked_list::{DoublyLinkedList, DoublyLinkedNode, drop_boxed_node},
};

#[derive(Debug)]
struct SieveNode {
    entry_id: EntryID,
    visited: bool,
}

type NodePtr = NonNull<DoublyLinkedNode<SieveNode>>;

#[derive(Debug, Default)]
struct SieveInternalState {
    map: HashMap<EntryID, NodePtr>,
    list: DoublyLinkedList<SieveNode>,
    hand: Option<NodePtr>,
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
}

unsafe impl Send for SievePolicy {}
unsafe impl Sync for SievePolicy {}

impl CachePolicy for SievePolicy {
    fn find_victim(&self, cnt: usize) -> Vec<EntryID> {
        let mut state = self.state.lock().unwrap();
        let mut advices = Vec::with_capacity(cnt);
        for _ in 0..cnt {
            let hand_ptr = match state.hand {
                Some(ptr) => Some(ptr),
                None => state.list.tail(),
            };
            let mut hand_ptr = match hand_ptr {
                Some(p) => p,
                None => break,
            };
            loop {
                if unsafe { hand_ptr.as_ref() }.data.visited {
                    unsafe { hand_ptr.as_mut() }.data.visited = false;
                    let prev = unsafe { hand_ptr.as_ref().prev };
                    let next_hand = prev
                        .or(state.list.tail())
                        .expect("non-empty list must have a tail");
                    hand_ptr = next_hand;
                    state.hand = Some(next_hand);
                } else {
                    let victim_id = unsafe { hand_ptr.as_ref().data.entry_id };
                    let prev = unsafe { hand_ptr.as_ref().prev };
                    let node_ptr = state.map.remove(&victim_id).unwrap();
                    unsafe {
                        state.list.unlink(node_ptr);
                        drop_boxed_node(node_ptr);
                    }
                    state.total_size -= self.entry_size(&victim_id);
                    advices.push(victim_id);
                    state.hand = prev.or(state.list.tail());
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
                    node_ptr.as_mut().data.visited = true;
                }
            }
            return;
        }

        let was_empty = state.list.head().is_none();
        let node = DoublyLinkedNode::new(SieveNode {
            entry_id: *entry_id,
            visited: false,
        });
        let node_ptr = NonNull::from(Box::leak(node));
        state.map.insert(*entry_id, node_ptr);
        unsafe {
            state.list.push_front(node_ptr);
        }
        if was_empty {
            state.hand = Some(node_ptr);
        }
        state.total_size += self.entry_size(entry_id);
    }

    fn notify_access(&self, entry_id: &EntryID) {
        let state = self.state.lock().unwrap();
        if let Some(mut node_ptr) = state.map.get(entry_id).copied() {
            unsafe {
                node_ptr.as_mut().data.visited = true;
            }
        }
    }
}

impl Drop for SievePolicy {
    fn drop(&mut self) {
        let mut state = self.state.lock().unwrap();
        let handles: Vec<_> = state.map.drain().map(|(_, ptr)| ptr).collect();
        for node_ptr in handles {
            unsafe {
                state.list.unlink(node_ptr);
                drop_boxed_node(node_ptr);
            }
        }
        unsafe {
            state.list.drop_all();
        }
        state.hand = None;
        state.total_size = 0;
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
        let advice = policy.find_victim(1);
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
        assert_eq!(policy.find_victim(1), vec![]);
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
