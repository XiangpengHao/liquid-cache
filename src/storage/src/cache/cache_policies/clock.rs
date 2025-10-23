//! CLOCK (second-chance) cache policy implementation with optional size awareness.

use std::{collections::HashMap, fmt, ptr::NonNull, sync::Arc};

use crate::{
    cache::{cached_batch::CachedBatchType, utils::EntryID},
    sync::Mutex,
};

use super::{
    CachePolicy,
    doubly_linked_list::{DoublyLinkedList, DoublyLinkedNode, drop_boxed_node},
};

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
}

type NodePtr = NonNull<DoublyLinkedNode<ClockNode>>;

#[derive(Debug, Default)]
struct ClockInternalState {
    map: HashMap<EntryID, NodePtr>,
    list: DoublyLinkedList<ClockNode>,
    hand: Option<NodePtr>,
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
}

unsafe impl Send for ClockPolicy {}
unsafe impl Sync for ClockPolicy {}

impl CachePolicy for ClockPolicy {
    fn find_victim(&self, cnt: usize) -> Vec<EntryID> {
        let mut state = self.state.lock().unwrap();
        if cnt == 0 {
            return Vec::new();
        }

        let mut evicted = Vec::with_capacity(cnt);
        let mut cursor = match state.hand {
            Some(ptr) => Some(ptr),
            None => state.list.head(),
        };

        for _ in 0..cnt {
            loop {
                let Some(handle) = cursor else {
                    state.hand = None;
                    break;
                };

                let mut handle_ptr = handle;
                if unsafe { handle_ptr.as_ref() }.data.referenced {
                    unsafe { handle_ptr.as_mut() }.data.referenced = false;
                    let next = unsafe { handle_ptr.as_ref().next }.or(state.list.head());
                    cursor = next;
                    state.hand = next;
                } else {
                    let victim_id = unsafe { handle_ptr.as_ref().data.entry_id };
                    let succ = unsafe { handle_ptr.as_ref().next };
                    state
                        .map
                        .remove(&victim_id)
                        .expect("pointer must exist in map");
                    unsafe {
                        state.list.unlink(handle_ptr);
                        drop_boxed_node(handle_ptr);
                    }
                    state.total_size -= self.entry_size(&victim_id);
                    state.hand = succ.or(state.list.head());
                    evicted.push(victim_id);
                    cursor = state.hand;
                    break;
                }
            }

            if state.hand.is_none() {
                break;
            }
        }

        evicted
    }

    fn notify_insert(&self, entry_id: &EntryID, _batch_type: CachedBatchType) {
        let mut state = self.state.lock().unwrap();

        if let Some(mut existing) = state.map.get(entry_id).copied() {
            unsafe {
                existing.as_mut().data.referenced = true;
            }
            return;
        }

        let node = DoublyLinkedNode::new(ClockNode {
            entry_id: *entry_id,
            referenced: true,
        });
        let new_ptr = NonNull::from(Box::leak(node));

        unsafe { state.list.push_back(new_ptr) };
        if state.hand.is_none() {
            state.hand = Some(new_ptr);
        }

        state.map.insert(*entry_id, new_ptr);
        state.total_size += self.entry_size(entry_id);
    }

    fn notify_access(&self, entry_id: &EntryID, _batch_type: CachedBatchType) {
        let state = self.state.lock().unwrap();
        if let Some(mut handle) = state.map.get(entry_id).copied() {
            unsafe {
                handle.as_mut().data.referenced = true;
            }
        }
    }
}

impl Drop for ClockPolicy {
    fn drop(&mut self) {
        if let Ok(mut state) = self.state.lock() {
            let handles: Vec<_> = state.map.drain().map(|(_, ptr)| ptr).collect();
            for ptr in handles {
                unsafe {
                    state.list.unlink(ptr);
                    drop_boxed_node(ptr);
                }
            }
            unsafe {
                state.list.drop_all();
            }
            state.hand = None;
            state.total_size = 0;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cache::{
        cached_batch::CachedBatch,
        utils::{EntryID, create_cache_store, create_test_arrow_array},
    };

    fn entry(id: usize) -> EntryID {
        id.into()
    }

    #[test]
    fn test_clock_policy_insertion_order() {
        let advisor = ClockPolicy::new();

        let entry_id1 = EntryID::from(1);
        let entry_id2 = EntryID::from(2);
        let entry_id3 = EntryID::from(3);

        advisor.notify_insert(&entry_id1, CachedBatchType::MemoryArrow);
        advisor.notify_insert(&entry_id2, CachedBatchType::MemoryArrow);
        advisor.notify_insert(&entry_id3, CachedBatchType::MemoryArrow);

        assert_eq!(advisor.find_victim(1), vec![entry_id1]);
    }

    #[test]
    fn test_clock_policy_sequential_evictions() {
        let advisor = ClockPolicy::new();

        let entry_id1 = EntryID::from(1);
        let entry_id2 = EntryID::from(2);
        let entry_id3 = EntryID::from(3);

        advisor.notify_insert(&entry_id1, CachedBatchType::MemoryArrow);
        advisor.notify_insert(&entry_id2, CachedBatchType::MemoryArrow);
        advisor.notify_insert(&entry_id3, CachedBatchType::MemoryArrow);

        assert_eq!(advisor.find_victim(1), vec![entry_id1]);
        assert_eq!(advisor.find_victim(1), vec![entry_id2]);
        assert_eq!(advisor.find_victim(1), vec![entry_id3]);
    }

    #[test]
    fn test_clock_policy_single_item() {
        let advisor = ClockPolicy::new();

        let entry_id1 = EntryID::from(1);
        advisor.notify_insert(&entry_id1, CachedBatchType::MemoryArrow);

        assert_eq!(advisor.find_victim(1), vec![entry_id1]);
    }

    #[test]
    fn test_clock_policy_advise_empty() {
        let advisor = ClockPolicy::new();

        assert_eq!(advisor.find_victim(1), vec![]);
    }

    #[tokio::test]
    async fn test_clock_policy_integration_with_store() {
        let advisor = ClockPolicy::new();
        let store = create_cache_store(3000, Box::new(advisor));

        let entry_id1 = EntryID::from(1);
        let entry_id2 = EntryID::from(2);
        let entry_id3 = EntryID::from(3);

        store.insert(entry_id1, create_test_arrow_array(100)).await;
        store.insert(entry_id2, create_test_arrow_array(100)).await;
        store.insert(entry_id3, create_test_arrow_array(100)).await;

        let entry_id4 = EntryID::from(4);
        store.insert(entry_id4, create_test_arrow_array(100)).await;

        if let Some(data) = store.index().get(&entry_id1) {
            assert!(matches!(data, CachedBatch::DiskLiquid(_)));
        }
        assert!(store.index().get(&entry_id2).is_some());
        assert!(store.index().get(&entry_id3).is_some());
        assert!(store.index().get(&entry_id4).is_some());
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

        policy.notify_insert(&e1, CachedBatchType::MemoryArrow);
        policy.notify_insert(&e2, CachedBatchType::MemoryArrow);
        policy.notify_insert(&e3, CachedBatchType::MemoryArrow);

        let state = policy.state.lock().unwrap();
        assert_eq!(state.total_size, 102);
    }

    #[test]
    fn test_clock_policy_size_awareness_without_closure() {
        let policy = ClockPolicy::new();

        let e1 = entry(1);
        let e2 = entry(2);
        let e3 = entry(11);

        policy.notify_insert(&e1, CachedBatchType::MemoryArrow);
        policy.notify_insert(&e2, CachedBatchType::MemoryArrow);
        policy.notify_insert(&e3, CachedBatchType::MemoryArrow);

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

        policy.notify_insert(&e1, CachedBatchType::MemoryArrow);
        policy.notify_insert(&e2, CachedBatchType::MemoryArrow);
        policy.notify_insert(&e3, CachedBatchType::MemoryArrow);

        {
            let state = policy.state.lock().unwrap();
            assert_eq!(state.total_size, 102);
        }

        let evicted = policy.find_victim(1);
        assert_eq!(evicted, vec![e1]);

        {
            let state = policy.state.lock().unwrap();
            assert_eq!(state.total_size, 101);
        }

        let evicted = policy.find_victim(1);
        assert_eq!(evicted, vec![e2]);

        {
            let state = policy.state.lock().unwrap();
            assert_eq!(state.total_size, 100);
        }
    }

    #[test]
    fn test_clock_policy_reinsert_sets_reference_bit() {
        let policy = ClockPolicy::new();
        let entry_id = entry(42);

        policy.notify_insert(&entry_id, CachedBatchType::MemoryArrow);

        {
            let state = policy.state.lock().unwrap();
            let mut node_ptr = state.map.get(&entry_id).copied().unwrap();
            unsafe {
                node_ptr.as_mut().data.referenced = false;
            }
        }

        policy.notify_insert(&entry_id, CachedBatchType::MemoryArrow);

        let state = policy.state.lock().unwrap();
        let node_ptr = state.map.get(&entry_id).copied().unwrap();
        unsafe {
            assert!(node_ptr.as_ref().data.referenced);
        }
        assert_eq!(state.map.len(), 1);
    }
}
