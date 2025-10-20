//! LRU cache policy implementation using a hash map and doubly linked list.

use std::{collections::HashMap, ptr::NonNull};

use crate::{
    cache::{cached_data::CachedBatchType, utils::EntryID},
    sync::Mutex,
};

use super::{
    CachePolicy,
    doubly_linked_list::{DoublyLinkedList, DoublyLinkedNode, drop_boxed_node},
};

#[derive(Debug)]
struct LruNode {
    entry_id: EntryID,
}

type NodePtr = NonNull<DoublyLinkedNode<LruNode>>;

#[derive(Debug, Default)]
struct HashList {
    map: HashMap<EntryID, NodePtr>,
    list: DoublyLinkedList<LruNode>,
}

impl HashList {
    fn tail(&self) -> Option<NodePtr> {
        self.list.tail()
    }

    unsafe fn move_to_front(&mut self, node_ptr: NodePtr) {
        unsafe { self.list.move_to_front(node_ptr) };
    }

    unsafe fn push_front(&mut self, node_ptr: NodePtr) {
        unsafe { self.list.push_front(node_ptr) };
    }

    unsafe fn remove_and_release(&mut self, node_ptr: NodePtr) {
        unsafe {
            self.list.unlink(node_ptr);
            drop_boxed_node(node_ptr);
        }
    }
}

impl Drop for HashList {
    fn drop(&mut self) {
        for (_, node_ptr) in self.map.drain() {
            unsafe {
                self.list.unlink(node_ptr);
                drop_boxed_node(node_ptr);
            }
        }
        // Any nodes not tracked in the map (shouldn't happen) get cleaned up here.
        unsafe {
            self.list.drop_all();
        }
    }
}

/// The policy that implement the LRU algorithm using a HashMap and a doubly linked list.
#[derive(Debug, Default)]
pub struct LruPolicy {
    state: Mutex<HashList>,
}

impl LruPolicy {
    /// Create a new [`LruPolicy`].
    pub fn new() -> Self {
        Self {
            state: Mutex::new(HashList::default()),
        }
    }
}

// SAFETY: The Mutex ensures that only one thread accesses the internal state
// (hash map and intrusive list containing NonNull pointers) at a time, making it safe
// to send and share across threads.
unsafe impl Send for LruPolicy {}
unsafe impl Sync for LruPolicy {}

impl CachePolicy for LruPolicy {
    fn find_victim(&self, cnt: usize) -> Vec<EntryID> {
        let mut state = self.state.lock().unwrap();
        if cnt == 0 {
            return vec![];
        }

        let mut advices = Vec::with_capacity(cnt);
        for _ in 0..cnt {
            let Some(tail_ptr) = state.tail() else {
                break;
            };
            let tail_entry_id = unsafe { tail_ptr.as_ref().data.entry_id };
            let node_ptr = state
                .map
                .remove(&tail_entry_id)
                .expect("tail node not found");
            unsafe {
                state.remove_and_release(node_ptr);
            }
            advices.push(tail_entry_id);
        }

        advices
    }

    fn notify_access(&self, entry_id: &EntryID, _batch_type: CachedBatchType) {
        let mut state = self.state.lock().unwrap();
        if let Some(node_ptr) = state.map.get(entry_id).copied() {
            unsafe { state.move_to_front(node_ptr) };
        }
    }

    fn notify_insert(&self, entry_id: &EntryID, _batch_type: CachedBatchType) {
        let mut state = self.state.lock().unwrap();

        if let Some(existing_node_ptr) = state.map.get(entry_id).copied() {
            unsafe { state.move_to_front(existing_node_ptr) };
            return;
        }

        let node = DoublyLinkedNode::new(LruNode {
            entry_id: *entry_id,
        });
        let node_ptr = NonNull::from(Box::leak(node));

        state.map.insert(*entry_id, node_ptr);
        unsafe {
            state.push_front(node_ptr);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cache::utils::{EntryID, create_cache_store, create_test_arrow_array};
    use crate::sync::{Arc, Barrier, thread};
    use std::sync::atomic::{AtomicUsize, Ordering};

    fn entry(id: usize) -> EntryID {
        id.into()
    }

    fn assert_evict_advice(policy: &LruPolicy, expect_evict: EntryID) {
        let advice = policy.find_victim(1);
        assert_eq!(advice, vec![expect_evict]);
    }

    #[test]
    fn test_lru_policy_insertion_order() {
        let policy = LruPolicy::new();
        let e1 = entry(1);
        let e2 = entry(2);
        let e3 = entry(3);

        policy.notify_insert(&e1, CachedBatchType::MemoryArrow);
        policy.notify_insert(&e2, CachedBatchType::MemoryArrow);
        policy.notify_insert(&e3, CachedBatchType::MemoryArrow);

        assert_evict_advice(&policy, e1);
    }

    #[test]
    fn test_lru_policy_access_moves_to_front() {
        let policy = LruPolicy::new();
        let e1 = entry(1);
        let e2 = entry(2);
        let e3 = entry(3);

        policy.notify_insert(&e1, CachedBatchType::MemoryArrow);
        policy.notify_insert(&e2, CachedBatchType::MemoryArrow);
        policy.notify_insert(&e3, CachedBatchType::MemoryArrow);

        policy.notify_access(&e1, CachedBatchType::MemoryArrow);
        assert_evict_advice(&policy, e2);
        policy.notify_access(&e2, CachedBatchType::MemoryArrow);
        assert_evict_advice(&policy, e3);
    }

    #[test]
    fn test_lru_policy_reinsert_moves_to_front() {
        let policy = LruPolicy::new();
        let e1 = entry(1);
        let e2 = entry(2);
        let e3 = entry(3);

        policy.notify_insert(&e1, CachedBatchType::MemoryArrow);
        policy.notify_insert(&e2, CachedBatchType::MemoryArrow);
        policy.notify_insert(&e3, CachedBatchType::MemoryArrow);

        policy.notify_insert(&e1, CachedBatchType::MemoryArrow);
        assert_evict_advice(&policy, e2);
    }

    #[test]
    fn test_lru_policy_advise_empty() {
        let policy = LruPolicy::new();
        assert_eq!(policy.find_victim(1), vec![]);
    }

    #[test]
    fn test_lru_policy_advise_single_item_self() {
        let policy = LruPolicy::new();
        let e1 = entry(1);
        policy.notify_insert(&e1, CachedBatchType::MemoryArrow);

        assert_evict_advice(&policy, e1);
    }

    #[test]
    fn test_lru_policy_advise_single_item_other() {
        let policy = LruPolicy::new();
        let e1 = entry(1);
        policy.notify_insert(&e1, CachedBatchType::MemoryArrow);
        assert_evict_advice(&policy, e1);
    }

    #[test]
    fn test_lru_policy_access_nonexistent() {
        let policy = LruPolicy::new();
        let e1 = entry(1);
        let e2 = entry(2);

        policy.notify_insert(&e1, CachedBatchType::MemoryArrow);
        policy.notify_insert(&e2, CachedBatchType::MemoryArrow);

        policy.notify_access(&entry(99), CachedBatchType::MemoryArrow);

        assert_evict_advice(&policy, e1);
    }

    impl HashList {
        fn check_integrity(&self) {
            let map_count = self.map.len();
            let forward_count = count_nodes_in_list(self);
            let backward_count = count_nodes_reverse(self);

            assert_eq!(map_count, forward_count);
            assert_eq!(map_count, backward_count);
        }
    }

    fn count_nodes_in_list(state: &HashList) -> usize {
        let mut count = 0;
        let mut current = state.list.head();

        while let Some(node_ptr) = current {
            count += 1;
            current = unsafe { node_ptr.as_ref().next };
        }

        count
    }

    fn count_nodes_reverse(state: &HashList) -> usize {
        let mut count = 0;
        let mut current = state.list.tail();

        while let Some(node_ptr) = current {
            count += 1;
            current = unsafe { node_ptr.as_ref().prev };
        }

        count
    }

    #[test]
    fn test_lru_policy_invariants() {
        let policy = LruPolicy::new();

        for i in 0..10 {
            policy.notify_insert(&entry(i), CachedBatchType::MemoryArrow);
        }
        policy.notify_access(&entry(2), CachedBatchType::MemoryArrow);
        policy.notify_access(&entry(5), CachedBatchType::MemoryArrow);
        policy.find_victim(1);
        policy.find_victim(1);

        let state = policy.state.lock().unwrap();
        state.check_integrity();

        let map_count = state.map.len();
        assert_eq!(map_count, 8);
        assert!(!state.map.contains_key(&entry(0)));
        assert!(!state.map.contains_key(&entry(1)));
        assert!(state.map.contains_key(&entry(2)));

        let head_id = unsafe { state.list.head().unwrap().as_ref().data.entry_id };
        assert_eq!(head_id, entry(5));
    }

    #[test]
    fn test_concurrent_lru_operations() {
        concurrent_lru_operations();
    }

    #[cfg(feature = "shuttle")]
    #[test]
    fn shuttle_lru_operations() {
        crate::utils::shuttle_test(concurrent_lru_operations);
    }

    fn concurrent_lru_operations() {
        let policy = Arc::new(LruPolicy::new());
        let num_threads = 4;
        let operations_per_thread = 100;

        let total_inserts = Arc::new(AtomicUsize::new(0));
        let total_evictions = Arc::new(AtomicUsize::new(0));

        let barrier = Arc::new(Barrier::new(num_threads));

        let mut handles = vec![];
        for thread_id in 0..num_threads {
            let policy_clone = policy.clone();
            let total_inserts_clone = total_inserts.clone();
            let total_evictions_clone = total_evictions.clone();
            let barrier_clone = barrier.clone();

            let handle = thread::spawn(move || {
                barrier_clone.wait();

                for i in 0..operations_per_thread {
                    let op_type = i % 3;
                    let entry_id = entry(thread_id * operations_per_thread + i);

                    match op_type {
                        0 => {
                            policy_clone.notify_insert(&entry_id, CachedBatchType::MemoryArrow);
                            total_inserts_clone.fetch_add(1, Ordering::SeqCst);
                        }
                        1 => {
                            policy_clone.notify_access(&entry_id, CachedBatchType::MemoryArrow);
                        }
                        _ => {
                            let advised = policy_clone.find_victim(1);
                            if !advised.is_empty() {
                                total_evictions_clone.fetch_add(1, Ordering::SeqCst);
                            }
                        }
                    }
                }
            });

            handles.push(handle);
        }

        for handle in handles {
            handle.join().unwrap();
        }

        let state = policy.state.lock().unwrap();
        state.check_integrity();

        let inserts = total_inserts.load(Ordering::SeqCst);
        let evictions = total_evictions.load(Ordering::SeqCst);
        assert!(inserts >= evictions);
    }

    #[tokio::test]
    async fn test_lru_integration() {
        let policy = LruPolicy::new();
        let store = create_cache_store(3000, Box::new(policy));

        let entry_id1 = EntryID::from(1);
        let entry_id2 = EntryID::from(2);
        let entry_id3 = EntryID::from(3);

        store.insert(entry_id1, create_test_arrow_array(100)).await;
        store.insert(entry_id2, create_test_arrow_array(100)).await;
        store.insert(entry_id3, create_test_arrow_array(100)).await;

        assert!(store.get(&entry_id1).is_some());
        assert!(store.get(&entry_id2).is_some());
        assert!(store.get(&entry_id3).is_some());
    }
}
