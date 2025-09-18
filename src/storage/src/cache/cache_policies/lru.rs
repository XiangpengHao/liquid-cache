//! LRU cache policy implementation using a hash map and doubly linked list.

use std::{collections::HashMap, ptr::NonNull};

use crate::{cache::utils::EntryID, sync::Mutex};

use super::CachePolicy;

#[derive(Debug)]
struct Node {
    entry_id: EntryID,
    prev: Option<NonNull<Node>>,
    next: Option<NonNull<Node>>,
}

#[derive(Debug, Default)]
struct HashList {
    map: HashMap<EntryID, NonNull<Node>>,
    head: Option<NonNull<Node>>,
    tail: Option<NonNull<Node>>,
}

impl HashList {
    unsafe fn unlink_node(&mut self, mut node_ptr: NonNull<Node>) {
        let node = unsafe { node_ptr.as_mut() };

        match node.prev {
            Some(mut prev) => unsafe { prev.as_mut().next = node.next },
            None => self.head = node.next,
        }

        match node.next {
            Some(mut next) => unsafe { next.as_mut().prev = node.prev },
            None => self.tail = node.prev,
        }

        node.prev = None;
        node.next = None;
    }

    /// Pushes the node to the front (head) of the list.
    /// Must be called within the lock.
    unsafe fn push_front(&mut self, mut node_ptr: NonNull<Node>) {
        let node = unsafe { node_ptr.as_mut() };

        node.next = self.head;
        node.prev = None;

        match self.head {
            Some(mut head) => unsafe { head.as_mut().prev = Some(node_ptr) },
            None => self.tail = Some(node_ptr),
        }

        self.head = Some(node_ptr);
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
            state: Mutex::new(HashList {
                map: HashMap::new(),
                head: None,
                tail: None,
            }),
        }
    }
}

// SAFETY: The Mutex ensures that only one thread accesses the internal state
// (map, head, tail containing NonNull pointers) at a time, making it safe
// to send and share across threads.
unsafe impl Send for LruPolicy {}
unsafe impl Sync for LruPolicy {}

impl CachePolicy for LruPolicy {
    fn advise(&self, cnt: usize) -> Vec<EntryID> {
        let mut state = self.state.lock().unwrap();
        if cnt == 0 {
            return vec![];
        }

        let mut advices = Vec::with_capacity(cnt);
        for _ in 0..cnt {
            let Some(tail_ptr) = state.tail else { break };
            let tail_entry_id = unsafe { tail_ptr.as_ref().entry_id };
            let node_ptr = state
                .map
                .remove(&tail_entry_id)
                .expect("tail node not found");
            unsafe {
                state.unlink_node(node_ptr);
                drop(Box::from_raw(node_ptr.as_ptr()));
            }
            advices.push(tail_entry_id);
        }

        advices
    }

    fn notify_access(&self, entry_id: &EntryID) {
        let mut state = self.state.lock().unwrap();
        if let Some(node_ptr) = state.map.get(entry_id).copied() {
            unsafe {
                state.unlink_node(node_ptr);
                state.push_front(node_ptr);
            }
        }
    }

    fn notify_insert(&self, entry_id: &EntryID) {
        let mut state = self.state.lock().unwrap();

        if let Some(existing_node_ptr) = state.map.get(entry_id).copied() {
            unsafe {
                state.unlink_node(existing_node_ptr);
                state.push_front(existing_node_ptr);
            }
            return;
        }

        let node = Node {
            entry_id: *entry_id,
            prev: None,
            next: None,
        };
        let node_ptr = match NonNull::new(Box::into_raw(Box::new(node))) {
            Some(ptr) => ptr,
            None => panic!("Failed to allocate memory for LRU node"),
        };

        state.map.insert(*entry_id, node_ptr);
        unsafe {
            state.push_front(node_ptr);
        }
    }
}

impl Drop for LruPolicy {
    fn drop(&mut self) {
        let mut state = self.state.lock().unwrap();
        for (_, node_ptr) in state.map.drain() {
            unsafe {
                drop(Box::from_raw(node_ptr.as_ptr()));
            }
        }
        state.head = None;
        state.tail = None;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cache::cached_data::CachedBatch;
    use crate::cache::utils::{EntryID, create_cache_store, create_test_arrow_array};
    use crate::sync::{Arc, Barrier, thread};
    use std::sync::atomic::{AtomicUsize, Ordering};

    fn entry(id: usize) -> EntryID {
        id.into()
    }

    fn assert_evict_advice(policy: &LruPolicy, expect_evict: EntryID) {
        let advice = policy.advise(1);
        assert_eq!(advice, vec![expect_evict]);
    }

    #[test]
    fn test_lru_policy_insertion_order() {
        let policy = LruPolicy::new();
        let e1 = entry(1);
        let e2 = entry(2);
        let e3 = entry(3);

        policy.notify_insert(&e1);
        policy.notify_insert(&e2);
        policy.notify_insert(&e3);

        assert_evict_advice(&policy, e1);
    }

    #[test]
    fn test_lru_policy_access_moves_to_front() {
        let policy = LruPolicy::new();
        let e1 = entry(1);
        let e2 = entry(2);
        let e3 = entry(3);

        policy.notify_insert(&e1);
        policy.notify_insert(&e2);
        policy.notify_insert(&e3);

        policy.notify_access(&e1);
        assert_evict_advice(&policy, e2);
        policy.notify_access(&e2);
        assert_evict_advice(&policy, e3);
    }

    #[test]
    fn test_lru_policy_reinsert_moves_to_front() {
        let policy = LruPolicy::new();
        let e1 = entry(1);
        let e2 = entry(2);
        let e3 = entry(3);

        policy.notify_insert(&e1);
        policy.notify_insert(&e2);
        policy.notify_insert(&e3);

        policy.notify_insert(&e1);
        assert_evict_advice(&policy, e2);
    }

    #[test]
    fn test_lru_policy_advise_empty() {
        let policy = LruPolicy::new();
        assert_eq!(policy.advise(1), vec![]);
    }

    #[test]
    fn test_lru_policy_advise_single_item_self() {
        let policy = LruPolicy::new();
        let e1 = entry(1);
        policy.notify_insert(&e1);

        assert_evict_advice(&policy, e1);
    }

    #[test]
    fn test_lru_policy_advise_single_item_other() {
        let policy = LruPolicy::new();
        let e1 = entry(1);
        policy.notify_insert(&e1);
        assert_evict_advice(&policy, e1);
    }

    #[test]
    fn test_lru_policy_access_nonexistent() {
        let policy = LruPolicy::new();
        let e1 = entry(1);
        let e2 = entry(2);

        policy.notify_insert(&e1);
        policy.notify_insert(&e2);

        policy.notify_access(&entry(99));

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
        let mut current = state.head;

        while let Some(node_ptr) = current {
            count += 1;
            current = unsafe { node_ptr.as_ref().next };
        }

        count
    }

    fn count_nodes_reverse(state: &HashList) -> usize {
        let mut count = 0;
        let mut current = state.tail;

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
            policy.notify_insert(&entry(i));
        }
        policy.notify_access(&entry(2));
        policy.notify_access(&entry(5));
        policy.advise(1);
        policy.advise(1);

        let state = policy.state.lock().unwrap();
        state.check_integrity();

        let map_count = state.map.len();
        assert_eq!(map_count, 8);
        assert!(!state.map.contains_key(&entry(0)));
        assert!(!state.map.contains_key(&entry(1)));
        assert!(state.map.contains_key(&entry(2)));

        let head_id = unsafe { state.head.unwrap().as_ref().entry_id };
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
                            policy_clone.notify_insert(&entry_id);
                            total_inserts_clone.fetch_add(1, Ordering::SeqCst);
                        }
                        1 => {
                            if i > 10 {
                                let other_thread = (thread_id + 1) % num_threads;
                                let other_id = entry(other_thread * operations_per_thread + i - 10);
                                policy_clone.notify_access(&other_id);
                            }
                            policy_clone.notify_access(&entry_id);
                        }
                        2 => {
                            if i > 20 {
                                policy_clone.advise(1);
                                total_evictions_clone.fetch_add(1, Ordering::SeqCst);
                            }
                        }
                        _ => unreachable!(),
                    }
                }
            });

            handles.push(handle);
        }

        for handle in handles {
            handle.join().unwrap();
        }

        let inserts = total_inserts.load(Ordering::SeqCst);
        let evictions = total_evictions.load(Ordering::SeqCst);
        let expected_count = inserts - evictions;

        let state = policy.state.lock().unwrap();
        state.check_integrity();

        let map_count = state.map.len();
        assert_eq!(map_count, expected_count);
    }

    #[test]
    fn test_lru_integration() {
        let advisor = LruPolicy::new();
        let store = create_cache_store(3000, Box::new(advisor));

        let entry_id1 = EntryID::from(1);
        let entry_id2 = EntryID::from(2);
        let entry_id3 = EntryID::from(3);

        store.insert(entry_id1, create_test_arrow_array(100));
        store.insert(entry_id2, create_test_arrow_array(100));
        store.insert(entry_id3, create_test_arrow_array(100));

        store.get(&entry_id1);

        let entry_id4 = EntryID::from(4);
        store.insert(entry_id4, create_test_arrow_array(100));

        assert!(store.get(&entry_id1).is_some());
        assert!(store.get(&entry_id3).is_some());

        if let Some(data) = store.get(&entry_id2) {
            match data.raw_data() {
                CachedBatch::DiskLiquid => {}
                _ => panic!("Expected OnDiskLiquid, got {:?}", data.raw_data()),
            }
        }
    }
}
