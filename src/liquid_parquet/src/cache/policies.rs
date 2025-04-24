use crate::sync::Mutex;
use std::{
    collections::{HashMap, VecDeque},
    ptr::NonNull,
};

use super::{CacheAdvice, CacheEntryID, CachedBatch};

/// The cache policy that guides the replacement of LiquidCache
pub trait CachePolicy: std::fmt::Debug + Send + Sync {
    /// Give advice on what to do when cache is full.
    fn advise(&self, entry_id: &CacheEntryID, to_insert: &CachedBatch) -> CacheAdvice;

    /// Notify the cache policy that an entry was inserted.
    fn notify_insert(&self, _entry_id: &CacheEntryID) {}

    /// Notify the cache policy that an entry was accessed.
    fn notify_access(&self, _entry_id: &CacheEntryID) {}

    /// Notify the cache policy that an entry was evicted.
    fn notify_evict(&self, _entry_id: &CacheEntryID);
}

/// The policy that implements the FILO (First In, Last Out) algorithm.
/// Newest entries are evicted first.
#[derive(Debug, Default)]
pub struct FiloPolicy {
    queue: Mutex<VecDeque<CacheEntryID>>,
}

impl FiloPolicy {
    /// Create a new [FiloPolicy].
    pub fn new() -> Self {
        Self {
            queue: Mutex::new(VecDeque::new()),
        }
    }

    fn add_entry(&self, entry_id: &CacheEntryID) {
        let mut queue = self.queue.lock().unwrap();
        queue.push_front(*entry_id);
    }

    fn get_newest_entry(&self) -> Option<CacheEntryID> {
        let mut queue = self.queue.lock().unwrap();
        queue.pop_front()
    }
}

impl CachePolicy for FiloPolicy {
    fn advise(&self, entry_id: &CacheEntryID, _to_insert: &CachedBatch) -> CacheAdvice {
        if let Some(newest_entry) = self.get_newest_entry() {
            return CacheAdvice::Evict(newest_entry);
        }
        CacheAdvice::TranscodeToDisk(*entry_id)
    }

    fn notify_evict(&self, entry_id: &CacheEntryID) {
        let mut queue = self.queue.lock().unwrap();
        queue.retain(|id| id != entry_id);
    }

    fn notify_insert(&self, entry_id: &CacheEntryID) {
        self.add_entry(entry_id);
    }
}

#[derive(Debug)]
struct Node {
    entry_id: CacheEntryID,
    prev: Option<NonNull<Node>>,
    next: Option<NonNull<Node>>,
}

#[derive(Debug, Default)]
struct LruInternalState {
    map: HashMap<CacheEntryID, NonNull<Node>>,
    head: Option<NonNull<Node>>,
    tail: Option<NonNull<Node>>,
}

/// The policy that implement the Lru algorithm using a HashMap and a Doubly Linked List.
#[derive(Debug, Default)]
pub struct LruPolicy {
    state: Mutex<LruInternalState>,
}

impl LruPolicy {
    /// Create a new [LruPolicy].
    pub fn new() -> Self {
        Self {
            state: Mutex::new(LruInternalState {
                map: HashMap::new(),
                head: None,
                tail: None,
            }),
        }
    }

    /// Unlinks the node from the doubly linked list.
    /// Must be called within the lock.
    unsafe fn unlink_node(&self, state: &mut LruInternalState, mut node_ptr: NonNull<Node>) {
        let node = unsafe { node_ptr.as_mut() };

        match node.prev {
            Some(mut prev) => unsafe { prev.as_mut().next = node.next },
            // Node is head
            None => state.head = node.next,
        }

        match node.next {
            Some(mut next) => unsafe { next.as_mut().prev = node.prev },
            // Node is tail
            None => state.tail = node.prev,
        }

        node.prev = None;
        node.next = None;
    }

    /// Pushes the node to the front (head) of the list.
    /// Must be called within the lock.
    unsafe fn push_front(&self, state: &mut LruInternalState, mut node_ptr: NonNull<Node>) {
        let node = unsafe { node_ptr.as_mut() };

        node.next = state.head;
        node.prev = None;

        match state.head {
            Some(mut head) => unsafe { head.as_mut().prev = Some(node_ptr) },
            // List was empty
            None => state.tail = Some(node_ptr),
        }

        state.head = Some(node_ptr);
    }
}

// SAFETY: The Mutex ensures that only one thread accesses the internal state
// (map, head, tail containing NonNull pointers) at a time, making it safe
// to send and share across threads.
unsafe impl Send for LruPolicy {}
unsafe impl Sync for LruPolicy {}

impl CachePolicy for LruPolicy {
    fn advise(&self, entry_id: &CacheEntryID, _to_insert: &CachedBatch) -> CacheAdvice {
        let state = self.state.lock().unwrap();
        if let Some(tail_ptr) = state.tail {
            let tail_entry_id = unsafe { tail_ptr.as_ref().entry_id };
            if tail_entry_id != *entry_id {
                return CacheAdvice::Evict(tail_entry_id);
            }
        }
        CacheAdvice::TranscodeToDisk(*entry_id)
    }

    fn notify_access(&self, entry_id: &CacheEntryID) {
        let mut state = self.state.lock().unwrap();
        if let Some(node_ptr) = state.map.get(entry_id).copied() {
            unsafe {
                self.unlink_node(&mut state, node_ptr);
                self.push_front(&mut state, node_ptr);
            }
        }
        // If not in map, it means it was already evicted or never inserted
    }

    fn notify_evict(&self, entry_id: &CacheEntryID) {
        let mut state = self.state.lock().unwrap();
        if let Some(node_ptr) = state.map.remove(entry_id) {
            unsafe {
                self.unlink_node(&mut state, node_ptr);
                drop(Box::from_raw(node_ptr.as_ptr()));
            }
        }
    }

    fn notify_insert(&self, entry_id: &CacheEntryID) {
        let mut state = self.state.lock().unwrap();

        // If entry already exists, move it to front (treat insert like access)
        if let Some(existing_node_ptr) = state.map.get(entry_id).copied() {
            unsafe {
                self.unlink_node(&mut state, existing_node_ptr);
                self.push_front(&mut state, existing_node_ptr);
            }
            return; // Already handled
        }

        // Allocate a new node on the heap
        let node = Node {
            entry_id: *entry_id,
            prev: None,
            next: None,
        };
        let node_ptr = match NonNull::new(Box::into_raw(Box::new(node))) {
            Some(ptr) => ptr,
            None => panic!("Failed to allocate memory for LRU node"), // Or handle allocation failure more gracefully
        };

        state.map.insert(*entry_id, node_ptr);
        unsafe {
            self.push_front(&mut state, node_ptr);
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
mod test {
    use crate::cache::utils::{create_cache_store, create_entry_id, create_test_array};
    use crate::policies::CachePolicy;

    use super::super::{CacheAdvice, CacheEntryID, CachedBatch};
    use super::{FiloPolicy, LruPolicy};

    // Helper to create entry IDs for tests
    fn entry(id: u64) -> CacheEntryID {
        create_entry_id(id, id, id, id as u16)
    }

    // Helper to assert eviction advice
    fn assert_evict_advice(
        policy: &LruPolicy,
        expect_evict: CacheEntryID,
        trigger_entry: CacheEntryID,
    ) {
        let dummy_batch = create_test_array(1);
        let advice = policy.advise(&trigger_entry, &dummy_batch);
        assert_eq!(advice, CacheAdvice::Evict(expect_evict));
    }

    // Helper to assert transcode advice
    fn assert_transcode_advice(policy: &LruPolicy, trigger_entry: CacheEntryID) {
        let dummy_batch = create_test_array(1);
        let advice = policy.advise(&trigger_entry, &dummy_batch);
        assert_eq!(advice, CacheAdvice::TranscodeToDisk(trigger_entry));
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

        // Oldest entry (e1) should be advised for eviction
        assert_evict_advice(&policy, e1, entry(4));
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

        // Access e1, making it the most recent
        policy.notify_access(&e1);

        // Now e2 should be the oldest
        assert_evict_advice(&policy, e2, entry(4));

        // Access e2
        policy.notify_access(&e2);

        // Now e3 should be the oldest
        assert_evict_advice(&policy, e3, entry(4));
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

        // Re-insert e1 (should act like access)
        policy.notify_insert(&e1);

        // Now e2 should be the oldest
        assert_evict_advice(&policy, e2, entry(4));
    }

    #[test]
    fn test_lru_policy_advise_empty() {
        let policy = LruPolicy::new();
        // Should advise transcode if empty
        assert_transcode_advice(&policy, entry(1));
    }

    #[test]
    fn test_lru_policy_advise_single_item_self() {
        let policy = LruPolicy::new();
        let e1 = entry(1);
        policy.notify_insert(&e1);

        // Should advise transcode if the only candidate is the item being inserted
        assert_transcode_advice(&policy, e1);
    }

    #[test]
    fn test_lru_policy_advise_single_item_other() {
        let policy = LruPolicy::new();
        let e1 = entry(1);
        policy.notify_insert(&e1);
        let e2 = entry(2);

        // If only one other item exists, it should be evicted
        assert_evict_advice(&policy, e1, e2);
    }

    #[test]
    fn test_lru_policy_access_nonexistent() {
        let policy = LruPolicy::new();
        let e1 = entry(1);
        let e2 = entry(2);

        policy.notify_insert(&e1);
        policy.notify_insert(&e2);

        // Access an entry not in the policy; should not panic or change order
        policy.notify_access(&entry(99));

        // e1 should still be the oldest
        assert_evict_advice(&policy, e1, entry(3));
    }

    // --- Keep existing tests for FiloPolicy and integration tests below ---
    // Existing test test_lru_policy is now more of an integration test for CacheStore + LruPolicy
    #[test]
    fn test_lru_integration() {
        // Renamed from test_lru_policy
        let advisor = LruPolicy::new();
        let store = create_cache_store(3000, Box::new(advisor));

        // Create three entries
        let entry_id1 = create_entry_id(1, 1, 1, 1);
        let entry_id2 = create_entry_id(1, 1, 1, 2);
        let entry_id3 = create_entry_id(1, 1, 1, 3);

        // Make sure on-disk paths exist
        let on_disk_path = entry_id1.on_disk_path(&store.config().cache_root_dir());
        std::fs::create_dir_all(on_disk_path.parent().unwrap()).unwrap();

        // Insert entries in order: 1, 2, 3
        store.insert(entry_id1, create_test_array(100));
        store.insert(entry_id2, create_test_array(100));
        store.insert(entry_id3, create_test_array(100));

        // Access entry 1 to move it to front
        store.get(&entry_id1);

        // Insert a fourth entry to force eviction
        let entry_id4 = create_entry_id(4, 4, 4, 4);
        store.insert(entry_id4, create_test_array(100));

        // Entry 2 should be evicted (it's now the oldest since entry 1 was moved to front)
        assert!(store.get(&entry_id1).is_some());
        assert!(store.get(&entry_id3).is_some());

        // Entry 2 should now be on disk (or possibly evicted entirely)
        match store.get(&entry_id2) {
            Some(CachedBatch::OnDiskLiquid) => {}
            None => {} // This is also acceptable if fully evicted
            other => panic!("Expected OnDiskLiquid or None, got {:?}", other),
        }
    }

    #[test]
    fn test_filo_advisor() {
        let advisor = FiloPolicy::new();
        let store = create_cache_store(3000, Box::new(advisor));

        // Create three entries
        let entry_id1 = create_entry_id(1, 1, 1, 1);
        let entry_id2 = create_entry_id(1, 1, 1, 2);
        let entry_id3 = create_entry_id(1, 1, 1, 3);

        // Make sure on-disk paths exist
        let on_disk_path = entry_id1.on_disk_path(&store.config().cache_root_dir());
        std::fs::create_dir_all(on_disk_path.parent().unwrap()).unwrap();

        // Insert entries in order: 1, 2, 3
        store.insert(entry_id1, create_test_array(100));
        store.insert(entry_id2, create_test_array(100));
        store.insert(entry_id3, create_test_array(100));

        // Insert a fourth entry to force eviction
        let entry_id4 = create_entry_id(4, 4, 4, 4);
        store.insert(entry_id4, create_test_array(100));

        // Entry 3 should be evicted (it's the newest before entry 4)
        assert!(store.get(&entry_id1).is_some());
        assert!(store.get(&entry_id2).is_some());
        assert!(store.get(&entry_id4).is_some());

        // Entry 3 should now be on disk (or possibly evicted entirely)
        match store.get(&entry_id3) {
            Some(CachedBatch::OnDiskLiquid) => {}
            None => {} // This is also acceptable if fully evicted
            other => panic!("Expected OnDiskLiquid or None, got {:?}", other),
        }
    }
}
