use liquid_cache_common::LiquidCacheMode;

use crate::{
    cache::{CacheEntryID, utils::CacheAdvice},
    sync::Mutex,
};
use rand::{Rng, rng};
use std::{
    collections::{HashMap, VecDeque},
    ptr::NonNull,
};

/// The cache policy that guides the replacement of LiquidCache
pub trait CachePolicy: std::fmt::Debug + Send + Sync {
    /// Give advice on what to do when cache is full.
    fn advise(&self, entry_id: &CacheEntryID, cache_mode: &LiquidCacheMode) -> CacheAdvice;

    /// Notify the cache policy that an entry was inserted.
    fn notify_insert(&self, _entry_id: &CacheEntryID) {}

    /// Notify the cache policy that an entry was accessed.
    fn notify_access(&self, _entry_id: &CacheEntryID) {}
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
    fn advise(&self, entry_id: &CacheEntryID, cache_mode: &LiquidCacheMode) -> CacheAdvice {
        if let Some(newest_entry) = self.get_newest_entry() {
            return CacheAdvice::Evict(newest_entry);
        }
        fallback_advice(entry_id, cache_mode)
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
    fn advise(&self, entry_id: &CacheEntryID, cache_mode: &LiquidCacheMode) -> CacheAdvice {
        let mut state = self.state.lock().unwrap();
        if let Some(tail_ptr) = state.tail {
            let tail_entry_id = unsafe { tail_ptr.as_ref().entry_id };
            let node_ptr = state
                .map
                .remove(&tail_entry_id)
                .expect("tail node not found");
            unsafe {
                self.unlink_node(&mut state, node_ptr);
                drop(Box::from_raw(node_ptr.as_ptr()));
            }
            return CacheAdvice::Evict(tail_entry_id);
        }
        fallback_advice(entry_id, cache_mode)
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

/// The policy that discards entries when the cache is full.
#[derive(Debug, Default)]
pub struct DiscardPolicy;

impl CachePolicy for DiscardPolicy {
    fn advise(&self, _entry_id: &CacheEntryID, _cache_mode: &LiquidCacheMode) -> CacheAdvice {
        CacheAdvice::Discard
    }
}

/// The policy that writes entries to disk when the cache is full.
/// This preserves the original format of the data (Arrow stays Arrow, Liquid stays Liquid).
#[derive(Debug, Default)]
pub struct ToDiskPolicy;

impl ToDiskPolicy {
    /// Create a new [ToDiskPolicy].
    pub fn new() -> Self {
        Self
    }
}

impl CachePolicy for ToDiskPolicy {
    fn advise(&self, entry_id: &CacheEntryID, _cache_mode: &LiquidCacheMode) -> CacheAdvice {
        CacheAdvice::ToDisk(*entry_id)
    }
}

/// SIEVE as implemented by (<https://cachemon.github.io/SIEVE-website/>)
#[derive(Debug, Default)]
pub struct SievePolicy {
    state: Mutex<SieveInternalState>,
}

#[derive(Debug)]
struct SieveNode {
    entry_id: CacheEntryID,
    visited: bool,
    prev: Option<NonNull<SieveNode>>,
    next: Option<NonNull<SieveNode>>,
}

#[derive(Debug, Default)]
struct SieveInternalState {
    map: HashMap<CacheEntryID, NonNull<SieveNode>>,
    head: Option<NonNull<SieveNode>>,
    tail: Option<NonNull<SieveNode>>,
    hand: Option<NonNull<SieveNode>>,
}

impl SievePolicy {
    /// Create a new SIEVE policy.
    pub fn new() -> Self {
        SievePolicy {
            state: Mutex::new(SieveInternalState::default()),
        }
    }

    /// Unlink `node_ptr` from the doubly-linked list.  
    /// Must hold the lock.
    unsafe fn unlink_node(&self, state: &mut SieveInternalState, mut node_ptr: NonNull<SieveNode>) {
        unsafe {
            let node = node_ptr.as_mut();
            // patch up prev → next
            if let Some(mut p) = node.prev {
                p.as_mut().next = node.next;
            } else {
                state.head = node.next;
            }
            // patch up next → prev
            if let Some(mut n) = node.next {
                n.as_mut().prev = node.prev;
            } else {
                state.tail = node.prev;
            }
            node.prev = None;
            node.next = None;
        }
    }

    /// Push `node_ptr` to the front (head) of the list.  
    /// Must hold the lock.
    unsafe fn push_front(&self, state: &mut SieveInternalState, mut node_ptr: NonNull<SieveNode>) {
        unsafe {
            let node = node_ptr.as_mut();
            node.next = state.head;
            node.prev = None;

            if let Some(mut old_head) = state.head {
                old_head.as_mut().prev = Some(node_ptr);
            } else {
                // was empty
                state.tail = Some(node_ptr);
            }
            state.head = Some(node_ptr);
        }
    }
}

// Safe to share across threads because we guard all pointer mutations.
unsafe impl Send for SievePolicy {}
unsafe impl Sync for SievePolicy {}

impl CachePolicy for SievePolicy {
    fn advise(&self, entry_id: &CacheEntryID, cache_mode: &LiquidCacheMode) -> CacheAdvice {
        let mut state = self.state.lock().unwrap();
        // if empty, fall back
        if state.hand.is_none() {
            return fallback_advice(entry_id, cache_mode);
        }

        // scan for the first unvisited node (wrapping tail→head)
        let mut cursor = state.hand;
        loop {
            if let Some(mut ptr) = cursor {
                let node = unsafe { ptr.as_ref() };
                if node.visited {
                    // clear and move on
                    unsafe {
                        ptr.as_mut().visited = false;
                    }
                    cursor = node.prev.or(state.tail);
                    continue;
                } else {
                    // evict this one
                    let evict_id = node.entry_id;
                    state.hand = node.prev.or(state.tail);
                    state.map.remove(&evict_id);
                    unsafe {
                        self.unlink_node(&mut state, ptr);
                        drop(Box::from_raw(ptr.as_ptr()));
                    }
                    return CacheAdvice::Evict(evict_id);
                }
            } else {
                // wrapped past head: restart from tail
                cursor = state.tail;
                if cursor.is_none() {
                    break;
                }
            }
        }

        // should not happen, but just in case
        fallback_advice(entry_id, cache_mode)
    }

    fn notify_access(&self, entry_id: &CacheEntryID) {
        let state = self.state.lock().unwrap();
        if let Some(&(mut ptr)) = state.map.get(entry_id) {
            unsafe {
                ptr.as_mut().visited = true;
            }
        }
    }

    fn notify_insert(&self, entry_id: &CacheEntryID) {
        let mut state = self.state.lock().unwrap();
        // allocate new node
        let boxed = Box::new(SieveNode {
            entry_id: *entry_id,
            visited: false,
            prev: None,
            next: None,
        });
        let ptr = unsafe { NonNull::new_unchecked(Box::into_raw(boxed)) };

        state.map.insert(*entry_id, ptr);
        unsafe {
            self.push_front(&mut state, ptr);
        }

        // initialize hand on first insert
        if state.hand.is_none() {
            state.hand = state.tail;
        }
    }
}

impl Drop for SievePolicy {
    fn drop(&mut self) {
        let mut state = self.state.lock().unwrap();
        for (_, ptr) in state.map.drain() {
            unsafe {
                drop(Box::from_raw(ptr.as_ptr()));
            }
        }
        state.head = None;
        state.tail = None;
        state.hand = None;
    }
}

/// The CLOCK (second‐chance) eviction policy.
#[derive(Debug, Default)]
pub struct ClockPolicy {
    state: Mutex<ClockInternalState>,
}

#[derive(Debug)]
struct ClockNode {
    entry_id: CacheEntryID,
    referenced: bool,
    prev: Option<NonNull<ClockNode>>,
    next: Option<NonNull<ClockNode>>,
}

#[derive(Debug, Default)]
struct ClockInternalState {
    map: HashMap<CacheEntryID, NonNull<ClockNode>>,
    head: Option<NonNull<ClockNode>>,
    tail: Option<NonNull<ClockNode>>,
    hand: Option<NonNull<ClockNode>>,
}

impl ClockPolicy {
    /// Create a new CLOCK policy.
    pub fn new() -> Self {
        ClockPolicy {
            state: Mutex::new(ClockInternalState::default()),
        }
    }

    /// Unlink `node_ptr` from the doubly‐linked list. Should hold lock.
    unsafe fn unlink_node(&self, state: &mut ClockInternalState, mut node_ptr: NonNull<ClockNode>) {
        unsafe {
            let node = node_ptr.as_mut();
            // patch up prev → next
            if let Some(mut p) = node.prev {
                p.as_mut().next = node.next;
            } else {
                state.head = node.next;
            }
            // patch up next → prev
            if let Some(mut n) = node.next {
                n.as_mut().prev = node.prev;
            } else {
                state.tail = node.prev;
            }
            node.prev = None;
            node.next = None;
        }
    }

    /// Insert `new_ptr` immediately after the current `hand_ptr`. Should hold lock.
    unsafe fn insert_after(
        &self,
        state: &mut ClockInternalState,
        mut new_ptr: NonNull<ClockNode>,
        mut hand_ptr: NonNull<ClockNode>,
    ) {
        unsafe {
            let hand_node = hand_ptr.as_mut();
            let next = hand_node.next;
            hand_node.next = Some(new_ptr);
            new_ptr.as_mut().prev = Some(hand_ptr);
            new_ptr.as_mut().next = next;

            if let Some(mut n) = next {
                n.as_mut().prev = Some(new_ptr);
            } else {
                // was tail, so update
                state.tail = Some(new_ptr);
            }
        }
    }
}

// Safe across threads since we guard pointer mutations.
unsafe impl Send for ClockPolicy {}
unsafe impl Sync for ClockPolicy {}

impl CachePolicy for ClockPolicy {
    fn advise(&self, entry_id: &CacheEntryID, cache_mode: &LiquidCacheMode) -> CacheAdvice {
        let mut state = self.state.lock().unwrap();
        let Some(mut ptr) = state.hand else {
            return fallback_advice(entry_id, cache_mode);
        };

        loop {
            let node = unsafe { ptr.as_ref() };
            if !node.referenced {
                // evict this one
                let evict_id = node.entry_id;
                // advance hand to next (wrap to head)
                let next = node.next.unwrap_or(state.head.unwrap());
                state.hand = Some(next);

                let node_ptr = ptr;
                state.map.remove(&evict_id);
                unsafe {
                    self.unlink_node(&mut state, node_ptr);
                    drop(Box::from_raw(node_ptr.as_ptr()));
                }
                return CacheAdvice::Evict(evict_id);
            }
            // give a second chance: clear bit and move on
            unsafe {
                ptr.as_mut().referenced = false;
            }
            ptr = node.next.unwrap_or(state.head.unwrap());
        }
    }

    fn notify_insert(&self, entry_id: &CacheEntryID) {
        let mut state = self.state.lock().unwrap();
        // if already present, treat as access
        if let Some(mut existing) = state.map.get(entry_id).copied() {
            unsafe {
                existing.as_mut().referenced = true;
            }
            return;
        }

        // allocate new node with referenced = true
        let boxed = Box::new(ClockNode {
            entry_id: *entry_id,
            referenced: true,
            prev: None,
            next: None,
        });
        let new_ptr = unsafe { NonNull::new_unchecked(Box::into_raw(boxed)) };

        if let Some(hand_ptr) = state.hand {
            // insert into rotation after the hand
            unsafe {
                self.insert_after(&mut state, new_ptr, hand_ptr);
            }
        } else {
            // first node in the list
            state.head = Some(new_ptr);
            state.tail = Some(new_ptr);
            state.hand = Some(new_ptr);
        }

        state.map.insert(*entry_id, new_ptr);
    }

    fn notify_access(&self, entry_id: &CacheEntryID) {
        let state = self.state.lock().unwrap();
        if let Some(&(mut ptr)) = state.map.get(entry_id) {
            unsafe {
                ptr.as_mut().referenced = true;
            }
        }
    }
}

impl Drop for ClockPolicy {
    fn drop(&mut self) {
        let mut state = self.state.lock().unwrap();
        for (_, ptr) in state.map.drain() {
            unsafe {
                drop(Box::from_raw(ptr.as_ptr()));
            }
        }
        state.head = None;
        state.tail = None;
        state.hand = None;
    }
}

/// The Random Replacement eviction policy.
#[derive(Debug, Default)]
pub struct RandomPolicy {
    state: Mutex<RandomInternalState>,
}

#[derive(Debug, Default)]
struct RandomInternalState {
    entries: Vec<CacheEntryID>,
}

// Safe to share since we guard all state mutations with a Mutex.
unsafe impl Send for RandomPolicy {}
unsafe impl Sync for RandomPolicy {}

impl RandomPolicy {
    /// Create a new RandomPolicy.
    pub fn new() -> Self {
        RandomPolicy {
            state: Mutex::new(RandomInternalState::default()),
        }
    }
}

impl CachePolicy for RandomPolicy {
    fn advise(&self, entry_id: &CacheEntryID, cache_mode: &LiquidCacheMode) -> CacheAdvice {
        let state = self.state.lock().unwrap();
        if state.entries.is_empty() {
            return fallback_advice(entry_id, cache_mode);
        }
        let mut rng = rng();
        let idx = rng.random_range(0..state.entries.len());
        CacheAdvice::Evict(state.entries[idx])
    }

    fn notify_insert(&self, entry_id: &CacheEntryID) {
        let mut state = self.state.lock().unwrap();
        if !state.entries.contains(entry_id) {
            state.entries.push(*entry_id);
        }
    }

    fn notify_access(&self, _entry_id: &CacheEntryID) {
        // Random policy does not change state on access
    }
}

fn fallback_advice(entry_id: &CacheEntryID, cache_mode: &LiquidCacheMode) -> CacheAdvice {
    match cache_mode {
        LiquidCacheMode::Arrow => CacheAdvice::Discard,
        _ => CacheAdvice::TranscodeToDisk(*entry_id),
    }
}

#[cfg(test)]
mod test {
    use liquid_cache_common::LiquidCacheMode;

    use crate::cache::policies::CachePolicy;
    use crate::cache::tracer::CacheAccessReason;
    use crate::cache::utils::{create_cache_store, create_entry_id, create_test_array};

    use super::super::{CacheAdvice, CacheEntryID, CachedBatch};
    use super::*;
    use crate::sync::{Arc, Barrier, thread};
    use std::sync::atomic::Ordering;

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
        let advice = policy.advise(&trigger_entry, &LiquidCacheMode::Arrow);
        assert_eq!(advice, CacheAdvice::Evict(expect_evict));
    }

    fn assert_discard_advice(policy: &LruPolicy, trigger_entry: CacheEntryID) {
        let advice = policy.advise(&trigger_entry, &LiquidCacheMode::Arrow);
        assert_eq!(advice, CacheAdvice::Discard);
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

        policy.notify_access(&e1);
        assert_evict_advice(&policy, e2, entry(4));
        policy.notify_access(&e2);
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

        policy.notify_insert(&e1);
        assert_evict_advice(&policy, e2, entry(4));
    }

    #[test]
    fn test_lru_policy_advise_empty() {
        let policy = LruPolicy::new();
        assert_discard_advice(&policy, entry(1));
    }

    #[test]
    fn test_lru_policy_advise_single_item_self() {
        let policy = LruPolicy::new();
        let e1 = entry(1);
        policy.notify_insert(&e1);

        assert_evict_advice(&policy, e1, entry(1));
    }

    #[test]
    fn test_lru_policy_advise_single_item_other() {
        let policy = LruPolicy::new();
        let e1 = entry(1);
        policy.notify_insert(&e1);
        let e2 = entry(2);

        assert_evict_advice(&policy, e1, e2);
    }

    #[test]
    fn test_lru_policy_access_nonexistent() {
        let policy = LruPolicy::new();
        let e1 = entry(1);
        let e2 = entry(2);

        policy.notify_insert(&e1);
        policy.notify_insert(&e2);

        policy.notify_access(&entry(99));

        assert_evict_advice(&policy, e1, entry(3));
    }

    impl LruInternalState {
        fn check_integrity(&self) {
            let map_count = self.map.len();
            let forward_count = count_nodes_in_list(self);
            let backward_count = count_nodes_reverse(self);

            assert_eq!(map_count, forward_count);
            assert_eq!(map_count, backward_count);
        }
    }

    /// Count nodes in the linked list by traversing from head to tail
    fn count_nodes_in_list(state: &super::LruInternalState) -> usize {
        let mut count = 0;
        let mut current = state.head;

        while let Some(node_ptr) = current {
            count += 1;
            current = unsafe { node_ptr.as_ref().next };
        }

        count
    }

    /// Count nodes in the linked list by traversing from tail to head
    fn count_nodes_reverse(state: &super::LruInternalState) -> usize {
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
        policy.advise(&entry(0), &LiquidCacheMode::Arrow);
        policy.advise(&entry(1), &LiquidCacheMode::Arrow);

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

    fn concurrent_invariant_advice_once(policy: Arc<dyn CachePolicy>) {
        let num_threads = 4;

        for i in 0..100 {
            policy.notify_insert(&entry(i));
        }

        let advised_entries = Arc::new(crate::sync::Mutex::new(Vec::new()));

        let mut handles = Vec::new();
        for _ in 0..num_threads {
            let policy_clone = policy.clone();
            let advised_entries_clone = advised_entries.clone();

            let handle = thread::spawn(move || {
                let advice = policy_clone.advise(&entry(999), &LiquidCacheMode::Arrow);
                if let CacheAdvice::Evict(entry_id) = advice {
                    let mut entries = advised_entries_clone.lock().unwrap();
                    entries.push(entry_id);
                }
            });

            handles.push(handle);
        }

        for handle in handles {
            handle.join().unwrap();
        }

        let entries = advised_entries.lock().unwrap();
        let mut unique_entries = entries.clone();
        unique_entries.sort();
        unique_entries.dedup();

        // If there are duplicates, entries.len() will be greater than unique_entries.len()
        assert_eq!(
            entries.len(),
            unique_entries.len(),
            "Some entries were advised for eviction multiple times: {entries:?}"
        );
    }

    #[test]
    fn test_concurrent_invariant_advice_once() {
        concurrent_invariant_advice_once(Arc::new(LruPolicy::new()));

        concurrent_invariant_advice_once(Arc::new(DiscardPolicy));

        concurrent_invariant_advice_once(Arc::new(FiloPolicy::new()));

        concurrent_invariant_advice_once(Arc::new(ToDiskPolicy::new()));
    }

    #[cfg(feature = "shuttle")]
    #[test]
    fn shuttle_concurrent_invariant_advice_once() {
        crate::utils::shuttle_test(test_concurrent_invariant_advice_once);
    }

    fn concurrent_lru_operations() {
        let policy = Arc::new(LruPolicy::new());
        let num_threads = 4;
        let operations_per_thread = 100;

        let total_inserts = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let total_evictions = Arc::new(std::sync::atomic::AtomicUsize::new(0));

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
                    let op_type = i % 3; // 0: insert, 1: access, 2: evict
                    let entry_id = entry((thread_id * operations_per_thread + i) as u64);

                    match op_type {
                        0 => {
                            policy_clone.notify_insert(&entry_id);
                            total_inserts_clone.fetch_add(1, Ordering::SeqCst);
                        }
                        1 => {
                            // Every thread also accesses entries created by other threads
                            if i > 10 {
                                let other_thread = (thread_id + 1) % num_threads;
                                let other_id =
                                    entry((other_thread * operations_per_thread + i - 10) as u64);
                                policy_clone.notify_access(&other_id);
                            }
                            policy_clone.notify_access(&entry_id);
                        }
                        2 => {
                            if i > 20 {
                                // Evict some earlier entries we created
                                let to_evict =
                                    entry((thread_id * operations_per_thread + i - 20) as u64);
                                policy_clone.advise(&to_evict, &LiquidCacheMode::Arrow);
                                total_evictions_clone.fetch_add(1, Ordering::SeqCst);
                            }
                        }
                        _ => unreachable!(),
                    }
                }
            });

            handles.push(handle);
        }

        // Wait for all threads to complete
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

        let entry_id1 = create_entry_id(1, 1, 1, 1);
        let entry_id2 = create_entry_id(1, 1, 1, 2);
        let entry_id3 = create_entry_id(1, 1, 1, 3);

        let on_disk_path = entry_id1.on_disk_path(store.config().cache_root_dir());
        std::fs::create_dir_all(on_disk_path.parent().unwrap()).unwrap();

        store.insert(entry_id1, create_test_array(100));
        store.insert(entry_id2, create_test_array(100));
        store.insert(entry_id3, create_test_array(100));

        store.get(&entry_id1, CacheAccessReason::Testing);

        let entry_id4 = create_entry_id(4, 4, 4, 4);
        store.insert(entry_id4, create_test_array(100));

        assert!(store.get(&entry_id1, CacheAccessReason::Testing).is_some());
        assert!(store.get(&entry_id3, CacheAccessReason::Testing).is_some());

        match store.get(&entry_id2, CacheAccessReason::Testing) {
            Some(CachedBatch::DiskLiquid) => {}
            None => {} // This is also acceptable if fully evicted
            other => panic!("Expected OnDiskLiquid or None, got {other:?}"),
        }
    }

    #[test]
    fn test_filo_advisor() {
        let advisor = FiloPolicy::new();
        let store = create_cache_store(3000, Box::new(advisor));

        let entry_id1 = create_entry_id(1, 1, 1, 1);
        let entry_id2 = create_entry_id(1, 1, 1, 2);
        let entry_id3 = create_entry_id(1, 1, 1, 3);

        let on_disk_path = entry_id1.on_disk_path(store.config().cache_root_dir());
        std::fs::create_dir_all(on_disk_path.parent().unwrap()).unwrap();

        store.insert(entry_id1, create_test_array(100));
        store.insert(entry_id2, create_test_array(100));
        store.insert(entry_id3, create_test_array(100));

        let entry_id4 = create_entry_id(4, 4, 4, 4);
        store.insert(entry_id4, create_test_array(100));

        assert!(store.get(&entry_id1, CacheAccessReason::Testing).is_some());
        assert!(store.get(&entry_id2, CacheAccessReason::Testing).is_some());
        assert!(store.get(&entry_id4, CacheAccessReason::Testing).is_some());

        match store.get(&entry_id3, CacheAccessReason::Testing) {
            Some(CachedBatch::DiskLiquid) => {}
            None => {} // This is also acceptable if fully evicted
            other => panic!("Expected OnDiskLiquid or None, got {other:?}"),
        }
    }

    #[test]
    fn test_sieve_policy_insertion_order() {
        let policy = SievePolicy::new();
        let e1 = entry(1);
        let e2 = entry(2);
        let e3 = entry(3);

        // Insert entries: e1 -> e2 -> e3
        policy.notify_insert(&e1);
        policy.notify_insert(&e2);
        policy.notify_insert(&e3);

        // No accesses yet; first eviction should target e1 (the oldest, at tail)
        let advice = policy.advise(&entry(4), &LiquidCacheMode::Arrow);
        assert_eq!(advice, CacheAdvice::Evict(e1));
    }

    #[test]
    fn test_sieve_policy_access_behavior() {
        let policy = SievePolicy::new();
        let e1 = entry(1);
        let e2 = entry(2);
        let e3 = entry(3);

        policy.notify_insert(&e1);
        policy.notify_insert(&e2);
        policy.notify_insert(&e3);

        // Access e1: mark visited
        policy.notify_access(&e1);

        // Now eviction should skip e1 (clearing its bit) and evict e2
        let advice = policy.advise(&entry(4), &LiquidCacheMode::Arrow);
        assert_eq!(advice, CacheAdvice::Evict(e2));
    }

    #[test]
    fn test_sieve_policy_multiple_access_then_evict() {
        let policy = SievePolicy::new();
        let e1 = entry(1);
        let e2 = entry(2);
        let e3 = entry(3);

        policy.notify_insert(&e1);
        policy.notify_insert(&e2);
        policy.notify_insert(&e3);

        // Access all to set visited bits
        policy.notify_access(&e1);
        policy.notify_access(&e2);
        policy.notify_access(&e3);

        // First advise clears all bits in one pass, second pass evicts e1
        let first = policy.advise(&entry(4), &LiquidCacheMode::Arrow);
        // Should clear bits, not evict yet
        assert_eq!(first, CacheAdvice::Evict(e1));
    }

    #[test]
    fn test_sieve_policy_single_item_self() {
        let policy = SievePolicy::new();
        let e1 = entry(1);
        policy.notify_insert(&e1);

        // With only one item, advising eviction should evict e1
        let advice = policy.advise(&e1, &LiquidCacheMode::Arrow);
        assert_eq!(advice, CacheAdvice::Evict(e1));
    }

    #[test]
    fn test_sieve_policy_advise_empty() {
        let policy = SievePolicy::new();
        // No entries inserted: fallback should discard
        let advice = policy.advise(&entry(1), &LiquidCacheMode::Arrow);
        assert_eq!(advice, CacheAdvice::Discard);
    }

    #[test]
    fn test_sieve_integration() {
        let advisor = SievePolicy::new();
        let store = create_cache_store(3000, Box::new(advisor));

        let entry_id1 = create_entry_id(1, 1, 1, 1);
        let entry_id2 = create_entry_id(1, 1, 1, 2);
        let entry_id3 = create_entry_id(1, 1, 1, 3);

        // Prepare on-disk directory
        let on_disk_path = entry_id1.on_disk_path(&store.config().cache_root_dir());
        std::fs::create_dir_all(on_disk_path.parent().unwrap()).unwrap();

        // Fill cache
        store.insert(entry_id1, create_test_array(100));
        store.insert(entry_id2, create_test_array(100));
        store.insert(entry_id3, create_test_array(100));

        // Trigger eviction with a fourth entry
        let entry_id4 = create_entry_id(4, 4, 4, 4);
        store.insert(entry_id4, create_test_array(100));

        // SIEVE should evict the oldest entry (entry_id1)
        match store.get(&entry_id1, CacheAccessReason::Testing) {
            Some(CachedBatch::DiskLiquid) => {}
            None => {} // Also acceptable if fully evicted
            other => panic!("Expected OnDiskLiquid or None, got {:?}", other),
        }

        // The other entries should still be resident
        assert!(store.get(&entry_id2, CacheAccessReason::Testing).is_some());
        assert!(store.get(&entry_id3, CacheAccessReason::Testing).is_some());
        assert!(store.get(&entry_id4, CacheAccessReason::Testing).is_some());
    }

    #[test]
    fn test_clock_policy_insertion_order() {
        let policy = ClockPolicy::new();
        let e1 = entry(1);
        let e2 = entry(2);
        let e3 = entry(3);

        // Insert entries: e1 -> e2 -> e3
        policy.notify_insert(&e1);
        policy.notify_insert(&e2);
        policy.notify_insert(&e3);

        // Without any prior evictions or accesses, the oldest (e1) should be evicted
        let advice = policy.advise(&entry(4), &LiquidCacheMode::Arrow);
        assert_eq!(advice, CacheAdvice::Evict(e1));
    }

    #[test]
    fn test_clock_policy_sequential_evictions() {
        let policy = ClockPolicy::new();
        let e1 = entry(1);
        let e2 = entry(2);
        let e3 = entry(3);

        policy.notify_insert(&e1);
        policy.notify_insert(&e2);
        policy.notify_insert(&e3);

        // First eviction should remove e1
        let adv1 = policy.advise(&entry(4), &LiquidCacheMode::Arrow);
        assert_eq!(adv1, CacheAdvice::Evict(e1));

        // Next eviction should remove e3 (next hand position)
        let adv2 = policy.advise(&entry(5), &LiquidCacheMode::Arrow);
        assert_eq!(adv2, CacheAdvice::Evict(e3));

        // Finally, e2 should be evicted
        let adv3 = policy.advise(&entry(6), &LiquidCacheMode::Arrow);
        assert_eq!(adv3, CacheAdvice::Evict(e2));
    }

    #[test]
    fn test_clock_policy_single_item() {
        let policy = ClockPolicy::new();
        let e1 = entry(1);
        policy.notify_insert(&e1);

        // With only one item, evicting should return that item
        let advice = policy.advise(&e1, &LiquidCacheMode::Arrow);
        assert_eq!(advice, CacheAdvice::Evict(e1));
    }

    #[test]
    fn test_clock_policy_advise_empty() {
        let policy = ClockPolicy::new();

        // No entries inserted: should discard
        let advice = policy.advise(&entry(1), &LiquidCacheMode::Arrow);
        assert_eq!(advice, CacheAdvice::Discard);
    }

    #[test]
    fn test_clock_policy_integration_with_store() {
        let advisor = ClockPolicy::new();
        let store = create_cache_store(3000, Box::new(advisor));

        let entry_id1 = create_entry_id(1, 1, 1, 1);
        let entry_id2 = create_entry_id(1, 1, 1, 2);
        let entry_id3 = create_entry_id(1, 1, 1, 3);

        // Prepare on-disk directory for eviction
        let on_disk_path = entry_id1.on_disk_path(&store.config().cache_root_dir());
        std::fs::create_dir_all(on_disk_path.parent().unwrap()).unwrap();

        // Fill cache to capacity
        store.insert(entry_id1, create_test_array(100));
        store.insert(entry_id2, create_test_array(100));
        store.insert(entry_id3, create_test_array(100));

        // Insert one more to trigger eviction
        let entry_id4 = create_entry_id(4, 4, 4, 4);
        store.insert(entry_id4, create_test_array(100));

        // Oldest (entry_id1) should be on-disk or evicted
        match store.get(&entry_id1, CacheAccessReason::Testing) {
            Some(CachedBatch::DiskLiquid) => {}
            None => {}
            other => panic!("Expected OnDiskLiquid or None, got {:?}", other),
        }

        // Others should still be retrieveable in-memory
        assert!(store.get(&entry_id2, CacheAccessReason::Testing).is_some());
        assert!(store.get(&entry_id3, CacheAccessReason::Testing).is_some());
        assert!(store.get(&entry_id4, CacheAccessReason::Testing).is_some());
    }
    fn test_to_disk_policy() {
        let advisor = ToDiskPolicy::new();
        let store = create_cache_store(3000, Box::new(advisor)); // Small budget to force disk storage

        let entry_id1 = create_entry_id(1, 1, 1, 1);
        let entry_id2 = create_entry_id(1, 1, 1, 2);

        let on_disk_liquid_path = entry_id1.on_disk_path(store.config().cache_root_dir());
        let on_disk_arrow_path = entry_id1.on_disk_arrow_path(store.config().cache_root_dir());
        std::fs::create_dir_all(on_disk_liquid_path.parent().unwrap()).unwrap();
        std::fs::create_dir_all(on_disk_arrow_path.parent().unwrap()).unwrap();

        store.insert(entry_id1, create_test_array(100));
        assert!(matches!(
            store.get(&entry_id1, CacheAccessReason::Testing).unwrap(),
            CachedBatch::MemoryArrow(_)
        ));

        store.insert(entry_id2, create_test_array(2000)); // Large enough to exceed budget

        assert!(matches!(
            store.get(&entry_id2, CacheAccessReason::Testing).unwrap(),
            CachedBatch::DiskArrow
        ));

        assert!(store.get(&entry_id1, CacheAccessReason::Testing).is_some());
    }

    #[test]
    fn test_to_disk_policy_advice() {
        let policy = ToDiskPolicy::new();
        let entry_id = entry(42);

        let advice = policy.advise(&entry_id, &LiquidCacheMode::Arrow);
        assert_eq!(advice, CacheAdvice::ToDisk(entry_id));

        let advice = policy.advise(
            &entry_id,
            &LiquidCacheMode::Liquid {
                transcode_in_background: false,
            },
        );
        assert_eq!(advice, CacheAdvice::ToDisk(entry_id));
    }
}
