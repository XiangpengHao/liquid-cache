//! FILO (First In, Last Out) and FIFO cache policy implementations.

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
struct QueueNode {
    entry_id: EntryID,
}

type NodePtr = NonNull<DoublyLinkedNode<QueueNode>>;

#[derive(Debug, Default)]
struct QueueState {
    map: HashMap<EntryID, NodePtr>,
    list: DoublyLinkedList<QueueNode>,
}

impl QueueState {
    fn is_empty(&self) -> bool {
        self.list.head().is_none()
    }

    fn insert_front(&mut self, entry_id: EntryID) {
        if let Some(ptr) = self.map.get(&entry_id).copied() {
            unsafe {
                self.list.unlink(ptr);
                self.list.push_front(ptr);
            }
            return;
        }

        let node = DoublyLinkedNode::new(QueueNode { entry_id });
        let ptr = NonNull::from(Box::leak(node));

        self.map.insert(entry_id, ptr);
        unsafe {
            self.list.push_front(ptr);
        }
    }

    fn insert_back(&mut self, entry_id: EntryID) {
        if let Some(ptr) = self.map.get(&entry_id).copied() {
            unsafe {
                self.list.unlink(ptr);
                self.list.push_back(ptr);
            }
            return;
        }

        let node = DoublyLinkedNode::new(QueueNode { entry_id });
        let ptr = NonNull::from(Box::leak(node));

        self.map.insert(entry_id, ptr);
        unsafe {
            self.list.push_back(ptr);
        }
    }

    fn pop_front(&mut self) -> Option<EntryID> {
        let head_ptr = self.list.head()?;
        let entry_id = unsafe { head_ptr.as_ref().data.entry_id };
        let node_ptr = self
            .map
            .remove(&entry_id)
            .expect("head pointer must have map entry");
        unsafe {
            self.list.unlink(node_ptr);
            drop_boxed_node(node_ptr);
        }
        Some(entry_id)
    }
}

impl Drop for QueueState {
    fn drop(&mut self) {
        let handles: Vec<_> = self.map.drain().map(|(_, ptr)| ptr).collect();
        for ptr in handles {
            unsafe {
                self.list.unlink(ptr);
                drop_boxed_node(ptr);
            }
        }
        unsafe {
            self.list.drop_all();
        }
    }
}

/// The policy that implements the FILO (First In, Last Out) algorithm.
/// Newest entries are evicted first.
#[derive(Debug, Default)]
pub struct FiloPolicy {
    state: Mutex<QueueState>,
}

impl FiloPolicy {
    /// Create a new [`FiloPolicy`].
    pub fn new() -> Self {
        Self {
            state: Mutex::new(QueueState::default()),
        }
    }
}

// SAFETY: Access to raw pointers is protected by the internal `Mutex`.
unsafe impl Send for FiloPolicy {}
unsafe impl Sync for FiloPolicy {}

impl CachePolicy for FiloPolicy {
    fn find_victim(&self, cnt: usize) -> Vec<EntryID> {
        if cnt == 0 {
            return vec![];
        }

        let mut state = self.state.lock().unwrap();
        if state.is_empty() {
            return vec![];
        }

        let mut victims = Vec::with_capacity(cnt);
        for _ in 0..cnt {
            let Some(entry) = state.pop_front() else {
                break;
            };
            victims.push(entry);
        }
        victims
    }

    fn notify_insert(&self, entry_id: &EntryID, _batch_type: CachedBatchType) {
        let mut state = self.state.lock().unwrap();
        state.insert_front(*entry_id);
    }
}

/// The policy that implements the FIFO (First In, First Out) algorithm.
/// Oldest entries are evicted first.
#[derive(Debug, Default)]
pub struct FifoPolicy {
    state: Mutex<QueueState>,
}

impl FifoPolicy {
    /// Create a new [`FifoPolicy`].
    pub fn new() -> Self {
        Self {
            state: Mutex::new(QueueState::default()),
        }
    }
}

// SAFETY: Access to raw pointers is protected by the internal `Mutex`.
unsafe impl Send for FifoPolicy {}
unsafe impl Sync for FifoPolicy {}

impl CachePolicy for FifoPolicy {
    fn find_victim(&self, cnt: usize) -> Vec<EntryID> {
        if cnt == 0 {
            return vec![];
        }

        let mut state = self.state.lock().unwrap();
        if state.is_empty() {
            return vec![];
        }

        let mut victims = Vec::with_capacity(cnt);
        for _ in 0..cnt {
            let Some(entry) = state.pop_front() else {
                break;
            };
            victims.push(entry);
        }
        victims
    }

    fn notify_insert(&self, entry_id: &EntryID, _batch_type: CachedBatchType) {
        let mut state = self.state.lock().unwrap();
        state.insert_back(*entry_id);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cache::cached_data::{CachedBatch, CachedBatchType};
    use crate::cache::utils::{EntryID, create_cache_store, create_test_arrow_array};

    fn entry(id: usize) -> EntryID {
        id.into()
    }

    #[test]
    fn test_filo_advisor() {
        let advisor = FiloPolicy::new();
        let store = create_cache_store(3000, Box::new(advisor));

        let entry_id1 = EntryID::from(1);
        let entry_id2 = EntryID::from(2);
        let entry_id3 = EntryID::from(3);

        store.insert(entry_id1, create_test_arrow_array(100));

        let data = store.get(&entry_id1).unwrap();
        let data = data.raw_data();
        assert!(matches!(data, CachedBatch::MemoryArrow(_)));
        store.insert(entry_id2, create_test_arrow_array(100));
        store.insert(entry_id3, create_test_arrow_array(100));

        let entry_id4: EntryID = EntryID::from(4);
        store.insert(entry_id4, create_test_arrow_array(100));

        assert!(store.get(&entry_id1).is_some());
        assert!(store.get(&entry_id2).is_some());
        assert!(store.get(&entry_id4).is_some());

        if let Some(data) = store.get(&entry_id3) {
            assert!(matches!(data.raw_data(), CachedBatch::DiskLiquid));
        }
    }

    #[test]
    fn test_filo_advise_empty() {
        let policy = FiloPolicy::new();
        assert!(policy.find_victim(1).is_empty());
    }

    #[test]
    fn test_filo_advise_order() {
        let policy = FiloPolicy::new();
        let e1 = entry(1);
        let e2 = entry(2);

        policy.notify_insert(&e1, CachedBatchType::MemoryArrow);
        policy.notify_insert(&e2, CachedBatchType::MemoryArrow);

        assert_eq!(policy.find_victim(1), vec![e2]);
        assert_eq!(policy.find_victim(1), vec![e1]);
    }

    #[test]
    fn test_filo_reinsert_moves_to_front() {
        let policy = FiloPolicy::new();
        let first = entry(1);
        let second = entry(2);

        policy.notify_insert(&first, CachedBatchType::MemoryArrow);
        policy.notify_insert(&second, CachedBatchType::MemoryArrow);
        policy.notify_insert(&first, CachedBatchType::MemoryArrow);

        assert_eq!(policy.find_victim(1), vec![first]);
        assert_eq!(policy.find_victim(1), vec![second]);
    }

    #[test]
    fn test_fifo_advise_empty() {
        let policy = FifoPolicy::new();
        assert!(policy.find_victim(1).is_empty());
    }

    #[test]
    fn test_fifo_advise_order() {
        let policy = FifoPolicy::new();
        let e1 = entry(1);
        let e2 = entry(2);

        policy.notify_insert(&e1, CachedBatchType::MemoryArrow);
        policy.notify_insert(&e2, CachedBatchType::MemoryArrow);

        assert_eq!(policy.find_victim(1), vec![e1]);
        assert_eq!(policy.find_victim(1), vec![e2]);
    }

    #[test]
    fn test_fifo_reinsert_moves_to_back() {
        let policy = FifoPolicy::new();
        let first = entry(1);
        let second = entry(2);

        policy.notify_insert(&first, CachedBatchType::MemoryArrow);
        policy.notify_insert(&second, CachedBatchType::MemoryArrow);
        policy.notify_insert(&first, CachedBatchType::MemoryArrow);

        assert_eq!(policy.find_victim(1), vec![second]);
        assert_eq!(policy.find_victim(1), vec![first]);
    }
}
