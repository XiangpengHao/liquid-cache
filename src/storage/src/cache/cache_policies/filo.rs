//! FILO (First In, Last Out) cache policy implementation.

use std::collections::VecDeque;

use crate::{cache::utils::EntryID, sync::Mutex};

use super::CachePolicy;

/// The policy that implements the FILO (First In, Last Out) algorithm.
/// Newest entries are evicted first.
#[derive(Debug, Default)]
pub struct FiloPolicy {
    queue: Mutex<VecDeque<EntryID>>,
}

impl FiloPolicy {
    /// Create a new [`FiloPolicy`].
    pub fn new() -> Self {
        Self {
            queue: Mutex::new(VecDeque::new()),
        }
    }

    fn add_entry(&self, entry_id: &EntryID) {
        let mut queue = self.queue.lock().unwrap();
        queue.push_front(*entry_id);
    }
}

impl CachePolicy for FiloPolicy {
    fn find_victim(&self, cnt: usize) -> Vec<EntryID> {
        let mut queue = self.queue.lock().unwrap();
        if cnt == 0 || queue.is_empty() {
            return vec![];
        }
        let k = cnt.min(queue.len());
        let mut out = vec![];
        for _i in 0..k {
            out.push(queue.pop_front().unwrap());
        }
        out
    }

    fn notify_insert(&self, entry_id: &EntryID) {
        self.add_entry(entry_id);
    }
}

/// The policy that implements the FIFO (First In, First Out) algorithm.
/// Oldest entries are evicted first.
#[derive(Debug, Default)]
pub struct FifoPolicy {
    queue: Mutex<VecDeque<EntryID>>,
}

impl FifoPolicy {
    /// Create a new [`FiloPolicy`].
    pub fn new() -> Self {
        Self {
            queue: Mutex::new(VecDeque::new()),
        }
    }

    fn add_entry(&self, entry_id: &EntryID) {
        let mut queue = self.queue.lock().unwrap();
        queue.push_back(*entry_id);
    }
}

impl CachePolicy for FifoPolicy {
    fn find_victim(&self, cnt: usize) -> Vec<EntryID> {
        let mut queue = self.queue.lock().unwrap();
        if cnt == 0 || queue.is_empty() {
            return vec![];
        }
        let k = cnt.min(queue.len());
        let mut out = vec![];
        for _i in 0..k {
            out.push(queue.pop_front().unwrap());
        }
        out
    }

    fn notify_insert(&self, entry_id: &EntryID) {
        self.add_entry(entry_id);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cache::cached_data::CachedBatch;
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
            assert!(matches!(data.raw_data(), CachedBatch::MemoryLiquid(_)));
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

        policy.notify_insert(&e1);
        policy.notify_insert(&e2);

        assert_eq!(policy.find_victim(1), vec![e2]);
        assert_eq!(policy.find_victim(1), vec![e1]);
    }
}
