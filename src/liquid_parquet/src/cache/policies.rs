use std::{collections::VecDeque, sync::Mutex};

use dashmap::DashMap;

use super::{CacheAdvice, CacheEntryID, CachedBatch};

/// The cache policy that guides the replacement of LiquidCache
pub trait CachePolicy: std::fmt::Debug + Send + Sync {
    /// Give advice on what to do when cache is full.
    fn advise(
        &self,
        entry_id: &CacheEntryID,
        to_insert: &CachedBatch,
        cached: &DashMap<CacheEntryID, CachedBatch>,
    ) -> CacheAdvice;

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
    fn advise(
        &self,
        entry_id: &CacheEntryID,
        _to_insert: &CachedBatch,
        cached: &DashMap<CacheEntryID, CachedBatch>,
    ) -> CacheAdvice {
        // Get the newest entry from the front of the queue
        if let Some(newest_entry) = self.get_newest_entry() {
            // Only evict if the entry still exists in the cache
            if cached.contains_key(&newest_entry) && newest_entry != *entry_id {
                return CacheAdvice::Evict(newest_entry);
            }
        }

        // If no entries to evict, transcode to disk as fallback
        CacheAdvice::TranscodeToDisk(*entry_id)
    }

    fn notify_insert(&self, entry_id: &CacheEntryID) {
        self.add_entry(entry_id);
    }
}

/// The policy that implement the Lru algorithm.
#[derive(Debug, Default)]
pub struct LruPolicy {
    queue: Mutex<VecDeque<CacheEntryID>>,
}

impl LruPolicy {
    /// Create a new [LruPolicy].
    pub fn new() -> Self {
        Self {
            queue: Mutex::new(VecDeque::new()),
        }
    }

    fn add_entry(&self, entry_id: &CacheEntryID) {
        let mut queue = self.queue.lock().unwrap();
        // Add to front of queue
        queue.push_front(*entry_id);
    }

    fn move_to_front(&self, entry_id: &CacheEntryID) {
        let mut queue = self.queue.lock().unwrap();
        if let Some(pos) = queue.iter().position(|id| id == entry_id) {
            let entry = queue.remove(pos).unwrap();
            queue.push_front(entry);
        } else {
            queue.push_front(*entry_id);
        }
    }

    fn get_oldest_entry(&self) -> Option<CacheEntryID> {
        let mut queue = self.queue.lock().unwrap();
        queue.pop_back()
    }
}

impl CachePolicy for LruPolicy {
    fn advise(
        &self,
        entry_id: &CacheEntryID,
        _to_insert: &CachedBatch,
        cached: &DashMap<CacheEntryID, CachedBatch>,
    ) -> CacheAdvice {
        // Get the oldest entry from the back of the queue
        if let Some(oldest_entry) = self.get_oldest_entry() {
            // Only evict if the entry still exists in the cache
            if cached.contains_key(&oldest_entry) && oldest_entry != *entry_id {
                return CacheAdvice::Evict(oldest_entry);
            }
        }

        // If no entries to evict, transcode to disk as fallback
        CacheAdvice::TranscodeToDisk(*entry_id)
    }

    fn notify_access(&self, entry_id: &CacheEntryID) {
        // Move the accessed entry to the front of the queue
        self.move_to_front(entry_id);
    }

    fn notify_insert(&self, entry_id: &CacheEntryID) {
        self.add_entry(entry_id);
    }
}

#[cfg(test)]
mod test {
    use crate::cache::utils::{create_cache_store, create_entry_id, create_test_array};

    use super::*;

    #[test]
    fn test_lru_policy() {
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
        store.insert(entry_id1, CachedBatch::ArrowMemory(create_test_array(100)));
        store.insert(entry_id2, CachedBatch::ArrowMemory(create_test_array(100)));
        store.insert(entry_id3, CachedBatch::ArrowMemory(create_test_array(100)));

        // Access entry 1 to move it to front
        store.get(&entry_id1);

        // Insert a fourth entry to force eviction
        let entry_id4 = create_entry_id(4, 4, 4, 4);
        store.insert(entry_id4, CachedBatch::ArrowMemory(create_test_array(100)));

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
        store.insert(entry_id1, CachedBatch::ArrowMemory(create_test_array(100)));
        store.insert(entry_id2, CachedBatch::ArrowMemory(create_test_array(100)));
        store.insert(entry_id3, CachedBatch::ArrowMemory(create_test_array(100)));

        // Insert a fourth entry to force eviction
        let entry_id4 = create_entry_id(4, 4, 4, 4);
        store.insert(entry_id4, CachedBatch::ArrowMemory(create_test_array(100)));

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
