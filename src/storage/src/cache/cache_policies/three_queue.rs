use std::collections::VecDeque;

use crate::{
    cache::{CachePolicy, EntryID, cached_data::CachedBatchType},
    sync::Mutex,
};

/// Cache policy that keeps independent FIFO queues per batch type.
#[derive(Debug, Default)]
pub struct LiquidPolicy {
    inner: Mutex<LiquidQueueInternalState>,
}

#[derive(Default, Debug)]
struct LiquidQueueInternalState {
    arrow: VecDeque<EntryID>,
    liquid: VecDeque<EntryID>,
    hybrid: VecDeque<EntryID>,
    disk: VecDeque<EntryID>,
}

impl LiquidPolicy {
    /// Create a new [`ThreeQueuePolicy`].
    pub fn new() -> Self {
        Self {
            inner: Mutex::new(LiquidQueueInternalState::default()),
        }
    }
}

impl CachePolicy for LiquidPolicy {
    fn find_victim(&self, cnt: usize) -> Vec<EntryID> {
        if cnt == 0 {
            return vec![];
        }

        let mut inner = self.inner.lock().unwrap();
        let mut victims = Vec::with_capacity(cnt);

        while victims.len() < cnt {
            if let Some(entry) = inner.arrow.pop_front() {
                victims.push(entry);
                continue;
            }

            if let Some(entry) = inner.liquid.pop_front() {
                victims.push(entry);
                continue;
            }

            if let Some(entry) = inner.hybrid.pop_front() {
                victims.push(entry);
                continue;
            }

            break;
        }

        victims
    }

    fn notify_insert(&self, entry_id: &EntryID, batch_type: CachedBatchType) {
        let mut inner = self.inner.lock().unwrap();

        match batch_type {
            CachedBatchType::MemoryArrow => {
                inner.arrow.push_back(*entry_id);
            }
            CachedBatchType::MemoryLiquid => {
                inner.liquid.push_back(*entry_id);
            }
            CachedBatchType::MemoryHybridLiquid => {
                inner.hybrid.push_back(*entry_id);
            }
            CachedBatchType::DiskLiquid | CachedBatchType::DiskArrow => {
                inner.disk.push_back(*entry_id);
            }
        }
    }

    fn notify_access(&self, _entry_id: &EntryID, _batch_type: CachedBatchType) {}
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cache::utils::EntryID;

    fn entry(id: usize) -> EntryID {
        id.into()
    }

    #[test]
    fn test_fifo_within_each_queue() {
        let policy = LiquidPolicy::new();

        let arrow_a = entry(1);
        let arrow_b = entry(2);
        let liquid_a = entry(3);
        let liquid_b = entry(4);

        policy.notify_insert(&arrow_a, CachedBatchType::MemoryArrow);
        policy.notify_insert(&arrow_b, CachedBatchType::MemoryArrow);
        policy.notify_insert(&liquid_a, CachedBatchType::MemoryLiquid);
        policy.notify_insert(&liquid_b, CachedBatchType::MemoryLiquid);

        assert_eq!(policy.find_victim(1), vec![arrow_a]);
        assert_eq!(policy.find_victim(2), vec![arrow_b, liquid_a]);
        assert_eq!(policy.find_victim(1), vec![liquid_b]);
    }

    #[test]
    fn test_queue_priority_order() {
        let policy = LiquidPolicy::new();

        let arrow_entry = entry(1);
        let liquid_entry = entry(2);
        let hybrid_entry = entry(3);

        policy.notify_insert(&liquid_entry, CachedBatchType::MemoryLiquid);
        policy.notify_insert(&hybrid_entry, CachedBatchType::MemoryHybridLiquid);
        policy.notify_insert(&arrow_entry, CachedBatchType::MemoryArrow);

        // Request more victims than available to ensure we only get what exists.
        let victims = policy.find_victim(5);
        assert_eq!(victims, vec![arrow_entry, liquid_entry, hybrid_entry]);
    }

    #[test]
    fn test_zero_victim_request_returns_empty() {
        let policy = LiquidPolicy::new();

        policy.notify_insert(&entry(1), CachedBatchType::MemoryArrow);
        assert!(policy.find_victim(0).is_empty());
    }

    #[test]
    fn test_disk_entries_not_evicted() {
        let policy = LiquidPolicy::new();

        let disk_entry = entry(1);
        let arrow_entry = entry(2);
        let liquid_entry = entry(3);

        policy.notify_insert(&disk_entry, CachedBatchType::DiskArrow);
        policy.notify_insert(&arrow_entry, CachedBatchType::MemoryArrow);
        policy.notify_insert(&liquid_entry, CachedBatchType::MemoryLiquid);

        let victims = policy.find_victim(5);
        assert_eq!(victims, vec![arrow_entry, liquid_entry]);

        // Only the disk entry remains and should still not be evicted.
        assert!(policy.find_victim(1).is_empty());
    }
}
