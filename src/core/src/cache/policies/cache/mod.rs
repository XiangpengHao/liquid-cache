//! Cache policies for liquid cache.

use crate::cache::cached_batch::CachedBatchType;
use crate::cache::utils::EntryID;

mod clock;
mod doubly_linked_list;
mod filo;
mod lru;
mod s3_fifo;
mod sieve;
mod three_queue;

pub use clock::ClockPolicy;
pub use filo::FifoPolicy;
pub use filo::FiloPolicy;
pub use lru::LruPolicy;
pub use s3_fifo::S3FifoPolicy;
pub use sieve::SievePolicy;
pub use three_queue::LiquidPolicy;

/// The cache policy that guides the replacement of LiquidCache
pub trait CachePolicy: std::fmt::Debug + Send + Sync {
    /// Give cnt amount of entries to evict when cache is full.
    fn find_victim(&self, cnt: usize) -> Vec<EntryID>;

    /// Notify the cache policy that an entry was inserted.
    fn notify_insert(&self, _entry_id: &EntryID, _batch_type: CachedBatchType) {}

    /// Notify the cache policy that an entry was accessed.
    fn notify_access(&self, _entry_id: &EntryID, _batch_type: CachedBatchType) {}
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cache::utils::EntryID;
    use crate::sync::{Arc, Mutex, thread};

    fn entry(id: usize) -> EntryID {
        id.into()
    }

    fn concurrent_invariant_advice_once(policy: Arc<dyn CachePolicy>) {
        let num_threads = 4;

        for i in 0..100 {
            policy.notify_insert(&entry(i), CachedBatchType::MemoryArrow);
        }

        let advised_entries = Arc::new(Mutex::new(Vec::new()));

        let mut handles = Vec::new();
        for _ in 0..num_threads {
            let policy_clone = policy.clone();
            let advised_entries_clone = advised_entries.clone();

            let handle = thread::spawn(move || {
                let advice = policy_clone.find_victim(1);
                if let Some(entry_id) = advice.first() {
                    let mut entries = advised_entries_clone.lock().unwrap();
                    entries.push(*entry_id);
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

        assert_eq!(
            entries.len(),
            unique_entries.len(),
            "Some entries were advised for eviction multiple times: {entries:?}"
        );
    }

    fn run_concurrent_invariant_tests() {
        concurrent_invariant_advice_once(Arc::new(LruPolicy::new()));
        concurrent_invariant_advice_once(Arc::new(FiloPolicy::new()));
    }

    #[test]
    fn test_concurrent_invariant_advice_once() {
        run_concurrent_invariant_tests();
    }

    #[cfg(feature = "shuttle")]
    #[test]
    fn shuttle_concurrent_invariant_advice_once() {
        crate::utils::shuttle_test(run_concurrent_invariant_tests);
    }
}
