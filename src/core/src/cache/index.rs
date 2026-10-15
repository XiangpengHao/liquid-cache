use congee::CongeeArc;
use std::{
    fmt::{Debug, Formatter},
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
};

use crate::cache::{cached_batch::CacheEntry, utils::EntryID};

/// What the index holds for a key: the entry, and the identity of the source
/// it was cached for.
///
/// A key does not name its source uniquely: callers pack fields into fixed
/// widths, so two sources can compute the same key. The identity is what tells
/// them apart, so a read under any other identity must treat the slot as
/// absent.
#[derive(Debug)]
pub(crate) struct Slot {
    pub(crate) identity: u64,
    pub(crate) entry: CacheEntry,
}

impl Slot {
    pub(crate) fn belongs_to(&self, identity: u64) -> bool {
        self.identity == identity
    }
}

pub(crate) struct ArtIndex {
    art: CongeeArc<EntryID, Slot>,
    entry_count: AtomicUsize,
}

impl Debug for ArtIndex {
    fn fmt(&self, _f: &mut Formatter<'_>) -> std::fmt::Result {
        Ok(())
    }
}

impl ArtIndex {
    pub(crate) fn new() -> Self {
        let art: CongeeArc<EntryID, Slot> = CongeeArc::new();
        Self {
            art,
            entry_count: AtomicUsize::new(0),
        }
    }

    /// The slot under `entry_id`, whichever identity holds it.
    pub(crate) fn get(&self, entry_id: &EntryID) -> Option<Arc<Slot>> {
        let guard = self.art.pin();
        self.art.get(*entry_id, &guard)
    }

    /// The slot under `entry_id`, only if `identity` holds it.
    pub(crate) fn get_for(&self, entry_id: &EntryID, identity: u64) -> Option<Arc<Slot>> {
        self.get(entry_id).filter(|slot| slot.belongs_to(identity))
    }

    pub(crate) fn contains(&self, entry_id: &EntryID, identity: u64) -> bool {
        self.get_for(entry_id, identity).is_some()
    }

    /// Store `batch` under `entry_id` for `identity`, replacing whatever held
    /// the key, including a slot held by another identity.
    pub(crate) fn insert(&self, entry_id: &EntryID, identity: u64, batch: CacheEntry) {
        let guard = self.art.pin();
        let slot = Slot {
            identity,
            entry: batch,
        };
        let existing = self
            .art
            .insert(*entry_id, Arc::new(slot), &guard)
            .expect("Insertion failed");
        if existing.is_none() {
            self.entry_count.fetch_add(1, Ordering::Relaxed);
        }
    }

    pub(crate) fn remove(&self, entry_id: &EntryID) -> Option<Arc<Slot>> {
        let guard = self.art.pin();
        let removed = self.art.remove(*entry_id, &guard);
        if removed.is_some() {
            self.entry_count.fetch_sub(1, Ordering::Relaxed);
        }
        removed
    }

    pub(crate) fn reset(&self) {
        let guard = self.art.pin();
        self.art.keys().into_iter().for_each(|k| {
            _ = self.art.remove(k, &guard).unwrap();
        });
        self.entry_count.store(0, Ordering::Relaxed);
    }

    pub(crate) fn for_each(&self, mut f: impl FnMut(&EntryID, &Slot)) {
        let guard = self.art.pin();
        for id in self.art.keys().into_iter() {
            f(
                &id,
                &self
                    .art
                    .get(id, &guard)
                    .expect("Failed to get value from ART"),
            );
        }
    }

    #[cfg(test)]
    pub(crate) fn keys(&self) -> Vec<EntryID> {
        self.art.keys()
    }

    pub(crate) fn entry_count(&self) -> usize {
        self.entry_count.load(Ordering::Relaxed)
    }
}

#[cfg(test)]
mod tests {
    use crate::cache::cached_batch::CacheEntry;
    use crate::cache::utils::create_test_array;

    use super::*;

    #[test]
    fn test_get_and_contains() {
        let store = ArtIndex::new();
        let entry_id1: EntryID = EntryID::from(1);
        let entry_id2: EntryID = EntryID::from(2);
        let array1 = create_test_array(100);

        // Initially, entries should not be cached
        assert!(!store.contains(&entry_id1, 0));
        assert!(!store.contains(&entry_id2, 0));
        assert!(store.get(&entry_id1).is_none());

        // Insert an entry and verify it's cached
        {
            store.insert(&entry_id1, 0, array1.clone());
        }

        assert!(store.contains(&entry_id1, 0));
        assert!(!store.contains(&entry_id2, 0));

        // Get should return the cached value
        match store.get_for(&entry_id1, 0) {
            Some(slot) => match &slot.entry {
                CacheEntry::MemoryArrow(arr) => assert_eq!(arr.len(), 100),
                _ => panic!("Expected ArrowMemory batch"),
            },
            None => panic!("Expected ArrowMemory batch"),
        }
    }

    #[test]
    fn test_reset() {
        let store = ArtIndex::new();
        let entry_id: EntryID = EntryID::from(1);
        let array = create_test_array(100);

        store.insert(&entry_id, 0, array.clone());

        let entry_id: EntryID = EntryID::from(1);
        assert!(store.contains(&entry_id, 0));

        store.reset();
        let entry_id: EntryID = EntryID::from(1);
        assert!(!store.contains(&entry_id, 0));
    }

    #[test]
    fn a_slot_is_visible_only_to_its_identity() {
        let store = ArtIndex::new();
        let entry_id = EntryID::from(1);
        store.insert(&entry_id, 7, create_test_array(10));

        assert!(store.contains(&entry_id, 7));
        assert!(!store.contains(&entry_id, 8));
        assert!(store.get_for(&entry_id, 8).is_none());

        store.insert(&entry_id, 8, create_test_array(20));
        assert!(!store.contains(&entry_id, 7));
        assert!(store.contains(&entry_id, 8));
        assert_eq!(store.entry_count(), 1);
    }
}
