use congee::CongeeArc;
use std::{
    fmt::{Debug, Formatter},
    sync::Arc,
};

use crate::cache::{cached_data::CachedBatch, utils::EntryID};

pub(crate) struct ArtIndex {
    art: CongeeArc<EntryID, CachedBatch>,
}

impl Debug for ArtIndex {
    fn fmt(&self, _f: &mut Formatter<'_>) -> std::fmt::Result {
        Ok(())
    }
}

impl ArtIndex {
    pub(crate) fn new() -> Self {
        let art: CongeeArc<EntryID, CachedBatch> = CongeeArc::new();
        Self { art }
    }

    pub(crate) fn get(&self, entry_id: &EntryID) -> Option<CachedBatch> {
        let guard = self.art.pin();
        let batch = self.art.get(*entry_id, &guard)?;
        Some(CachedBatch::clone(&batch))
    }

    pub(crate) fn is_cached(&self, entry_id: &EntryID) -> bool {
        let guard = self.art.pin();
        self.art.get(*entry_id, &guard).is_some()
    }

    pub(crate) fn insert(&self, entry_id: &EntryID, batch: CachedBatch) {
        let guard = self.art.pin();
        _ = self
            .art
            .insert(*entry_id, Arc::new(batch), &guard)
            .expect("Insertion failed");
    }

    pub(crate) fn reset(&self) {
        let guard = self.art.pin();
        self.art.keys().into_iter().for_each(|k| {
            _ = self.art.remove(k, &guard).unwrap();
        });
    }

    pub(crate) fn for_each(&self, mut f: impl FnMut(&EntryID, &CachedBatch)) {
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
}

#[cfg(test)]
mod tests {
    use crate::cache::utils::create_test_array;

    use super::*;

    #[test]
    fn test_get_and_is_cached() {
        let store = ArtIndex::new();
        let entry_id1: EntryID = EntryID::from(1);
        let entry_id2: EntryID = EntryID::from(2);
        let array1 = create_test_array(100);

        // Initially, entries should not be cached
        assert!(!store.is_cached(&entry_id1));
        assert!(!store.is_cached(&entry_id2));
        assert!(store.get(&entry_id1).is_none());

        // Insert an entry and verify it's cached
        {
            store.insert(&entry_id1, array1.clone());
        }

        assert!(store.is_cached(&entry_id1));
        assert!(!store.is_cached(&entry_id2));

        // Get should return the cached value
        match store.get(&entry_id1) {
            Some(CachedBatch::MemoryArrow(arr)) => assert_eq!(arr.len(), 100),
            _ => panic!("Expected ArrowMemory batch"),
        }
    }

    #[test]
    fn test_reset() {
        let store = ArtIndex::new();
        let entry_id: EntryID = EntryID::from(1);
        let array = create_test_array(100);

        store.insert(&entry_id, array.clone());

        let entry_id: EntryID = EntryID::from(1);
        assert!(store.is_cached(&entry_id));

        store.reset();
        let entry_id: EntryID = EntryID::from(1);
        assert!(!store.is_cached(&entry_id));
    }
}
