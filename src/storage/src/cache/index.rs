use congee::CongeeArc;
use std::{
    fmt::{Debug, Formatter},
    sync::Arc,
};

use crate::cache::{CacheEntryID, CachedBatch};

pub(crate) struct ArtIndex {
    art: CongeeArc<CacheEntryID, CachedBatch>,
}

impl Debug for ArtIndex {
    fn fmt(&self, _f: &mut Formatter<'_>) -> std::fmt::Result {
        Ok(())
    }
}

impl ArtIndex {
    pub(crate) fn new() -> Self {
        let art: CongeeArc<CacheEntryID, CachedBatch> = CongeeArc::new();
        Self { art }
    }

    pub(crate) fn get(&self, entry_id: &CacheEntryID) -> Option<CachedBatch> {
        let guard = self.art.pin();
        let batch = self.art.get(*entry_id, &guard)?;
        Some(CachedBatch::clone(&batch))
    }

    pub(crate) fn is_cached(&self, entry_id: &CacheEntryID) -> bool {
        let guard = self.art.pin();
        self.art.get(*entry_id, &guard).is_some()
    }

    pub(crate) fn insert(&self, entry_id: &CacheEntryID, batch: CachedBatch) {
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

    pub(crate) fn for_each(&self, mut f: impl FnMut(&CacheEntryID, &CachedBatch)) {
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
    pub(crate) fn keys(&self) -> Vec<CacheEntryID> {
        self.art.keys()
    }
}
