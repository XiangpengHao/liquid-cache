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
