use std::{
    collections::VecDeque,
    ops::Range,
    sync::{Arc, RwLock},
};

use ahash::AHashMap;
use bytes::Bytes;
use liquid_cache::cache::{CacheExpression, EntryID, IoContext, LiquidCompressorStates};

use crate::cache::{ColumnAccessPath, ParquetArrayID};

/// Convert an [`EntryID`] to a t4 key (8-byte little-endian representation).
fn entry_id_to_key(entry_id: &EntryID) -> Vec<u8> {
    usize::from(*entry_id).to_le_bytes().to_vec()
}

#[derive(Debug)]
pub(crate) struct ParquetIoContext {
    compressor_states: RwLock<AHashMap<ColumnAccessPath, Arc<LiquidCompressorStates>>>,
    expression_hints: RwLock<AHashMap<ColumnAccessPath, ColumnExpressionTracker>>,
    store: t4::Store,
}

impl ParquetIoContext {
    pub fn new(store: t4::Store) -> Self {
        Self {
            compressor_states: RwLock::new(AHashMap::new()),
            expression_hints: RwLock::new(AHashMap::new()),
            store,
        }
    }
}

const COLUMN_EXPRESSION_HISTORY: usize = 16;

#[derive(Debug, Default, Clone)]
struct ColumnExpressionTracker {
    history: VecDeque<Arc<CacheExpression>>,
}

impl ColumnExpressionTracker {
    fn record(&mut self, expression: Arc<CacheExpression>) {
        if self.history.len() == COLUMN_EXPRESSION_HISTORY {
            self.history.pop_front();
        }
        self.history.push_back(expression);
    }

    fn majority(&self) -> Option<Arc<CacheExpression>> {
        use std::cmp::Ordering;
        let mut counts: AHashMap<Arc<CacheExpression>, (usize, usize)> = AHashMap::new();
        for (idx, expr) in self.history.iter().enumerate() {
            let entry = counts.entry(expr.clone()).or_insert((0, idx));
            entry.0 += 1;
            entry.1 = idx;
        }

        counts
            .into_iter()
            .max_by(|a, b| match a.1.0.cmp(&b.1.0) {
                Ordering::Less => Ordering::Less,
                Ordering::Greater => Ordering::Greater,
                Ordering::Equal => a.1.1.cmp(&b.1.1),
            })
            .map(|(expr, _)| expr)
    }
}

#[async_trait::async_trait]
impl IoContext for ParquetIoContext {
    fn add_squeeze_hint(&self, entry_id: &EntryID, expression: Arc<CacheExpression>) {
        let column_path = ColumnAccessPath::from(ParquetArrayID::from(*entry_id));
        let mut guard = self.expression_hints.write().unwrap();
        let expression_tracker = guard.entry(column_path).or_default();
        expression_tracker.record(expression.clone());
    }

    fn squeeze_hint(&self, entry_id: &EntryID) -> Option<Arc<CacheExpression>> {
        let column_path = ColumnAccessPath::from(ParquetArrayID::from(*entry_id));
        let guard = self.expression_hints.read().unwrap();
        guard
            .get(&column_path)
            .and_then(ColumnExpressionTracker::majority)
    }

    fn get_compressor(&self, entry_id: &EntryID) -> Arc<LiquidCompressorStates> {
        let column_path = ColumnAccessPath::from(ParquetArrayID::from(*entry_id));
        let mut states = self.compressor_states.write().unwrap();
        states
            .entry(column_path)
            .or_insert_with(|| Arc::new(LiquidCompressorStates::new()))
            .clone()
    }

    #[inline(never)]
    #[fastrace::trace]
    async fn read(
        &self,
        entry_id: &EntryID,
        range: Option<Range<u64>>,
    ) -> Result<Bytes, std::io::Error> {
        let key = entry_id_to_key(entry_id);
        match range {
            Some(range) => {
                let len = range.end - range.start;
                let bytes = self
                    .store
                    .get_range(&key, range.start, len)
                    .await
                    .map_err(|e| std::io::Error::other(e.to_string()))?;
                Ok(Bytes::from(bytes))
            }
            None => {
                let bytes = self
                    .store
                    .get(&key)
                    .await
                    .map_err(|e| std::io::Error::other(e.to_string()))?;
                Ok(Bytes::from(bytes))
            }
        }
    }

    #[inline(never)]
    #[fastrace::trace]
    async fn write(&self, entry_id: &EntryID, data: Bytes) -> Result<(), std::io::Error> {
        let key = entry_id_to_key(entry_id);
        self.store
            .put(key, data.to_vec())
            .await
            .map_err(|e| std::io::Error::other(e.to_string()))?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use liquid_cache::liquid_array::Date32Field;

    fn entry(file: u64, rg: u64, col: u64) -> EntryID {
        let id = ParquetArrayID::new(file, rg, col, crate::cache::BatchID::from_raw(0));
        EntryID::from(usize::from(id))
    }

    fn make_ctx() -> ParquetIoContext {
        let tmp = tempfile::tempdir().unwrap();
        let store = pollster::block_on(t4::mount(tmp.path().join("liquid_cache.t4"))).unwrap();
        ParquetIoContext::new(store)
    }

    #[test]
    fn squeeze_hint_tracks_majority() {
        let ctx = make_ctx();
        let e = entry(1, 2, 3);
        let month = Arc::new(CacheExpression::extract_date32(Date32Field::Month));
        let year = Arc::new(CacheExpression::extract_date32(Date32Field::Year));

        ctx.add_squeeze_hint(&e, month.clone());
        ctx.add_squeeze_hint(&e, month.clone());
        ctx.add_squeeze_hint(&e, year.clone());

        let majority = ctx.squeeze_hint(&e).expect("hint");
        assert_eq!(majority, month);
    }

    #[test]
    fn squeeze_hint_prefers_recent_on_tie() {
        let ctx = make_ctx();
        let e = entry(9, 9, 9);
        let year = Arc::new(CacheExpression::extract_date32(Date32Field::Year));
        let day = Arc::new(CacheExpression::extract_date32(Date32Field::Day));

        ctx.add_squeeze_hint(&e, year.clone());
        ctx.add_squeeze_hint(&e, day.clone());

        let majority = ctx.squeeze_hint(&e).expect("hint");
        assert_eq!(majority, day);
    }
}
