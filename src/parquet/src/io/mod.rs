use std::{
    collections::VecDeque,
    path::PathBuf,
    sync::{Arc, RwLock},
};

use ahash::AHashMap;
use bytes::Bytes;
use liquid_cache_common::IoMode;
use liquid_cache_storage::cache::{
    CacheEntry, CacheExpression, EntryID, IoContext, LiquidCompressorStates,
};
use liquid_cache_storage::liquid_array::SqueezedBacking;

use crate::cache::{ColumnAccessPath, ParquetArrayID};

#[cfg(target_os = "linux")]
mod io_uring;

mod io_backend;

#[derive(Debug)]
pub(crate) struct ParquetIoContext {
    compressor_states: RwLock<AHashMap<ColumnAccessPath, Arc<LiquidCompressorStates>>>,
    expression_hints: RwLock<AHashMap<ColumnAccessPath, ColumnExpressionTracker>>,
    base_dir: PathBuf,
    io_mode: IoMode,
}

impl ParquetIoContext {
    pub fn new(base_dir: PathBuf, io_mode: IoMode) -> Self {
        if matches!(
            io_mode,
            IoMode::UringDirect | IoMode::Uring | IoMode::UringBlocking
        ) {
            #[cfg(target_os = "linux")]
            {
                crate::io::io_uring::initialize_uring_pool(io_mode);
            }
            #[cfg(not(target_os = "linux"))]
            {
                panic!("io_mode {:?} is only supported on Linux", io_mode);
            }
        }

        Self {
            compressor_states: RwLock::new(AHashMap::new()),
            expression_hints: RwLock::new(AHashMap::new()),
            base_dir,
            io_mode,
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

    fn disk_path(&self, entry: &CacheEntry, entry_id: &EntryID) -> PathBuf {
        let parquet_array_id = ParquetArrayID::from(*entry_id);
        match entry {
            CacheEntry::DiskArrow(_) | CacheEntry::MemoryArrow(_) => {
                parquet_array_id.on_disk_arrow_path(&self.base_dir)
            }
            CacheEntry::DiskLiquid(_) | CacheEntry::MemoryLiquid(_) => {
                parquet_array_id.on_disk_path(&self.base_dir)
            }
            CacheEntry::MemorySqueezedLiquid(array) => match array.disk_backing() {
                SqueezedBacking::Arrow => parquet_array_id.on_disk_arrow_path(&self.base_dir),
                SqueezedBacking::Liquid => parquet_array_id.on_disk_path(&self.base_dir),
            },
        }
    }

    #[inline(never)]
    #[fastrace::trace]
    async fn read(
        &self,
        path: PathBuf,
        range: Option<std::ops::Range<u64>>,
    ) -> Result<Bytes, std::io::Error> {
        io_backend::read(self.io_mode, path, range).await
    }

    #[inline(never)]
    #[fastrace::trace]
    async fn write_file(&self, path: PathBuf, data: Bytes) -> Result<(), std::io::Error> {
        io_backend::write(self.io_mode, path, data).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use liquid_cache_storage::liquid_array::Date32Field;
    use tempfile::tempdir;

    fn entry(file: u64, rg: u64, col: u64) -> EntryID {
        let id = ParquetArrayID::new(file, rg, col, crate::cache::BatchID::from_raw(0));
        EntryID::from(usize::from(id))
    }

    #[test]
    fn squeeze_hint_tracks_majority() {
        let tmp = tempdir().unwrap();
        let ctx = ParquetIoContext::new(tmp.path().to_path_buf(), IoMode::StdBlocking);
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
        let tmp = tempdir().unwrap();
        let ctx = ParquetIoContext::new(tmp.path().to_path_buf(), IoMode::StdBlocking);
        let e = entry(9, 9, 9);
        let year = Arc::new(CacheExpression::extract_date32(Date32Field::Year));
        let day = Arc::new(CacheExpression::extract_date32(Date32Field::Day));

        ctx.add_squeeze_hint(&e, year.clone());
        ctx.add_squeeze_hint(&e, day.clone());

        let majority = ctx.squeeze_hint(&e).expect("hint");
        assert_eq!(majority, day);
    }
}
