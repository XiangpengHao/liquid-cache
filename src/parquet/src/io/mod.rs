use std::{
    path::{Path, PathBuf},
    sync::{Arc, RwLock},
};

use ahash::AHashMap;
use bytes::Bytes;
use liquid_cache_common::IoMode;
use liquid_cache_storage::cache::{EntryID, IoContext, LiquidCompressorStates};

use crate::cache::{ColumnAccessPath, ParquetArrayID};

#[cfg(target_os = "linux")]
mod io_uring;

mod io_backend;

#[derive(Debug)]
pub(crate) struct ParquetIoContext {
    compressor_states: RwLock<AHashMap<ColumnAccessPath, Arc<LiquidCompressorStates>>>,
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
            base_dir,
            io_mode,
        }
    }
}

#[async_trait::async_trait]
impl IoContext for ParquetIoContext {
    fn base_dir(&self) -> &Path {
        &self.base_dir
    }

    fn get_compressor(&self, entry_id: &EntryID) -> Arc<LiquidCompressorStates> {
        let column_path = ColumnAccessPath::from(ParquetArrayID::from(*entry_id));
        let mut states = self.compressor_states.write().unwrap();
        states
            .entry(column_path)
            .or_insert_with(|| Arc::new(LiquidCompressorStates::new()))
            .clone()
    }

    fn arrow_path(&self, entry_id: &EntryID) -> PathBuf {
        let parquet_array_id = ParquetArrayID::from(*entry_id);
        parquet_array_id.on_disk_arrow_path(self.base_dir())
    }

    fn liquid_path(&self, entry_id: &EntryID) -> PathBuf {
        let parquet_array_id = ParquetArrayID::from(*entry_id);
        parquet_array_id.on_disk_path(self.base_dir())
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
