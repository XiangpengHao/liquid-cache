use std::{
    path::{Path, PathBuf},
    sync::Arc,
};

use ahash::AHashMap;
use bytes::Bytes;
use liquid_cache_common::IoMode;
use liquid_cache_storage::cache::{EntryID, IoContext, LiquidCompressorStates};

use crate::{
    cache::{ColumnAccessPath, ParquetArrayID},
    sync::RwLock,
};

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

mod io_backend {
    use super::*;
    use std::{
        io::{Read, Seek, Write},
        ops::Range,
        path::PathBuf,
    };
    use tokio::io::{AsyncReadExt, AsyncSeekExt, AsyncWriteExt};

    pub(super) async fn read_file(io_mode: IoMode, path: PathBuf) -> Result<Bytes, std::io::Error> {
        match io_mode {
            IoMode::Uring | IoMode::UringDirect => read_file_uring(path).await,
            IoMode::UringBlocking => read_file_blocking_uring(path).await,
            IoMode::StdSpawnBlocking => read_file_spawn_blocking(path).await,
            IoMode::StdBlocking => read_file_blocking(path),
            IoMode::TokioIO => read_file_tokio(path).await,
        }
    }

    pub(super) async fn read_range(
        io_mode: IoMode,
        path: PathBuf,
        range: Range<u64>,
    ) -> Result<Bytes, std::io::Error> {
        match io_mode {
            IoMode::Uring | IoMode::UringDirect => read_range_uring(path, range).await,
            IoMode::UringBlocking => read_range_blocking_uring(path, range).await,
            IoMode::StdSpawnBlocking => read_range_spawn_blocking(path, range).await,
            IoMode::StdBlocking => read_range_blocking(path, range),
            IoMode::TokioIO => read_range_tokio(path, range).await,
        }
    }

    pub(super) async fn write_file(
        io_mode: IoMode,
        path: PathBuf,
        data: Bytes,
    ) -> Result<(), std::io::Error> {
        match io_mode {
            IoMode::Uring | IoMode::UringDirect => write_file_uring(path, data).await,
            IoMode::UringBlocking => write_file_blocking_uring(path, data).await,
            IoMode::StdSpawnBlocking => write_file_spawn_blocking(path, data).await,
            IoMode::StdBlocking => write_file_blocking(path, data),
            IoMode::TokioIO => write_file_tokio(path, data).await,
        }
    }

    fn read_file_blocking_impl(path: PathBuf) -> Result<Bytes, std::io::Error> {
        let bytes = std::fs::read(path)?;
        Ok(Bytes::from(bytes))
    }

    fn read_range_blocking_impl(path: PathBuf, range: Range<u64>) -> Result<Bytes, std::io::Error> {
        let mut file = std::fs::File::open(path)?;
        let len = (range.end - range.start) as usize;
        let mut bytes = vec![0u8; len];
        file.seek(std::io::SeekFrom::Start(range.start))?;
        file.read_exact(&mut bytes)?;
        Ok(Bytes::from(bytes))
    }

    fn write_file_blocking_impl(path: PathBuf, data: Bytes) -> Result<(), std::io::Error> {
        let mut file = std::fs::File::create(path)?;
        file.write_all(data.as_ref())?;
        Ok(())
    }

    fn read_file_blocking(path: PathBuf) -> Result<Bytes, std::io::Error> {
        read_file_blocking_impl(path)
    }

    fn read_range_blocking(path: PathBuf, range: Range<u64>) -> Result<Bytes, std::io::Error> {
        read_range_blocking_impl(path, range)
    }

    fn write_file_blocking(path: PathBuf, data: Bytes) -> Result<(), std::io::Error> {
        write_file_blocking_impl(path, data)
    }

    async fn read_file_spawn_blocking(path: PathBuf) -> Result<Bytes, std::io::Error> {
        maybe_spawn_blocking(move || read_file_blocking_impl(path)).await
    }

    async fn read_range_spawn_blocking(
        path: PathBuf,
        range: Range<u64>,
    ) -> Result<Bytes, std::io::Error> {
        maybe_spawn_blocking(move || read_range_blocking_impl(path, range)).await
    }

    async fn write_file_spawn_blocking(path: PathBuf, data: Bytes) -> Result<(), std::io::Error> {
        maybe_spawn_blocking(move || write_file_blocking_impl(path, data)).await
    }

    async fn read_file_tokio(path: PathBuf) -> Result<Bytes, std::io::Error> {
        let mut file = tokio::fs::File::open(path).await?;
        let mut bytes = Vec::new();
        file.read_to_end(&mut bytes).await?;
        Ok(Bytes::from(bytes))
    }

    async fn read_range_tokio(path: PathBuf, range: Range<u64>) -> Result<Bytes, std::io::Error> {
        let mut file = tokio::fs::File::open(path).await?;
        let len = (range.end - range.start) as usize;
        let mut bytes = vec![0u8; len];
        file.seek(tokio::io::SeekFrom::Start(range.start)).await?;
        file.read_exact(&mut bytes).await?;
        Ok(Bytes::from(bytes))
    }

    async fn write_file_tokio(path: PathBuf, data: Bytes) -> Result<(), std::io::Error> {
        let mut file = tokio::fs::File::create(path).await?;
        file.write_all(data.as_ref()).await?;
        Ok(())
    }

    #[cfg(target_os = "linux")]
    async fn read_file_uring(path: PathBuf) -> Result<Bytes, std::io::Error> {
        crate::io::io_uring::read_range_from_uring(path, None).await
    }

    #[cfg(not(target_os = "linux"))]
    async fn read_file_uring(_path: PathBuf) -> Result<Bytes, std::io::Error> {
        panic!("io_uring modes are only supported on Linux");
    }

    #[cfg(target_os = "linux")]
    async fn read_range_uring(path: PathBuf, range: Range<u64>) -> Result<Bytes, std::io::Error> {
        crate::io::io_uring::read_range_from_uring(path, Some(range)).await
    }

    #[cfg(not(target_os = "linux"))]
    async fn read_range_uring(_path: PathBuf, _range: Range<u64>) -> Result<Bytes, std::io::Error> {
        panic!("io_uring modes are only supported on Linux");
    }

    #[cfg(target_os = "linux")]
    async fn read_file_blocking_uring(path: PathBuf) -> Result<Bytes, std::io::Error> {
        crate::io::io_uring::read_range_from_blocking_uring(path, None)
    }

    #[cfg(not(target_os = "linux"))]
    async fn read_file_blocking_uring(_path: PathBuf) -> Result<Bytes, std::io::Error> {
        panic!("io_uring modes are only supported on Linux");
    }

    #[cfg(target_os = "linux")]
    async fn read_range_blocking_uring(
        path: PathBuf,
        range: Range<u64>,
    ) -> Result<Bytes, std::io::Error> {
        crate::io::io_uring::read_range_from_blocking_uring(path, Some(range))
    }

    #[cfg(not(target_os = "linux"))]
    async fn read_range_blocking_uring(
        _path: PathBuf,
        _range: Range<u64>,
    ) -> Result<Bytes, std::io::Error> {
        panic!("io_uring modes are only supported on Linux");
    }

    #[cfg(target_os = "linux")]
    async fn write_file_uring(path: PathBuf, data: Bytes) -> Result<(), std::io::Error> {
        crate::io::io_uring::write_to_uring(path, &data).await
    }

    #[cfg(not(target_os = "linux"))]
    async fn write_file_uring(_path: PathBuf, _data: Bytes) -> Result<(), std::io::Error> {
        panic!("io_uring modes are only supported on Linux");
    }

    #[cfg(target_os = "linux")]
    async fn write_file_blocking_uring(path: PathBuf, data: Bytes) -> Result<(), std::io::Error> {
        crate::io::io_uring::write_to_blocking_uring(path, &data)
    }

    #[cfg(not(target_os = "linux"))]
    async fn write_file_blocking_uring(_path: PathBuf, _data: Bytes) -> Result<(), std::io::Error> {
        panic!("io_uring modes are only supported on Linux");
    }

    async fn maybe_spawn_blocking<F, T>(f: F) -> Result<T, std::io::Error>
    where
        F: FnOnce() -> Result<T, std::io::Error> + Send + 'static,
        T: Send + 'static,
    {
        match tokio::runtime::Handle::try_current() {
            Ok(runtime) => match runtime.spawn_blocking(f).await {
                Ok(result) => result,
                Err(err) => Err(std::io::Error::other(err)),
            },
            Err(_) => f(),
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
    async fn read_file(&self, path: PathBuf) -> Result<Bytes, std::io::Error> {
        io_backend::read_file(self.io_mode, path).await
    }

    #[inline(never)]
    #[fastrace::trace]
    async fn read_range(
        &self,
        path: PathBuf,
        range: std::ops::Range<u64>,
    ) -> Result<Bytes, std::io::Error> {
        io_backend::read_range(self.io_mode, path, range).await
    }

    #[inline(never)]
    #[fastrace::trace]
    async fn write_file(&self, path: PathBuf, data: Bytes) -> Result<(), std::io::Error> {
        io_backend::write_file(self.io_mode, path, data).await
    }
}
