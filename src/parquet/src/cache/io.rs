use std::{
    path::{Path, PathBuf},
    sync::Arc,
};

use ahash::AHashMap;
use bytes::Bytes;
use liquid_cache_common::IoMode;
use liquid_cache_storage::cache::{EntryID, IoContext, LiquidCompressorStates};

use crate::{
    cache::id::{BatchID, ParquetArrayID},
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
        if matches!(io_mode, IoMode::UringDirectIO | IoMode::Uring) {
            #[cfg(target_os = "linux")]
            {
                crate::cache::io_uring::initialize_uring_pool(io_mode);
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
            IoMode::Uring | IoMode::UringDirectIO => read_file_uring(path).await,
            IoMode::StdSpawnBlockingIO => read_file_spawn_blocking(path).await,
            IoMode::StdBlockingIO => read_file_blocking(path),
            IoMode::TokioIO => read_file_tokio(path).await,
        }
    }

    pub(super) async fn read_range(
        io_mode: IoMode,
        path: PathBuf,
        range: Range<u64>,
    ) -> Result<Bytes, std::io::Error> {
        match io_mode {
            IoMode::Uring | IoMode::UringDirectIO => read_range_uring(path, range).await,
            IoMode::StdSpawnBlockingIO => read_range_spawn_blocking(path, range).await,
            IoMode::StdBlockingIO => read_range_blocking(path, range),
            IoMode::TokioIO => read_range_tokio(path, range).await,
        }
    }

    pub(super) async fn write_file(
        io_mode: IoMode,
        path: PathBuf,
        data: Bytes,
    ) -> Result<(), std::io::Error> {
        match io_mode {
            IoMode::Uring | IoMode::UringDirectIO => write_file_uring(path, data).await,
            IoMode::StdSpawnBlockingIO => write_file_spawn_blocking(path, data).await,
            IoMode::StdBlockingIO => write_file_blocking(path, data),
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
        crate::cache::io_uring::read_range_from_uring(path, None).await
    }

    #[cfg(not(target_os = "linux"))]
    async fn read_file_uring(_path: PathBuf) -> Result<Bytes, std::io::Error> {
        panic!("io_uring modes are only supported on Linux");
    }

    #[cfg(target_os = "linux")]
    async fn read_range_uring(path: PathBuf, range: Range<u64>) -> Result<Bytes, std::io::Error> {
        crate::cache::io_uring::read_range_from_uring(path, Some(range)).await
    }

    #[cfg(not(target_os = "linux"))]
    async fn read_range_uring(_path: PathBuf, _range: Range<u64>) -> Result<Bytes, std::io::Error> {
        panic!("io_uring modes are only supported on Linux");
    }

    #[cfg(target_os = "linux")]
    async fn write_file_uring(path: PathBuf, data: Bytes) -> Result<(), std::io::Error> {
        crate::cache::io_uring::write_to_uring(path, &data).await
    }

    #[cfg(not(target_os = "linux"))]
    async fn write_file_uring(_path: PathBuf, _data: Bytes) -> Result<(), std::io::Error> {
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

    async fn read_file(&self, path: PathBuf) -> Result<Bytes, std::io::Error> {
        io_backend::read_file(self.io_mode, path).await
    }

    async fn read_range(
        &self,
        path: PathBuf,
        range: std::ops::Range<u64>,
    ) -> Result<Bytes, std::io::Error> {
        io_backend::read_range(self.io_mode, path, range).await
    }

    async fn write_file(&self, path: PathBuf, data: Bytes) -> Result<(), std::io::Error> {
        io_backend::write_file(self.io_mode, path, data).await
    }
}

/// Column access path.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Ord, PartialOrd)]
pub struct ColumnAccessPath {
    file_id: u16,
    rg_id: u16,
    col_id: u16,
}

impl ColumnAccessPath {
    /// Create a new instance of ColumnAccessPath.
    pub fn new(file_id: u64, row_group_id: u64, column_id: u64) -> Self {
        debug_assert!(file_id <= u16::MAX as u64);
        debug_assert!(row_group_id <= u16::MAX as u64);
        debug_assert!(column_id <= u16::MAX as u64);
        Self {
            file_id: file_id as u16,
            rg_id: row_group_id as u16,
            col_id: column_id as u16,
        }
    }

    /// Initialize the directory for the column access path.
    pub fn initialize_dir(&self, cache_root_dir: &Path) {
        let path = cache_root_dir
            .join(format!("file_{}", self.file_id_inner()))
            .join(format!("rg_{}", self.row_group_id_inner()))
            .join(format!("col_{}", self.column_id_inner()));
        std::fs::create_dir_all(&path).expect("Failed to create cache directory");
    }

    /// Get the file id.
    fn file_id_inner(&self) -> u64 {
        self.file_id as u64
    }

    /// Get the row group id.
    fn row_group_id_inner(&self) -> u64 {
        self.rg_id as u64
    }

    /// Get the column id.
    fn column_id_inner(&self) -> u64 {
        self.col_id as u64
    }

    /// Get the entry id.
    pub fn entry_id(&self, batch_id: BatchID) -> ParquetArrayID {
        ParquetArrayID::new(
            self.file_id_inner(),
            self.row_group_id_inner(),
            self.column_id_inner(),
            batch_id,
        )
    }
}

impl From<ParquetArrayID> for ColumnAccessPath {
    fn from(value: ParquetArrayID) -> Self {
        Self {
            file_id: value.file_id_inner() as u16,
            rg_id: value.row_group_id_inner() as u16,
            col_id: value.column_id_inner() as u16,
        }
    }
}

#[cfg(test)]
mod tests {
    use tempfile::tempdir;

    use super::*;

    #[test]
    fn test_column_path_from_cache_entry_id() {
        let entry_id = ParquetArrayID::new(1, 2, 3, BatchID::from_raw(4));
        let column_path: ColumnAccessPath = entry_id.into();

        assert_eq!(column_path.file_id, 1);
        assert_eq!(column_path.rg_id, 2);
        assert_eq!(column_path.col_id, 3);
    }

    #[test]
    fn test_column_path_directory_hosts_cache_entry_path() {
        let temp_dir = tempdir().unwrap();
        let cache_root = temp_dir.path();

        // Create a column path
        let file_id = 5u64;
        let row_group_id = 6u64;
        let column_id = 7u64;
        let column_path = ColumnAccessPath::new(file_id, row_group_id, column_id);

        // Initialize the directory
        column_path.initialize_dir(cache_root);

        // Create a cache entry ID from the column path
        let batch_id = BatchID::from_raw(8);
        let entry_id = column_path.entry_id(batch_id);

        // Get the on-disk path
        let entry_path = entry_id.on_disk_path(cache_root);

        // Verify the parent directory of the entry path exists
        assert!(entry_path.parent().unwrap().exists());

        // Verify the directory structure matches
        let expected_dir = cache_root
            .join(format!("file_{file_id}"))
            .join(format!("rg_{row_group_id}"))
            .join(format!("col_{column_id}"));

        assert_eq!(entry_path.parent().unwrap(), &expected_dir);

        // Verify we can create a file at the entry path
        std::fs::write(&entry_path, b"test data").unwrap();
        assert!(entry_path.exists());
    }
}
