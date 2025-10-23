use std::{
    path::{Path, PathBuf},
    sync::Arc,
};

use ahash::AHashMap;
use bytes::Bytes;
use liquid_cache_storage::cache::{EntryID, IoContext, LiquidCompressorStates};

use crate::{
    cache::id::{BatchID, ParquetArrayID},
    sync::RwLock,
};

#[derive(Debug)]
pub(crate) struct ParquetIoContext {
    compressor_states: RwLock<AHashMap<ColumnAccessPath, Arc<LiquidCompressorStates>>>,
    base_dir: PathBuf,
}

impl ParquetIoContext {
    pub fn new(base_dir: PathBuf) -> Self {
        Self {
            compressor_states: RwLock::new(AHashMap::new()),
            base_dir,
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
        #[cfg(target_os = "linux")]
        {
            read_range_from_uring(path, None).await
        }
        #[cfg(not(target_os = "linux"))]
        {
            maybe_spawn_blocking(move || {
                use std::io::Read;
                let mut file = std::fs::File::open(path)?;
                let mut bytes = Vec::new();
                file.read_to_end(&mut bytes)?;
                Ok(Bytes::from(bytes))
            })
            .await
        }
    }

    async fn read_range(
        &self,
        path: PathBuf,
        range: std::ops::Range<u64>,
    ) -> Result<Bytes, std::io::Error> {
        #[cfg(target_os = "linux")]
        {
            read_range_from_uring(path, Some(range)).await
        }
        #[cfg(not(target_os = "linux"))]
        {
            maybe_spawn_blocking(move || {
                use std::io::{Read, Seek};
                let mut file = std::fs::File::open(path)?;
                let len = (range.end - range.start) as usize;
                let mut bytes = vec![0u8; len];
                file.seek(std::io::SeekFrom::Start(range.start))?;
                file.read_exact(&mut bytes)?;
                Ok(Bytes::from(bytes))
            })
            .await
        }
    }

    async fn write_file(&self, path: PathBuf, data: Bytes) -> Result<(), std::io::Error> {
        #[cfg(target_os = "linux")]
        {
            write_to_uring(path, &data).await
        }
        #[cfg(not(target_os = "linux"))]
        {
            maybe_spawn_blocking(move || {
                use std::io::Write;
                let mut file = std::fs::File::create(path)?;
                file.write_all(&data)?;
                Ok(())
            })
            .await
        }
    }
}

#[cfg(target_os = "linux")]
async fn read_range_from_uring(path: PathBuf, mut range: Option<std::ops::Range::<u64>>) -> Result<Bytes, std::io::Error> {
    use std::os::fd::AsRawFd;
    use std::{fs::OpenOptions, os::unix::fs::OpenOptionsExt as _};

    use liquid_cache_storage::cache::io_backend::{IoMode, get_io_mode, FileReadTask, UringFuture};

    let flags = if get_io_mode() == IoMode::Direct {
        libc::O_DIRECT
    } else {
        0
    };
    let file = OpenOptions::new()
        .read(true)
        .custom_flags(flags)
        .open(path)
        .expect("failed to open file");

    if range.is_none() {
        range = Some(std::ops::Range::<u64> {
            start: 0,
            end: file.metadata()?.len(),
        });
    }

    let task = Arc::new(FileReadTask::new(range.unwrap(), file.as_raw_fd()));
    // UringFuture will be responsible for submitting and driving the future to completion
    let uring_fut = UringFuture::new(task.clone());
    uring_fut.await;
    Ok(task.get_bytes())
}

#[cfg(target_os = "linux")]
async fn write_to_uring(path: PathBuf, data: &Bytes) -> Result<(), std::io::Error> {
    use liquid_cache_storage::cache::io_backend::{FileWriteTask, IoMode, get_io_mode, UringFuture};
    use std::os::fd::AsRawFd;
    use std::{fs::OpenOptions, os::unix::fs::OpenOptionsExt as _};

    let flags = if get_io_mode() == IoMode::Direct {
        libc::O_DIRECT
    } else {
        0
    };
    let file = OpenOptions::new()
        .create(true)
        .write(true)
        .custom_flags(flags)
        .open(path)
        .expect("failed to create file");

    let task = Arc::new(FileWriteTask::new(
        data.as_ptr(),
        data.len(),
        file.as_raw_fd(),
    ));
    // UringFuture will be responsible for submitting and driving the future to completion
    let uring_fut = UringFuture::new(task);
    uring_fut.await;
    Ok(())
}

#[allow(dead_code)]
pub(crate) async fn maybe_spawn_blocking<F, T>(f: F) -> Result<T, std::io::Error>
where
    F: FnOnce() -> Result<T, std::io::Error> + Send + 'static,
    T: Send + 'static,
{
    match tokio::runtime::Handle::try_current() {
        Ok(runtime) => runtime.spawn_blocking(f).await?,
        Err(_) => f(),
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
