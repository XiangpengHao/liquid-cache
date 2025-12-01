use std::{fmt::Debug, ops::Range, path::PathBuf};

use bytes::Bytes;

use crate::cache::{
    cached_batch::CacheEntry,
    utils::{EntryID, LiquidCompressorStates},
};
use crate::liquid_array::HybridBacking;
use crate::sync::Arc;

/// A trait for objects that can handle IO operations for the cache.
#[async_trait::async_trait]
pub trait IoContext: Debug + Send + Sync {
    /// Get the compressor for an entry.
    fn get_compressor(&self, entry_id: &EntryID) -> Arc<LiquidCompressorStates>;

    /// Get the disk path for a cache entry.
    ///
    /// The implementation should determine the appropriate path based on the entry type
    /// (e.g., different extensions for Arrow vs Liquid formats).
    fn disk_path(&self, entry: &CacheEntry, entry_id: &EntryID) -> PathBuf;

    /// Read bytes from the file at the given path, optionally restricted to the provided range.
    async fn read(&self, path: PathBuf, range: Option<Range<u64>>)
    -> Result<Bytes, std::io::Error>;

    /// Write the entire buffer to a file at the given path.
    async fn write_file(&self, path: PathBuf, data: Bytes) -> Result<(), std::io::Error>;
}

/// A default implementation of [IoContext] that uses the default compressor.
#[derive(Debug)]
pub struct DefaultIoContext {
    compressor_state: Arc<LiquidCompressorStates>,
    base_dir: PathBuf,
}

impl DefaultIoContext {
    /// Create a new instance of [DefaultIoContext].
    pub fn new(base_dir: PathBuf) -> Self {
        Self {
            compressor_state: Arc::new(LiquidCompressorStates::new()),
            base_dir,
        }
    }
}

#[async_trait::async_trait]
impl IoContext for DefaultIoContext {
    fn get_compressor(&self, _entry_id: &EntryID) -> Arc<LiquidCompressorStates> {
        self.compressor_state.clone()
    }

    fn disk_path(&self, entry: &CacheEntry, entry_id: &EntryID) -> PathBuf {
        let ext = match entry {
            CacheEntry::DiskArrow(_) | CacheEntry::MemoryArrow(_) => "arrow",
            CacheEntry::DiskLiquid(_) | CacheEntry::MemoryLiquid(_) => "liquid",
            CacheEntry::MemoryHybridLiquid(array) => match array.disk_backing() {
                HybridBacking::Arrow => "arrow",
                HybridBacking::Liquid => "liquid",
            },
        };
        self.base_dir
            .join(format!("{:016x}.{}", usize::from(*entry_id), ext))
    }

    async fn read(
        &self,
        path: PathBuf,
        range: Option<Range<u64>>,
    ) -> Result<Bytes, std::io::Error> {
        use tokio::io::AsyncReadExt;
        use tokio::io::AsyncSeekExt;
        let mut file = tokio::fs::File::open(path).await?;

        match range {
            Some(range) => {
                let len = (range.end - range.start) as usize;
                let mut bytes = vec![0u8; len];
                file.seek(tokio::io::SeekFrom::Start(range.start)).await?;
                file.read_exact(&mut bytes).await?;
                Ok(Bytes::from(bytes))
            }
            None => {
                let mut bytes = Vec::new();
                file.read_to_end(&mut bytes).await?;
                Ok(Bytes::from(bytes))
            }
        }
    }

    async fn write_file(&self, path: PathBuf, data: Bytes) -> Result<(), std::io::Error> {
        use tokio::io::AsyncWriteExt;
        let mut file = tokio::fs::File::create(path).await?;
        file.write_all(&data).await?;
        Ok(())
    }
}

/// A blocking implementation of [IoContext] that uses the default compressor.
/// This is used for testing purposes as all io operations are blocking.
#[derive(Debug)]
pub struct BlockingIoContext {
    compressor_state: Arc<LiquidCompressorStates>,
    base_dir: PathBuf,
}

impl BlockingIoContext {
    /// Create a new instance of [BlockingIoContext].
    pub fn new(base_dir: PathBuf) -> Self {
        Self {
            compressor_state: Arc::new(LiquidCompressorStates::new()),
            base_dir,
        }
    }
}

#[async_trait::async_trait]
impl IoContext for BlockingIoContext {
    fn get_compressor(&self, _entry_id: &EntryID) -> Arc<LiquidCompressorStates> {
        self.compressor_state.clone()
    }

    fn disk_path(&self, entry: &CacheEntry, entry_id: &EntryID) -> PathBuf {
        let ext = match entry {
            CacheEntry::DiskArrow(_) | CacheEntry::MemoryArrow(_) => "arrow",
            CacheEntry::DiskLiquid(_) | CacheEntry::MemoryLiquid(_) => "liquid",
            CacheEntry::MemoryHybridLiquid(array) => match array.disk_backing() {
                HybridBacking::Arrow => "arrow",
                HybridBacking::Liquid => "liquid",
            },
        };
        self.base_dir
            .join(format!("{:016x}.{}", usize::from(*entry_id), ext))
    }

    async fn read(
        &self,
        path: PathBuf,
        range: Option<Range<u64>>,
    ) -> Result<Bytes, std::io::Error> {
        let mut file = std::fs::File::open(path)?;
        match range {
            Some(range) => {
                let len = (range.end - range.start) as usize;
                let mut bytes = vec![0u8; len];
                std::io::Seek::seek(&mut file, std::io::SeekFrom::Start(range.start))?;
                std::io::Read::read_exact(&mut file, &mut bytes)?;
                Ok(Bytes::from(bytes))
            }
            None => {
                let mut bytes = Vec::new();
                std::io::Read::read_to_end(&mut file, &mut bytes)?;
                Ok(Bytes::from(bytes))
            }
        }
    }

    async fn write_file(&self, path: PathBuf, data: Bytes) -> Result<(), std::io::Error> {
        let mut file = std::fs::File::create(path)?;
        std::io::Write::write_all(&mut file, &data)?;
        Ok(())
    }
}
