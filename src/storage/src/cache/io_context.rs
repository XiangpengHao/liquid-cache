use std::path::Path;
use std::{fmt::Debug, ops::Range, path::PathBuf};

use ahash::AHashMap;
use bytes::Bytes;

use crate::liquid_array::SqueezedBacking;
use crate::sync::{Arc, RwLock};
use crate::{
    cache::{
        CacheExpression,
        cached_batch::CacheEntry,
        utils::{EntryID, LiquidCompressorStates},
    },
    liquid_array::SqueezeIoHandler,
};

/// A trait for objects that can handle IO operations for the cache.
#[async_trait::async_trait]
pub trait IoContext: Debug + Send + Sync {
    /// Add a squeeze hint for an entry.
    fn add_squeeze_hint(&self, _entry_id: &EntryID, _expression: Arc<CacheExpression>) {
        // Do nothing by default
    }

    /// Get the squeeze hint for an entry.
    /// If None, the entry will be evicted to disk entirely.
    /// If Some, the entry will be squeezed according to the cache expressions previously recorded for this column.
    /// For example, if expression is ExtractDate32 { field: Date32Field::Year },
    /// the entry will be squeezed to a [crate::liquid_array::SqueezedDate32Array] with the year component.
    fn squeeze_hint(&self, _entry_id: &EntryID) -> Option<Arc<CacheExpression>> {
        None
    }

    /// Get the compressor for an entry.
    fn get_compressor(&self, entry_id: &EntryID) -> Arc<LiquidCompressorStates>;

    /// Get the disk path for a cache entry.
    ///
    /// The implementation should determine the appropriate path based on the entry type
    /// (e.g., different extensions for Arrow vs Liquid formats).
    fn disk_path(&self, entry: &CacheEntry, entry_id: &EntryID) -> PathBuf;

    /// Read bytes from the file at the given path, optionally restricted to the provided range.
    async fn read<'p>(
        &self,
        path: &'p Path,
        range: Option<Range<u64>>,
    ) -> Result<Bytes, std::io::Error>;

    /// Write the entire buffer to a file at the given path.
    async fn write_file<'p>(&self, path: &'p Path, data: Bytes) -> Result<(), std::io::Error>;
}

/// A default implementation of [IoContext] that uses the default compressor.
/// It uses tokio's async IO by default.
#[derive(Debug)]
pub struct DefaultIoContext {
    compressor_state: Arc<LiquidCompressorStates>,
    squeeze_hints: RwLock<AHashMap<EntryID, Arc<CacheExpression>>>,
    base_dir: PathBuf,
}

impl DefaultIoContext {
    /// Create a new instance of [DefaultIoContext].
    pub fn new(base_dir: PathBuf) -> Self {
        Self {
            compressor_state: Arc::new(LiquidCompressorStates::new()),
            base_dir,
            squeeze_hints: RwLock::new(AHashMap::new()),
        }
    }
}

#[async_trait::async_trait]
impl IoContext for DefaultIoContext {
    fn add_squeeze_hint(&self, entry_id: &EntryID, expression: Arc<CacheExpression>) {
        let mut guard = self.squeeze_hints.write().unwrap();
        guard.insert(*entry_id, expression);
    }

    fn squeeze_hint(&self, entry_id: &EntryID) -> Option<Arc<CacheExpression>> {
        let guard = self.squeeze_hints.read().unwrap();
        guard.get(entry_id).cloned()
    }

    fn get_compressor(&self, _entry_id: &EntryID) -> Arc<LiquidCompressorStates> {
        self.compressor_state.clone()
    }

    fn disk_path(&self, entry: &CacheEntry, entry_id: &EntryID) -> PathBuf {
        let ext = match entry {
            CacheEntry::DiskArrow(_) | CacheEntry::MemoryArrow(_) => "arrow",
            CacheEntry::DiskLiquid(_) | CacheEntry::MemoryLiquid(_) => "liquid",
            CacheEntry::MemorySqueezedLiquid(array) => match array.disk_backing() {
                SqueezedBacking::Arrow => "arrow",
                SqueezedBacking::Liquid => "liquid",
            },
        };
        self.base_dir
            .join(format!("{:016x}.{}", usize::from(*entry_id), ext))
    }

    async fn read<'p>(
        &self,
        path: &'p Path,
        range: Option<Range<u64>>,
    ) -> Result<Bytes, std::io::Error> {
        if cfg!(test) {
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
        } else {
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
    }

    async fn write_file<'p>(&self, path: &'p Path, data: Bytes) -> Result<(), std::io::Error> {
        if cfg!(test) {
            std::fs::write(path, data.as_ref())?;
            Ok(())
        } else {
            use tokio::io::AsyncWriteExt;
            let mut file = tokio::fs::File::create(path).await?;
            file.write_all(&data).await?;
            file.sync_all().await?;
            Ok(())
        }
    }
}

/// A default implementation of [SqueezeIo] that uses the default [IoContext].
pub struct DefaultSqueezeIo {
    path: PathBuf,
    io_context: Arc<dyn IoContext>,
}

impl DefaultSqueezeIo {
    /// Create a new instance of [DefaultSqueezeIo].
    pub fn new(path: PathBuf, io_context: Arc<dyn IoContext>) -> Self {
        Self { path, io_context }
    }
}

#[async_trait::async_trait]
impl SqueezeIoHandler for DefaultSqueezeIo {
    async fn read(&self, range: Option<Range<u64>>) -> std::io::Result<Bytes> {
        self.io_context.read(&self.path, range).await
    }
}

#[cfg(test)]
pub(crate) struct TestingSqueezeIo;

#[cfg(test)]
#[async_trait::async_trait]
impl SqueezeIoHandler for TestingSqueezeIo {
    async fn read(&self, _range: Option<Range<u64>>) -> std::io::Result<Bytes> {
        unreachable!()
    }
}
