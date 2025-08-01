use crate::liquid_array::LiquidArrayRef;
use crate::sync::{Arc, RwLock};
use arrow::array::ArrayRef;
use liquid_cache_common::LiquidCacheMode;
use std::fmt::Display;
use std::fs::File;
use std::io::Write;
use std::ops::Deref;
use std::path::{Path, PathBuf};

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
    pub fn entry_id(&self, batch_id: BatchID) -> CacheEntryID {
        CacheEntryID::new(
            self.file_id_inner(),
            self.row_group_id_inner(),
            self.column_id_inner(),
            batch_id,
        )
    }
}

impl From<CacheEntryID> for ColumnAccessPath {
    fn from(value: CacheEntryID) -> Self {
        Self {
            file_id: value.file_id_inner() as u16,
            rg_id: value.row_group_id_inner() as u16,
            col_id: value.column_id_inner() as u16,
        }
    }
}

#[derive(Debug)]
pub struct CacheConfig {
    batch_size: usize,
    max_cache_bytes: usize,
    cache_root_dir: PathBuf,
    cache_mode: LiquidCacheMode,
}

impl CacheConfig {
    pub(super) fn new(
        batch_size: usize,
        max_cache_bytes: usize,
        cache_root_dir: PathBuf,
        cache_mode: LiquidCacheMode,
    ) -> Self {
        Self {
            batch_size,
            max_cache_bytes,
            cache_root_dir,
            cache_mode,
        }
    }

    pub fn batch_size(&self) -> usize {
        self.batch_size
    }

    pub fn max_cache_bytes(&self) -> usize {
        self.max_cache_bytes
    }

    pub fn cache_root_dir(&self) -> &PathBuf {
        &self.cache_root_dir
    }

    pub fn cache_mode(&self) -> &LiquidCacheMode {
        &self.cache_mode
    }
}

// Helper methods
#[cfg(test)]
pub(crate) fn create_test_array(size: usize) -> CachedBatch {
    use arrow::array::Int64Array;
    use std::sync::Arc;

    CachedBatch::MemoryArrow(Arc::new(Int64Array::from_iter_values(0..size as i64)))
}

// Helper methods
#[cfg(test)]
pub(crate) fn create_test_arrow_array(size: usize) -> ArrayRef {
    use arrow::array::Int64Array;
    Arc::new(Int64Array::from_iter_values(0..size as i64))
}

#[cfg(test)]
pub(crate) fn create_cache_store(
    max_cache_bytes: usize,
    policy: Box<dyn super::policies::CachePolicy>,
) -> Arc<super::core::CacheStorage> {
    use tempfile::tempdir;

    use crate::cache::core::CacheStorage;

    let temp_dir = tempdir().unwrap();
    let batch_size = 128;

    Arc::new(CacheStorage::new(
        batch_size,
        max_cache_bytes,
        temp_dir.keep(),
        LiquidCacheMode::LiquidBlocking,
        policy,
    ))
}

/// Advice given by the cache policy.
#[derive(PartialEq, Eq, Debug)]
pub enum CacheAdvice {
    /// Evict the entry with the given ID.
    Evict(CacheEntryID),
    /// Transcode the entry to disk.
    TranscodeToDisk(CacheEntryID),
    /// Transcode the entry to liquid memory.
    Transcode(CacheEntryID),
    /// Write the entry to disk as-is (preserve format).
    ToDisk(CacheEntryID),
    /// Discard the incoming entry, do not cache.
    // Note that discarding a previously cached entry is disallowed,
    // as it creates race conditions when one thread reads that a entry is cached,
    // and later only to find it is not cached.
    Discard,
}

/// This is a unique identifier for a row in a parquet file.
#[repr(C, align(8))]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Ord, PartialOrd)]
pub struct CacheEntryID {
    file_id: u16,
    rg_id: u16,
    col_id: u16,
    batch_id: BatchID,
}

impl From<CacheEntryID> for usize {
    fn from(id: CacheEntryID) -> Self {
        (id.file_id as usize) << 48
            | (id.rg_id as usize) << 32
            | (id.col_id as usize) << 16
            | (id.batch_id.v as usize)
    }
}

impl From<usize> for CacheEntryID {
    fn from(value: usize) -> Self {
        Self {
            file_id: (value >> 48) as u16,
            rg_id: ((value >> 32) & 0xFFFF) as u16,
            col_id: ((value >> 16) & 0xFFFF) as u16,
            batch_id: BatchID::from_raw((value & 0xFFFF) as u16),
        }
    }
}

impl CacheEntryID {
    /// returns row group id
    pub fn row_group_id(&self) -> u16 {
        self.rg_id
    }
}

const _: () = assert!(std::mem::size_of::<CacheEntryID>() == 8);
const _: () = assert!(std::mem::align_of::<CacheEntryID>() == 8);

/// BatchID is a unique identifier for a batch of rows,
/// it is row id divided by the batch size.
///
// It's very easy to misinterpret this as row id, so we use new type idiom to avoid confusion:
// https://doc.rust-lang.org/rust-by-example/generics/new_types.html
#[repr(C, align(2))]
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Ord, PartialOrd)]
pub struct BatchID {
    v: u16,
}

impl BatchID {
    /// Creates a new BatchID from a row id and a batch size.
    /// The row id is at the boundary of the batch.
    pub fn from_row_id(row_id: usize, batch_size: usize) -> Self {
        Self {
            v: (row_id / batch_size) as u16,
        }
    }

    /// Creates a new BatchID from a raw value.
    pub fn from_raw(v: u16) -> Self {
        Self { v }
    }

    /// Increment the batch id.
    pub fn inc(&mut self) {
        debug_assert!(self.v < u16::MAX);
        self.v += 1;
    }
}

impl Deref for BatchID {
    type Target = u16;

    fn deref(&self) -> &Self::Target {
        &self.v
    }
}

impl CacheEntryID {
    /// Creates a new CacheEntryID.
    pub fn new(file_id: u64, row_group_id: u64, column_id: u64, batch_id: BatchID) -> Self {
        debug_assert!(file_id <= u16::MAX as u64);
        debug_assert!(row_group_id <= u16::MAX as u64);
        debug_assert!(column_id <= u16::MAX as u64);
        Self {
            file_id: file_id as u16,
            rg_id: row_group_id as u16,
            col_id: column_id as u16,
            batch_id,
        }
    }

    /// Get the batch id.
    pub fn batch_id_inner(&self) -> u64 {
        self.batch_id.v as u64
    }

    /// Get the file id.
    pub fn file_id_inner(&self) -> u64 {
        self.file_id as u64
    }

    /// Get the row group id.
    pub fn row_group_id_inner(&self) -> u64 {
        self.rg_id as u64
    }

    /// Get the column id.
    pub fn column_id_inner(&self) -> u64 {
        self.col_id as u64
    }

    /// Get the on-disk path.
    pub fn on_disk_path(&self, cache_root_dir: &Path) -> PathBuf {
        let batch_id = self.batch_id_inner();
        cache_root_dir
            .join(format!("file_{}", self.file_id_inner()))
            .join(format!("rg_{}", self.row_group_id_inner()))
            .join(format!("col_{}", self.column_id_inner()))
            .join(format!("batch_{batch_id}.liquid"))
    }

    /// Get the on-disk arrow path.
    pub fn on_disk_arrow_path(&self, cache_root_dir: &Path) -> PathBuf {
        let batch_id = self.batch_id_inner();
        cache_root_dir
            .join(format!("file_{}", self.file_id_inner()))
            .join(format!("rg_{}", self.row_group_id_inner()))
            .join(format!("col_{}", self.column_id_inner()))
            .join(format!("batch_{batch_id}.arrow"))
    }

    /// Write the liquid array to disk.
    pub(super) fn write_liquid_to_disk(
        &self,
        cache_root_dir: &Path,
        array: &LiquidArrayRef,
    ) -> Result<usize, std::io::Error> {
        let path = self.on_disk_path(cache_root_dir);
        let bytes = array.to_bytes();
        let mut file = File::create(&path)?;
        file.write_all(&bytes)?;
        Ok(bytes.len())
    }

    /// Write an arrow array to disk in IPC format.
    pub(super) fn write_arrow_to_disk(
        &self,
        cache_root_dir: &Path,
        array: &arrow::array::ArrayRef,
    ) -> Result<usize, std::io::Error> {
        use arrow::array::RecordBatch;
        use arrow::ipc::writer::StreamWriter;
        use std::fs::File;
        use std::io::BufWriter;

        let file_path = self.on_disk_arrow_path(cache_root_dir);

        // Ensure parent directory exists
        if let Some(parent) = file_path.parent() {
            std::fs::create_dir_all(parent)?;
        }

        let file = File::create(file_path)?;
        let buf_writer = BufWriter::new(file);

        // Create a record batch with the single array
        // We need to create a dummy field since we don't have the original field here
        let field =
            arrow_schema::Field::new("column", array.data_type().clone(), array.null_count() > 0);
        let schema = std::sync::Arc::new(arrow_schema::Schema::new(vec![field]));
        let batch = RecordBatch::try_new(schema.clone(), vec![array.clone()]).unwrap();

        let mut stream_writer = StreamWriter::try_new(buf_writer, &schema).unwrap();
        stream_writer.write(&batch).unwrap();
        stream_writer.finish().unwrap();

        // Return approximate size for disk usage tracking
        Ok(array.get_array_memory_size())
    }
}

#[cfg(test)]
pub(crate) fn create_entry_id(
    file_id: u64,
    row_group_id: u64,
    column_id: u64,
    batch_id: u16,
) -> CacheEntryID {
    CacheEntryID::new(
        file_id,
        row_group_id,
        column_id,
        BatchID::from_raw(batch_id),
    )
}

/// States for liquid compressor.
pub struct LiquidCompressorStates {
    fsst_compressor: RwLock<Option<Arc<fsst::Compressor>>>,
}

impl std::fmt::Debug for LiquidCompressorStates {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "EtcCompressorStates")
    }
}

impl Default for LiquidCompressorStates {
    fn default() -> Self {
        Self::new()
    }
}

impl LiquidCompressorStates {
    /// Create a new instance of LiquidCompressorStates.
    pub fn new() -> Self {
        Self {
            fsst_compressor: RwLock::new(None),
        }
    }

    pub fn new_with_fsst_compressor(fsst_compressor: Arc<fsst::Compressor>) -> Self {
        Self {
            fsst_compressor: RwLock::new(Some(fsst_compressor)),
        }
    }

    /// Get the fsst compressor.
    pub fn fsst_compressor(&self) -> Option<Arc<fsst::Compressor>> {
        self.fsst_compressor.read().unwrap().clone()
    }

    /// Get the fsst compressor .
    pub fn fsst_compressor_raw(&self) -> &RwLock<Option<Arc<fsst::Compressor>>> {
        &self.fsst_compressor
    }
}

/// Cached batch.
#[derive(Debug, Clone)]
pub enum CachedBatch {
    /// Cached batch in memory as Arrow array.
    MemoryArrow(ArrayRef),
    /// Cached batch in memory as liquid array.
    MemoryLiquid(LiquidArrayRef),
    /// Cached batch on disk as liquid array.
    DiskLiquid,
    /// Cached batch on disk as Arrow array.
    DiskArrow,
}

impl CachedBatch {
    /// Get the memory usage of the cached batch.
    pub fn memory_usage_bytes(&self) -> usize {
        match self {
            Self::MemoryArrow(array) => array.get_array_memory_size(),
            Self::MemoryLiquid(array) => array.get_array_memory_size(),
            Self::DiskLiquid => 0,
            Self::DiskArrow => 0,
        }
    }

    /// Get the reference count of the cached batch.
    pub fn reference_count(&self) -> usize {
        match self {
            Self::MemoryArrow(array) => Arc::strong_count(array),
            Self::MemoryLiquid(array) => Arc::strong_count(array),
            Self::DiskLiquid => 0,
            Self::DiskArrow => 0,
        }
    }
}

impl Display for CachedBatch {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::MemoryArrow(_) => write!(f, "MemoryArrow"),
            Self::MemoryLiquid(_) => write!(f, "MemoryLiquid"),
            Self::DiskLiquid => write!(f, "DiskLiquid"),
            Self::DiskArrow => write!(f, "DiskArrow"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn test_column_path_from_cache_entry_id() {
        let entry_id = CacheEntryID::new(1, 2, 3, BatchID::from_raw(4));
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

    #[test]
    fn test_batch_id_from_row_id() {
        let batch_id = BatchID::from_row_id(256, 128);
        assert_eq!(batch_id.v, 2);
    }

    #[test]
    fn test_batch_id_from_raw() {
        let batch_id = BatchID::from_raw(5);
        assert_eq!(batch_id.v, 5);
    }

    #[test]
    fn test_batch_id_inc() {
        let mut batch_id = BatchID::from_raw(10);
        batch_id.inc();
        assert_eq!(batch_id.v, 11);
    }

    #[test]
    #[should_panic]
    fn test_batch_id_inc_overflow() {
        let mut batch_id = BatchID::from_raw(u16::MAX);
        // Should panic because incrementing exceeds u16::MAX
        batch_id.inc();
    }

    #[test]
    fn test_batch_id_deref() {
        let batch_id = BatchID::from_raw(15);
        assert_eq!(*batch_id, 15);
    }

    #[test]
    fn test_cache_entry_id_new_and_getters() {
        let file_id = 10u64;
        let row_group_id = 20u64;
        let column_id = 30u64;
        let batch_id = BatchID::from_raw(40);
        let entry_id = CacheEntryID::new(file_id, row_group_id, column_id, batch_id);

        assert_eq!(entry_id.file_id_inner(), file_id);
        assert_eq!(entry_id.row_group_id_inner(), row_group_id);
        assert_eq!(entry_id.column_id_inner(), column_id);
        assert_eq!(entry_id.batch_id_inner(), *batch_id as u64);
    }

    #[test]
    fn test_cache_entry_id_boundaries() {
        let file_id = u16::MAX as u64;
        let row_group_id = 0u64;
        let column_id = u16::MAX as u64;
        let batch_id = BatchID::from_raw(0);
        let entry_id = CacheEntryID::new(file_id, row_group_id, column_id, batch_id);

        assert_eq!(entry_id.file_id_inner(), file_id);
        assert_eq!(entry_id.row_group_id_inner(), row_group_id);
        assert_eq!(entry_id.column_id_inner(), column_id);
        assert_eq!(entry_id.batch_id_inner(), *batch_id as u64);
    }

    #[test]
    #[should_panic]
    fn test_cache_entry_id_new_panic_file_id() {
        CacheEntryID::new((u16::MAX as u64) + 1, 0, 0, BatchID::from_raw(0));
    }

    #[test]
    #[should_panic]
    fn test_cache_entry_id_new_panic_row_group_id() {
        CacheEntryID::new(0, (u16::MAX as u64) + 1, 0, BatchID::from_raw(0));
    }

    #[test]
    #[should_panic]
    fn test_cache_entry_id_new_panic_column_id() {
        CacheEntryID::new(0, 0, (u16::MAX as u64) + 1, BatchID::from_raw(0));
    }

    #[test]
    fn test_cache_entry_id_on_disk_path() {
        let temp_dir = tempdir().unwrap();
        let cache_root = temp_dir.path();
        let entry_id = CacheEntryID::new(1, 2, 3, BatchID::from_raw(4));
        let expected_path = cache_root
            .join("file_1")
            .join("rg_2")
            .join("col_3")
            .join("batch_4.liquid");
        assert_eq!(entry_id.on_disk_path(cache_root), expected_path);
    }
}
