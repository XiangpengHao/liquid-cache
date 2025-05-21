use std::{
    ops::Deref,
    path::{Path, PathBuf},
};

use liquid_cache_common::LiquidCacheMode;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Ord, PartialOrd)]
pub(super) struct ColumnAccessPath {
    file_id: u16,
    rg_id: u16,
    row_id: u16,
}

impl ColumnAccessPath {
    pub(super) fn new(file_id: u64, row_group_id: u64, column_id: u64) -> Self {
        debug_assert!(file_id <= u16::MAX as u64);
        debug_assert!(row_group_id <= u16::MAX as u64);
        debug_assert!(column_id <= u16::MAX as u64);
        Self {
            file_id: file_id as u16,
            rg_id: row_group_id as u16,
            row_id: column_id as u16,
        }
    }

    pub(super) fn initialize_dir(&self, cache_root_dir: &Path) {
        let path = cache_root_dir
            .join(format!("file_{}", self.file_id_inner()))
            .join(format!("rg_{}", self.row_group_id_inner()))
            .join(format!("col_{}", self.column_id_inner()));
        std::fs::create_dir_all(&path).expect("Failed to create cache directory");
    }

    fn file_id_inner(&self) -> u64 {
        self.file_id as u64
    }

    fn row_group_id_inner(&self) -> u64 {
        self.rg_id as u64
    }

    fn column_id_inner(&self) -> u64 {
        self.row_id as u64
    }

    pub(super) fn entry_id(&self, batch_id: BatchID) -> CacheEntryID {
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
            row_id: value.column_id_inner() as u16,
        }
    }
}

/// This is a unique identifier for a row in a parquet file.
#[repr(C, align(8))]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Ord, PartialOrd)]
pub struct CacheEntryID {
    file_id: u16,
    rg_id: u16,
    row_id: u16,
    batch_id: BatchID,
}

const _: () = assert!(std::mem::size_of::<CacheEntryID>() == 8);
const _: () = assert!(std::mem::align_of::<CacheEntryID>() == 8);

impl From<CacheEntryID> for usize {
    fn from(id: CacheEntryID) -> Self {
        (id.file_id as usize) << 48
            | (id.rg_id as usize) << 32
            | (id.row_id as usize) << 16
            | (id.batch_id.v as usize)
    }
}

impl From<usize> for CacheEntryID {
    fn from(value: usize) -> Self {
        Self {
            file_id: (value >> 48) as u16,
            rg_id: ((value >> 32) & 0xFFFF) as u16,
            row_id: ((value >> 16) & 0xFFFF) as u16,
            batch_id: BatchID::from_raw((value & 0xFFFF) as u16),
        }
    }
}

impl CacheEntryID {
    pub fn row_group_id(&self) -> u16 {
        self.rg_id
    }
}

const _: () = assert!(std::mem::size_of::<CacheEntryID>() == 8);

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
    pub(crate) fn from_row_id(row_id: usize, batch_size: usize) -> Self {
        Self {
            v: (row_id / batch_size) as u16,
        }
    }

    pub(crate) fn from_raw(v: u16) -> Self {
        Self { v }
    }

    pub(crate) fn inc(&mut self) {
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
    pub(super) fn new(file_id: u64, row_group_id: u64, column_id: u64, batch_id: BatchID) -> Self {
        debug_assert!(file_id <= u16::MAX as u64);
        debug_assert!(row_group_id <= u16::MAX as u64);
        debug_assert!(column_id <= u16::MAX as u64);
        Self {
            file_id: file_id as u16,
            rg_id: row_group_id as u16,
            row_id: column_id as u16,
            batch_id,
        }
    }

    pub(super) fn batch_id_inner(&self) -> u64 {
        self.batch_id.v as u64
    }

    pub(super) fn file_id_inner(&self) -> u64 {
        self.file_id as u64
    }

    pub(super) fn row_group_id_inner(&self) -> u64 {
        self.rg_id as u64
    }

    pub(super) fn column_id_inner(&self) -> u64 {
        self.row_id as u64
    }

    pub(super) fn on_disk_path(&self, cache_root_dir: &Path) -> PathBuf {
        let batch_id = self.batch_id_inner();
        cache_root_dir
            .join(format!("file_{}", self.file_id_inner()))
            .join(format!("rg_{}", self.row_group_id_inner()))
            .join(format!("col_{}", self.column_id_inner()))
            .join(format!("batch_{batch_id}.bin"))
    }
}

#[derive(Debug)]
pub(super) struct CacheConfig {
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
pub(crate) fn create_test_array(size: usize) -> super::CachedBatch {
    use arrow::array::Int64Array;
    use std::sync::Arc;

    super::CachedBatch::ArrowMemory(Arc::new(Int64Array::from_iter_values(0..size as i64)))
}

#[cfg(test)]
pub(crate) fn create_cache_store(
    max_cache_bytes: usize,
    policy: Box<dyn super::policies::CachePolicy>,
) -> super::store::CacheStore {
    use tempfile::tempdir;

    use crate::cache::store::CacheStore;

    let temp_dir = tempdir().unwrap();
    let batch_size = 128;

    CacheStore::new(
        batch_size,
        max_cache_bytes,
        temp_dir.into_path(),
        LiquidCacheMode::InMemoryLiquid {
            transcode_in_background: false,
        },
        policy,
    )
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
        crate::cache::BatchID::from_raw(batch_id),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

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
            .join("batch_4.bin");
        assert_eq!(entry_id.on_disk_path(cache_root), expected_path);
    }

    #[test]
    fn test_column_path_from_cache_entry_id() {
        let entry_id = CacheEntryID::new(1, 2, 3, BatchID::from_raw(4));
        let column_path: ColumnAccessPath = entry_id.into();

        assert_eq!(column_path.file_id, 1);
        assert_eq!(column_path.rg_id, 2);
        assert_eq!(column_path.row_id, 3);
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
            .join(format!("file_{}", file_id))
            .join(format!("rg_{}", row_group_id))
            .join(format!("col_{}", column_id));

        assert_eq!(entry_path.parent().unwrap(), &expected_dir);

        // Verify we can create a file at the entry path
        std::fs::write(&entry_path, b"test data").unwrap();
        assert!(entry_path.exists());
    }
}
