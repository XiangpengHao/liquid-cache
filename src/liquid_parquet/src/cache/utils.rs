use std::{
    ops::Deref,
    path::{Path, PathBuf},
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Ord, PartialOrd)]
pub(super) struct ColumnAccessPath {
    v: u64,
}

impl ColumnAccessPath {
    pub(super) fn new(file_id: u64, row_group_id: u64, column_id: u64) -> Self {
        Self {
            v: (file_id << 48) | (row_group_id << 32) | (column_id << 16),
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
        self.v >> 48
    }

    fn row_group_id_inner(&self) -> u64 {
        (self.v >> 32) & 0x0000_0000_0000_FFFF
    }

    fn column_id_inner(&self) -> u64 {
        (self.v >> 16) & 0x0000_0000_0000_FFFF
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
            v: (value.file_id_inner() << 48)
                | (value.row_group_id_inner() << 32)
                | (value.column_id_inner() << 16),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Ord, PartialOrd)]
pub(super) struct CacheEntryID {
    // This is a unique identifier for a row in a parquet file.
    // It is composed of 8 bytes:
    // - 2 bytes for the file id
    // - 2 bytes for the row group id
    // - 2 bytes for the column id
    // - 2 bytes for the batch id
    // The numerical order of val is meaningful: sorted by each of the fields.
    val: u64,
}

/// BatchID is a unique identifier for a batch of rows,
/// it is row id divided by the batch size.
///
// It's very easy to misinterpret this as row id, so we use new type idiom to avoid confusion:
// https://doc.rust-lang.org/rust-by-example/generics/new_types.html
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Ord, PartialOrd)]
pub struct BatchID {
    v: u16,
}

impl BatchID {
    /// Creates a new BatchID from a row id and a batch size.
    /// row id must be on the batch boundary.
    pub(crate) fn from_row_id(row_id: usize, batch_size: usize) -> Self {
        debug_assert!(row_id % batch_size == 0);
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
            val: (file_id) << 48 | (row_group_id) << 32 | (column_id) << 16 | batch_id.v as u64,
        }
    }

    pub(super) fn batch_id_inner(&self) -> u64 {
        self.val & 0x0000_0000_0000_FFFF
    }

    pub(super) fn file_id_inner(&self) -> u64 {
        self.val >> 48
    }

    pub(super) fn row_group_id_inner(&self) -> u64 {
        (self.val >> 32) & 0x0000_0000_FFFF
    }

    pub(super) fn column_id_inner(&self) -> u64 {
        (self.val >> 16) & 0x0000_0000_FFFF
    }

    pub(super) fn on_disk_path(&self, cache_root_dir: &Path) -> PathBuf {
        let batch_id = self.batch_id_inner();
        cache_root_dir
            .join(format!("file_{}", self.file_id_inner()))
            .join(format!("rg_{}", self.row_group_id_inner()))
            .join(format!("col_{}", self.column_id_inner()))
            .join(format!("batch_{}.bin", batch_id))
    }
}

#[derive(Debug)]
pub(super) struct CacheConfig {
    batch_size: usize,
    max_cache_bytes: usize,
    cache_root_dir: PathBuf,
}

impl CacheConfig {
    pub(super) fn new(batch_size: usize, max_cache_bytes: usize, cache_root_dir: PathBuf) -> Self {
        Self {
            batch_size,
            max_cache_bytes,
            cache_root_dir,
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
    #[should_panic]
    fn test_batch_id_from_row_id_panic() {
        // Should panic because row_id is not a multiple of batch_size
        BatchID::from_row_id(100, 128);
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

        // Reconstruct the expected value based on the bit shifting logic
        let expected_v = (1u64 << 48) | (2u64 << 32) | (3u64 << 16);
        assert_eq!(column_path.v, expected_v);
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
