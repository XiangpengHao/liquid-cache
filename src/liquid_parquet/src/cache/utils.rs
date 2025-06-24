use std::path::{Path, PathBuf};

use crate::lib::{BatchID, CacheEntryID};
use liquid_cache_common::LiquidCacheMode;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Ord, PartialOrd)]
pub(super) struct ColumnAccessPath {
    file_id: u16,
    rg_id: u16,
    col_id: u16,
}

impl ColumnAccessPath {
    pub(super) fn new(file_id: u64, row_group_id: u64, column_id: u64) -> Self {
        debug_assert!(file_id <= u16::MAX as u64);
        debug_assert!(row_group_id <= u16::MAX as u64);
        debug_assert!(column_id <= u16::MAX as u64);
        Self {
            file_id: file_id as u16,
            rg_id: row_group_id as u16,
            col_id: column_id as u16,
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
        self.col_id as u64
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
            col_id: value.column_id_inner() as u16,
        }
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

    super::CachedBatch::MemoryArrow(Arc::new(Int64Array::from_iter_values(0..size as i64)))
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
        temp_dir.keep(),
        LiquidCacheMode::Liquid {
            transcode_in_background: false,
        },
        policy,
    )
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
            .join(format!("file_{}", file_id))
            .join(format!("rg_{}", row_group_id))
            .join(format!("col_{}", column_id));

        assert_eq!(entry_path.parent().unwrap(), &expected_dir);

        // Verify we can create a file at the entry path
        std::fs::write(&entry_path, b"test data").unwrap();
        assert!(entry_path.exists());
    }
}
