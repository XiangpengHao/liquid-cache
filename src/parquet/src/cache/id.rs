use std::{
    ops::Deref,
    path::{Path, PathBuf},
};

use liquid_cache_storage::cache::EntryID;

/// This is a unique identifier for a row in a parquet file.
#[repr(C, align(8))]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Ord, PartialOrd)]
pub struct ParquetArrayID {
    file_id: u16,
    rg_id: u16,
    col_id: u16,
    batch_id: BatchID,
}

impl From<ParquetArrayID> for usize {
    fn from(id: ParquetArrayID) -> Self {
        (id.file_id as usize) << 48
            | (id.rg_id as usize) << 32
            | (id.col_id as usize) << 16
            | (id.batch_id.v as usize)
    }
}

impl From<usize> for ParquetArrayID {
    fn from(value: usize) -> Self {
        Self {
            file_id: (value >> 48) as u16,
            rg_id: ((value >> 32) & 0xFFFF) as u16,
            col_id: ((value >> 16) & 0xFFFF) as u16,
            batch_id: BatchID::from_raw((value & 0xFFFF) as u16),
        }
    }
}

impl ParquetArrayID {}

impl From<ParquetArrayID> for EntryID {
    fn from(id: ParquetArrayID) -> Self {
        EntryID::from(usize::from(id))
    }
}

impl From<EntryID> for ParquetArrayID {
    fn from(id: EntryID) -> Self {
        ParquetArrayID::from(usize::from(id))
    }
}

const _: () = assert!(std::mem::size_of::<ParquetArrayID>() == 8);
const _: () = assert!(std::mem::align_of::<ParquetArrayID>() == 8);

impl ParquetArrayID {
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
}

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

#[cfg(test)]
mod tests {
    use tempfile::tempdir;

    use super::*;

    #[test]
    fn test_cache_entry_id_new_and_getters() {
        let file_id = 10u64;
        let row_group_id = 20u64;
        let column_id = 30u64;
        let batch_id = BatchID::from_raw(40);
        let entry_id = ParquetArrayID::new(file_id, row_group_id, column_id, batch_id);

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
        let entry_id = ParquetArrayID::new(file_id, row_group_id, column_id, batch_id);

        assert_eq!(entry_id.file_id_inner(), file_id);
        assert_eq!(entry_id.row_group_id_inner(), row_group_id);
        assert_eq!(entry_id.column_id_inner(), column_id);
        assert_eq!(entry_id.batch_id_inner(), *batch_id as u64);
    }

    #[test]
    #[should_panic]
    fn test_cache_entry_id_new_panic_file_id() {
        ParquetArrayID::new((u16::MAX as u64) + 1, 0, 0, BatchID::from_raw(0));
    }

    #[test]
    #[should_panic]
    fn test_cache_entry_id_new_panic_row_group_id() {
        ParquetArrayID::new(0, (u16::MAX as u64) + 1, 0, BatchID::from_raw(0));
    }

    #[test]
    #[should_panic]
    fn test_cache_entry_id_new_panic_column_id() {
        ParquetArrayID::new(0, 0, (u16::MAX as u64) + 1, BatchID::from_raw(0));
    }

    #[test]
    fn test_cache_entry_id_on_disk_path() {
        let temp_dir = tempdir().unwrap();
        let cache_root = temp_dir.path();
        let entry_id = ParquetArrayID::new(1, 2, 3, BatchID::from_raw(4));
        let expected_path = cache_root
            .join("file_1")
            .join("rg_2")
            .join("col_3")
            .join("batch_4.liquid");
        assert_eq!(entry_id.on_disk_path(cache_root), expected_path);
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
}
