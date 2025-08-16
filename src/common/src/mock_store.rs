//! Mock object store for testing purposes.

use async_trait::async_trait;
use bytes::Bytes;
use chrono::{DateTime, Utc};
use futures::{StreamExt, stream::BoxStream};
use object_store::PutMultipartOptions;
use object_store::{
    Attributes, Error, GetOptions, GetResult, GetResultPayload, ListResult, MultipartUpload,
    ObjectMeta, ObjectStore, PutMode, PutOptions, PutPayload, PutResult, Result, path::Path,
};
use std::collections::{BTreeMap, BTreeSet};
use std::ops::Range;
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::RwLock;
use std::sync::atomic::{AtomicUsize, Ordering};

/// In-memory storage for testing purposes.
///
/// This can be used as a mock object store for testing purposes instead of using a real object store like S3.
/// It is not meant to be used in production.
///
/// # Usage
///
/// The following example shows how to create a store, list items, get an object,
/// and put a new object.
///
/// ```rust
/// # use liquid_cache_common::mock_store::MockStore;
/// # use object_store::{path::Path, GetOptions, PutPayload, ObjectMeta, ObjectStore};
/// # use futures::TryStreamExt;
/// # use bytes::Bytes;
/// # async fn test() -> Result<(), object_store::Error> {
/// let store = MockStore::new_with_files(10, 1024 * 10); // 10 files of 10KB each
/// let paths: Vec<ObjectMeta> = store.list(None).try_collect().await?;
///
/// let options = GetOptions {
///     range: Some((0..(1024 * 10)).into()),
///     ..Default::default()
/// };
/// let path = Path::from("1.parquet");
/// let result = store.get_opts(&path, options).await?;
/// let bytes = result.bytes().await?;
///
/// let path = Path::from("11.parquet");
/// let payload = PutPayload::from(Bytes::from_static(b"test data"));
/// store.put(&path, payload).await?;
/// # Ok(())
/// # }
/// ```
///
#[derive(Debug, Default)]
pub struct MockStore {
    storage: SharedStorage,
}

/// A specialized `Error` for in-memory object store-related errors
#[derive(Debug, thiserror::Error)]
enum MockStoreError {
    #[error("No data in memory found. Location: {path}")]
    NoDataInMemory { path: String },

    #[error("Object already exists at that location: {path}")]
    AlreadyExists { path: String },

    #[error("Invalid range")]
    InvalidGetRange,
}

impl From<MockStoreError> for object_store::Error {
    fn from(source: MockStoreError) -> Self {
        match source {
            MockStoreError::NoDataInMemory { ref path } => Self::NotFound {
                path: path.into(),
                source: source.into(),
            },
            MockStoreError::AlreadyExists { ref path } => Self::AlreadyExists {
                path: path.into(),
                source: source.into(),
            },
            _ => Self::Generic {
                store: "MockStore",
                source: Box::new(source),
            },
        }
    }
}

#[derive(Debug, Clone)]
struct Entry {
    data: Bytes,
    last_modified: DateTime<Utc>,
    attributes: Attributes,
    e_tag: usize,
    access_count: Arc<AtomicUsize>,
    access_ranges: Arc<Mutex<Vec<Range<u64>>>>,
}

impl Entry {
    fn new(
        data: Bytes,
        last_modified: DateTime<Utc>,
        e_tag: usize,
        attributes: Attributes,
    ) -> Self {
        Self {
            data,
            last_modified,
            e_tag,
            attributes,
            access_count: Arc::new(AtomicUsize::new(0)),
            access_ranges: Arc::new(Mutex::new(Vec::new())),
        }
    }
}

#[derive(Debug, Default, Clone)]
struct Storage {
    next_etag: usize,
    map: BTreeMap<Path, Entry>,
}

type SharedStorage = Arc<RwLock<Storage>>;

impl Storage {
    fn insert(&mut self, location: &Path, bytes: Bytes, attributes: Attributes) -> usize {
        let etag = self.next_etag;
        self.next_etag += 1;
        let entry = Entry::new(bytes, Utc::now(), etag, attributes);
        self.overwrite(location, entry);
        etag
    }

    fn overwrite(&mut self, location: &Path, entry: Entry) {
        self.map.insert(location.clone(), entry);
    }

    fn create(&mut self, location: &Path, entry: Entry) -> Result<()> {
        use std::collections::btree_map;
        match self.map.entry(location.clone()) {
            btree_map::Entry::Occupied(_) => Err(MockStoreError::AlreadyExists {
                path: location.to_string(),
            }
            .into()),
            btree_map::Entry::Vacant(v) => {
                v.insert(entry);
                Ok(())
            }
        }
    }
}

impl std::fmt::Display for MockStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "MockStore")
    }
}

impl MockStore {
    /// Create new in-memory storage.
    pub fn new() -> Self {
        Self::default()
    }

    /// Initialize the object store with preset values
    pub fn new_with_files(file_count: usize, file_size: usize) -> Self {
        let store = Self::new();
        {
            let mut storage = store.storage.write().unwrap();
            let data = vec![0u8; file_size];

            // Fill the data with a pattern: index % 256
            // This makes it easy to verify ranges
            let data: Vec<u8> = data
                .iter()
                .enumerate()
                .map(|(i, _)| (i % 256) as u8)
                .collect();

            for file_name in 0..file_count {
                let path = Path::from(format!("{file_name}.parquet"));
                storage.insert(&path, Bytes::from(data.clone()), Attributes::default());
            }
        }
        store
    }

    /// Creates a fork of the store, with the current content copied into the
    /// new store.
    pub fn fork(&self) -> Self {
        let storage = self.storage.read().unwrap();
        let storage = Arc::new(RwLock::new(storage.clone()));
        Self { storage }
    }

    /// Get the access count(no. of calls to `get_opts`) for a specific file
    pub fn get_access_count(&self, location: &Path) -> Option<usize> {
        self.storage
            .read()
            .unwrap()
            .map
            .get(location)
            .map(|entry| entry.access_count.load(Ordering::SeqCst))
    }

    /// Get a list of ranges that have been requested via `get_opts`
    pub fn get_access_ranges(&self, location: &Path) -> Option<Vec<Range<u64>>> {
        self.storage
            .read()
            .unwrap()
            .map
            .get(location)
            .map(|entry| entry.access_ranges.lock().unwrap().clone())
    }

    /// Get the number of objects stored in the store
    pub fn get_file_count(&self) -> usize {
        self.storage.read().unwrap().map.len()
    }

    /// Get the total size of all objects in the store
    pub fn get_store_size(&self) -> usize {
        self.storage
            .read()
            .unwrap()
            .map
            .values()
            .map(|entry| entry.data.len())
            .sum()
    }

    fn entry(&self, location: &Path) -> Result<Entry> {
        let storage = self.storage.read().unwrap();
        let value =
            storage
                .map
                .get(location)
                .cloned()
                .ok_or_else(|| MockStoreError::NoDataInMemory {
                    path: location.to_string(),
                })?;

        Ok(value)
    }
}

#[async_trait]
impl ObjectStore for MockStore {
    async fn put_opts(
        &self,
        location: &Path,
        payload: PutPayload,
        opts: PutOptions,
    ) -> Result<PutResult> {
        let mut storage = self.storage.write().unwrap();
        let etag = storage.next_etag;
        let entry = Entry::new(payload.into(), Utc::now(), etag, opts.attributes);

        match opts.mode {
            PutMode::Overwrite => storage.overwrite(location, entry),
            PutMode::Create => storage.create(location, entry)?,
            PutMode::Update(_) => unreachable!("MockStore does not support update"),
        }
        storage.next_etag += 1;

        Ok(PutResult {
            e_tag: Some(etag.to_string()),
            version: None,
        })
    }

    async fn get_opts(&self, location: &Path, options: GetOptions) -> Result<GetResult> {
        let entry = self.entry(location)?;

        // Atomically increment the count. This is a fast, lock-free operation.
        entry.access_count.fetch_add(1, Ordering::SeqCst);

        let e_tag = entry.e_tag.to_string();

        let meta = ObjectMeta {
            location: location.clone(),
            last_modified: entry.last_modified,
            size: entry.data.len() as u64,
            e_tag: Some(e_tag),
            version: None,
        };
        options.check_preconditions(&meta)?;

        let (range, data) = match options.range {
            Some(range) => {
                let r = range
                    .as_range(entry.data.len() as u64)
                    .map_err(|_| Error::Generic {
                        store: "MockStore",
                        source: Box::new(MockStoreError::InvalidGetRange),
                    })?;
                (
                    r.clone(),
                    entry.data.slice(r.start as usize..r.end as usize),
                )
            }
            None => (0..entry.data.len() as u64, entry.data),
        };
        entry.access_ranges.lock().unwrap().push(range.clone());
        let stream = futures::stream::once(futures::future::ready(Ok(data)));

        Ok(GetResult {
            payload: GetResultPayload::Stream(stream.boxed()),
            attributes: entry.attributes,
            meta,
            range,
        })
    }

    async fn head(&self, location: &Path) -> Result<ObjectMeta> {
        let entry = self.entry(location)?;

        Ok(ObjectMeta {
            location: location.clone(),
            last_modified: entry.last_modified,
            size: entry.data.len() as u64,
            e_tag: Some(entry.e_tag.to_string()),
            version: None,
        })
    }

    async fn delete(&self, location: &Path) -> Result<()> {
        self.storage.write().unwrap().map.remove(location);
        Ok(())
    }

    fn list(&self, prefix: Option<&Path>) -> BoxStream<'static, Result<ObjectMeta>> {
        let root = Path::default();
        let prefix = prefix.unwrap_or(&root);

        let storage = self.storage.read().unwrap();
        let values: Vec<_> = storage
            .map
            .range((prefix)..)
            .take_while(|(key, _)| key.as_ref().starts_with(prefix.as_ref()))
            .filter(|(key, _)| {
                // Don't return for exact prefix match
                key.prefix_match(prefix)
                    .map(|mut x| x.next().is_some())
                    .unwrap_or(false)
            })
            .map(|(key, value)| {
                Ok(ObjectMeta {
                    location: key.clone(),
                    last_modified: value.last_modified,
                    size: value.data.len() as u64,
                    e_tag: Some(value.e_tag.to_string()),
                    version: None,
                })
            })
            .collect();

        futures::stream::iter(values).boxed()
    }

    async fn list_with_delimiter(&self, prefix: Option<&Path>) -> Result<ListResult> {
        let root = Path::default();
        let prefix = prefix.unwrap_or(&root);

        let mut common_prefixes = BTreeSet::new();

        // Only objects in this base level should be returned in the
        // response. Otherwise, we just collect the common prefixes.
        let mut objects = vec![];
        for (k, v) in self.storage.read().unwrap().map.range((prefix)..) {
            if !k.as_ref().starts_with(prefix.as_ref()) {
                break;
            }

            let mut parts = match k.prefix_match(prefix) {
                Some(parts) => parts,
                None => continue,
            };

            // Pop first element
            let common_prefix = match parts.next() {
                Some(p) => p,
                // Should only return children of the prefix
                None => continue,
            };

            if parts.next().is_some() {
                common_prefixes.insert(prefix.child(common_prefix));
            } else {
                let object = ObjectMeta {
                    location: k.clone(),
                    last_modified: v.last_modified,
                    size: v.data.len() as u64,
                    e_tag: Some(v.e_tag.to_string()),
                    version: None,
                };
                objects.push(object);
            }
        }

        Ok(ListResult {
            objects,
            common_prefixes: common_prefixes.into_iter().collect(),
        })
    }

    async fn put_multipart_opts(
        &self,
        _location: &Path,
        _opts: PutMultipartOptions,
    ) -> Result<Box<dyn MultipartUpload>> {
        unreachable!("MockStore does not support multipart upload")
    }

    async fn copy(&self, _from: &Path, _to: &Path) -> Result<()> {
        unreachable!("MockStore does not support copy")
    }

    async fn copy_if_not_exists(&self, _from: &Path, _to: &Path) -> Result<()> {
        unreachable!("MockStore does not support copy_if_not_exists")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::TryStreamExt;

    async fn setup_test_store() -> MockStore {
        let store = MockStore::new_with_files(10, 1024 * 10); // 10 files of 10KB
        let paths: Vec<ObjectMeta> = store.list(None).try_collect().await.unwrap();
        assert_eq!(paths.len(), 10, "Initial store should have 10 files");
        store
    }

    #[tokio::test]
    async fn test_new_with_files() {
        let store = setup_test_store().await;
        let paths: Vec<ObjectMeta> = store.list(None).try_collect().await.unwrap();
        assert_eq!(paths.len(), 10);

        // Test if the files are present and have the correct size
        for (i, meta) in paths.iter().enumerate() {
            let path = Path::from(format!("{i}.parquet"));
            let loaded_meta = store.head(&path).await.unwrap();
            assert_eq!(
                loaded_meta.size,
                1024 * 10,
                "Expected size of 10KB, got {}",
                loaded_meta.size
            );
            assert_eq!(
                meta.location, path,
                "Expected location to be {path}, got {}",
                meta.location
            );
        }
    }

    #[tokio::test]
    async fn test_get_opts() {
        let store = setup_test_store().await;
        let path = Path::from("1.parquet");

        // Test for a range that is fully in bounds
        let options = GetOptions {
            range: Some((0..(1024 * 10)).into()),
            ..GetOptions::default()
        };
        let result = store.get_opts(&path, options).await.unwrap();
        let bytes = result.bytes().await.unwrap();
        assert_eq!(bytes.len(), 1024 * 10);

        // Test for a range that is partially in bounds
        let options = GetOptions {
            range: Some((1024..4096).into()),
            ..GetOptions::default()
        };
        let result = store.get_opts(&path, options).await.unwrap();
        let bytes = result.bytes().await.unwrap();
        assert_eq!(bytes.len(), 3072);

        // Test for a range that is partially out of bounds
        let options = GetOptions {
            range: Some((8192..12288).into()),
            ..GetOptions::default()
        };
        let result = store.get_opts(&path, options).await.unwrap();
        let bytes = result.bytes().await.unwrap();
        // The store should return only the valid part of the range.
        assert_eq!(bytes.len(), 2048);

        // Test for a range that is fully out of bounds
        let options = GetOptions {
            range: Some((20480..30720).into()),
            ..GetOptions::default()
        };
        let err = store.get_opts(&path, options).await.unwrap_err();
        assert!(
            matches!(err, Error::Generic { .. }),
            "Expected an error for out-of-bounds request, got {err:?}"
        );
    }

    #[tokio::test]
    async fn test_insert_and_list() {
        let store = setup_test_store().await;

        // Test for a new file insertion
        let new_path = Path::from("11.parquet");
        let payload = PutPayload::from(Bytes::from_static(b"test data"));
        store
            .put_opts(&new_path, payload, PutOptions::default())
            .await
            .unwrap();

        let paths: Vec<ObjectMeta> = store.list(None).try_collect().await.unwrap();
        assert_eq!(
            paths.len(),
            11,
            "Store should have 11 files after insertion"
        );

        let meta = store.head(&new_path).await.unwrap();
        assert_eq!(meta.size, 9);
        assert_eq!(meta.location, new_path);
    }

    #[tokio::test]
    async fn test_delete() {
        let store = setup_test_store().await;
        let path_to_delete = Path::from("5.parquet");

        store.delete(&path_to_delete).await.unwrap();

        // Verify the file is deleted
        let err = store.head(&path_to_delete).await.unwrap_err();
        assert!(
            matches!(err, Error::NotFound { .. }),
            "Expected NotFound error after delete, but got {err:?}"
        );

        let paths: Vec<ObjectMeta> = store.list(None).try_collect().await.unwrap();
        assert_eq!(paths.len(), 9, "Store should have 9 files after deletion");
    }

    #[tokio::test]
    async fn test_list_uses_directories_correctly() {
        let store = setup_test_store().await;
        let folder_path = Path::from("folder/");
        let file_path = Path::from("folder/file.parquet");
        store
            .put_opts(
                &file_path,
                PutPayload::from(Bytes::from_static(b"test")),
                PutOptions::default(),
            )
            .await
            .unwrap();

        // Check listing the root
        let list_result = store.list_with_delimiter(None).await.unwrap();
        assert_eq!(list_result.objects.len(), 10, "Root should have 10 objects");
        assert_eq!(
            list_result.common_prefixes.len(),
            1,
            "Root should have 1 common prefix (folder)"
        );
        assert_eq!(
            list_result.common_prefixes[0], folder_path,
            "Common prefix should be 'folder/'"
        );

        // Check listing the sub-directory
        let list_result = store.list_with_delimiter(Some(&folder_path)).await.unwrap();
        assert_eq!(
            list_result.objects.len(),
            1,
            "Folder should contain 1 object"
        );
        assert_eq!(
            list_result.common_prefixes.len(),
            0,
            "Folder should have no common prefixes"
        );
        assert_eq!(list_result.objects[0].location, file_path);
    }

    #[tokio::test]
    async fn test_fork() {
        let original_store = setup_test_store().await;
        let forked_store = original_store.fork();

        // Modify the original store
        original_store
            .put_opts(
                &Path::from("11.parquet"),
                PutPayload::from(Bytes::from_static(b"new data")),
                PutOptions::default(),
            )
            .await
            .unwrap();

        // Verify the original store has changed
        let original_paths: Vec<ObjectMeta> =
            original_store.list(None).try_collect().await.unwrap();
        assert_eq!(original_paths.len(), 11);

        // Verify the forked store has NOT changed
        let forked_paths: Vec<ObjectMeta> = forked_store.list(None).try_collect().await.unwrap();
        assert_eq!(
            forked_paths.len(),
            10,
            "Forked store should not be affected by changes to the original"
        );
    }

    #[tokio::test]
    async fn test_access_count() {
        let store = setup_test_store().await;
        let path = Path::from("3.parquet");

        let count = store.get_access_count(&path).unwrap();
        assert_eq!(count, 0, "Initial access count should be 0, got {count}");

        // First get
        let _ = store.get_opts(&path, GetOptions::default()).await.unwrap();
        let count = store.get_access_count(&path).unwrap();
        assert_eq!(
            count, 1,
            "Access count should be 1 after one get, got {count}"
        );

        // Second get
        let _ = store.get_opts(&path, GetOptions::default()).await.unwrap();
        let count = store.get_access_count(&path).unwrap();
        assert_eq!(
            count, 2,
            "Access count should be 2 after two gets, got {count}"
        );
    }

    #[tokio::test]
    async fn test_store_metrics() {
        let store = setup_test_store().await;

        // Initial state from setup_test_store
        assert_eq!(store.get_file_count(), 10);
        assert_eq!(store.get_store_size(), 10 * 1024 * 10);

        // Add a new file
        let new_path = Path::from("new_file.parquet");
        let new_data = Bytes::from_static(b"some new data");
        let new_data_len = new_data.len();
        store
            .put_opts(&new_path, PutPayload::from(new_data), PutOptions::default())
            .await
            .unwrap();

        assert_eq!(store.get_file_count(), 11);
        assert_eq!(store.get_store_size(), 10 * 1024 * 10 + new_data_len);

        // Delete one of the original files
        let path_to_delete = Path::from("5.parquet");
        store.delete(&path_to_delete).await.unwrap();

        assert_eq!(store.get_file_count(), 10);
        assert_eq!(
            store.get_store_size(),
            9 * 1024 * 10 + new_data_len,
            "Store size should be reduced by the size of the deleted file"
        );
    }
}
