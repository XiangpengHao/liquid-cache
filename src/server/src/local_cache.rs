use std::{
    fmt::{Display, Formatter},
    fs,
    ops::{Range, RangeInclusive},
    path::PathBuf,
    sync::Arc,
};

use async_stream::stream;
use async_trait::async_trait;
use bytes::Bytes;
use futures::{Stream, stream::BoxStream};
use object_store::{
    Error, GetOptions, GetRange, GetResult, GetResultPayload, ListResult, MultipartUpload,
    ObjectMeta, ObjectStore, PutMultipartOpts, PutOptions, PutPayload, PutResult, Result,
    path::Path,
};
use tokio::io::{AsyncReadExt, AsyncSeekExt, AsyncWriteExt};

const CACHE_BLOCK_SIZE: u64 = 1024 * 1024 * 4; // 4MB

#[derive(Debug, Clone)]
pub struct LocalCache {
    inner: Arc<dyn ObjectStore>,
    cache_dir: PathBuf,
}

impl LocalCache {
    /// Create a new local cache, the cache_dir is the directory to store the cached files
    /// `LocalCache` can read from a previously initialized cache directory
    pub fn new(inner: Arc<dyn ObjectStore>, cache_dir: PathBuf) -> Self {
        if !cache_dir.exists() {
            fs::create_dir_all(&cache_dir).expect("Failed to create cache directory");
        }
        Self { inner, cache_dir }
    }

    /// Convert a path to a cached directory path
    fn get_cache_dir_for_path(&self, path: &Path) -> PathBuf {
        let path_str = path.as_ref().replace("/", "_");
        self.cache_dir.join(path_str)
    }

    /// Get the path for a specific chunk of a file
    fn get_chunk_path(&self, path: &Path, chunk_index: u64) -> PathBuf {
        let cache_dir = self.get_cache_dir_for_path(path);
        cache_dir.join(format!("chunk_{chunk_index}.bin"))
    }

    /// Calculate which chunks are needed for a given range
    fn chunks_for_range(&self, range: &Range<u64>) -> RangeInclusive<u64> {
        let start_chunk = range.start / CACHE_BLOCK_SIZE;
        let end_chunk = (range.end - 1) / CACHE_BLOCK_SIZE; // -1 because end is exclusive
        start_chunk..=end_chunk
    }

    /// Read data from a cached chunk
    async fn read_from_cached_chunk(
        &self,
        chunk_path: PathBuf,
        offset: u64,
        len: usize,
    ) -> Result<Bytes> {
        let mut file = tokio::fs::File::open(chunk_path)
            .await
            .map_err(|e| Error::Generic {
                store: "LocalCache",
                source: Box::new(e),
            })?;

        let mut buffer = vec![0u8; len];
        file.seek(tokio::io::SeekFrom::Start(offset))
            .await
            .map_err(|e| Error::Generic {
                store: "LocalCache",
                source: Box::new(e),
            })?;

        file.read_exact(&mut buffer)
            .await
            .map_err(|e| Error::Generic {
                store: "LocalCache",
                source: Box::new(e),
            })?;

        Ok(Bytes::from(buffer))
    }

    /// Save a chunk to the cache
    async fn save_chunk(&self, path: &Path, chunk_index: u64, data: Bytes) -> Result<()> {
        let cache_dir = self.get_cache_dir_for_path(path);
        if !cache_dir.exists() {
            fs::create_dir_all(&cache_dir).map_err(|e| Error::Generic {
                store: "LocalCache",
                source: Box::new(e),
            })?;
        }

        let chunk_path = self.get_chunk_path(path, chunk_index);
        let mut file = tokio::fs::File::create(chunk_path)
            .await
            .map_err(|e| Error::Generic {
                store: "LocalCache",
                source: Box::new(e),
            })?;

        file.write_all(&data).await.map_err(|e| Error::Generic {
            store: "LocalCache",
            source: Box::new(e),
        })?;

        // This sync is necessary to ensure that an immediate read from cache will return the data
        file.sync_all().await.map_err(|e| Error::Generic {
            store: "LocalCache",
            source: Box::new(e),
        })?;

        Ok(())
    }

    /// Get range data from cached chunks
    fn get_range_from_cache_stream(
        &self,
        location: &Path,
        range: &Range<u64>,
    ) -> impl Stream<Item = Result<Bytes>> + Send + 'static {
        let this = self.clone();
        let location = location.clone();
        let range = range.clone();
        stream! {
            let chunks_needed = this.chunks_for_range(&range);
            for chunk_idx in chunks_needed {
                let chunk_path = this.get_chunk_path(&location, chunk_idx);
                let chunk_start = chunk_idx * CACHE_BLOCK_SIZE;
                let chunk_end = chunk_start + CACHE_BLOCK_SIZE;

                let overlap_start = std::cmp::max(chunk_start, range.start);
                let overlap_end = std::cmp::min(chunk_end, range.end);

                if overlap_start < overlap_end {
                    let offset_in_chunk = overlap_start - chunk_start;
                    let length = overlap_end - overlap_start;

                    yield this
                        .read_from_cached_chunk(chunk_path, offset_in_chunk, length as usize)
                        .await;
                }
            }
        }
    }
}

impl Display for LocalCache {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "LocalCache(cache_dir: {:?})", self.cache_dir)
    }
}

#[async_trait]
impl ObjectStore for LocalCache {
    fn list(&self, prefix: Option<&Path>) -> BoxStream<'static, Result<ObjectMeta>> {
        self.inner.list(prefix)
    }

    async fn list_with_delimiter(&self, prefix: Option<&Path>) -> Result<ListResult> {
        self.inner.list_with_delimiter(prefix).await
    }

    async fn get_opts(&self, location: &Path, options: GetOptions) -> Result<GetResult> {
        // Get the metadata first to know the file size
        let meta = self.inner.head(location).await?;
        let file_size = meta.size;

        // If this is just a HEAD request, return the metadata
        if options.head {
            return self.inner.get_opts(location, options).await;
        }

        // Determine the range we need to fetch
        let range = match &options.range {
            Some(GetRange::Bounded(range)) => range.clone(),
            Some(GetRange::Suffix(suffix)) => (file_size.saturating_sub(*suffix))..file_size,
            Some(GetRange::Offset(offset)) => *offset..file_size,
            None => 0..file_size,
        };

        // Calculate which chunks we need
        let chunks_needed = self.chunks_for_range(&range);

        // Check which chunks are already cached
        let mut missing_chunks = Vec::new();
        for chunk_idx in chunks_needed {
            let chunk_path = self.get_chunk_path(location, chunk_idx);
            if !chunk_path.exists() {
                missing_chunks.push(chunk_idx);
            }
        }

        // Fetch missing chunks from the underlying store
        for chunk_idx in missing_chunks {
            let chunk_start = chunk_idx * CACHE_BLOCK_SIZE;
            let chunk_end = std::cmp::min(chunk_start + CACHE_BLOCK_SIZE, file_size);

            let chunk_range = GetRange::Bounded(chunk_start..chunk_end);
            let chunk_options = GetOptions {
                range: Some(chunk_range),
                ..options.clone()
            };

            let chunk_result = self.inner.get_opts(location, chunk_options).await?;
            let chunk_data = chunk_result.bytes().await?;

            // Save the chunk to cache
            self.save_chunk(location, chunk_idx, chunk_data).await?;
        }

        // Return a GetResult with the stream of bytes from cache
        Ok(GetResult {
            payload: GetResultPayload::Stream(Box::pin(
                self.get_range_from_cache_stream(location, &range),
            )),
            meta,
            range,
            attributes: Default::default(),
        })
    }

    async fn put_opts(
        &self,
        _location: &Path,
        _payload: PutPayload,
        _opts: PutOptions,
    ) -> Result<PutResult> {
        unreachable!("LocalCache does not support put")
    }

    async fn put_multipart_opts(
        &self,
        _location: &Path,
        _opts: PutMultipartOpts,
    ) -> Result<Box<dyn MultipartUpload>> {
        unreachable!("LocalCache does not support multipart upload")
    }

    async fn delete(&self, _location: &Path) -> Result<()> {
        unreachable!("LocalCache does not support delete")
    }

    async fn copy(&self, _from: &Path, _to: &Path) -> Result<()> {
        unreachable!("LocalCache does not support copy")
    }

    async fn copy_if_not_exists(&self, _from: &Path, _to: &Path) -> Result<()> {
        unreachable!("LocalCache does not support copy_if_not_exists")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bytes::Bytes;
    use object_store::memory::InMemory;
    use std::ops::Range;
    use tempfile::tempdir;

    // Helper function to create a test file of specified size in the in-memory store
    async fn create_test_file(store: &InMemory, path: &str, size: u64) -> Result<()> {
        let data = vec![0u8; size as usize];
        // Fill the data with a pattern: index % 256
        // This makes it easy to verify ranges
        let data: Vec<u8> = data
            .iter()
            .enumerate()
            .map(|(i, _)| (i % 256) as u8)
            .collect();

        let path = Path::from(path);
        store.put(&path, Bytes::from(data).into()).await?;
        Ok(())
    }

    // Helper function to read a range from the store and verify it
    async fn verify_range(store: &dyn ObjectStore, path: &str, range: Range<u64>) -> Result<()> {
        let path = Path::from(path);
        let result = store.get_range(&path, range.clone()).await?;

        // Verify that the returned data matches the expected pattern
        for (i, byte) in result.iter().enumerate() {
            let expected = ((range.start + i as u64) % 256) as u8;
            assert_eq!(*byte, expected, "Mismatch at position {}", i);
        }

        Ok(())
    }

    // Test reading a small file (less than one chunk)
    #[tokio::test]
    async fn test_small_file() -> Result<()> {
        let inner = Arc::new(InMemory::new());
        let temp_dir = tempdir().unwrap();
        let cache = LocalCache::new(inner.clone(), temp_dir.path().to_path_buf());

        // Create a small file (10KB)
        let file_path = "small_file.bin";
        create_test_file(&inner, file_path, 10 * 1024).await?;

        // Read the entire file
        verify_range(&cache, file_path, 0..10 * 1024).await?;

        // Verify the file was cached
        let cache_dir = cache.get_cache_dir_for_path(&Path::from(file_path));
        assert!(cache_dir.exists(), "Cache directory should exist");

        let chunk_path = cache.get_chunk_path(&Path::from(file_path), 0);
        assert!(chunk_path.exists(), "Chunk file should exist");

        Ok(())
    }

    // Test reading a large file (multiple chunks)
    #[tokio::test]
    async fn test_large_file() -> Result<()> {
        let inner = Arc::new(InMemory::new());
        let temp_dir = tempdir().unwrap();
        let cache = LocalCache::new(inner.clone(), temp_dir.path().to_path_buf());

        // Create a file slightly larger than 2 chunks (9MB)
        let file_path = "large_file.bin";
        let file_size = CACHE_BLOCK_SIZE * 2 + 1024 * 1024; // 9MB
        create_test_file(&inner, file_path, file_size).await?;

        // Read the entire file
        verify_range(&cache, file_path, 0..file_size).await?;

        // Verify all chunks were cached
        for chunk_idx in 0..=2 {
            let chunk_path = cache.get_chunk_path(&Path::from(file_path), chunk_idx);
            assert!(chunk_path.exists(), "Chunk {} should exist", chunk_idx);
        }

        Ok(())
    }

    // Test reading a range within a single chunk
    #[tokio::test]
    async fn test_range_within_chunk() -> Result<()> {
        let inner = Arc::new(InMemory::new());
        let temp_dir = tempdir().unwrap();
        let cache = LocalCache::new(inner.clone(), temp_dir.path().to_path_buf());

        // Create a file larger than one chunk
        let file_path = "range_test.bin";
        let file_size = CACHE_BLOCK_SIZE * 3; // 12MB
        create_test_file(&inner, file_path, file_size).await?;

        // Read a range entirely within the second chunk
        let start = CACHE_BLOCK_SIZE + 1024;
        let end = CACHE_BLOCK_SIZE + 2048;
        verify_range(&cache, file_path, start..end).await?;

        // Verify only the requested chunk was cached
        let chunk1_path = cache.get_chunk_path(&Path::from(file_path), 1);
        assert!(chunk1_path.exists(), "Chunk 1 should exist");

        // Other chunks should not be cached yet
        let chunk0_path = cache.get_chunk_path(&Path::from(file_path), 0);
        let chunk2_path = cache.get_chunk_path(&Path::from(file_path), 2);
        assert!(!chunk0_path.exists(), "Chunk 0 should not exist yet");
        assert!(!chunk2_path.exists(), "Chunk 2 should not exist yet");

        Ok(())
    }

    // Test reading a range that spans multiple chunks
    #[tokio::test]
    async fn test_range_across_chunks() -> Result<()> {
        let inner = Arc::new(InMemory::new());
        let temp_dir = tempdir().unwrap();
        let cache = LocalCache::new(inner.clone(), temp_dir.path().to_path_buf());

        // Create a file larger than two chunks
        let file_path = "multi_chunk_range.bin";
        let file_size = CACHE_BLOCK_SIZE * 3; // 12MB
        create_test_file(&inner, file_path, file_size).await?;

        // Read a range that spans chunk 1 and chunk 2
        let start = CACHE_BLOCK_SIZE - 1024;
        let end = CACHE_BLOCK_SIZE * 2 + 1024;
        verify_range(&cache, file_path, start..end).await?;

        // Verify the chunks were cached
        let chunk0_path = cache.get_chunk_path(&Path::from(file_path), 0);
        let chunk1_path = cache.get_chunk_path(&Path::from(file_path), 1);
        let chunk2_path = cache.get_chunk_path(&Path::from(file_path), 2);
        assert!(chunk0_path.exists(), "Chunk 0 should exist");
        assert!(chunk1_path.exists(), "Chunk 1 should exist");
        assert!(chunk2_path.exists(), "Chunk 2 should exist");

        Ok(())
    }

    // Test cache hit (read the same file twice)
    #[tokio::test]
    async fn test_cache_hit() -> Result<()> {
        let inner = Arc::new(InMemory::new());
        let temp_dir = tempdir().unwrap();
        let cache = LocalCache::new(inner.clone(), temp_dir.path().to_path_buf());

        // Create a file
        let file_path = "cache_hit.bin";
        let file_size = CACHE_BLOCK_SIZE + 1024; // Slightly more than one chunk
        create_test_file(&inner, file_path, file_size).await?;

        // Read the file to populate the cache
        verify_range(&cache, file_path, 0..file_size).await?;

        // Modify the original file in the inner store to verify we're reading from cache
        let modified_data = vec![255u8; file_size as usize];
        let path = Path::from(file_path);
        inner.put(&path, Bytes::from(modified_data).into()).await?;

        // Read the same range again - should get the original data from cache, not the modified data
        verify_range(&cache, file_path, 0..file_size).await?;

        Ok(())
    }

    // Test partial range requests
    #[tokio::test]
    async fn test_suffix_range() -> Result<()> {
        let inner = Arc::new(InMemory::new());
        let temp_dir = tempdir().unwrap();
        let cache = LocalCache::new(inner.clone(), temp_dir.path().to_path_buf());

        // Create a file
        let file_path = "suffix_range.bin";
        let file_size = CACHE_BLOCK_SIZE * 2; // 8MB
        create_test_file(&inner, file_path, file_size).await?;

        // Request the last 1MB of the file using GetRange::Suffix
        let path = Path::from(file_path);
        let options = GetOptions {
            range: Some(GetRange::Suffix(1024 * 1024)),
            ..Default::default()
        };

        let result = cache.get_opts(&path, options).await?;
        let data = result.bytes().await?;

        // Verify we got the right data size
        assert_eq!(data.len(), 1024 * 1024);

        // Verify the content matches expected pattern
        let start = file_size - 1024 * 1024;
        for (i, byte) in data.iter().enumerate() {
            let expected = ((start + i as u64) % 256) as u8;
            assert_eq!(*byte, expected, "Mismatch at position {}", i);
        }

        Ok(())
    }

    #[tokio::test]
    async fn test_persistent_cache() -> Result<()> {
        // Create an InMemory store as the inner store
        let inner = Arc::new(InMemory::new());
        let temp_dir = tempdir().unwrap();
        let cache_dir_path = temp_dir.path().to_path_buf();

        // Create a file in the inner store
        let file_path = "persistent_test.bin";
        let file_size = CACHE_BLOCK_SIZE + 1024; // Slightly more than one chunk
        create_test_file(&inner, file_path, file_size).await?;

        // First cache instance - read data to populate cache
        {
            let first_cache = LocalCache::new(inner.clone(), cache_dir_path.clone());
            verify_range(&first_cache, file_path, 0..file_size).await?;

            // Verify data was cached
            let chunk_path = first_cache.get_chunk_path(&Path::from(file_path), 0);
            assert!(chunk_path.exists(), "First chunk should be cached");
        }

        // Modify the data in the inner store to verify the second cache uses cached data
        let modified_data = vec![255u8; file_size as usize];
        let path = Path::from(file_path);
        inner.put(&path, Bytes::from(modified_data).into()).await?;

        // Create a new cache instance pointing to the same directory
        let second_cache = LocalCache::new(inner.clone(), cache_dir_path);

        // Read the data through the second cache - should get original data from cache
        verify_range(&second_cache, file_path, 0..file_size).await?;

        // Additional verification - check if a specific range is read correctly
        let mid_range = file_size / 2;
        verify_range(&second_cache, file_path, mid_range..mid_range + 1024).await?;

        Ok(())
    }
}
