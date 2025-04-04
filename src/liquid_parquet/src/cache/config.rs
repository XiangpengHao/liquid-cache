use std::{
    path::PathBuf,
    sync::atomic::{AtomicUsize, Ordering},
};

use log::warn;

#[derive(Debug)]
pub(crate) struct CacheConfig {
    batch_size: usize,
    max_cache_bytes: usize,
    used_memory_bytes: AtomicUsize,
    used_disk_bytes: AtomicUsize,
    cache_dir: PathBuf,
}

impl CacheConfig {
    pub fn new(batch_size: usize, max_cache_bytes: usize, cache_dir: PathBuf) -> Self {
        Self {
            batch_size,
            max_cache_bytes,
            used_memory_bytes: AtomicUsize::new(0),
            used_disk_bytes: AtomicUsize::new(0),
            cache_dir,
        }
    }

    pub fn batch_size(&self) -> usize {
        self.batch_size
    }

    pub fn max_cache_bytes(&self) -> usize {
        self.max_cache_bytes
    }

    pub fn memory_usage_bytes(&self) -> usize {
        self.used_memory_bytes.load(Ordering::Relaxed)
    }

    pub fn disk_usage_bytes(&self) -> usize {
        self.used_disk_bytes.load(Ordering::Relaxed)
    }

    #[allow(unused)]
    pub fn update_usage_after_eviction(&self, evicted_size: usize) {
        self.used_memory_bytes
            .fetch_sub(evicted_size, Ordering::Relaxed);
        self.used_disk_bytes
            .fetch_add(evicted_size, Ordering::Relaxed);
    }

    pub fn add_used_disk_bytes(&self, bytes: usize) {
        self.used_disk_bytes.fetch_add(bytes, Ordering::Relaxed);
    }

    /// Try to reserve space in the cache.
    /// Returns true if the space was reserved, false if the cache is full.
    pub fn try_reserve_memory(&self, request_bytes: usize) -> Result<(), ()> {
        let used = self.used_memory_bytes.load(Ordering::Relaxed);
        if used + request_bytes > self.max_cache_bytes {
            return Err(());
        }

        match self.used_memory_bytes.compare_exchange(
            used,
            used + request_bytes,
            Ordering::Relaxed,
            Ordering::Relaxed,
        ) {
            Ok(_) => Ok(()),
            Err(_) => self.try_reserve_memory(request_bytes),
        }
    }

    /// Adjust the cache size after transcoding.
    /// Returns true if the size was adjusted, false if the cache is full, when new_size is larger than old_size.
    pub fn try_update_memory_usage_after_transcoding(
        &self,
        old_size: usize,
        new_size: usize,
    ) -> Result<(), ()> {
        if old_size < new_size {
            let diff = new_size - old_size;
            if diff > 1024 * 1024 {
                warn!(
                    "Transcoding increased the size of the array by at least 1MB, previous size: {}, new size: {}, double check this is correct",
                    old_size, new_size
                );
            }

            self.try_reserve_memory(diff)?;
            Ok(())
        } else {
            self.used_memory_bytes
                .fetch_sub(old_size - new_size, Ordering::Relaxed);
            Ok(())
        }
    }

    pub fn reset_usage(&self) {
        self.used_memory_bytes.store(0, Ordering::Relaxed);
        self.used_disk_bytes.store(0, Ordering::Relaxed);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_reservation_and_accounting() {
        let config = CacheConfig::new(100, 1000);

        // Check initial state
        assert_eq!(config.memory_usage_bytes(), 0);

        // Basic reservation
        assert!(config.try_reserve_memory(500).is_ok());
        assert_eq!(config.memory_usage_bytes(), 500);

        // Additional reservation within limits
        assert!(config.try_reserve_memory(300).is_ok());
        assert_eq!(config.memory_usage_bytes(), 800);

        // Reservation exceeding limit
        assert!(config.try_reserve_memory(300).is_err());
        assert_eq!(config.memory_usage_bytes(), 800);

        // Test eviction accounting
        config.update_usage_after_eviction(200);
        assert_eq!(config.memory_usage_bytes(), 600);

        // Reset usage
        config.reset_usage();
        assert_eq!(config.memory_usage_bytes(), 0);
    }

    #[test]
    fn test_memory_transcoding_accounting() {
        let config = CacheConfig::new(100, 1000);

        // Initial reservation
        assert!(config.try_reserve_memory(400).is_ok());
        assert_eq!(config.memory_usage_bytes(), 400);

        // Update after transcoding - size decrease
        assert!(
            config
                .try_update_memory_usage_after_transcoding(300, 200)
                .is_ok()
        );
        assert_eq!(config.memory_usage_bytes(), 300); // 400 - (300 - 200)

        // Update after transcoding - size increase within limits
        assert!(
            config
                .try_update_memory_usage_after_transcoding(200, 500)
                .is_ok()
        );
        assert_eq!(config.memory_usage_bytes(), 600); // 300 + (500 - 200)

        // Update after transcoding - size increase exceeding limits
        assert!(
            config
                .try_update_memory_usage_after_transcoding(100, 600)
                .is_err()
        );
        assert_eq!(config.memory_usage_bytes(), 600); // Unchanged because update failed
    }
}
