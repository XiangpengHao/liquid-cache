use std::sync::atomic::{AtomicUsize, Ordering};

use log::warn;

#[derive(Debug)]
pub(crate) struct CacheConfig {
    batch_size: usize,
    max_cache_bytes: usize,
    remaining_memory_bytes: AtomicUsize,
    used_disk_bytes: AtomicUsize,
}

impl CacheConfig {
    pub fn new(batch_size: usize, max_cache_bytes: usize) -> Self {
        Self {
            batch_size,
            max_cache_bytes,
            remaining_memory_bytes: AtomicUsize::new(max_cache_bytes),
            used_disk_bytes: AtomicUsize::new(0),
        }
    }

    pub fn batch_size(&self) -> usize {
        self.batch_size
    }

    #[cfg(test)]
    pub fn memory_usage_bytes(&self) -> usize {
        self.max_cache_bytes - self.remaining_memory_bytes.load(Ordering::Relaxed)
    }

    pub fn update_usage_after_eviction(&self, evicted_size: usize) {
        self.remaining_memory_bytes
            .fetch_add(evicted_size, Ordering::Relaxed);
        self.used_disk_bytes
            .fetch_sub(evicted_size, Ordering::Relaxed);
    }

    /// Try to reserve space in the cache.
    /// Returns true if the space was reserved, false if the cache is full.
    pub fn try_reserve_memory(&self, request_bytes: usize) -> bool {
        let remaining = self.remaining_memory_bytes.load(Ordering::Relaxed);
        if remaining < request_bytes {
            return false;
        }

        match self.remaining_memory_bytes.compare_exchange(
            remaining,
            remaining - request_bytes,
            Ordering::Relaxed,
            Ordering::Relaxed,
        ) {
            Ok(_) => true,
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
            if (new_size - old_size) > 1024 * 1024 {
                warn!(
                    "Transcoding increased the size of the array by at least 1MB, previous size: {}, new size: {}, double check this is correct",
                    old_size, new_size
                );
            }

            if !self.try_reserve_memory(new_size - old_size) {
                self.remaining_memory_bytes
                    .fetch_add(old_size, Ordering::Relaxed);
                return Err(());
            }
            Ok(())
        } else {
            self.remaining_memory_bytes
                .fetch_add(old_size - new_size, Ordering::Relaxed);
            Ok(())
        }
    }

    pub fn reset_usage(&self) {
        self.remaining_memory_bytes
            .store(self.max_cache_bytes, Ordering::Relaxed);
        self.used_disk_bytes.store(0, Ordering::Relaxed);
    }
}
