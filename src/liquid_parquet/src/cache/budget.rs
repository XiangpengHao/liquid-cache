use crate::sync::atomic::{AtomicUsize, Ordering};

use log::warn;

#[derive(Debug)]
pub(super) struct BudgetAccounting {
    max_memory_bytes: usize,
    used_memory_bytes: AtomicUsize,
    used_disk_bytes: AtomicUsize,
}

impl BudgetAccounting {
    pub(super) fn new(max_memory_bytes: usize) -> Self {
        Self {
            max_memory_bytes,
            used_memory_bytes: AtomicUsize::new(0),
            used_disk_bytes: AtomicUsize::new(0),
        }
    }

    pub(super) fn reset_usage(&self) {
        self.used_memory_bytes.store(0, Ordering::Relaxed);
        self.used_disk_bytes.store(0, Ordering::Relaxed);
    }

    /// Try to reserve space in the cache.
    /// Returns ok if the space was reserved, err if the cache is full.
    pub(super) fn try_reserve_memory(&self, request_bytes: usize) -> Result<(), ()> {
        let used = self.used_memory_bytes.load(Ordering::Relaxed);
        if used + request_bytes > self.max_memory_bytes {
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
    pub(super) fn try_update_memory_usage(
        &self,
        old_size: usize,
        new_size: usize,
    ) -> Result<(), ()> {
        if old_size < new_size {
            let diff = new_size - old_size;
            if diff > 1024 * 1024 {
                warn!(
                    "Transcoding increased the size of the array by at least 1MB, previous size: {old_size}, new size: {new_size}, double check this is correct"
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

    pub fn memory_usage_bytes(&self) -> usize {
        self.used_memory_bytes.load(Ordering::Relaxed)
    }

    pub fn disk_usage_bytes(&self) -> usize {
        self.used_disk_bytes.load(Ordering::Relaxed)
    }

    pub fn add_used_disk_bytes(&self, bytes: usize) {
        self.used_disk_bytes.fetch_add(bytes, Ordering::Relaxed);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sync::{Arc, Barrier, thread};

    #[test]
    fn test_memory_reservation_and_accounting() {
        let config = BudgetAccounting::new(1000);

        assert_eq!(config.memory_usage_bytes(), 0);

        assert!(config.try_reserve_memory(500).is_ok());
        assert_eq!(config.memory_usage_bytes(), 500);

        assert!(config.try_reserve_memory(300).is_ok());
        assert_eq!(config.memory_usage_bytes(), 800);

        assert!(config.try_reserve_memory(300).is_err());
        assert_eq!(config.memory_usage_bytes(), 800);

        config.reset_usage();
        assert_eq!(config.memory_usage_bytes(), 0);
    }

    #[test]
    fn test_concurrent_memory_operations() {
        test_concurrent_memory_budget();
    }

    #[cfg(feature = "shuttle")]
    #[test]
    fn shuttle_memory_budget_operations() {
        crate::utils::shuttle_test(test_concurrent_memory_budget);
    }

    fn test_concurrent_memory_budget() {
        let num_threads = 3;
        let max_memory = 10000;
        let operations_per_thread = 100;

        let budget = Arc::new(BudgetAccounting::new(max_memory));
        let barrier = Arc::new(Barrier::new(num_threads));

        let mut thread_handles = vec![];

        for _ in 0..num_threads {
            let budget_clone = budget.clone();
            let barrier_clone = barrier.clone();

            let handle = thread::spawn(move || {
                let mut successful_reservations = Vec::new();

                barrier_clone.wait();

                for i in 0..operations_per_thread {
                    let reserve_size = 10 + (i % 20) * 5; // 10 to 105 bytes
                    if budget_clone.try_reserve_memory(reserve_size).is_ok() {
                        successful_reservations.push(reserve_size);
                    }

                    if i % 5 == 0 && !successful_reservations.is_empty() {
                        let idx = i % successful_reservations.len();
                        let old_size = successful_reservations[idx];
                        let new_size = if i % 2 == 0 {
                            old_size + 5 // Grow
                        } else {
                            old_size.saturating_sub(5) // Shrink
                        };

                        if budget_clone
                            .try_update_memory_usage(old_size, new_size)
                            .is_ok()
                        {
                            successful_reservations[idx] = new_size;
                        }
                    }
                }
                successful_reservations
            });

            thread_handles.push(handle);
        }

        let mut expected_memory_usage = 0;
        for handle in thread_handles {
            let reservations = handle.join().unwrap();
            for size in reservations {
                expected_memory_usage += size;
            }
        }

        assert_eq!(budget.memory_usage_bytes(), expected_memory_usage);
        assert!(budget.memory_usage_bytes() <= max_memory);
    }
}
