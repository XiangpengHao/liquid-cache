use super::observer::Observer;
use crate::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};

#[derive(Debug)]
pub struct BudgetAccounting {
    max_memory_bytes: usize,
    max_disk_bytes: usize,
    used_memory_bytes: AtomicUsize,
    used_disk_bytes: AtomicUsize,
    observer: Arc<Observer>,
}

impl BudgetAccounting {
    pub(super) fn new(
        max_memory_bytes: usize,
        max_disk_bytes: usize,
        observer: Arc<Observer>,
    ) -> Self {
        Self {
            max_memory_bytes,
            max_disk_bytes,
            used_memory_bytes: AtomicUsize::new(0),
            used_disk_bytes: AtomicUsize::new(0),
            observer,
        }
    }

    pub(super) fn reset_usage(&self) {
        self.used_memory_bytes.store(0, Ordering::Relaxed);
        self.used_disk_bytes.store(0, Ordering::Relaxed);
    }

    /// Try to reserve memory in the cache.
    /// Returns ok if the memory was reserved, err if the memory budget is full.
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

    /// Adjust memory usage after transcoding.
    /// Returns ok if the usage was adjusted, err if the memory budget is full when new_size is larger than old_size.
    pub(super) fn try_update_memory_usage(
        &self,
        old_size: usize,
        new_size: usize,
    ) -> Result<(), ()> {
        if old_size < new_size {
            let diff = new_size - old_size;
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

    pub(super) fn try_reserve_disk(&self, request_bytes: usize) -> Result<(), ()> {
        let used = self.used_disk_bytes.load(Ordering::Relaxed);
        if used + request_bytes > self.max_disk_bytes {
            self.observer.on_disk_reservation_failure();
            return Err(());
        }

        match self.used_disk_bytes.compare_exchange(
            used,
            used + request_bytes,
            Ordering::Relaxed,
            Ordering::Relaxed,
        ) {
            Ok(_) => Ok(()),
            Err(_) => self.try_reserve_disk(request_bytes),
        }
    }

    pub(super) fn release_disk(&self, bytes: usize) {
        self.used_disk_bytes.fetch_sub(bytes, Ordering::Relaxed);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sync::{Arc, Barrier, thread};

    fn test_budget(max_memory_bytes: usize, max_disk_bytes: usize) -> BudgetAccounting {
        BudgetAccounting::new(max_memory_bytes, max_disk_bytes, Arc::new(Observer::new()))
    }

    #[test]
    fn test_memory_reservation_and_accounting() {
        let config = test_budget(1000, usize::MAX);

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

        let budget = Arc::new(test_budget(max_memory, usize::MAX));
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

                    if i.is_multiple_of(5) && !successful_reservations.is_empty() {
                        let idx = i % successful_reservations.len();
                        let old_size = successful_reservations[idx];
                        let new_size = if i.is_multiple_of(2) {
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

    #[test]
    fn disk_reservation_and_release() {
        let budget = test_budget(usize::MAX, 1000);

        assert_eq!(budget.disk_usage_bytes(), 0);
        assert!(budget.try_reserve_disk(400).is_ok());
        assert_eq!(budget.disk_usage_bytes(), 400);
        assert!(budget.try_reserve_disk(600).is_ok());
        assert_eq!(budget.disk_usage_bytes(), 1000);
        assert!(budget.try_reserve_disk(1).is_err());
        assert_eq!(budget.disk_usage_bytes(), 1000);

        budget.release_disk(250);
        assert_eq!(budget.disk_usage_bytes(), 750);
        assert!(budget.try_reserve_disk(250).is_ok());
        assert_eq!(budget.disk_usage_bytes(), 1000);
    }
}
