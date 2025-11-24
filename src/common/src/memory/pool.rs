extern crate io_uring;

use std::sync::{Arc, Mutex, OnceLock, atomic::Ordering};

use futures::io;
use io_uring::IoUring;

use crate::memory::{arena::Arena, segment::Segment, tcache::{TCache, TCacheStats}};

static FIXED_BUFFER_POOL: OnceLock<FixedBufferPool> = OnceLock::new();

pub struct FixedBufferPool {
    local_caches: Vec<Mutex<TCache>>,
    arena: Arc<Mutex<Arena>>,
}

impl FixedBufferPool {
    fn new(capacity_mb: usize) -> FixedBufferPool {
        let num_cpus = std::thread::available_parallelism().unwrap();
        let arena = Self::allocate_arena(capacity_mb << 20);
        let mut local_caches = Vec::<Mutex<TCache>>::new();
        for i in 0..num_cpus.get() {
            local_caches.push(Mutex::new(TCache::new(arena.clone(), i)));
        }
        FixedBufferPool { local_caches, arena }
    }

    pub fn allocate_arena(capacity: usize) -> Arc<Mutex<Arena>> {
        Arc::new(Mutex::new(Arena::new(capacity)))
    }

    pub fn init(capacity_mb: usize) {
        FIXED_BUFFER_POOL.get_or_init(|| FixedBufferPool::new(capacity_mb));
    }

    fn get_thread_local_cache() -> &'static Mutex<TCache> {
        let cpu = unsafe { libc::sched_getcpu() };
        &FIXED_BUFFER_POOL.get().unwrap().local_caches[cpu as usize]
    }

    pub fn malloc(size: usize) -> (*mut u8, Option<usize>) {
        let local_cache = Self::get_thread_local_cache();
        local_cache.lock().unwrap().allocate(size)
    }

    pub fn free(ptr: *mut u8) {
        let segment_ptr = Segment::get_segment_from_ptr(ptr);
        let page_ptr = unsafe { (*segment_ptr).get_page_from_ptr(ptr) };
        unsafe {
            (*page_ptr).free(ptr);
        }
        // If page is local and unused after free, return it to segment
        let thread_id = unsafe { (*segment_ptr).thread_id };
        let cur_cpu = unsafe { libc::sched_getcpu() as usize };
        if cur_cpu == thread_id {
            let should_free_page = unsafe { (*page_ptr).used.load(Ordering::Relaxed) == 0 };
            if should_free_page {
                let local_cache = Self::get_thread_local_cache();
                let mut guard = local_cache.lock().unwrap();
                guard.retire_page(page_ptr);
            }
        }
    }

    pub fn register_buffers_with_ring(ring: &IoUring) -> io::Result<()> {
        let pool = FIXED_BUFFER_POOL.get().unwrap();
        let mut arena_guard = pool.arena.lock().unwrap();
        arena_guard.register_buffers_with_ring(ring)
    }

    pub(crate) fn get_stats(cpu: usize) -> TCacheStats {
        let pool = FIXED_BUFFER_POOL.get().unwrap();
        let tcache = pool.local_caches[cpu].lock().unwrap();
        tcache.get_stats()
    }
}

impl Drop for FixedBufferPool {
    fn drop(self: &mut Self) {
        println!("Drop called");
        let arena = self.arena.lock().unwrap();
        drop(arena);
    }
}


mod tests {
    use std::ptr::null_mut;

    use crate::memory::pool::FixedBufferPool;

    #[test]
    fn test_basic_alloc_and_free() {
        FixedBufferPool::init(128);

        let buffer_lengths = [4096, 4096, 4096 * 4];       // 2 different size classes
        let mut ptrs = Vec::<*mut u8>::new();
        for len in buffer_lengths {
            let (ptr, fixed_buffer) = FixedBufferPool::malloc(len);
            assert_ne!(ptr, null_mut());
            assert_eq!(fixed_buffer, None);
            // 4096 byte alignment is necessary for direct IO
            assert_eq!(ptr as usize % 4096, 0);

            let buffer = unsafe { std::slice::from_raw_parts_mut(ptr, len) };
            buffer[0] = 1;
            buffer[len-1] = 1;
            ptrs.push(ptr);
        }

        for ptr in ptrs {
            FixedBufferPool::free(ptr);
        }

        let cur_cpu = unsafe { libc::sched_getcpu() as usize };
        let stats = FixedBufferPool::get_stats(cur_cpu);

        assert_eq!(stats.allocations_from_arena, 1);
        assert_eq!(stats.fast_allocations, 1);
        assert_eq!(stats.pages_retired, 2);
        assert_eq!(stats.segments_retired, 1);
    }

    #[test]
    fn test_free_from_different_thread() {
        FixedBufferPool::init(128);

        let buffer_lengths = [4096, 4096 * 4];
        let mut buffers = Vec::<&mut [u8]>::new();
        for len in buffer_lengths {
            let (ptr, fixed_buffer) = FixedBufferPool::malloc(len);
            assert_ne!(ptr, null_mut());
            assert_eq!(fixed_buffer, None);
            // 4096 byte alignment is necessary for direct IO
            assert_eq!(ptr as usize % 4096, 0);

            let buffer = unsafe { std::slice::from_raw_parts_mut(ptr, len) };
            buffer[0] = 1;
            buffer[len-1] = 1;
            buffers.push(buffer);
        }

        std::thread::spawn(move || {
            for buffer in buffers {
                let ptr = buffer.as_mut_ptr();
                FixedBufferPool::free(ptr);
            }
        });

        let cur_cpu = unsafe { libc::sched_getcpu() as usize };
        let stats = FixedBufferPool::get_stats(cur_cpu);
        assert_eq!(stats.allocations_from_arena, 1);
        assert_eq!(stats.allocations_from_segment, 1);
        assert_eq!(stats.fast_allocations, 0);
        assert_eq!(stats.pages_retired, 0);
        assert_eq!(stats.segments_retired, 0);
    }

    #[test]
    fn test_large_alloc_and_free() {
        FixedBufferPool::init(128);
        let len = 1024 * 1024;      // 1 MB
        let (ptr, fixed_buffer) = FixedBufferPool::malloc(len);
        assert_ne!(ptr, null_mut());
        assert_eq!(fixed_buffer, None);
        // 4096 byte alignment is necessary for direct IO
        assert_eq!(ptr as usize % 4096, 0);
        let buffer = unsafe { std::slice::from_raw_parts_mut(ptr, len) };
        buffer[0] = 1;
        buffer[len-1] = 1;
        FixedBufferPool::free(ptr);

        let cur_cpu = unsafe { libc::sched_getcpu() as usize };
        let stats = FixedBufferPool::get_stats(cur_cpu);

        assert_eq!(stats.allocations_from_arena, 1);
        assert_eq!(stats.pages_retired, 1);
        assert_eq!(stats.segments_retired, 1);
    }

    #[test]
    fn test_large_alloc_and_free2() {
        FixedBufferPool::init(128);
        let len = 3 * 1024 * 1024;      // 1 MB
        let (ptr, fixed_buffer) = FixedBufferPool::malloc(len);
        assert_ne!(ptr, null_mut());
        assert_eq!(fixed_buffer, None);
        // 4096 byte alignment is necessary for direct IO
        assert_eq!(ptr as usize % 4096, 0);
        let buffer = unsafe { std::slice::from_raw_parts_mut(ptr, len) };
        buffer[0] = 1;
        buffer[len-1] = 1;
        FixedBufferPool::free(ptr);

        let cur_cpu = unsafe { libc::sched_getcpu() as usize };
        let stats = FixedBufferPool::get_stats(cur_cpu);

        assert_eq!(stats.allocations_from_arena, 1);
        assert_eq!(stats.pages_retired, 1);
        assert_eq!(stats.segments_retired, 1);
    }

    #[test]
    fn test_very_large_alloc_fails() {
        FixedBufferPool::init(128);
        let len = 32 * 1024 * 1024;      // 32 MB
        let (ptr, _fixed_buffer) = FixedBufferPool::malloc(len);
        assert_eq!(ptr, null_mut());
    }
}