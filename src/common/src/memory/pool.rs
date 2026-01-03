extern crate io_uring;

use core::slice;
use std::{cmp::min, sync::{Arc, Mutex, OnceLock, atomic::{AtomicBool, AtomicU64, Ordering}}};

use futures::io;
use io_uring::IoUring;

use crate::memory::{arena::Arena, page::PAGE_SIZE, segment::Segment, tcache::{TCache, TCacheStats}};

static FIXED_BUFFER_POOL: OnceLock<FixedBufferPool> = OnceLock::new();

pub const FIXED_BUFFER_SIZE_BYTES: usize = 1 << 20;
pub const FIXED_BUFFER_BITS: u32 = FIXED_BUFFER_SIZE_BYTES.trailing_zeros();

#[derive(Debug)]
pub struct FixedBuffer {
    pub ptr: *mut u8,
    pub buf_id: usize,
    pub bytes: usize,
}

#[derive(Debug)]
pub struct FixedBufferAllocation {
    pub ptr: *mut u8,
    pub size: usize,
}

unsafe impl Send for FixedBufferAllocation {}

impl AsRef<[u8]> for FixedBufferAllocation {
    fn as_ref(&self) -> &[u8] {
        unsafe { slice::from_raw_parts(self.ptr, self.size) }
    }
}

impl Drop for FixedBufferAllocation {
    fn drop(&mut self) {
        FixedBufferPool::free(self.ptr);
    }
}

pub struct FixedBufferPool {
    local_caches: Vec<Mutex<TCache>>,
    arena: Arc<Mutex<Arena>>,
    start_ptr: *mut u8,
    capacity: usize,
    registered: AtomicBool,         // Whether buffers have been registered
    foreign_free: AtomicU64,
}

unsafe impl Send for FixedBufferPool {}

unsafe impl Sync for FixedBufferPool {}

impl FixedBufferPool {
    fn new(capacity_mb: usize) -> FixedBufferPool {
        log::info!("Initializing fixed buffer pool with capacity: {} MB", capacity_mb);
        let num_cpus = std::thread::available_parallelism().unwrap();
        let capacity = capacity_mb << 20;
        let arena = Self::allocate_arena(capacity.clone());
        let start_ptr = {
            let guard = arena.try_lock().unwrap();
            guard.start_ptr()
        };
        let mut local_caches = Vec::<Mutex<TCache>>::new();
        for i in 0..num_cpus.get() {
            local_caches.push(Mutex::new(TCache::new(arena.clone(), i)));
        }
        FixedBufferPool { 
            local_caches, 
            arena, 
            start_ptr, 
            capacity, 
            registered: AtomicBool::new(false),
            foreign_free: AtomicU64::new(0),
        }
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

    pub fn malloc(size: usize) -> *mut u8 {
        let cpu = unsafe { libc::sched_getcpu() };
        let local_cache = Self::get_thread_local_cache();
        let ptr = local_cache.lock().unwrap().allocate(size);
        log::debug!("Allocated pointer: {:?}, size: {}, cpu: {}", ptr, size, cpu);
        if ptr.is_null() {
            let pool = FIXED_BUFFER_POOL.get().unwrap();
            log::info!("Foreign frees: {}", pool.foreign_free.load(Ordering::Relaxed));
        }
        ptr
    }

    pub fn register_buffers_with_ring(ring: &IoUring) -> io::Result<()> {
        let pool = FIXED_BUFFER_POOL.get().unwrap();
        let mut arena_guard = pool.arena.lock().unwrap();
        let res = arena_guard.register_buffers_with_ring(ring);
        if res.is_ok() {
            log::info!("Registered buffers with io-uring ring");
            pool.registered.store(true, Ordering::Relaxed);
        }
        res
    }

    pub(crate) fn get_stats(cpu: usize) -> TCacheStats {
        let pool = FIXED_BUFFER_POOL.get().unwrap();
        let tcache = pool.local_caches[cpu].lock().unwrap();
        tcache.get_stats()
    }

    pub fn get_fixed_buffers(alloc: &FixedBufferAllocation) -> Vec<FixedBuffer> {
        let ptr = alloc.ptr;
        let size = alloc.size;
        let pool = FIXED_BUFFER_POOL.get().unwrap();
        debug_assert!(ptr >= pool.start_ptr && ptr < pool.start_ptr.wrapping_add(pool.capacity),
            "Pointer doesn't lie within the arena");
        let mut remaining = size;
        let mut vec = Vec::<FixedBuffer>::new();
        let mut current = ptr.clone();
        let mut buffer_id = (current.wrapping_sub(pool.start_ptr as usize) as usize) >> FIXED_BUFFER_BITS;
        while remaining > 0 {
            let next_buffer_start = pool.start_ptr.wrapping_add((buffer_id + 1) << FIXED_BUFFER_BITS);
            let bytes = min(remaining, next_buffer_start as usize - current as usize);
            let fb = FixedBuffer {
                ptr: current,
                buf_id: buffer_id,
                bytes: bytes,
            };
            current = next_buffer_start;
            vec.push(fb);
            remaining -= bytes;
            buffer_id += 1;
        }
        vec
    }

    #[inline]
    pub fn buffers_registered() -> bool {
        let pool = FIXED_BUFFER_POOL.get().unwrap();
        pool.registered.load(Ordering::Relaxed)
    }

    fn free(ptr: *mut u8) {
        let segment_ptr = Segment::get_segment_from_ptr(ptr);
        let page_ptr = unsafe { (*segment_ptr).get_page_from_ptr(ptr) };
        let thread_id = unsafe { (*segment_ptr).thread_id };
        log::debug!("Freed pointer: {:?}, size: {}, owner thread id: {}", ptr, unsafe { (*page_ptr).block_size }, thread_id);
        unsafe {
            (*page_ptr).free(ptr);
        }
        // If page is local and unused after free, return it to segment        
        let cur_cpu = unsafe { libc::sched_getcpu() as usize };
        if cur_cpu == thread_id {
            let should_free_page = unsafe { (*page_ptr).used.load(Ordering::Relaxed) == 0 };
            if should_free_page {
                let local_cache = Self::get_thread_local_cache();
                let mut guard = local_cache.lock().unwrap();
                guard.retire_page(page_ptr);
            }
        } else {
            log::debug!("Freeing from foreign thread");
            let pool = FIXED_BUFFER_POOL.get().unwrap();
            pool.foreign_free.fetch_add(1, Ordering::Relaxed);
        }
    }
}

impl Drop for FixedBufferPool {
    fn drop(self: &mut Self) {
        let arena = self.arena.lock().unwrap();
        drop(arena);
    }
}

mod tests {
    use std::{io::Write, os::fd::AsRawFd, ptr::{null, null_mut}};

    use bytes::Bytes;
    use io_uring::{IoUring, cqueue, opcode, squeue};
    use libc::rlimit;
    use rand::RngCore as _;

    use crate::memory::pool::{FIXED_BUFFER_SIZE_BYTES, FixedBufferAllocation, FixedBufferPool};

    #[test]
    fn test_basic_alloc_and_free() {
        FixedBufferPool::init(128);

        let buffer_lengths = [4096, 4096, 4096 * 4];       // 2 different size classes
        let mut ptrs = Vec::<*mut u8>::new();
        for len in buffer_lengths {
            let ptr = FixedBufferPool::malloc(len);
            assert_ne!(ptr, null_mut());
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
    fn test_basic_alloc_and_free_bytes() {
        FixedBufferPool::init(128);

        let buffer_lengths = [4096, 4096, 4096 * 4];       // 2 different size classes
        // let mut ptrs = Vec::<*mut u8>::new();
        let mut bytes_vec = Vec::<Bytes>::new();
        for len in buffer_lengths {
            let ptr = FixedBufferPool::malloc(len);
            assert_ne!(ptr, null_mut());
            // 4096 byte alignment is necessary for direct IO
            assert_eq!(ptr as usize % 4096, 0);

            let buffer = unsafe { std::slice::from_raw_parts_mut(ptr, len) };
            buffer[0] = 1;
            buffer[len-1] = 1;
            let alloc = FixedBufferAllocation {ptr: ptr, size: len};
            let bytes = Bytes::from_owner(alloc);
            bytes_vec.push(bytes);
        }

        drop(bytes_vec);

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
            let ptr = FixedBufferPool::malloc(len);
            assert_ne!(ptr, null_mut());
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
        let ptr = FixedBufferPool::malloc(len);
        assert_ne!(ptr, null_mut());
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
        let ptr = FixedBufferPool::malloc(len);
        assert_ne!(ptr, null_mut());
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
        let ptr = FixedBufferPool::malloc(len);

        assert_eq!(ptr, null_mut());
    }

    #[test]
    fn test_with_uring_basic() {
        let mut rlimit = libc::rlimit{
            rlim_cur: 0,
            rlim_max: 0,
        };
        unsafe { libc::getrlimit(libc::RLIMIT_MEMLOCK, &mut rlimit); }
        assert!(64 * 1024 <= rlimit.rlim_max, "rlimit.MEMLOCK should be at least 64 MB to test the fixed-buffer pool. Current rlimit is: {} KB", rlimit.rlim_max);
        FixedBufferPool::init(64);
        
        let mut ring = IoUring::<squeue::Entry, cqueue::Entry>::builder().build(32).unwrap();
        let res = FixedBufferPool::register_buffers_with_ring(&ring);
        assert!(res.is_ok());

        const LEN: usize = 1 << 20; // 1 MB
        let mut file = tempfile::tempfile().unwrap();
        let ptr = FixedBufferPool::malloc(LEN);
        assert_ne!(ptr, null_mut());
        let alloc = FixedBufferAllocation {ptr: ptr, size: LEN};
        let buffers = FixedBufferPool::get_fixed_buffers(&alloc);
        assert!(buffers.len() <= (LEN / FIXED_BUFFER_SIZE_BYTES) + 1);

        let mut total = 0;
        for fixed_buffer in buffers.iter().as_ref() {
            total += fixed_buffer.bytes;
        }
        assert_eq!(total, LEN);

        let mut random_bytes = [0u8; LEN];
        let mut rng = rand::rng();
        rng.fill_bytes(&mut random_bytes);
        let mut res = file.write(&random_bytes);
        assert!(res.is_ok(), "Failed to write to temp file");
        assert_eq!(res.unwrap(), LEN, "Failed to write to temp file");

        let mut file_offset = 0;
        for fixed_buffer in buffers.iter().as_ref() {
            let sqe = opcode::ReadFixed::new(
                io_uring::types::Fd(file.as_raw_fd()),
                fixed_buffer.ptr,
                fixed_buffer.bytes as u32,
                fixed_buffer.buf_id as u16)
                .offset(file_offset).build();
            file_offset += fixed_buffer.bytes as u64;
            let mut sq = ring.submission();
            let res = unsafe { sq.push(&sqe) };
            assert!(res.is_ok(), "Failed to submit to io uring");
            sq.sync();
        }

        res = ring.submit_and_wait(buffers.len());
        assert!(res.is_ok(), "Failed to submit");
        let mut total_bytes_read = 0;

        for _i in 0..buffers.len() {
            let mut cq = ring.completion();
            let cqe = cq.next();
            assert!(cqe.is_some());
            let res = cqe.as_ref().unwrap().result();
            assert!( res > 0, "Read failed: {}", std::io::Error::from_raw_os_error(-cqe.unwrap().result()));
            total_bytes_read += res as usize;
        }
        assert_eq!(total_bytes_read, LEN, "Expected to read {} bytes, but read {}", LEN, total_bytes_read);
        let buffer = Bytes::from_owner(alloc);
        assert_eq!(buffer, &random_bytes[..]);
    }

    #[test]
    fn test_edge_case() {
        FixedBufferPool::init(128);
        let len = 4 * 1024;
        let ptr1 = FixedBufferPool::malloc(len);
        let ptr2 = FixedBufferPool::malloc(len << 1);
        let ptr3 = FixedBufferPool::malloc(len << 2);
        let ptr4 = FixedBufferPool::malloc(len << 4);

        FixedBufferPool::free(ptr1);
        FixedBufferPool::free(ptr3);
        FixedBufferPool::free(ptr2);
        FixedBufferPool::free(ptr4);
        let cur_cpu = unsafe { libc::sched_getcpu() as usize };
        let stats = FixedBufferPool::get_stats(cur_cpu);

        assert_eq!(stats.allocations_from_arena, 1);
        assert_eq!(stats.pages_retired, 4);
        assert_eq!(stats.segments_retired, 1);
        // assert_eq
    }
}