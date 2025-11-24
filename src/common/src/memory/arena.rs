use std::{io, os::raw::c_void, ptr::{null, null_mut}};

use io_uring::IoUring;

use crate::memory::{
    page::Slice,
    segment::{SEGMENT_SIZE, SEGMENT_SIZE_BITS, Segment}, tcache::MIN_SIZE_FROM_PAGES,
};

const FIXED_BUFFERS_PER_SEGMENT: usize = SEGMENT_SIZE / MIN_SIZE_FROM_PAGES;

pub struct Arena {
    size: usize,
    slices: Vec<Slice>,
    used_bitmap: Vec<u8>,
    /**
     * Segments need to be aligned to 32MB boundaries. Hence the first segment's starting address
     * could be different from the starting address of the allocated memory
     */
    aligned_start_ptr: *mut u8,     
    actual_start_ptr: *mut u8,
    buffers_registered: bool,
}

unsafe impl Send for Arena {}
unsafe impl Sync for Arena {}

impl Arena {
    pub fn new(capacity: usize) -> Arena {
        let mem_start = Self::allocate_memory_from_os(capacity);
        assert_ne!(mem_start, null_mut());
        let mem_end = mem_start.wrapping_add(capacity);
        let ptr_aligned = (mem_start as usize >> SEGMENT_SIZE_BITS) << SEGMENT_SIZE_BITS;
        let mut slice_start = ptr_aligned;
        if ptr_aligned != (mem_start as usize) {
            slice_start = ptr_aligned + SEGMENT_SIZE;
        }
        let mut slices = Vec::new();
        while slice_start + SEGMENT_SIZE <= mem_end as usize {
            slices.push(Slice {
                ptr: slice_start as *mut u8,
                size: SEGMENT_SIZE,
            });
            slice_start += SEGMENT_SIZE;
        }
        let mut used_bitmap = Vec::new();
        used_bitmap.resize(slices.len(), 0);

        Arena {
            size: capacity,
            slices: slices,
            used_bitmap: used_bitmap,
            aligned_start_ptr: ptr_aligned as *mut u8,
            actual_start_ptr: mem_start,
            buffers_registered: false,
        }
    }

    fn allocate_memory_from_os(capacity: usize) -> *mut u8 {
        let prot = libc::PROT_READ | libc::PROT_WRITE;
        let flags = libc::MAP_ANONYMOUS | libc::MAP_PRIVATE;
        unsafe { libc::mmap64(null_mut(), capacity, prot, flags, -1, 0) as *mut u8 }
    }

    pub fn allocate_segment(self: &mut Self, size: usize) -> Option<*mut Segment> {
        let num_slices = (size + SEGMENT_SIZE - 1) / SEGMENT_SIZE;
        let mut contiguous = 0;
        let mut result: i32 = -1;

        for index in 0..self.used_bitmap.len() {
            let bit = self.used_bitmap[index];
            if bit == 0 {
                contiguous += 1;
                if contiguous == num_slices {
                    result = (index + 1 - contiguous) as i32;
                    break;
                }
            } else {
                contiguous = 0;
            }
        }
        if result == -1 {
            return None;
        }
        for i in 0..contiguous {
            self.used_bitmap[result as usize + i] = 1;
        }
        let combined_slice = Slice {
            ptr: self.slices[result as usize].ptr,
            size: num_slices * SEGMENT_SIZE,
        };
        let start_buffer_id = if self.buffers_registered {
            Some(result as usize * FIXED_BUFFERS_PER_SEGMENT)
        } else {
            None
        };
        Some(Segment::new_from_slice(combined_slice, start_buffer_id))
    }

    pub(crate) fn retire_segment(self: &mut Self, segment: *mut Segment) {
        debug_assert!((self.slices[0].ptr as usize) <= segment as usize);
        let segment_idx = (segment as usize - self.slices[0].ptr as usize) / SEGMENT_SIZE;
        self.used_bitmap[segment_idx] = 0;
    }

    pub(crate) fn register_buffers_with_ring(self: &mut Self, ring: &IoUring) -> io::Result<()> {
        let num_buffers = self.size / MIN_SIZE_FROM_PAGES;
        let mut buffers = Vec::<libc::iovec>::new();
        buffers.reserve(num_buffers);
        let mut base_ptr = self.aligned_start_ptr;
        for _i in 0..num_buffers {
            buffers.push(libc::iovec {iov_base: base_ptr as *mut std::ffi::c_void, iov_len: MIN_SIZE_FROM_PAGES});
            base_ptr = base_ptr.wrapping_add(MIN_SIZE_FROM_PAGES);
        }
        let res = unsafe {
            ring.submitter().register_buffers(&buffers)
        };
        self.buffers_registered = res.is_ok();
        res
    }    
}

impl Drop for Arena {
    fn drop(self: &mut Self) {
        unsafe {
            libc::munmap(self.actual_start_ptr as *mut c_void, self.size);
        }
    }
}