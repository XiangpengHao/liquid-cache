use std::{collections::VecDeque, ptr::null_mut, sync::{Mutex, atomic::{AtomicUsize, Ordering}}};

#[derive(Clone, Copy)]
pub struct Block {
    ptr: *mut u8,
}

pub const PAGE_SIZE: usize = 256<<10;

pub struct Page {
    pub(crate) block_size: usize,                  // Size of objects that are being allocated to this page
    // TODO(): Remove dependency on dynamically allocated memory
    free_list: Mutex<VecDeque<Block>>,
    pub(crate) used: AtomicUsize,
    // local_free_list: VecDeque<Block>,
    // thread_free_list: VecDeque<Block>,
    pub(crate) capacity: usize,
    pub(crate) slice_count: usize,      // No. of pages in the slice containing this page
    pub(crate) slice_offset: usize,     // Offset of this page from the start of this slice
    pub(crate) page_start: *mut u8,
}

impl Page {
    pub fn from_slice(slice: Slice) -> Page {
        // let mut start_ptr = slice.ptr;
        let free_list = VecDeque::<Block>::new();
        Page {
            block_size: 0usize,
            free_list: Mutex::new(free_list),
            used: AtomicUsize::new(0),
            // local_free_list: VecDeque::new(),
            // thread_free_list: VecDeque::new(),
            capacity: slice.size,
            slice_count: 1,
            slice_offset: 0,
            page_start: slice.ptr,
        }
    }

    pub fn set_block_size(self: &mut Self, block_size: usize) {
        self.block_size = block_size;
        let mut offset: usize = 0;
        let mut guard = self.free_list.lock().unwrap();
        while offset < self.capacity {
            let ptr = unsafe { self.page_start.add(offset) };
            guard.push_back(Block {ptr});
            offset += self.block_size;
        }
    }

    /**
     * Returns (block, buffer id) pair
     */
    #[inline]
    pub fn get_free_block(self: &mut Self) -> *mut u8 {
        let mut guard = self.free_list.lock().unwrap();
        let block = guard.pop_front();
        if block.is_none() {
            return null_mut()
        }
        self.used.fetch_add(1usize, Ordering::Relaxed);
        block.unwrap().ptr
    }

    #[inline]
    pub fn is_full(self: &Self) -> bool {
        let guard = self.free_list.lock().unwrap();
        guard.is_empty()
    }

    #[inline]
    pub fn get_size(self: &Self) -> usize {
        self.capacity
    }

    #[inline]
    pub fn free(self: &mut Self, ptr: *mut u8) {
        let blk = Block {ptr};
        let mut guard = self.free_list.lock().unwrap();
        guard.push_back(blk);
        self.used.fetch_sub(1usize, Ordering::Relaxed);
    }
}

pub struct Slice {
    pub ptr: *mut u8,
    pub size: usize,
}

impl Slice {
    pub fn split(self: Self) -> (Slice, Slice) {
        let new_size = self.size >> 1;
        let slice1 = Slice {
            ptr: self.ptr,
            size: new_size,
        };
        let slice2 = Slice {
            ptr: self.ptr.wrapping_add(new_size),
            size: new_size,
        };
        (slice1, slice2)
    }
}

// pub struct PageQueue {
//     page: *mut Page,
//     next: *mut PageQueue,
// }

// impl PageQueue {
//     pub fn new() -> PageQueue {
//         PageQueue { page: null_mut(), next: null_mut() }
//     }

//     pub(crate) fn get_page(self: &mut Self) -> Option<*mut Page> {
//         if self.page.is_null() {
//             return None;
//         }
//         let result = self.page;
//         self = *self.next;
//         Some(result)
//     }
// }