use std::{collections::VecDeque, ptr::null_mut};

use crate::memory::tcache::MIN_SIZE_FROM_PAGES;

#[derive(Clone, Copy)]
pub struct Block {
    ptr: *mut u8,
}

pub const PAGE_SIZE: usize = 256<<10;

pub struct Page {
    pub(crate) block_size: usize,                  // Size of objects that are being allocated to this page
    // TODO(): Remove dependency on dynamically allocated memory
    free_list: VecDeque<Block>,
    pub(crate) used: usize,
    pub(crate) thread_free_list: crossbeam::queue::ArrayQueue<Block>,
    pub(crate) capacity: usize,
    pub(crate) slice_count: usize,      // No. of pages in the slice containing this page
    pub(crate) slice_offset: usize,     // Offset of this page from the start of this slice
    pub(crate) page_start: *mut u8,
}

impl Page {
    pub fn from_slice(slice: Slice) -> Page {
        Page {
            block_size: 0usize,
            free_list: VecDeque::<Block>::with_capacity(PAGE_SIZE/MIN_SIZE_FROM_PAGES),
            used: 0,
            thread_free_list: crossbeam::queue::ArrayQueue::new(PAGE_SIZE/MIN_SIZE_FROM_PAGES),
            capacity: slice.size,
            slice_count: 1,
            slice_offset: 0,
            page_start: slice.ptr,
        }
    }

    pub fn set_block_size(self: &mut Self, block_size: usize) {
        self.block_size = block_size;
        let mut offset: usize = 0;
        self.free_list.clear();
        while offset < self.capacity {
            let ptr = unsafe { self.page_start.add(offset) };
            self.free_list.push_back(Block {ptr});
            offset += self.block_size;
        }
    }

    #[inline]
    pub fn get_free_block(self: &mut Self) -> *mut u8 {
        let block = self.free_list.pop_front();
        if block.is_none() {
            return null_mut()
        }
        self.used += 1;
        block.unwrap().ptr
    }

    #[inline(always)]
    pub fn is_full(self: &Self) -> bool {
        self.free_list.is_empty()
    }

    #[inline(always)]
    pub fn is_unused(self: &Self) -> bool {
        self.used == 0
    }

    #[inline(always)]
    pub fn get_size(self: &Self) -> usize {
        self.capacity
    }

    /// Pointer freed on the same core
    #[inline(always)]
    pub fn free(self: &mut Self, ptr: *mut u8) {
        self.free_list.push_back(Block {ptr});
        self.used -= 1;
    }

    /// Pointer freed on a different core
    #[inline(always)]
    pub(crate) fn foreign_free(self: &mut Self, ptr: *mut u8) {
        let blk = Block {ptr};
        let r = self.thread_free_list.push(blk);
        debug_assert!(r.is_ok());
    }

    /// Collect pointers freed by other threads
    #[inline]
    pub(crate) fn collect_foreign_frees(self: &mut Self) {
        while !self.thread_free_list.is_empty() {
            let blk = self.thread_free_list.pop().unwrap();
            self.free_list.push_back(blk);
            self.used -= 1;
        }
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