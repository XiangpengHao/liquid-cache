use std::{ptr::null_mut, sync::atomic::{AtomicU8, Ordering}, u8};

use crossbeam::utils::CachePadded;

use crate::memory::tcache::MIN_SIZE_FROM_PAGES;

pub const PAGE_SIZE: usize = 64<<10;    // 64KB
const MAX_BLOCKS_PER_PAGE: usize = PAGE_SIZE/MIN_SIZE_FROM_PAGES;

struct LocalFreeList {
    head: u8,
    tail: u8,
    num_blocks: u8,
    /**
     * Stores the block indices within the page for a compact representation, rather than storing pointers.
     * That is, if block index=i, it represents ith block from the start of the page.
     */
    blocks: [u8; MAX_BLOCKS_PER_PAGE],
}

impl LocalFreeList {
    fn empty() -> LocalFreeList {
        LocalFreeList {
            head: 0,
            tail: 0,
            num_blocks: 0,
            blocks: [0; MAX_BLOCKS_PER_PAGE],
        }
    }

    fn new(num_blocks: usize) -> LocalFreeList {
        debug_assert!(num_blocks <= MAX_BLOCKS_PER_PAGE);
        let mut blocks = [0u8; MAX_BLOCKS_PER_PAGE];
        for i in 0..num_blocks {
            blocks[i] = i as u8;
        }
        LocalFreeList { head: 0, tail: num_blocks as u8, num_blocks: num_blocks as u8, blocks: blocks }
    }

    fn push(&mut self, block: u8) {
        debug_assert!(self.tail.wrapping_sub(self.head) < self.num_blocks);
        self.blocks[self.tail as usize & (MAX_BLOCKS_PER_PAGE - 1)] = block;
        self.tail = self.tail.wrapping_add(1);
    }

    fn is_empty(&self) -> bool {
        self.head == self.tail
    }

    fn pop(&mut self) -> Option<u8> {
        if self.head == self.tail {
            return None
        }
        let ret = self.blocks[self.head as usize & (MAX_BLOCKS_PER_PAGE - 1)];
        self.head = self.head.wrapping_add(1);
        Some(ret)
    }
}

struct MPSCQueue {
    head: u8,
    tail: CachePadded<AtomicU8>,
    num_blocks: u8,
    blocks: [u8; MAX_BLOCKS_PER_PAGE],
}

impl MPSCQueue {
    const HAZARD: u8 = u8::MAX;

    fn new(num_blocks: usize) -> MPSCQueue {
        debug_assert!(num_blocks <= MAX_BLOCKS_PER_PAGE);
        MPSCQueue {
            head: 0,
            num_blocks: num_blocks as u8,
            tail: CachePadded::new(AtomicU8::new(0)),
            blocks: [Self::HAZARD; MAX_BLOCKS_PER_PAGE],
        }
    }

    fn push(&mut self, block: u8) {
        loop {
            let cur_tail = self.tail.load(Ordering::Relaxed);
            assert!(cur_tail.wrapping_sub(self.head) < self.num_blocks);
            let new_tail = cur_tail.wrapping_add(1);
            if self.tail.compare_exchange(cur_tail, new_tail, Ordering::Relaxed, Ordering::Relaxed).is_ok() {
                unsafe {
                    std::ptr::write_volatile(&mut self.blocks[cur_tail as usize & (MAX_BLOCKS_PER_PAGE - 1)] as *mut u8, block);
                }
                return
            }
        }
    }

    fn pop(&mut self) -> Option<u8> {
        if self.head == self.tail.load(Ordering::Relaxed) {
            return None
        }
        let idx = self.head as usize & (MAX_BLOCKS_PER_PAGE - 1);
        loop {
            let ret = unsafe { std::ptr::read_volatile(&self.blocks[idx] as *const u8) };
            /*
             * The hazard value prevents the following race condition:
             * The producer has reserved a slot, but before it can write to the slot, the consumer calls pop.
             */
            if ret != Self::HAZARD {
                unsafe {
                    std::ptr::write_volatile(&mut self.blocks[idx] as *mut u8, Self::HAZARD);
                }
                self.head = self.head.wrapping_add(1);
                return Some(ret);
            }
        }
    }
}

pub struct Page {
    pub(crate) block_size: usize,                  // Size of objects that are being allocated to this page
    free_list: LocalFreeList,
    pub(crate) used: usize,
    thread_free_list: MPSCQueue,
    pub(crate) capacity: usize,
    pub(crate) slice_count: usize,      // No. of pages in the slice containing this page
    pub(crate) slice_offset: usize,     // Offset of this page from the start of this slice
    pub(crate) page_start: *mut u8,
    // Next and previous pages in the span which is a doubly-linked list  
    pub(crate) next_page: *mut Page,
    pub(crate) previous_page: *mut Page,
}

impl Page {
    pub fn from_slice(slice: Slice) -> Page {
        Page {
            block_size: 0usize,
            free_list: LocalFreeList::empty(),
            used: 0,
            thread_free_list: MPSCQueue::new(PAGE_SIZE/MIN_SIZE_FROM_PAGES),
            capacity: slice.size,
            slice_count: 1,
            slice_offset: 0,
            page_start: slice.ptr,
            next_page: null_mut(),
            previous_page: null_mut(),
        }
    }

    pub fn set_block_size(self: &mut Self, block_size: usize) {
        self.block_size = block_size;
        let num_blocks = (self.slice_count * PAGE_SIZE) / block_size;
        self.free_list = LocalFreeList::new(num_blocks);
    }

    #[inline]
    pub fn get_free_block(self: &mut Self) -> *mut u8 {
        let block_idx = self.free_list.pop();
        let block_idx = match block_idx {
            Some(i) => i,
            None => return null_mut(),
        };
        self.used += 1;
        unsafe { self.page_start.add(block_idx as usize * self.block_size) }
    }

    #[inline(always)]
    pub fn is_full(self: &Self) -> bool {
        self.free_list.is_empty()
    }

    #[inline(always)]
    pub fn is_unused(self: &Self) -> bool {
        self.used == 0
    }

    /// Pointer freed on the same core
    #[inline(always)]
    pub fn free(self: &mut Self, ptr: *mut u8) {
        let block_idx = (ptr as usize - self.page_start as usize) / self.block_size;
        self.free_list.push(block_idx as u8);
        self.used -= 1;
    }

    /// Pointer freed on a different core
    #[inline(always)]
    pub(crate) fn foreign_free(self: &mut Self, ptr: *mut u8) {
        let blk_idx = unsafe {ptr.offset_from(self.page_start) as usize / self.block_size};
        self.thread_free_list.push(blk_idx as u8);
    }

    /// Collect pointers freed by other threads
    #[inline]
    pub(crate) fn collect_foreign_frees(self: &mut Self) {
        while let Some(blk) = self.thread_free_list.pop() {
            self.free_list.push(blk as u8);
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