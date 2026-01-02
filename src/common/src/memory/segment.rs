use std::ptr::null_mut;

use crate::memory::{page::{PAGE_SIZE, Page, Slice}};

pub const SEGMENT_SIZE: usize = 32 * 1024 * 1024;
pub const SEGMENT_SIZE_BITS: usize = SEGMENT_SIZE.ilog2() as usize;

// The metadata is stored at the beginning of the slice. So we don't get the entirety of it for pages
pub const PAGES_PER_SEGMENT: usize = (SEGMENT_SIZE / PAGE_SIZE) - 1;

pub struct Segment {
    pub(crate) allocated: usize,
    pub(crate) num_slices: usize,
    pub(crate) pages: [Page; PAGES_PER_SEGMENT],
    pub(crate) thread_id: usize,
}

impl Segment {
    pub fn new_from_slice(slice: Slice) -> *mut Segment {
        // First sizeof(Segment) bytes should hold the Segment object
        let segment_ptr = slice.ptr as *mut Segment;
        let mut start_ptr = unsafe { slice.ptr.add(PAGE_SIZE) };
        
        unsafe {
            (*segment_ptr).allocated = 0;
            (*segment_ptr).num_slices = PAGES_PER_SEGMENT;
            for i in 0..(*segment_ptr).num_slices {
                (*segment_ptr).pages[i] = Page::from_slice(Slice {ptr: start_ptr, size: PAGE_SIZE});
                start_ptr = start_ptr.wrapping_add(PAGE_SIZE);
            }
        }
        segment_ptr
    }

    #[inline]
    pub fn full(self: &mut Self) -> bool {
        self.allocated == self.num_slices 
    }

    pub fn reset(self: &mut Self) -> () {
        for page in self.pages.iter_mut() {
            page.slice_count = 1;
            page.slice_offset = 0;
        }
    }

    pub fn get_segment_from_ptr(ptr: *mut u8) -> *mut Segment {
        let aligned_ptr = (ptr as usize >> SEGMENT_SIZE_BITS) << SEGMENT_SIZE_BITS;
        aligned_ptr as *mut Segment
    }

    pub fn get_page_from_ptr(self: &mut Self, ptr: *mut u8) -> *mut Page {
        let base_page_ptr = self.pages[0].page_start;
        debug_assert!(ptr >= base_page_ptr);
        let index = unsafe {
            ptr.sub(base_page_ptr as usize) as usize / PAGE_SIZE    
        };
        debug_assert!(index < PAGES_PER_SEGMENT);
        &mut self.pages[index] as *mut Page
    }

    /**
     * Split `page` into 2, with the first partition having `num_slices` pages.
     * Returns a pointer to the first page of the second slice.
     */
    pub fn split_page(self: &mut Self, page: *mut Page, num_slices: usize) -> *mut Page {
        debug_assert_ne!(page, null_mut());
        let base_page_ptr = unsafe { (*page).page_start };
        let base_segment_page_ptr = self.pages[0].page_start;
        debug_assert!(base_page_ptr >= base_segment_page_ptr);
        let index = unsafe {
            base_page_ptr.sub(base_segment_page_ptr as usize) as usize / PAGE_SIZE    
        };
        
        // Read original slice_count before modifying anything
        let original_slice_count = unsafe { (*page).slice_count };
        debug_assert!(num_slices > 0 && num_slices < original_slice_count, 
            "num_slices: {}, slice_count: {}", num_slices, original_slice_count);
        debug_assert!(index + original_slice_count <= PAGES_PER_SEGMENT);
        // log::info!("[thread_id: {}, segment_id: {}] Splitting page with {} slices", self.thread_id, self.segment_id, original_slice_count);
        
        /*
         * ASSUMPTION: Pointer to the beginning of the slice is passed in.
         * We don't need to modify all the intermediate pages while splitting. Only update the following:
         * - slice_offset for the first page of each slice (should be 0).
         * - slice_offset for the last page of each slice.
         * - slice_count for the first page of each slice.
         */
        // Use raw pointers to avoid borrow checker issues with multiple mutable references
        unsafe {
            // Update slice1: the original slice becomes the first part
            (*page).slice_offset = 0;
            (*page).slice_count = num_slices;
            
            let pages_ptr = self.pages.as_mut_ptr();
            let last_page_in_slice1 = pages_ptr.add(index + num_slices - 1);
            (*last_page_in_slice1).slice_offset = num_slices - 1;

            // Update slice2: the remaining pages become the second slice
            let slice2_count = original_slice_count - num_slices;
            let slice2 = pages_ptr.add(index + num_slices);
            (*slice2).slice_offset = 0;
            (*slice2).slice_count = slice2_count;
            assert!((*slice2).block_size == 0, "block size: {}", (*slice2).block_size);
            
            let last_page_in_slice2 = pages_ptr.add(index + original_slice_count - 1);
            (*last_page_in_slice2).slice_offset = slice2_count - 1;
            
            slice2
        }
    }

    pub fn coalesce_slices(self: &mut Self, left_slice: &mut Page, right_slice: &mut Page) {
        debug_assert!(left_slice.page_start >= self.pages[0].page_start && 
            left_slice.page_start <= self.pages[PAGES_PER_SEGMENT - 1].page_start);
        debug_assert!(right_slice.page_start >= self.pages[0].page_start && 
            right_slice.page_start <= self.pages[PAGES_PER_SEGMENT - 1].page_start);

        let left_slice_idx = (left_slice.page_start as usize - self.pages[0].page_start as usize) / PAGE_SIZE;
        let right_slice_idx = (right_slice.page_start as usize - self.pages[0].page_start as usize) / PAGE_SIZE;
        debug_assert!(left_slice_idx + left_slice.slice_count == right_slice_idx, 
            "left slice count: {}, left slice idx: {}, right slice idx: {}, thread_id: {}", 
            left_slice.slice_count, left_slice_idx, right_slice_idx, self.thread_id);
        debug_assert!(right_slice_idx + right_slice.slice_count <= PAGES_PER_SEGMENT);

        /*
         * ASSUMPTION: Pointer to the beginning of the slice is passed in free().
         * We don't need to modify all the intermediate pages while coalescing. Only update the following:
         * - slice_count for the first page of the combined slice (left_slice).
         * - slice_offset for the last page in the combined slice.
         * Note: right_slice becomes an intermediate page after merging, so we don't update its metadata.
         */
        left_slice.slice_offset = 0;
        left_slice.slice_count += right_slice.slice_count;

        let last_page = &mut self.pages[left_slice_idx + left_slice.slice_count - 1];
        last_page.slice_offset = left_slice.slice_count - 1;
    }

    pub fn check_valid_segment(self: &mut Self) {
        let mut idx = 0;
        while idx < PAGES_PER_SEGMENT {
            let page = &mut self.pages[idx];
            debug_assert!(page.slice_offset == 0 && idx + page.slice_count <= PAGES_PER_SEGMENT);
            let slice_count = page.slice_count;
            let last_page_in_slice = &mut self.pages[idx + slice_count - 1];
            debug_assert!(last_page_in_slice.slice_offset == slice_count - 1, 
                "slice count: {}, last page slice offset: {}, thread_id: {}", slice_count, last_page_in_slice.slice_offset, self.thread_id);
            idx += slice_count;
        }
    }
}