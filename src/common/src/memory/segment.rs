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

    // pub fn try_allocate_page(self: &mut Self, page_size: usize) -> Slice {
    //     let min_bin = page_size / PAGE_SIZE;
    //     for i in min_bin..NUM_SPANS {
    //         let slice_opt = self.spans[i].pop_front();
    //         if slice_opt.is_none() {
    //             continue;
    //         }
    //         let mut slice = slice_opt.unwrap();
    //         let mut j = i;
    //         while j > min_bin &&  slice.size >= 2 * page_size {
    //             // split slice
    //             let (slice1, slice2) = slice.split();
    //             self.spans[i-1].push_back(slice2);
    //             slice = slice1;
    //             j -= 1;
    //         }
    //         self.allocated += slice.size;
    //         return slice;
    //     }
    //     // Allocate from arena

    //     Slice {ptr: null_mut(), size: 0}
    // }

    pub fn reset(self: &mut Self) -> () {
        for page in self.pages.iter_mut() {
            page.slice_count = 0;
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
     * Split `page` into 2, with the first partition having `num_slices` slices
     */
    pub fn split_page(self: &mut Self, page: *mut Page, num_slices: usize) -> *mut Page {
        debug_assert_ne!(page, null_mut());
        let page_ref = unsafe {&mut (*page)};
        let base_page_ptr = self.pages[0].page_start;
        debug_assert!(page_ref.page_start >= base_page_ptr);
        let index = unsafe {
            page_ref.page_start.sub(base_page_ptr as usize) as usize / PAGE_SIZE    
        };
        debug_assert!(index + num_slices < PAGES_PER_SEGMENT);
        /*
         * ASSUMPTION: Pointer to the beginning of the slice is passed in free().
         * We don't need to modify all the intermediate pages while splitting. Only update the following:
         * - slice_offsets for last pages in each slice.
         * - slice_count for the first pages in each slice.
         */
        let last_page_in_slice1 = &mut self.pages[index + num_slices - 1];
        last_page_in_slice1.slice_offset = num_slices - 1;

        let last_page_in_slice2 = &mut self.pages[index + page_ref.slice_count - 1];
        last_page_in_slice2.slice_offset = page_ref.slice_count - num_slices - 1;

        let slice2 = &mut self.pages[index + num_slices];
        slice2.slice_offset = 0;
        slice2.slice_count = page_ref.slice_count - num_slices;
        page_ref.slice_count = num_slices;
        slice2 as *mut Page
    }

    pub fn coalesce_slices(self: &mut Self, left_slice: &mut Page, right_slice: &mut Page) {
        debug_assert!(left_slice.page_start >= self.pages[0].page_start && 
            left_slice.page_start <= self.pages[PAGES_PER_SEGMENT - 1].page_start);
        debug_assert!(right_slice.page_start >= self.pages[0].page_start && 
            right_slice.page_start <= self.pages[PAGES_PER_SEGMENT - 1].page_start);

        let left_slice_idx = (left_slice.page_start as usize - self.pages[0].page_start as usize) / PAGE_SIZE;

        /*
         * ASSUMPTION: Pointer to the beginning of the slice is passed in free().
         * We don't need to modify all the intermediate pages while splitting. Only update the following:
         * - slice_count for the first pages in combined slice.
         * - slice_offset for the last page in the combined slice.
         */
        left_slice.slice_offset = 0;
        right_slice.slice_offset = left_slice.slice_count;
        left_slice.slice_count += right_slice.slice_count;

        let last_page = &mut self.pages[left_slice_idx + left_slice.slice_count - 1];
        last_page.slice_offset = left_slice.slice_count - 1;
        
    }

    pub fn debug_print(self: &mut Self) {
        log::info!("------Segment debug print--------");
        let mut idx = 0;
        while idx < PAGES_PER_SEGMENT {
            let page = &mut self.pages[idx];
            log::info!("Page {}: slice_count: {}, slice_offset: {}, block_size: {}", idx, page.slice_count, page.slice_offset, page.block_size);
            idx += page.slice_count;
        }
        log::info!("------end--------");
    }
}