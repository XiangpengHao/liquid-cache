use std::ptr::null_mut;

use crate::memory::{page::{PAGE_SIZE, Page, Slice}, tcache::MIN_SIZE_FROM_PAGES};

pub const SEGMENT_SIZE: usize = 32 * 1024 * 1024;
pub const SEGMENT_SIZE_BITS: usize = SEGMENT_SIZE.ilog2() as usize;

pub const PAGES_PER_SEGMENT: usize = SEGMENT_SIZE / PAGE_SIZE;
const FIXED_BUFFERS_PER_PAGE: usize = PAGE_SIZE / MIN_SIZE_FROM_PAGES;

pub struct Segment {
    pub(crate) allocated: usize,
    pub(crate) num_slices: usize,
    pub(crate) pages: [Page; PAGES_PER_SEGMENT - 1],
    pub(crate) thread_id: usize,
    pub(crate) start_buffer_id: Option<usize>,
}

impl Segment {
    pub fn new_from_slice(slice: Slice, start_buffer_id: Option<usize>) -> *mut Segment {
        // First sizeof(Segment) bytes should hold the Segment object
        let segment_ptr = slice.ptr as *mut Segment;
        let mut start_ptr = unsafe { slice.ptr.add(PAGE_SIZE) };
        
        unsafe {
            (*segment_ptr).allocated = 0;
            (*segment_ptr).num_slices = PAGES_PER_SEGMENT - 1;
            for i in 0..(*segment_ptr).num_slices {
                let page_start_buffer_id = if start_buffer_id.is_some() {
                    let offset = (start_ptr as usize - segment_ptr as usize) / MIN_SIZE_FROM_PAGES;
                    Some(start_buffer_id.unwrap() + offset)
                } else {
                    start_buffer_id
                };
                (*segment_ptr).pages[i] = Page::from_slice(Slice {ptr: start_ptr, size: PAGE_SIZE}, page_start_buffer_id);
                start_ptr = start_ptr.wrapping_add(PAGE_SIZE);
            }
            (*segment_ptr).start_buffer_id = start_buffer_id;
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

    pub fn retire(self: &mut Self) -> () {
        todo!()
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
        debug_assert!(index < PAGES_PER_SEGMENT - 1);
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
        debug_assert!(index < PAGES_PER_SEGMENT - 1);
        let next_slice = &mut self.pages[index + num_slices];
        next_slice.slice_offset = 0;
        next_slice.slice_count = page_ref.slice_count - num_slices;
        page_ref.slice_count = num_slices;
        next_slice as *mut Page
    }
}