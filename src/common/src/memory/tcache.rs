use std::{
    ptr::null_mut,
    sync::{Arc, Mutex},
};

use crate::memory::{
    arena::Arena,
    page::{PAGE_SIZE, Page},
    segment::{PAGES_PER_SEGMENT, SEGMENT_SIZE, Segment},
};

const SIZE_CLASSES: &'static [usize] = &[
    4 << 10,
    8 << 10,
    16 << 10,
    32 << 10,
    64 << 10,
    128 << 10,
    256 << 10,
];

const NUM_SIZE_CLASSES: usize = SIZE_CLASSES.len();

pub(crate) const MIN_SIZE_FROM_PAGES: usize = SIZE_CLASSES[0];

const SEGMENT_BINS: usize = (SEGMENT_SIZE/PAGE_SIZE).ilog2() as usize + 1;

#[derive(Default, Clone)]
pub(crate) struct TCacheStats {
    // Allocation stats
    pub(crate) total_allocations: usize,
    pub(crate) unsuccessful_allocations: usize,
    pub(crate) total_segments_allocated: usize,
    pub(crate) fast_allocations: usize,             // Allocations from self.free_pages
    pub(crate) allocations_from_pages: usize,       // Allocations from self.used_pages
    pub(crate) allocations_from_segment: usize,
    pub(crate) allocations_from_arena: usize,

    // Deallocation stats
    pub(crate) pages_retired: usize,
    pub(crate) segments_retired: usize,
    // TODO(): Add more stats such as number of local frees and frees from another thread
}

impl TCacheStats {
    pub(crate) fn new() -> TCacheStats {
        TCacheStats::default()
    }

    #[allow(unused)]
    pub(crate) fn print(self: &Self) {
        println!("Total allocations: {}", self.total_allocations);
        println!("Unsuccessful allocations: {}", self.unsuccessful_allocations);
        println!("Fast allocations: {}", self.fast_allocations);
        println!("Allocations from pages: {}", self.allocations_from_pages);
        println!("Allocations from segment: {}", self.allocations_from_segment);
        println!("Allocations from arena: {}", self.allocations_from_arena);
        println!("Pages retired: {}", self.pages_retired);
        println!("Segments retired: {}", self.segments_retired);
    }
}

pub(crate) struct TCache {
    free_pages: [*mut Page; NUM_SIZE_CLASSES],
    // Last size class holds slices that serve large allocations (>256KB)
    used_pages: [Vec<*mut Page>; NUM_SIZE_CLASSES + 1],
    // TODO: Use a linked list for O(1) deletion
    spans: [Vec<*mut Page>; SEGMENT_BINS],
    arena: Arc<Mutex<Arena>>,
    thread_id: usize,
    stats: TCacheStats,
}

unsafe impl Send for TCache {}
unsafe impl Sync for TCache {}

impl TCache {
    pub(crate) fn new(arena: Arc<Mutex<Arena>>, thread_id: usize) -> TCache {
        TCache {
            free_pages: [const { null_mut() }; NUM_SIZE_CLASSES],
            used_pages: [const { Vec::<*mut Page>::new() }; NUM_SIZE_CLASSES + 1],
            spans: [const { Vec::<*mut Page>::new() }; SEGMENT_BINS],
            arena: arena.clone(),
            thread_id,
            stats: TCacheStats::new(),
        }
    }

    #[inline]
    fn get_size_class(size: usize) -> usize {
        if size <= MIN_SIZE_FROM_PAGES {
            return 0;
        }
        (size.next_power_of_two() / MIN_SIZE_FROM_PAGES).trailing_zeros() as usize
    }

    /**
     * Get the smallest bin which can hold contiguous runs of `slice_count` pages
     */
    #[inline]
    fn get_span_idx_from_slice_count(slice_count: usize) -> usize {
        (slice_count + 1).next_power_of_two().trailing_zeros() as usize - 1usize
    }

    // TODO(): Use per-page pointers to speed up the removal
    fn remove_slice_from_span(self: &mut Self, slice: &mut Page) -> bool {
        let span_idx = Self::get_span_idx_from_slice_count(slice.slice_count);
        for i in 0..self.spans[span_idx].len() {
            let page_start = unsafe { (*self.spans[span_idx][i]).page_start };
            if page_start == slice.page_start {
                self.spans[span_idx].remove(i);
                return true;
            }
        }
        log::info!("[thread_id: {}] Slice not found in span with index: {}, slice count: {}", self.thread_id, span_idx, slice.slice_count);
        return false;
    }

    fn retire_segment(self: &mut Self, segment: *mut Segment) {
        // log::info!("Retiring segment from thread with id: {}", self.thread_id);
        unsafe { (*segment).check_valid_segment(); }
        self.stats.segments_retired += 1;
        let pages = unsafe { &mut (*segment).pages };
        let mut slice_idx: usize = 0;
        while slice_idx < PAGES_PER_SEGMENT {
            assert!(pages[slice_idx].block_size != 0 || self.remove_slice_from_span(&mut pages[slice_idx]));
            slice_idx += pages[slice_idx].slice_count;
        }
        let mut guard = self.arena.lock().unwrap();
        guard.retire_segment(segment);
    }

    fn remove_page_from_used_queue(self: &mut Self, page_ptr: *mut Page) {
        let mut size_class = Self::get_size_class(unsafe { (*page_ptr).block_size });
        if size_class >= NUM_SIZE_CLASSES {
            size_class = NUM_SIZE_CLASSES;
        }
        for i in 0..self.used_pages[size_class].len() {
            if self.used_pages[size_class][i] == page_ptr {
                self.used_pages[size_class].remove(i);
                return
            }
        }
    }

    fn remove_page_from_free_queue(self: &mut Self, page_ptr: *mut Page) {
        let size_class = Self::get_size_class(unsafe { (*page_ptr).block_size });
        if size_class < NUM_SIZE_CLASSES && self.free_pages[size_class] == page_ptr {
            self.free_pages[size_class] = null_mut();
        }
    }

    pub(crate) fn retire_page(self: &mut Self, page: *mut Page) {
        assert!(unsafe { (*page).is_unused() });
        self.stats.pages_retired += 1;
        self.remove_page_from_used_queue(page);
        self.remove_page_from_free_queue(page);
        let page_ref = unsafe { &mut (*page) };

        let segment_ptr = Segment::get_segment_from_ptr(page as *mut u8);
        let segment = unsafe { &mut *segment_ptr };
        segment.allocated -= page_ref.slice_count;
        if segment.allocated == 0 {
            // Return segment to arena
            self.retire_segment(segment_ptr);
            return;
        }
        page_ref.block_size = 0;

        let next_slice = page.wrapping_add(page_ref.slice_count);
        if next_slice <= (&mut segment.pages[PAGES_PER_SEGMENT - 1]) as *mut Page {
            let next_slice_ref = unsafe { &mut (*next_slice) };
            if next_slice_ref.block_size == 0 {
                log::debug!("[thread_id: {}] Merging released slice with next slice. Slice count of next slice: {}", self.thread_id, next_slice_ref.slice_count);
                // Page is not in use, remove it
                assert!(self.remove_slice_from_span(next_slice_ref));
                segment.coalesce_slices(page_ref, unsafe { &mut (*next_slice) });
            }
        }

        let mut merged_with_prev = false;

        if unsafe { page.offset_from(&mut segment.pages[0] as *mut Page) > 0 } {
            let mut prev_slice = page.wrapping_sub(1);
            prev_slice = prev_slice.wrapping_sub(unsafe { (*prev_slice).slice_offset });
            let prev_slice_ref = unsafe { &mut (*prev_slice) };
            if prev_slice_ref.block_size == 0 {
                // Merge with the previous slice
                log::debug!("[thread_id: {}] Merging slice with previous slice. Slice count of previous slice: {}", self.thread_id, prev_slice_ref.slice_count);
                assert!(self.remove_slice_from_span(prev_slice_ref));
                segment.coalesce_slices(prev_slice_ref, page_ref);
                let span_idx = Self::get_span_idx_from_slice_count(prev_slice_ref.slice_count);
                self.spans[span_idx].push(prev_slice);
                log::debug!("[thread_id: {}] Added page with slice count {} to span with index: {}", self.thread_id, prev_slice_ref.slice_count, span_idx);
                merged_with_prev = true;
            }
        }
        if !merged_with_prev {
            let span_idx = Self::get_span_idx_from_slice_count(page_ref.slice_count);
            self.spans[span_idx].push(page);
            log::debug!("[thread_id: {}] Added page with slice count {} to span with index: {}", self.thread_id, page_ref.slice_count, span_idx);
        }
        segment.check_valid_segment();
    }

    fn cleanup_pages(self: &mut Self) {
        for i in 0..self.free_pages.len() {
            let page = self.free_pages[i];
            if page != null_mut() {
                unsafe {
                    (*page).collect_foreign_frees();
                    if (*page).is_unused() {
                        self.retire_page(page);
                        self.free_pages[i] = null_mut();
                    }
                }
            }
        }
        for i in 0..self.used_pages.len() {
            let mut page_idx = 0;
            while page_idx < self.used_pages[i].len() {
                let page = self.used_pages[i][page_idx];
                unsafe {
                    (*page).collect_foreign_frees();
                    if (*page).is_unused() {
                        self.retire_page(page);
                    } else {
                        page_idx += 1;
                    }
                }
            }
        }
    }

    fn find_page_from_used(self: &mut Self, bin: usize) -> *mut u8 {
        for i in 0..self.used_pages[bin].len() {
            unsafe {
                (*self.used_pages[bin][i]).collect_foreign_frees();
                if (*self.used_pages[bin][i]).is_full() {
                    continue;
                }
                let page = self.used_pages[bin].remove(i);
                let block = (*page).get_free_block();
                self.free_pages[bin] = page;
                return block
            }
        }
        null_mut()
    }

    fn find_page_from_spans(self: &mut Self, num_slices_required: usize, block_size: usize) -> *mut Page {
        debug_assert!(block_size >= MIN_SIZE_FROM_PAGES);
        let min_bin = Self::get_span_idx_from_slice_count(num_slices_required);
        for i in min_bin..SEGMENT_BINS {
            let bin = &mut self.spans[i];
            for j in 0..bin.len() {
                let slice = bin[j];
                let num_slices_original = unsafe { (*slice).slice_count };
                debug_assert!(num_slices_original >= 1 << i);
                if num_slices_original < num_slices_required {
                    continue;
                }
                bin.remove(j);
                let segment = Segment::get_segment_from_ptr(slice as *mut u8);
                unsafe {
                    (*segment).allocated += num_slices_required;
                }
                if num_slices_original > num_slices_required {
                    // split slice
                    let next_slice = unsafe { (*segment).split_page(slice, num_slices_required) };
                    debug_assert!(unsafe { (*slice).slice_count == num_slices_required});
                    unsafe { (*segment).check_valid_segment() } ;
                    let bin = Self::get_span_idx_from_slice_count(num_slices_original - num_slices_required);
                    self.spans[bin].push(next_slice);
                    log::debug!("[thread_id: {}] Added page with slice count {} to span with index: {}", self.thread_id, num_slices_original - num_slices_required, bin);
                }
                unsafe {
                    (*slice).set_block_size(block_size);
                }
                return slice;
            }
        }
        null_mut()
    }

    fn add_segment_to_spans(self: &mut Self, segment: *mut Segment) {
        let segment_ref = unsafe { &mut (*segment) };
        let slice_count = segment_ref.num_slices;
        let span_idx = Self::get_span_idx_from_slice_count(slice_count);
        let page = &mut segment_ref.pages[0];
        page.slice_count = slice_count;
        page.slice_offset = 0;
        self.spans[span_idx].push(page as *mut Page);

        let last_page = &mut segment_ref.pages[PAGES_PER_SEGMENT - 1];
        last_page.slice_offset = PAGES_PER_SEGMENT - 1;
    }

    fn allocate_segment_from_arena(self: &mut Self, thread_id: usize) -> bool {
        self.stats.total_segments_allocated += 1;
        let segment_opt = {
            let mut guard = self.arena.lock().unwrap();
            guard.allocate_segment(SEGMENT_SIZE)
        };
        if segment_opt.is_none() {
            return false;
        }
        // log::info!("Allocating segment to thread with id: {}", thread_id);
        unsafe {
            (*segment_opt.unwrap()).thread_id = thread_id;
        }
        
        self.add_segment_to_spans(segment_opt.unwrap());
        true
    }

    fn allocate_large(self: &mut Self, size: usize) -> *mut u8 {
        // Directly get page from segment
        let num_pages = (size + PAGE_SIZE - 1) / PAGE_SIZE;
        let block_size = num_pages * PAGE_SIZE;
        let mut free_page = self.find_page_from_spans(num_pages, block_size);
        if free_page != null_mut() {
            self.stats.allocations_from_segment += 1;
            let free_block = unsafe { (*free_page).get_free_block() };
            return free_block
        }
        self.cleanup_pages();
        // Retry after cleanup
        free_page = self.find_page_from_spans(num_pages, block_size);
        if free_page != null_mut() {
            self.stats.allocations_from_segment += 1;
            self.used_pages[NUM_SIZE_CLASSES].push(free_page);
            let free_block = unsafe { (*free_page).get_free_block() };
            return free_block
        }

        let res = self.allocate_segment_from_arena(self.thread_id);
        if !res {
            return null_mut()
        }
        self.stats.allocations_from_arena += 1;
        free_page = self.find_page_from_spans(num_pages, block_size);
        if free_page == null_mut() {
            self.stats.unsuccessful_allocations += 1;
            return null_mut()
        }
        self.used_pages[NUM_SIZE_CLASSES].push(free_page);
        assert_ne!(free_page, null_mut());
        let free_block = unsafe { (*free_page).get_free_block() };
        return free_block
    }

    pub(crate) fn allocate(self: &mut Self, size: usize) -> *mut u8 {
        self.stats.total_allocations = self.stats.total_allocations.wrapping_add(1);
        if self.stats.total_allocations & 0x7f == 0 {
            // Periodically cleanup pages
            self.cleanup_pages();
        }
        if size > PAGE_SIZE {
            return self.allocate_large(size)
        }
        let size_class = Self::get_size_class(size);
        debug_assert!(size_class < NUM_SIZE_CLASSES);

        let block_size = SIZE_CLASSES[size_class];
        let mut free_page = self.free_pages[size_class];
        if !free_page.is_null() {
            debug_assert_eq!(unsafe {(*free_page).block_size}, block_size);
            // allocate from free page
            let page = free_page.clone();
            unsafe {
                if !(*page).is_full() {
                    self.stats.fast_allocations += 1;
                    return (*page).get_free_block()
                } else {
                    // Try collecting frees from other threads and retrying
                    (*page).collect_foreign_frees();
                    if !(*page).is_full() {
                        return (*page).get_free_block()
                    }
                    self.used_pages[size_class].push(page);
                    self.free_pages[size_class] = null_mut();
                }
            }
        }
        let block = self.find_page_from_used(size_class);
        if !block.is_null() {
            self.stats.allocations_from_pages += 1;
            return block
        }
        free_page = self.find_page_from_spans(1, block_size);
        if free_page != null_mut() {
            self.stats.allocations_from_segment += 1;
            let free_block = unsafe { (*free_page).get_free_block() };
            self.free_pages[size_class] = free_page;
            return free_block;
        }
        // No space available in segments, allocate a new one
        let res = self.allocate_segment_from_arena(self.thread_id);
        if !res {
            self.stats.unsuccessful_allocations += 1;
            return null_mut()
        }
        self.stats.allocations_from_arena += 1;
        free_page = self.find_page_from_spans(1, block_size);
        assert_ne!(free_page, null_mut());
        let free_block = unsafe { (*free_page).get_free_block() };
        self.free_pages[size_class] = free_page;
        return free_block
    }

    #[allow(unused)]
    pub(crate) fn get_stats(self: &Self) -> TCacheStats {
        self.stats.clone()
    }
}