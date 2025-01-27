use std::collections::hash_map::Entry;
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::MutexGuard;

use ahash::HashMap;
use ahash::HashMapExt;
use parquet::basic::PageType;
use parquet::column::page::Page;
use parquet::column::page::PageMetadata;
use parquet::column::page::PageReader;
use parquet::errors::ParquetError;
use parquet::file::reader::ChunkReader;
use parquet::file::reader::SerializedPageReader;

struct CachedPage {
    dict: Option<(usize, Page)>, // page offset -> page
    data: Option<(usize, Page)>, // page offset -> page
}

struct PredicatePageCacheInner {
    pages: HashMap<usize, CachedPage>, // col_id (Parquet's leaf column index) -> CachedPage
}

impl PredicatePageCacheInner {
    #[allow(unused)]
    pub(crate) fn get_page(&self, col_id: usize, offset: usize) -> Option<Page> {
        self.pages.get(&col_id).and_then(|pages| {
            pages
                .dict
                .iter()
                .chain(pages.data.iter())
                .find(|(page_offset, _)| *page_offset == offset)
                .map(|(_, page)| page.clone())
        })
    }

    /// Insert a page into the cache.
    /// Inserting a page will override the existing page, if any.
    /// This is because we only need to cache 2 pages per column, see below.
    #[allow(unused)]
    pub(crate) fn insert_page(&mut self, col_id: usize, offset: usize, page: Page) {
        let is_dict = page.page_type() == PageType::DICTIONARY_PAGE;

        let cached_pages = self.pages.entry(col_id);
        match cached_pages {
            Entry::Occupied(mut entry) => {
                if is_dict {
                    entry.get_mut().dict = Some((offset, page));
                } else {
                    entry.get_mut().data = Some((offset, page));
                }
            }
            Entry::Vacant(entry) => {
                let cached_page = if is_dict {
                    CachedPage {
                        dict: Some((offset, page)),
                        data: None,
                    }
                } else {
                    CachedPage {
                        dict: None,
                        data: Some((offset, page)),
                    }
                };
                entry.insert(cached_page);
            }
        }
    }
}

/// A simple cache to avoid double-decompressing pages with filter pushdown.
/// In filter pushdown, we first decompress a page, apply the filter, and then decompress the page again.
/// This double decompression is expensive, so we cache the decompressed page.
///
/// This implementation contains subtle dynamics that can be hard to understand.
///
/// ## Which columns to cache
///
/// Let's consider this example: SELECT B, C FROM table WHERE A = 42 and B = 37;
/// We have 3 columns, and the predicate is applied to column A and B, and projection is on B and C.
///
/// For column A, we need to decompress it, apply the filter (A=42), and never have to decompress it again, as it's not in the projection.
/// For column B, we need to decompress it, apply the filter (B=37), and then decompress it again, as it's in the projection.
/// For column C, we don't have predicate, so we only decompress it once.
///
/// A, C is only decompressed once, and B is decompressed twice (as it appears in both the predicate and the projection).
/// The PredicatePageCache will only cache B.
/// We use B's col_id (Parquet's leaf column index) to identify the cache entry.
///
/// ## How many pages to cache
///
/// Now we identified the columns to cache, next question is to determine the **minimal** number of pages to cache.
///
/// Let's revisit our decoding pipeline:
/// Load batch 1 -> evaluate predicates -> filter 1 -> load & emit batch 1
/// Load batch 2 -> evaluate predicates -> filter 2 -> load & emit batch 2
/// ...
/// Load batch N -> evaluate predicates -> filter N -> load & emit batch N
///
/// Assumption & observation: each page consists multiple batches.
/// Then our pipeline looks like this:
/// Load Page 1
/// Load batch 1 -> evaluate predicates -> filter 1 -> load & emit batch 1
/// Load batch 2 -> evaluate predicates -> filter 2 -> load & emit batch 2
/// Load batch 3 -> evaluate predicates -> filter 3 -> load & emit batch 3
/// Load Page 2
/// Load batch 4 -> evaluate predicates -> filter 4 -> load & emit batch 4
/// Load batch 5 -> evaluate predicates -> filter 5 -> load & emit batch 5
/// ...
///
/// This means that we only need to cache one page per column,
/// because the page that is used by the predicate is the same page, and is immediately used in loading the batch.
///
/// The only exception is the dictionary page -- the first page of each column.
/// If we encountered a dict page, we will need to immediately read next page, and cache it.
///
/// To summarize, the cache only contains 2 pages per column: one dict page and one data page.
/// This is a nice property as it means the caching memory consumption is negligible and constant to the number of columns.
///
/// ## How to identify a page
/// We use the page offset (the offset to the Parquet file) to uniquely identify a page.
pub(crate) struct PredicatePageCache {
    inner: Mutex<PredicatePageCacheInner>,
}

impl PredicatePageCache {
    pub(crate) fn new() -> Self {
        Self {
            inner: Mutex::new(PredicatePageCacheInner {
                pages: HashMap::new(),
            }),
        }
    }

    #[allow(unused)]
    fn get(&self) -> MutexGuard<PredicatePageCacheInner> {
        self.inner.lock().unwrap()
    }
}

pub(crate) struct CachedPageReader<R: ChunkReader> {
    inner: SerializedPageReader<R>,
    #[allow(unused)]
    cache: Arc<PredicatePageCache>,
    #[allow(unused)]
    col_id: usize,
}

impl<R: ChunkReader> CachedPageReader<R> {
    pub(crate) fn new(
        inner: SerializedPageReader<R>,
        cache: Arc<PredicatePageCache>,
        col_id: usize,
    ) -> Self {
        Self {
            inner,
            cache,
            col_id,
        }
    }
}

impl<R: ChunkReader> Iterator for CachedPageReader<R> {
    type Item = Result<Page, ParquetError>;

    fn next(&mut self) -> Option<Self::Item> {
        self.get_next_page().transpose()
    }
}

impl<R: ChunkReader> PageReader for CachedPageReader<R> {
    fn get_next_page(&mut self) -> Result<Option<Page>, ParquetError> {
        // We need to wait for peek_next_page_offset() hit next release.
        // I think it will be late Jan 2025.
        //
        // let next_page_offset = self.inner.peek_next_page_offset()?;

        // let Some(offset) = next_page_offset else {
        //     return Ok(None);
        // };

        // let mut cache = self.cache.get();

        // let page = cache.get_page(self.col_id, offset);
        // if let Some(page) = page {
        //     self.inner.skip_next_page()?;
        //     Ok(Some(page))
        // } else {
        //     let inner_page = self.inner.get_next_page()?;
        //     let Some(inner_page) = inner_page else {
        //         return Ok(None);
        //     };
        //     cache.insert_page(self.col_id, offset, inner_page.clone());
        //     Ok(Some(inner_page))
        // }

        self.inner.get_next_page()
    }

    fn peek_next_page(&mut self) -> Result<Option<PageMetadata>, ParquetError> {
        self.inner.peek_next_page()
    }

    fn skip_next_page(&mut self) -> Result<(), ParquetError> {
        self.inner.skip_next_page()
    }

    fn at_record_boundary(&mut self) -> Result<bool, ParquetError> {
        self.inner.at_record_boundary()
    }
}
