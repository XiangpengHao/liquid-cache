use liquid_cache_common::LiquidCacheMode;
use parquet::file::reader::{FileReader, SerializedFileReader};
use parquet::record::RowAccessor;
use std::collections::HashMap;
use std::fs;
use std::fs::File;

use liquid_cache_parquet::cache::store::CacheAdvice;
use liquid_cache_parquet::cache::utils::CacheEntryID;
use liquid_cache_parquet::policies::{
    CachePolicy, ClockPolicy, DiscardPolicy, FiloPolicy, LruPolicy, RandomPolicy, SievePolicy,
};

// Import the Cache trait and implementations from eviction_cache.rs
//mod eviction_cache;
//use eviction_cache::{Cache, LruCache, FifoCache, ClockCache, SieveCache};

pub struct CacheImpl {
    p: Box<dyn CachePolicy>,
    data: HashMap<u64, bool>,
    size: usize,
    hits: u64,
    total: u64,
}

impl CacheImpl {
    fn new(p: Box<dyn CachePolicy>, size: usize) -> Self {
        Self {
            p,
            data: HashMap::new(),
            size,
            hits: 0,
            total: 0,
        }
    }
    fn insert(&mut self, key: u64) {
        if self.data.len() >= self.size {
            let advice = self.p.advise(
                &CacheEntryID::from(key as usize),
                &LiquidCacheMode::InMemoryArrow,
            );
            match advice {
                CacheAdvice::Evict(id) => {
                    // Convert CacheEntryID back to usize and then to u64
                    let evicted_key: usize = id.into();
                    self.data.remove(&(evicted_key as u64));
                    self.p.notify_evict(&id);
                    //println!("Evicted: {:?}", id);
                }
                _ => {
                    //panic!("Unexpected advice: {:?}", advice);
                    return;
                }
            }
        }
        self.data.insert(key, true);
        self.p.notify_insert(&CacheEntryID::from(key as usize));
    }
    fn access(&mut self, key: u64) {
        if self.data.contains_key(&key) {
            self.hits += 1;
            self.p.notify_access(&CacheEntryID::from(key as usize));
        } else {
            self.insert(key);
        }
        self.total += 1;
    }
    fn result(&self) -> (u64, u64) {
        (self.hits, self.total)
    }
}

fn pack_u16s(a: u16, b: u16, c: u16, d: u16) -> u64 {
    ((a as u64) << 48) | ((b as u64) << 32) | ((c as u64) << 16) | (d as u64)
}

fn simulate_belady(trace: &[u64], cache_size: usize) -> (u64, u64) {
    let mut cache = std::collections::HashSet::new();
    let mut hits = 0;
    let mut misses = 0;

    // Precompute future positions for each item
    let mut future_positions = std::collections::HashMap::new();
    for (pos, &item) in trace.iter().enumerate() {
        future_positions
            .entry(item)
            .or_insert_with(Vec::new)
            .push(pos);
    }

    // Process each access in the trace
    for (current_pos, &item) in trace.iter().enumerate() {
        if cache.contains(&item) {
            hits += 1;
        } else {
            misses += 1;

            if cache.len() >= cache_size {
                // Find the item in cache that will be accessed furthest in the future
                let mut victim = None;
                let mut furthest_future = 0;

                for &cached_item in &cache {
                    let positions = future_positions.get(&cached_item).unwrap();
                    let next_pos = positions
                        .iter()
                        .find(|&&p| p > current_pos)
                        .copied()
                        .unwrap_or(usize::MAX);

                    if next_pos > furthest_future {
                        furthest_future = next_pos;
                        victim = Some(cached_item);
                    }
                }

                if let Some(v) = victim {
                    cache.remove(&v);
                }
            }
            cache.insert(item);
        }

        // Remove the current position from future positions
        if let Some(positions) = future_positions.get_mut(&item) {
            if let Some(pos) = positions.iter().position(|&p| p == current_pos) {
                positions.remove(pos);
            }
        }
    }

    (hits, misses)
}

fn bench(create: impl Fn(usize) -> CacheImpl, name: String) {
    // Read all trace data into memory first
    let mut trace_data = Vec::new();
    let mut unique_items = std::collections::HashSet::new();
    let paths = fs::read_dir("./trace").expect("Failed to read directory");

    for fname in paths {
        let file = File::open(fname.expect("Failed to open file").path())
            .expect("Failed to open parquet file");
        let parquet_reader =
            SerializedFileReader::new(file).expect("Failed to create parquet reader");
        let row_iter = parquet_reader
            .get_row_iter(None)
            .expect("Failed to get row iterator");

        for row in row_iter {
            let row = row.expect("Failed to read row");
            let file_id: u16 = row.get_ulong(0).expect("Failed to get file_id") as u16;
            let row_group_id: u16 = row.get_ulong(1).expect("Failed to get row_group_id") as u16;
            let column_id: u16 = row.get_ulong(2).expect("Failed to get column_id") as u16;
            let batch_id: u16 = row.get_ulong(3).expect("Failed to get batch_id") as u16;
            let cache_memory_bytes: u64 =
                row.get_ulong(4).expect("Failed to get cache_memory_bytes");

            let key = pack_u16s(file_id, row_group_id, column_id, batch_id);
            trace_data.push((key, cache_memory_bytes));
            unique_items.insert(key);
        }
    }

    /*println!("Total accesses: {}", trace_data.len());
    println!("Unique items: {}", unique_items.len());
    println!("Maximum possible hits: {}", trace_data.len() - unique_items.len());*/

    // Start with 100% of total memory and reduce by 5% each iteration
    let mut cache_size = unique_items.len() as i64;
    let initial_cache_size = cache_size;

    while cache_size > 0 {
        let mut cache = create(cache_size as usize);

        // Run the cache simulation with the stored trace data
        for (key, _size) in &trace_data {
            cache.access(*key);
        }

        let (hits, total) = cache.result();
        println!(
            "{},{},{}",
            name,
            cache_size as f64 / initial_cache_size as f64,
            hits as f64 / total as f64
        );
        //cache_size -= (initial_cache_size as f64 / 20.0).ceil() as i64;
        cache_size -= (initial_cache_size as f64 / 20.0).ceil() as i64;
    }
}

fn main() {
    // First run all the regular cache policies
    bench(
        |size: usize| CacheImpl::new(Box::new(LruPolicy::new()), size),
        "LRU".to_string(),
    );
    bench(
        |size: usize| CacheImpl::new(Box::new(FiloPolicy::new()), size),
        "FILO".to_string(),
    );
    bench(
        |size: usize| CacheImpl::new(Box::new(ClockPolicy::new()), size),
        "CLOCK".to_string(),
    );
    bench(
        |size: usize| CacheImpl::new(Box::new(SievePolicy::new()), size),
        "SIEVE".to_string(),
    );
    bench(
        |size: usize| CacheImpl::new(Box::new(RandomPolicy::new()), size),
        "RANDOM".to_string(),
    );
    bench(
        |size: usize| CacheImpl::new(Box::new(DiscardPolicy), size),
        "DISCARD".to_string(),
    );

    let mut trace_data = Vec::new();
    let mut unique_items = std::collections::HashSet::new();
    let paths = fs::read_dir("./trace").expect("Failed to read directory");

    for fname in paths {
        let file = File::open(fname.expect("Failed to open file").path())
            .expect("Failed to open parquet file");
        let parquet_reader =
            SerializedFileReader::new(file).expect("Failed to create parquet reader");
        let row_iter = parquet_reader
            .get_row_iter(None)
            .expect("Failed to get row iterator");

        for row in row_iter {
            let row = row.expect("Failed to read row");
            let file_id: u16 = row.get_ulong(0).expect("Failed to get file_id") as u16;
            let row_group_id: u16 = row.get_ulong(1).expect("Failed to get row_group_id") as u16;
            let column_id: u16 = row.get_ulong(2).expect("Failed to get column_id") as u16;
            let batch_id: u16 = row.get_ulong(3).expect("Failed to get batch_id") as u16;
            let cache_memory_bytes: u64 =
                row.get_ulong(4).expect("Failed to get cache_memory_bytes");

            let key = pack_u16s(file_id, row_group_id, column_id, batch_id);
            trace_data.push((key, cache_memory_bytes));
            unique_items.insert(key);
        }
    }

    // Extract just the keys for Belady's algorithm
    let trace: Vec<u64> = trace_data.iter().map(|(key, _)| *key).collect();
    let initial_cache_size = unique_items.len() as i64;
    let mut cache_size = initial_cache_size;

    while cache_size > 0 {
        let (hits, misses) = simulate_belady(&trace, cache_size as usize);
        let hit_rate = hits as f64 / (hits + misses) as f64;
        println!(
            "BELADY,{},{}",
            cache_size as f64 / initial_cache_size as f64,
            hit_rate
        );
        cache_size -= (initial_cache_size as f64 / 20.0).ceil() as i64;
    }
}
