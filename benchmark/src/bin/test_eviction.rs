use std::collections::HashSet;

use liquid_cache_benchmarks::eviction::eviction_cache::{
    ClockCache, Cache, FifoCache, LfuCache, LruCache,
};
use std::fs::File;
use std::io::{BufRead, BufReader};

fn pack_u16s(a: u16, b: u16, c: u16) -> u64 {
    ((a as u64) << 32) | ((b as u64) << 16) | (c as u64)
}

fn bench<C: Cache>(total_size: u64, create: impl Fn(u64) -> C, name: String) {
    let mut cache_size = total_size;

    while cache_size > 0 {
        let mut cache = create(cache_size);
        let file = File::open("./cache_trace.csv").expect("Failed to reopen cache_trace.csv");
        let reader = BufReader::new(file);

        for line in reader.lines().skip(1) {
            let line = line.expect("Failed to read line");
            let fields: Vec<&str> = line.split(',').collect();
            if fields.len() == 6 {
                let file_id: u16 = fields[0].parse().expect("Failed to parse file_id");
                let row_group: u16 = fields[1].parse().expect("Failed to parse row_group");
                let col: u16 = fields[2].parse().expect("Failed to parse col");
                let size: u64 = fields[4].parse().expect("Failed to parse size");

                let key = pack_u16s(file_id, row_group, col);

                cache.get(key, size);
            }
        }

        let (hits, total) = cache.result();
        println!("{},{},{},{}", name, cache_size, hits, total);

        cache_size /= 10;
    }
}

fn main() {
    // Read and parse the cache trace file
    let file = File::open("./cache_trace.csv").expect("Failed to open cache_trace.csv");
    let reader = BufReader::new(file);

    let mut total_size: u64 = 0;
    let mut count = 0;
    let mut cols = HashSet::new();

    for line in reader.lines().skip(1) {
        let line = line.expect("Failed to read line");
        let fields: Vec<&str> = line.split(',').collect();
        if fields.len() == 6 {
            let file_id: u16 = fields[0].parse().expect("Failed to parse file_id");
            let row_group: u16 = fields[1].parse().expect("Failed to parse row_group");
            let col: u16 = fields[2].parse().expect("Failed to parse col");
            let size: u64 = fields[4].parse().expect("Failed to parse size");

            let new = cols.insert(pack_u16s(file_id, row_group, col));
            if new {
                total_size += size;
            }
            count += 1;
        }
    }

    println!("Read {} inserts, total size: {}", count, total_size);

    bench(total_size, LruCache::new, "LRU".to_string());
    bench(total_size, ClockCache::new, "CLOCK".to_string());
    bench(total_size, LfuCache::new, "LFU".to_string());
    bench(total_size, FifoCache::new, "FIFO".to_string());
}
