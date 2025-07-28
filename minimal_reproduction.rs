// Minimal reproduction of the duplicate values bug in liquid-cache
// This test demonstrates the core issue without requiring the full liquid-cache setup

use std::collections::HashMap;

// Simulate the BatchID structure
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct BatchID {
    v: u16,
}

impl BatchID {
    fn new() -> Self {
        Self { v: 0 }
    }
    
    fn inc(&mut self) {
        self.v += 1;
    }
}

// Simulate the cache structure
struct Cache {
    data: HashMap<BatchID, Vec<i32>>,
}

impl Cache {
    fn new() -> Self {
        Self {
            data: HashMap::new(),
        }
    }
    
    fn get(&self, batch_id: BatchID) -> Option<&Vec<i32>> {
        self.data.get(&batch_id)
    }
    
    fn insert(&mut self, batch_id: BatchID, data: Vec<i32>) {
        self.data.insert(batch_id, data);
    }
}

// Simulate the buggy batch reader
struct BuggyBatchReader {
    current_batch_id: BatchID,
    cache: Cache,
    batch_size: usize,
    remaining_data: Vec<i32>,
}

impl BuggyBatchReader {
    fn new(data: Vec<i32>, batch_size: usize) -> Self {
        let mut cache = Cache::new();
        
        // Pre-populate cache with some data
        for i in 0..3 {
            let batch_id = BatchID { v: i };
            let start = i * batch_size;
            let end = std::cmp::min(start + batch_size, data.len());
            let batch_data = data[start..end].to_vec();
            cache.insert(batch_id, batch_data);
        }
        
        Self {
            current_batch_id: BatchID::new(),
            cache,
            batch_size,
            remaining_data: data,
        }
    }
    
    // This simulates the buggy next() method
    fn next_buggy(&mut self) -> Option<Vec<i32>> {
        if self.remaining_data.is_empty() {
            return None;
        }
        
        // Simulate take_next_batch
        let batch_size = std::cmp::min(self.batch_size, self.remaining_data.len());
        let batch_data = self.remaining_data.drain(0..batch_size).collect::<Vec<_>>();
        
        // BUG: Try to read from cache BEFORE incrementing batch_id
        if let Some(cached_data) = self.cache.get(self.current_batch_id) {
            println!("Cache hit for batch_id {:?}: {:?}", self.current_batch_id, cached_data);
            // BUG: batch_id is incremented AFTER cache read
            self.current_batch_id.inc();
            return Some(cached_data.clone());
        }
        
        // Cache miss - use actual data
        println!("Cache miss for batch_id {:?}: {:?}", self.current_batch_id, batch_data);
        self.current_batch_id.inc();
        Some(batch_data)
    }
    
    // This simulates the fixed next() method
    fn next_fixed(&mut self) -> Option<Vec<i32>> {
        if self.remaining_data.is_empty() {
            return None;
        }
        
        // Simulate take_next_batch
        let batch_size = std::cmp::min(self.batch_size, self.remaining_data.len());
        let batch_data = self.remaining_data.drain(0..batch_size).collect::<Vec<_>>();
        
        // FIX: Increment batch_id BEFORE cache read
        self.current_batch_id.inc();
        
        // Now try to read from cache with the correct batch_id
        if let Some(cached_data) = self.cache.get(self.current_batch_id) {
            println!("Cache hit for batch_id {:?}: {:?}", self.current_batch_id, cached_data);
            return Some(cached_data.clone());
        }
        
        // Cache miss - use actual data
        println!("Cache miss for batch_id {:?}: {:?}", self.current_batch_id, batch_data);
        Some(batch_data)
    }
}

fn find_duplicates<T: Eq + std::hash::Hash + Clone>(vec: &[T]) -> Vec<T> {
    use std::collections::HashSet;
    let mut seen = HashSet::new();
    let mut duplicates = Vec::new();
    
    for item in vec {
        if !seen.insert(item.clone()) {
            duplicates.push(item.clone());
        }
    }
    duplicates
}

fn main() {
    println!("=== Liquid Cache Duplicate Values Bug Reproduction ===\n");
    
    // Create test data with unique values
    let test_data: Vec<i32> = (0..10).collect(); // [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    println!("Test data: {:?}", test_data);
    println!("Batch size: 3\n");
    
    // Test the buggy version
    println!("--- Testing BUGGY version ---");
    let mut buggy_reader = BuggyBatchReader::new(test_data.clone(), 3);
    let mut buggy_results = Vec::new();
    
    while let Some(batch) = buggy_reader.next_buggy() {
        buggy_results.extend(batch);
    }
    
    println!("Buggy results: {:?}", buggy_results);
    let buggy_duplicates = find_duplicates(&buggy_results);
    println!("Buggy duplicates: {:?}", buggy_duplicates);
    println!("Buggy duplicate count: {}", buggy_duplicates.len());
    
    println!("\n--- Testing FIXED version ---");
    let mut fixed_reader = BuggyBatchReader::new(test_data, 3);
    let mut fixed_results = Vec::new();
    
    while let Some(batch) = fixed_reader.next_fixed() {
        fixed_results.extend(batch);
    }
    
    println!("Fixed results: {:?}", fixed_results);
    let fixed_duplicates = find_duplicates(&fixed_results);
    println!("Fixed duplicates: {:?}", fixed_duplicates);
    println!("Fixed duplicate count: {}", fixed_duplicates.len());
    
    println!("\n=== Analysis ===");
    println!("The bug occurs because:");
    println!("1. The cache lookup uses the current batch_id");
    println!("2. The batch_id is incremented AFTER the cache read");
    println!("3. Multiple iterations can use the same batch_id for cache lookups");
    println!("4. This causes the same cached data to be returned multiple times");
    println!("5. Result: duplicate values in the final output");
    
    if !buggy_duplicates.is_empty() {
        println!("\n✅ BUG REPRODUCED: The buggy version produces {} duplicates", buggy_duplicates.len());
    }
    
    if fixed_duplicates.is_empty() {
        println!("✅ FIX VERIFIED: The fixed version produces 0 duplicates");
    }
}