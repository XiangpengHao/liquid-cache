#![allow(unused)]
pub trait Cache {
    fn new(budget: u64) -> Self;
    fn get(&mut self, id: u64, size: u64);
    fn result(&self) -> (u64, u64);
}

pub struct LruCache {
    capacity: u64,     // Maximum total size allowed.
    current_size: u64, // Current sum of sizes in the cache.
    // The cache now holds (id, size) with index 0 being the most recently used.
    cache: Vec<(u64, u64)>,
    hits: u64,
    gets: u64,
}

impl Cache for LruCache {
    fn new(budget: u64) -> Self {
        LruCache {
            capacity: budget,
            current_size: 0,
            cache: Vec::new(),
            hits: 0,
            gets: 0,
        }
    }

    fn get(&mut self, id: u64, size: u64) {
        self.gets += 1;
        if let Some(pos) = self.cache.iter().position(|&(key, _)| key == id) {
            // Cache hit: move the item to the front (most recently used)
            self.hits += 1;
            let item = self.cache.remove(pos);
            self.cache.insert(0, item);
        } else {
            // If the item is larger than the capacity, skip caching.
            if size > self.capacity {
                return;
            }
            // Evict least recently used items until there is room for the new item.
            while self.current_size + size > self.capacity {
                if let Some((_, removed_size)) = self.cache.pop() {
                    self.current_size -= removed_size;
                } else {
                    break; // Should not occur since size <= capacity.
                }
            }
            // Insert the new item at the front and update the total used size.
            self.cache.insert(0, (id, size));
            self.current_size += size;
        }
    }

    fn result(&self) -> (u64, u64) {
        (self.hits, self.gets)
    }
}

struct ClockEntry {
    id: u64,
    size: u64,
    ref_bit: bool,
}

pub struct ClockCache {
    capacity: u64,     // Maximum total size allowed.
    current_size: u64, // Current sum of sizes in the cache.
    // The clock cache holds ClockEntry items.
    cache: Vec<ClockEntry>,
    // Hand pointer to the next candidate for eviction.
    hand: usize,
    hits: u64,
    gets: u64,
}

impl Cache for ClockCache {
    fn new(budget: u64) -> Self {
        ClockCache {
            capacity: budget,
            current_size: 0,
            cache: Vec::new(),
            hand: 0,
            hits: 0,
            gets: 0,
        }
    }

    fn get(&mut self, id: u64, size: u64) {
        self.gets += 1;

        // Check for a cache hit: if found, mark it as referenced.
        if let Some(pos) = self.cache.iter_mut().position(|entry| entry.id == id) {
            self.hits += 1;
            self.cache[pos].ref_bit = true;
            return;
        }

        // If the new item is larger than our total capacity, skip caching.
        if size > self.capacity {
            return;
        }

        // Evict items using the CLOCK algorithm until there's enough space.
        while self.current_size + size > self.capacity {
            // If the cache is empty, nothing to evict (should not occur because size <= capacity).
            if self.cache.is_empty() {
                break;
            }
            // Look at the candidate at the hand pointer.
            if self.cache[self.hand].ref_bit {
                // Give this entry a "second chance": clear the bit and move the hand.
                self.cache[self.hand].ref_bit = false;
                self.hand = (self.hand + 1) % self.cache.len();
            } else {
                // Evict the candidate (its reference bit is false).
                let removed_size = self.cache[self.hand].size;
                self.cache.remove(self.hand);
                self.current_size -= removed_size;
                // After removal, the vector has shrunk.
                // If the hand now points past the end, wrap it to 0.
                if !self.cache.is_empty() && self.hand >= self.cache.len() {
                    self.hand = 0;
                }
                // Note: we do not advance the hand here because the next element
                // shifted into the current position.
            }
        }

        // Insert the new item with its reference bit set.
        let new_entry = ClockEntry {
            id,
            size,
            ref_bit: true,
        };

        if self.cache.is_empty() {
            // If the cache is empty, just push the new entry.
            self.cache.push(new_entry);
            self.hand = 0;
        } else {
            // Insert at the hand position (the candidate for replacement).
            self.cache.insert(self.hand, new_entry);
            // Advance the hand to avoid immediately scanning the new entry.
            self.hand = (self.hand + 1) % self.cache.len();
        }
        self.current_size += size;
    }

    fn result(&self) -> (u64, u64) {
        (self.hits, self.gets)
    }
}

struct LFUEntry {
    id: u64,
    size: u64,
    freq: u64,
}

pub struct LfuCache {
    capacity: u64,     // Maximum total size allowed.
    current_size: u64, // Current sum of sizes in the cache.
    // The cache holds LFUEntry items.
    cache: Vec<LFUEntry>,
    hits: u64,
    gets: u64,
}

impl Cache for LfuCache {
    fn new(budget: u64) -> Self {
        LfuCache {
            capacity: budget,
            current_size: 0,
            cache: Vec::new(),
            hits: 0,
            gets: 0,
        }
    }

    fn get(&mut self, id: u64, size: u64) {
        self.gets += 1;

        // Check if the item is already in the cache.
        if let Some(entry) = self.cache.iter_mut().find(|entry| entry.id == id) {
            // Cache hit: increase the frequency count.
            self.hits += 1;
            entry.freq += 1;
            return;
        }

        // If the item is larger than the total capacity, skip caching.
        if size > self.capacity {
            return;
        }

        // Evict entries until there is room for the new item.
        while self.current_size + size > self.capacity {
            // Find the index of the entry with the lowest frequency.
            if let Some((min_index, _)) = self
                .cache
                .iter()
                .enumerate()
                .min_by_key(|&(_, entry)| entry.freq)
            {
                let removed_size = self.cache.remove(min_index).size;
                self.current_size -= removed_size;
            } else {
                break; // This should not happen as size <= capacity.
            }
        }

        // Insert the new item with an initial frequency count of 1.
        self.cache.push(LFUEntry { id, size, freq: 1 });
        self.current_size += size;
    }

    fn result(&self) -> (u64, u64) {
        (self.hits, self.gets)
    }
}

pub struct FifoCache {
    capacity: u64,     // Maximum total size allowed.
    current_size: u64, // Current sum of sizes in the cache.
    // The cache holds (id, size) pairs.
    // Index 0 holds the oldest entry.
    cache: Vec<(u64, u64)>,
    hits: u64,
    gets: u64,
}

impl Cache for FifoCache {
    fn new(budget: u64) -> Self {
        FifoCache {
            capacity: budget,
            current_size: 0,
            cache: Vec::new(),
            hits: 0,
            gets: 0,
        }
    }

    fn get(&mut self, id: u64, size: u64) {
        self.gets += 1;

        // Check if the item is already in the cache.
        if self.cache.iter().any(|&(key, _)| key == id) {
            // Cache hit: simply count the hit.
            self.hits += 1;
            return;
        }

        // If the item is larger than our total capacity, skip caching.
        if size > self.capacity {
            return;
        }

        // Evict items from the front (oldest) until there is enough room.
        while self.current_size + size > self.capacity {
            if let Some((_, removed_size)) = self.cache.first() {
                self.current_size -= *removed_size;
                self.cache.remove(0);
            } else {
                break;
            }
        }

        // Insert the new item at the back (newest end).
        self.cache.push((id, size));
        self.current_size += size;
    }

    fn result(&self) -> (u64, u64) {
        (self.hits, self.gets)
    }
}
