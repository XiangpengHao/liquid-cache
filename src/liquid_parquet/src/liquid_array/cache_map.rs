use std::collections::{HashMap, VecDeque};
use std::fmt::{self, Debug, Display};
use std::hash::{Hash, Hasher};
use std::sync::Arc;
use std::sync::atomic::Ordering;

use crate::cache::{CacheConfig, LiquidCachedColumn, LiquidCachedFile, LiquidCachedRowGroup};

#[derive(Clone, Debug)]
pub struct ColumnKey {
    pub file: String,
    pub row_group: usize,
    pub column: usize,
}

impl PartialEq for ColumnKey {
    fn eq(&self, other: &Self) -> bool {
        self.file == other.file && self.row_group == other.row_group && self.column == other.column
    }
}

impl Eq for ColumnKey {}

impl Hash for ColumnKey {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.file.hash(state);
        self.row_group.hash(state);
        self.column.hash(state);
    }
}

pub trait LiquidCacheMap: Debug + Send + Sync {
    fn new(config: CacheConfig) -> Self
    where
        Self: Sized;
    fn put(&mut self, key: ColumnKey, value: Arc<LiquidCachedColumn>);
    fn contains(&self, key: &ColumnKey) -> bool;
    fn get(&mut self, key: &ColumnKey) -> Option<Arc<LiquidCachedColumn>>;
    fn iter<'a>(
        &'a self,
    ) -> Box<dyn Iterator<Item = (&'a ColumnKey, &'a Arc<LiquidCachedColumn>)> + 'a>;
    fn memory_usage(&self) -> u64;
}

pub struct LRUCache {
    pub map: HashMap<ColumnKey, Arc<LiquidCachedColumn>>,
    order: VecDeque<ColumnKey>,
    config: CacheConfig,
    current_memory_bytes: usize, // Track current memory usage
}

impl LRUCache {
    pub fn new(config: CacheConfig) -> Self {
        <Self as LiquidCacheMap>::new(config)
    }
}

impl Debug for LRUCache {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("LRUCache")
            .field("curr_size", &self.map.len())
            .field("map", &self.map)
            .finish()
    }
}

impl LiquidCacheMap for LRUCache {
    fn new(config: CacheConfig) -> Self {
        println!("New LRU Cache");
        Self {
            map: HashMap::new(),
            order: VecDeque::new(),
            config: config,
            current_memory_bytes: 0,
        }
    }

    fn put(&mut self, key: ColumnKey, value: Arc<LiquidCachedColumn>) {
        let new_value_size = value.memory_usage();

        // If item already exists, remove it first
        if let Some(old_value) = self.map.remove(&key) {
            self.current_memory_bytes = self
                .current_memory_bytes
                .saturating_sub(old_value.memory_usage());
            self.order.retain(|k| k != &key);
        }

        // Evict entries until we have enough space
        while self.current_memory_bytes + new_value_size > self.config.max_cache_bytes {
            if let Some(lru_key) = self.order.pop_front() {
                if let Some(removed_value) = self.map.remove(&lru_key) {
                    self.current_memory_bytes = self
                        .current_memory_bytes
                        .saturating_sub(removed_value.memory_usage());
                }
            } else {
                // Cache is empty but new value still too large
                return;
            }
        }

        // Insert new entry
        self.current_memory_bytes += new_value_size;
        self.map.insert(key.clone(), value);
        self.order.push_back(key);
    }

    fn contains(&self, key: &ColumnKey) -> bool {
        self.map.contains_key(key)
    }

    fn get(&mut self, key: &ColumnKey) -> Option<Arc<LiquidCachedColumn>> {
        println!("Cache Dump - Keys:");
        /*for key in self.map.keys() {
            println!("  - {:?}", key);
        }*/
        println!(
            "Cache size: {}/{} bytes ({:.2}%)",
            self.current_memory_bytes,
            self.config.max_cache_bytes,
            (self.current_memory_bytes as f64 / self.config.max_cache_bytes as f64) * 100.0
        );

        if let Some(value) = self.map.get(key) {
            println!("CACHE HIT");
            // Update the order to mark the key as most recently used.
            self.order.retain(|k| k != key);
            self.order.push_back(key.clone());
            Some(Arc::clone(value))
        } else {
            println!("CACHE MISS");
            None
        }
    }

    fn iter<'a>(
        &'a self,
    ) -> Box<dyn Iterator<Item = (&'a ColumnKey, &'a Arc<LiquidCachedColumn>)> + 'a> {
        Box::new(self.map.iter())
    }

    fn memory_usage(&self) -> u64 {
        self.current_memory_bytes as u64
    }
}
