use liquid_cache_parquet::LiquidCacheRef;
use liquid_cache_server::StatsCollector;
use pprof::ProfilerGuard;
use std::{
    path::PathBuf,
    sync::{
        Mutex,
        atomic::{AtomicU32, AtomicUsize},
    },
};

pub struct FlameGraphReport {
    output_dir: PathBuf,
    guard: Mutex<Option<ProfilerGuard<'static>>>,
    running_count: AtomicUsize,
    flame_graph_id: AtomicU32,
}

impl FlameGraphReport {
    pub fn new(output: PathBuf) -> Self {
        Self {
            output_dir: output,
            guard: Mutex::new(None),
            running_count: AtomicUsize::new(0),
            flame_graph_id: AtomicU32::new(0),
        }
    }
}

impl StatsCollector for FlameGraphReport {
    fn start(&self) {
        self.running_count
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let mut guard = self.guard.lock().unwrap();
        if guard.is_some() {
            return;
        }
        let old = guard.take();
        assert!(old.is_none(), "FlameGraphReport is already started");
        *guard = Some(
            pprof::ProfilerGuardBuilder::default()
                .frequency(500)
                .blocklist(&["libpthread.so.0", "libm.so.6", "libgcc_s.so.1"])
                .build()
                .unwrap(),
        );
    }

    fn stop(&self) {
        let previous = self
            .running_count
            .fetch_sub(1, std::sync::atomic::Ordering::Relaxed);
        if previous != 1 {
            return;
        }

        let mut lock_guard = self.guard.lock().unwrap();
        if lock_guard.is_none() {
            log::warn!("FlameGraphReport is not started");
            return;
        }
        let profiler = lock_guard.take().unwrap();
        drop(lock_guard);
        let report = profiler.report().build().unwrap();
        drop(profiler);

        let now = std::time::SystemTime::now();
        let datetime = now.duration_since(std::time::UNIX_EPOCH).unwrap();
        let minute = (datetime.as_secs() / 60) % 60;
        let second = datetime.as_secs() % 60;
        let flame_graph_id = self
            .flame_graph_id
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let filename = format!(
            "flamegraph-id{:02}-{:02}-{:03}.svg",
            flame_graph_id, minute, second
        );
        let filepath = self.output_dir.join(filename);
        let file = std::fs::File::create(&filepath).unwrap();
        report.flamegraph(file).unwrap();
        log::info!("Flamegraph saved to {}", filepath.display());
    }
}

pub struct StatsReport {
    output_dir: PathBuf,
    cache: LiquidCacheRef,
    running_count: AtomicUsize,
    stats_id: AtomicU32,
}

impl StatsReport {
    pub fn new(output: PathBuf, cache: LiquidCacheRef) -> Self {
        Self {
            output_dir: output,
            cache,
            running_count: AtomicUsize::new(0),
            stats_id: AtomicU32::new(0),
        }
    }
}

impl StatsCollector for StatsReport {
    fn start(&self) {
        self.running_count
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }

    fn stop(&self) {
        let previous = self
            .running_count
            .fetch_sub(1, std::sync::atomic::Ordering::Relaxed);
        // We only write stats once, after every thread finishes.
        if previous != 1 {
            return;
        }
        let now = std::time::SystemTime::now();
        let datetime = now.duration_since(std::time::UNIX_EPOCH).unwrap();
        let minute = (datetime.as_secs() / 60) % 60;
        let second = datetime.as_secs() % 60;
        let stats_id = self
            .stats_id
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let filename = format!(
            "stats-id{:02}-{:02}-{:03}.parquet",
            stats_id, minute, second
        );
        let parquet_file_path = self.output_dir.join(filename);
        self.cache.write_stats(&parquet_file_path).unwrap();
        log::info!("Stats saved to {}", parquet_file_path.display());
    }
}

pub struct CacheTraceReport {
    output_dir: PathBuf,
    liquid_cache: LiquidCacheRef,
    running_count: AtomicUsize,
    trace_id: AtomicU32,
}

impl CacheTraceReport {
    pub fn new(output: PathBuf, liquid_cache: LiquidCacheRef) -> Self {
        Self {
            output_dir: output,
            liquid_cache,
            running_count: AtomicUsize::new(0),
            trace_id: AtomicU32::new(0),
        }
    }
}

impl StatsCollector for CacheTraceReport {
    fn start(&self) {
        self.liquid_cache.enable_trace();
        self.running_count
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }

    fn stop(&self) {
        self.liquid_cache.disable_trace();
        let previous = self
            .running_count
            .fetch_sub(1, std::sync::atomic::Ordering::Relaxed);
        if previous != 1 {
            return;
        }
        let now = std::time::SystemTime::now();
        let datetime = now.duration_since(std::time::UNIX_EPOCH).unwrap();
        let minute = (datetime.as_secs() / 60) % 60;
        let second = datetime.as_secs() % 60;
        let trace_id = self
            .trace_id
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let filename = format!(
            "cache-trace-id{:02}-{:02}-{:03}.parquet",
            trace_id, minute, second
        );
        let parquet_file_path = self.output_dir.join(filename);
        self.liquid_cache.flush_trace(&parquet_file_path);
        log::info!("Cache trace saved to {}", parquet_file_path.display());
    }
}
