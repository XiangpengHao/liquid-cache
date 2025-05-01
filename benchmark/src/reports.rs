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
        let filename = format!("flamegraph-id{flame_graph_id:02}-{minute:02}-{second:03}.svg");
        let filepath = self.output_dir.join(filename);
        let file = std::fs::File::create(&filepath).unwrap();
        report.flamegraph(file).unwrap();
        log::info!("Flamegraph saved to {}", filepath.display());
    }
}
