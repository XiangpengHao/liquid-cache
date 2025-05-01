use std::{
    path::PathBuf,
    sync::{Mutex, atomic::AtomicU32},
};

use pprof::ProfilerGuard;

pub(super) struct FlameGraph {
    id: AtomicU32,
    guard: Mutex<Option<ProfilerGuard<'static>>>,
}

impl FlameGraph {
    pub fn new() -> Self {
        Self {
            id: AtomicU32::new(0),
            guard: Mutex::new(None),
        }
    }

    pub fn start(&self) {
        let mut guard = self.guard.lock().unwrap();
        let old = guard.take();
        assert!(old.is_none(), "FlameGraph is already started");
        *guard = Some(
            pprof::ProfilerGuardBuilder::default()
                .frequency(500)
                .blocklist(&["libpthread.so.0", "libm.so.6", "libgcc_s.so.1"])
                .build()
                .unwrap(),
        );
    }

    pub fn stop(&self, output_dir: &PathBuf) -> PathBuf {
        let mut guard = self.guard.lock().unwrap();
        let old = guard.take();
        assert!(old.is_some(), "FlameGraph is not started");
        let profiler = old.unwrap();
        drop(guard);

        let report = profiler.report().build().unwrap();
        let now = std::time::SystemTime::now();
        let datetime = now.duration_since(std::time::UNIX_EPOCH).unwrap();
        let minute = (datetime.as_secs() / 60) % 60;
        let second = datetime.as_secs() % 60;
        let id = self.id.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let filename = format!("flamegraph-id{id:02}-{minute:02}-{second:03}.svg",);
        let filepath = output_dir.join(filename);
        let file = std::fs::File::create(&filepath).unwrap();
        report.flamegraph(file).unwrap();
        filepath
    }
}
