use std::{path::PathBuf, sync::Mutex};

use liquid_cache_server::StatsCollector;
use pprof::ProfilerGuard;

pub struct FlameGraphReport {
    output_dir: PathBuf,
    guard: Mutex<Option<ProfilerGuard<'static>>>,
}

impl FlameGraphReport {
    pub fn new(output: PathBuf) -> Self {
        Self {
            output_dir: output,
            guard: Mutex::new(None),
        }
    }
}

impl StatsCollector for FlameGraphReport {
    fn start(&self) {
        let mut guard = self.guard.lock().unwrap();
        if guard.is_some() {
            return;
        }
        let old = guard.take();
        assert!(old.is_none(), "FlameGraphReport is already started");
        *guard = Some(pprof::ProfilerGuardBuilder::default().build().unwrap());
    }

    fn stop(&self) {
        let mut lock_guard = self.guard.lock().unwrap();
        if lock_guard.is_none() {
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
        let filename = format!("flamegraph-{:02}-{:02}.svg", minute, second);
        let filepath = self.output_dir.join(filename);
        let file = std::fs::File::create(&filepath).unwrap();
        report.flamegraph(file).unwrap();
        log::info!("Flamegraph saved to {}", filepath.display());
    }
}
