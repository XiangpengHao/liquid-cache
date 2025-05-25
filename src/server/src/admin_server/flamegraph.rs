use std::sync::Mutex;

use pprof::ProfilerGuard;

pub(super) struct FlameGraph {
    guard: Mutex<Option<ProfilerGuard<'static>>>,
}

impl FlameGraph {
    pub fn new() -> Self {
        Self {
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

    pub fn stop_to_string(&self) -> Result<String, Box<dyn std::error::Error>> {
        let mut guard = self.guard.lock().unwrap();
        let old = guard.take();
        if old.is_none() {
            return Err("FlameGraph is not started".into());
        }
        let profiler = old.unwrap();
        drop(guard);

        let report = profiler.report().build()?;
        let mut svg_data = Vec::new();
        report.flamegraph(&mut svg_data)?;
        Ok(String::from_utf8(svg_data)?)
    }
}
