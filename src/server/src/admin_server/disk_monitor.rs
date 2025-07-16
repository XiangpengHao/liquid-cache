use std::{
    path::PathBuf,
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering},
    },
    thread,
    time::Duration,
};

use hdrhistogram::Histogram;
use std::fs::File;
use std::io::{BufRead, BufReader};
use sysinfo::{Pid, ProcessesToUpdate, System};

pub(super) struct DiskMonitor {
    server_pid: Pid,
    enabled: AtomicBool,
    recorder_thread: Mutex<Option<thread::JoinHandle<()>>>,
    histogram: Mutex<Histogram<u64>>,
    path: Mutex<Option<PathBuf>>,
}

unsafe impl Send for DiskMonitor {}

unsafe impl Sync for DiskMonitor {}

impl DiskMonitor {
    const SAMPLING_INTERVAL: u64 = 100;

    pub(crate) fn new() -> DiskMonitor {
        let histogram =
            Histogram::<u64>::new_with_max(20000, 4).expect("Failed to create histogram instance");
        DiskMonitor {
            server_pid: Pid::from(std::process::id() as usize),
            enabled: AtomicBool::new(false),
            recorder_thread: Mutex::new(None),
            histogram: Mutex::new(histogram),
            path: Mutex::new(None),
        }
    }

    fn set_path(self: &Arc<Self>, path: String) {
        let path_buf = PathBuf::from(path);
        let mut path_mutex = self.path.lock().unwrap();
        *path_mutex = Some(path_buf);
    }

    pub(crate) fn start_recording(self: Arc<Self>, path: String) {
        self.enabled.store(true, Ordering::Relaxed);
        self.set_path(path);
        let self_clone = Arc::clone(&self);
        let handle = Some(thread::spawn(move || {
            self_clone.thread_loop();
        }));
        let mut recorder_thread = self.recorder_thread.lock().unwrap();
        *recorder_thread = handle;
    }

    fn thread_loop(self: Arc<Self>) {
        let mut sys = System::new_all();
        let device_name = "nvme1n1p3";
        let mut inflight_samples = Vec::new();
        loop {
            if !self.enabled.load(Ordering::Relaxed) {
                break;
            }
            sys.refresh_processes(ProcessesToUpdate::Some(&[self.server_pid]), true);
            let process = match sys.process(self.server_pid) {
                Some(process) => process,
                None => {
                    eprintln!("Process with PID {:?} not found.", self.server_pid);
                    break;
                }
            };
            let disk_usage = process.disk_usage();
            let usage = disk_usage.read_bytes + disk_usage.written_bytes;
            let usage_bytes = (usage as f64) * 1000.0 / (Self::SAMPLING_INTERVAL as f64);
            let usage_mb = usage_bytes / (1024f64 * 1024f64);
            let inflight = get_inflight_requests(&device_name).unwrap_or(0);
            inflight_samples.push(inflight);
            {
                let mut histogram = self.histogram.lock().unwrap();
                (*histogram)
                    .record(usage_mb as u64)
                    .expect("Failed to record disk usage sample");
            }
            thread::sleep(Duration::from_millis(Self::SAMPLING_INTERVAL));
        }
        let mut histogram = self.histogram.lock().unwrap();
        for i in (50..=90).step_by(5) {
            let quantile = i as f64 / 100.0;
            log::info!("p{} disk usage: {}", i, histogram.value_at_quantile(quantile));
        }
        log::info!("p99 disk usage: {}", histogram.value_at_quantile(0.99));

        if !inflight_samples.is_empty() {
            inflight_samples.sort_unstable();
            let percentiles = [50, 60, 70, 80, 90, 99];
            for &p in &percentiles {
                let idx = ((p as f64 / 100.0) * (inflight_samples.len() as f64 - 1.0)).round() as usize;
                let val = inflight_samples[idx];
                log::info!("p{} inflight requests: {}", p, val);
            }
        }
    }

    fn dump_histogram(self: &Arc<Self>) {
        let path_mutex = self.path.lock().unwrap();
        let path = (*path_mutex).as_ref().unwrap();
        let filename = path.join("histogram.hgrm");
        // let mut file = File::create(&filename).expect("Failed to create .hgrm file");
        let mut histogram = self.histogram.lock().unwrap();
        // V2Serializer::new()
        //     .serialize(&histogram, &mut file)
        //     .expect("Failed to serialize histogram");
        // for i in (50..=90).step_by(5) {
        //     log::info!("p{} disk usage: {}", i, histogram.value_at_quantile(i as f64 / 100.0));
        // }
        histogram.reset();
    }

    pub(crate) fn stop_recording(self: Arc<Self>) {
        self.enabled.store(false, Ordering::Relaxed);
        let mut recorder_thread = self.recorder_thread.lock().unwrap();
        if let Some(handle) = recorder_thread.take() {
            handle.join().expect("Failed to join recorder thread");
        }
        self.dump_histogram();
    }
}

/// Returns the number of in-flight requests for a given device by parsing /proc/diskstats.
fn get_inflight_requests(device: &str) -> Option<u64> {
    let file = File::open("/proc/diskstats").ok()?;
    let reader = BufReader::new(file);
    for line in reader.lines() {
        let line = line.ok()?;
        if line.contains(&format!(" {}", device)) {
            let fields: Vec<&str> = line.split_whitespace().collect();
            // io_in_progress is the 12th field (index 11) in /proc/diskstats
            if fields.len() > 11 {
                return fields[11].parse().ok();
            }
        }
    }
    None
}
