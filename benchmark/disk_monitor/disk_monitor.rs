use std::{env, sync::atomic::{AtomicU64}, thread, time::Duration};
use sysinfo::{Pid, System, ProcessesToUpdate};
use std::sync::{Arc, atomic::Ordering};
use ctrlc;

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() != 3 {
        print!("Usage:\n
            cargo run --release --bin disk_monitor -- <pid to monitor> <sampling interval in ms>");
    }
    let mut sys = System::new_all();
    let pid = Pid::from(args[1].parse::<usize>().unwrap());
    let sampling_interval = args[2].parse::<u64>().unwrap();     // Sampling interval in ms
    let peak_usage = Arc::new(AtomicU64::new(0));
    let peak_usage_ctrlc = Arc::clone(&peak_usage);

    let _ = ctrlc::set_handler(move || {
        println!("Disk monitor exiting....");
        let peak_usage_bytes = (peak_usage_ctrlc.load(Ordering::Relaxed) as f64) * 1000.0/(sampling_interval as f64);
        let peak_usage_mb = peak_usage_bytes / (1024f64 * 1024f64);
        println!("Peak disk usage: {}", peak_usage_mb);
        std::process::exit(0);
    });
    sys.refresh_processes(ProcessesToUpdate::Some(&[pid]), true);
    
    loop {
        sys.refresh_processes(ProcessesToUpdate::Some(&[pid]), true);
        let process = match sys.process(pid) {
            Some(process) => process,
            None => {
                eprintln!("Process with PID {:?} not found.", pid);
                break;
            }
        };
        let disk_usage = process.disk_usage();
        if disk_usage.written_bytes > 0 || disk_usage.read_bytes > 0 {
            // println!("Total written bytes: {}", disk_usage.written_bytes);
            // println!("Total read bytes: {}", disk_usage.read_bytes);
            peak_usage.store(std::cmp::max(disk_usage.read_bytes + disk_usage.written_bytes, peak_usage.load(Ordering::Relaxed)), Ordering::Relaxed);
        }
        thread::sleep(Duration::from_millis(sampling_interval));
    }

    println!("Disk monitor exiting....");

    let peak_usage_bytes = (peak_usage.load(Ordering::Relaxed) as f64) * 1000.0/(sampling_interval as f64);
    let peak_usage_mb = peak_usage_bytes / (1024f64 * 1024f64);
    println!("Peak disk usage: {}", peak_usage_mb);
}