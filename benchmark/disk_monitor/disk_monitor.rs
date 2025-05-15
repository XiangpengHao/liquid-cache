use std::{env, thread, time::Duration};
use sysinfo::{Pid, System, ProcessesToUpdate};

fn main() {
    let args: Vec<String> = env::args().collect();
    let mut sys = System::new_all();
    let pid = Pid::from(args[1].parse::<usize>().unwrap());
    
    loop {
        sys.refresh_processes(ProcessesToUpdate::Some(&[pid]), true);
        let process = match sys.process(pid) {
            Some(process) => process,
            None => {
                eprintln!("Process with PID {:?} not found.", pid);
                return;
            }
        };
        let disk_usage = process.disk_usage();
        println!("Total written bytes: {}", disk_usage.written_bytes);
        println!("Total read bytes: {}", disk_usage.read_bytes);
        thread::sleep(Duration::from_secs(1));
    }
}