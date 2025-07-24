//! Microbenchmark for sequential read performance of IO engines: io_uring (polled, kernel polling) and POSIX O_DIRECT.
//! Each benchmark launches N threads to read a set of files sequentially.

use std::fs::{File, OpenOptions};
use std::os::unix::fs::OpenOptionsExt;
use std::os::fd::AsRawFd;
use std::path::PathBuf;
use std::sync::Arc;
use std::{num, thread};
use std::time::{Duration, Instant};
use std::alloc::{Layout, alloc};
use io_uring::{cqueue, opcode, squeue, IoUring};
use std::sync::mpsc::{self, Receiver};
use clap::Parser;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
pub struct Cli {
    /// IO engine: "uring" or "posix"
    #[arg(long)]
    pub engine: String,
    /// Number of threads
    #[arg(long)]
    pub num_threads: usize,
    /// File size in bytes
    #[arg(long)]
    pub file_size: usize,
    /// Chunk size in bytes
    #[arg(long, default_value_t = 4096)]
    pub chunk_size: usize,
    /// List of files to read
    #[arg(long, required = true)]
    pub files: String,
}

fn main() {
    let cli = Cli::parse();
    let engine = match cli.engine.as_str() {
        "uring" => IoEngine::IoUring,
        "posix" => IoEngine::PosixODirect,
        _ => panic!("Unknown engine: {}", cli.engine),
    };
    let files: Vec<std::path::PathBuf> = cli.files
        .split_whitespace()
        .map(PathBuf::from)
        .collect();
    // for file in &files {
    //     println!("{}", file.to_str().unwrap());
    // }
    // println!("here");
    let config = MicrobenchConfig {
        files,
        num_threads: cli.num_threads,
        file_size: cli.file_size,
        chunk_size: cli.chunk_size,
        engine,
    };
    run_microbench(config);
}

pub struct MicrobenchConfig {
    pub files: Vec<PathBuf>,
    pub num_threads: usize,
    pub file_size: usize,
    pub chunk_size: usize,
    pub engine: IoEngine,
}

pub enum IoEngine {
    IoUring,
    PosixODirect,
}

pub fn run_microbench(config: MicrobenchConfig) {
    match config.engine {
        IoEngine::IoUring => run_uring_bench(&config),
        IoEngine::PosixODirect => run_posix_odirect_bench(&config),
    }
}

pub struct IoUringThreadpool {
    pub workers: Vec<thread::JoinHandle<()>>,
    pub rx: Receiver<Vec<(Instant, Option<Instant>)>>,
}

// Check out ebpf -> get familiar with observability tools (bcc)
// Start with one single large file 
impl IoUringThreadpool {
    const NUM_ENTRIES: u32 = 4096;
    pub const BUFFER_ALIGNMENT: usize = 4096;

    pub fn new(files_per_thread: &Vec<Vec<PathBuf>>, file_size: usize, chunk_size: usize) -> IoUringThreadpool {
        let (tx, rx) = mpsc::channel();
        let mut workers = Vec::<thread::JoinHandle<()>>::new();
        let mut first_iter = true;
        for files in files_per_thread.iter() {
            let mut builder = IoUring::<squeue::Entry, cqueue::Entry>::builder();
            builder.setup_iopoll();
            // Add a similar argument to the worker thread as well to sleep when not busy
            builder.setup_sqpoll(50000);
            let ring = builder
                .build(Self::NUM_ENTRIES)
                .expect("Failed to build IoUring instance");
            if first_iter {
                builder.setup_attach_wq(ring.as_raw_fd());
                first_iter = false;
            }
            let files = files.clone();
            let tx = tx.clone();
            let worker = thread::spawn(move || {
                let mut uring_worker = UringWorker::new(ring, files, file_size, chunk_size);
                uring_worker.thread_loop();
                let timings = uring_worker.get_timings().clone();
                tx.send(timings).unwrap();
            });
            workers.push(worker);
        }
        drop(tx);
        IoUringThreadpool { workers, rx }
    }
}

#[derive(Debug)]
pub enum FileIoOp {
    FileRead,
    FileWrite,
}

pub struct IoTask {
    pub base_ptr: *mut u8,
    pub num_bytes: usize,
    pub file: File,
}

fn create_io_task(file_path: &PathBuf, file_size: usize) -> Arc<IoTask> {
    let file = OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_DIRECT)
        .open(file_path)
        .expect("Failed to open file with O_DIRECT");
    let layout = Layout::from_size_align(file_size, IoUringThreadpool::BUFFER_ALIGNMENT).unwrap();
    let base_ptr = unsafe { alloc(layout) };
    Arc::new(IoTask {
        base_ptr,
        num_bytes: file_size,
        file,
    })
}

struct UringWorker {
    ring: io_uring::IoUring,
    completions_array: Vec<usize>,
    tasks: Vec<Option<Arc<IoTask>>>,
    file_paths: Vec<PathBuf>,
    file_size: usize,
    timings: Vec<(Instant, Option<Instant>)>, // (submit_time, completion_time), indexed by file_idx
    chunk_size: usize,
}

impl UringWorker {
    fn new(ring: io_uring::IoUring, file_paths: Vec<PathBuf>, file_size: usize, chunk_size: usize) -> UringWorker {
        let mut completions_array = Vec::<usize>::new();
        completions_array.resize(1<<16, 0);
        let mut tasks = Vec::<Option<Arc<IoTask>>>::new();
        tasks.resize(1<<16, None);
        let timings = vec![(Instant::now(), None); file_paths.len()];
        
        UringWorker {
            ring,
            completions_array,
            tasks,
            file_paths,
            file_size,
            timings,
            chunk_size,
        }
    }

    fn thread_loop(&mut self) {
        let mut inflight = 0;
        let mut file_idx = 0;
        let num_files = self.file_paths.len();
        let mut last_io: Option<Arc<IoTask>> = None;
        while file_idx < num_files || inflight > 0 {
            // Submit next file if available
            if file_idx < num_files {
                let mut task: Option<Arc<IoTask>> = None;
                if last_io.is_some() {
                    task = last_io.take();
                } else {
                    task = Some(create_io_task(&self.file_paths[file_idx], self.file_size));
                }
                let submit_time = Instant::now();
                let num_chunks = self.submit_reads(&task.as_ref().unwrap(), file_idx.clone());
                if num_chunks > 0 {
                    self.tasks[file_idx] = task;
                    self.completions_array[file_idx] = num_chunks;
                    self.timings[file_idx].0 = submit_time;
                    inflight += 1;
                    file_idx += 1;
                    let _submitted = self.ring.submit();
                } else {
                    last_io = task;
                }
            }
            // Poll completions
            let completed = self.poll_completions();

            inflight -= completed;
        }
    }

    fn submit_reads(&mut self, task: &IoTask, file_idx: usize) -> usize {
        let ring = &mut self.ring;
        let sq = &mut (ring.submission());
        let mut buf_ptr = task.base_ptr;
        let num_chunks = (task.num_bytes + self.chunk_size - 1) / self.chunk_size;
        sq.sync();
        let remaining = sq.capacity() - sq.len();
        if num_chunks > remaining {
            return 0;
        }
        let user_data = (file_idx as u64)<<48;
        for i in 0..num_chunks {
            // println!("{}", i);
            let read_op = opcode::Read::new(
                io_uring::types::Fd(task.file.as_raw_fd()),
                buf_ptr,
                self.chunk_size as _,
            );
            let sqe = read_op
                .offset((i * self.chunk_size) as u64)
                .build()
                .user_data(user_data + i as u64);
            unsafe {
                while let Err(_) = sq.push(&sqe) {
                    sq.sync(); // Submit to kernel to make space
                }
                buf_ptr = buf_ptr.add(self.chunk_size);
            }
        }
        sq.sync();
        num_chunks
    }

    /// Returns number of tasks completed in this poll
    fn poll_completions(&mut self) -> usize {
        let cq = &mut self.ring.completion();
        let mut completed = 0;
        loop {
            cq.sync();
            match cq.next() {
                Some(cqe) => {
                    let errno = -cqe.result();
                    let err = std::io::Error::from_raw_os_error(errno);
                    assert!(
                        cqe.result() == self.chunk_size as i32 || cqe.result() == 0,
                        "Read cqe result error: {err}"
                    );
                    let opcode = (cqe.user_data()>>48) as usize;
                    let remaining = &mut self.completions_array[opcode];
                    *remaining -= 1;
                    if *remaining == 0 {
                        self.timings[opcode].1 = Some(Instant::now());
                        completed += 1;
                    }
                    self.tasks[opcode].take();
                },
                None => {
                    break;
                }
            }
        }
        completed
    }

    pub fn get_timings(&self) -> &Vec<(Instant, Option<Instant>)> {
        &self.timings
    }
}


/// Benchmark sequential read using io_uring (polled, kernel polling)
pub fn run_uring_bench(config: &MicrobenchConfig) {
    println!("Running io_uring sequential read benchmark with {} threads", config.num_threads);
    println!("File size: {}", config.file_size);
    let files_per_thread = split_files(&config.files, config.num_threads);
    let mut threadpool = IoUringThreadpool::new(&files_per_thread, config.file_size, config.chunk_size);
    for worker in threadpool.workers.drain(..) {
        let _ = worker.join();
    }
    // Gather timings
    let mut all_latencies = Vec::new();
    for timings in threadpool.rx.iter() {
        for (start, end_opt) in timings {
            if let Some(end) = end_opt {
                let ms = (end - start).as_secs_f64() * 1000.0;
                all_latencies.push(ms);
            }
        }
    }
    if all_latencies.is_empty() {
        println!("No timings collected");
        return;
    }
    all_latencies.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mean = all_latencies.iter().sum::<f64>() / all_latencies.len() as f64;
    let p50 = percentile(&all_latencies, 0.5);
    let p70 = percentile(&all_latencies, 0.7);
    let p90 = percentile(&all_latencies, 0.9);
    println!("io_uring benchmark completed");
    println!("Mean: {:.2} ms, p50: {:.2} ms, p70: {:.2} ms, p90: {:.2} ms", mean, p50, p70, p90);
}

fn percentile(sorted: &[f64], pct: f64) -> f64 {
    let idx = ((sorted.len() as f64) * pct).ceil() as usize - 1;
    sorted.get(idx).copied().unwrap_or(*sorted.last().unwrap())
}

/// Benchmark sequential read using POSIX O_DIRECT
pub fn run_posix_odirect_bench(config: &MicrobenchConfig) {
    println!("Running POSIX O_DIRECT sequential read benchmark with {} threads", config.num_threads);
    let files_per_thread = split_files(&config.files, config.num_threads);
    use std::sync::mpsc;
    let (tx, rx) = mpsc::channel();
    let mut workers = Vec::new();
    for files in files_per_thread.iter() {
        let files = files.clone();
        let tx = tx.clone();
        let file_size = config.file_size;
        let worker = thread::spawn(move || {
            let mut latencies = Vec::with_capacity(files.len());
            for file_path in files {
                let file = OpenOptions::new()
                    .read(true)
                    .custom_flags(libc::O_DIRECT)
                    .open(&file_path)
                    .expect("Failed to open file with O_DIRECT");
                let start = Instant::now();
                let fd = file.as_raw_fd();
                let layout = Layout::from_size_align(file_size, 4096).unwrap();
                let buf = unsafe { alloc(layout) };
                let mut total_read = 0;
                while total_read < file_size {
                    let to_read = std::cmp::min(4096, file_size - total_read);
                    let ret = unsafe {
                        libc::pread(
                            fd,
                            buf.add(total_read) as *mut libc::c_void,
                            to_read,
                            total_read as libc::off_t,
                        )
                    };
                    if ret < 0 {
                        panic!("pread failed: {}", std::io::Error::last_os_error());
                    }
                    total_read += ret as usize;
                }
                let end = Instant::now();
                let ms = (end - start).as_secs_f64() * 1000.0;
                latencies.push(ms);
            }
            tx.send(latencies).unwrap();
        });
        workers.push(worker);
    }
    drop(tx);
    for worker in workers.drain(..) {
        let _ = worker.join();
    }
    // Gather timings
    let mut all_latencies = Vec::new();
    for latencies in rx {
        all_latencies.extend(latencies);
    }
    if all_latencies.is_empty() {
        println!("No timings collected");
        return;
    }
    all_latencies.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mean = all_latencies.iter().sum::<f64>() / all_latencies.len() as f64;
    let p50 = percentile(&all_latencies, 0.5);
    let p70 = percentile(&all_latencies, 0.7);
    let p90 = percentile(&all_latencies, 0.9);
    println!("POSIX O_DIRECT benchmark completed");
    println!("Mean: {:.2} ms, p50: {:.2} ms, p70: {:.2} ms, p90: {:.2} ms", mean, p50, p70, p90);
}

/// Helper: Split files among threads
fn split_files(files: &[PathBuf], num_threads: usize) -> Vec<Vec<PathBuf>> {
    let mut out = vec![vec![]; num_threads];
    for (i, file) in files.iter().enumerate() {
        out[i % num_threads].push(file.clone());
    }
    out
} 