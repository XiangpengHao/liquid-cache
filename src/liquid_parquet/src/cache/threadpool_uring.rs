use std::{os::fd::RawFd, pin::Pin, sync::{atomic::Ordering, Arc, LazyLock}, task::{Context, Poll}, thread};

use std::sync::atomic::AtomicBool;
use io_uring::{cqueue, opcode, squeue, IoUring};
use std::os::fd::AsRawFd;

pub enum FileIoOp {
    FileRead,
    FileWrite,
}

pub struct IoTask {
    op: FileIoOp,
    base_ptr: *mut u8,
    num_bytes: usize,
    fd: RawFd,
    completed: AtomicBool
}

impl IoTask {
    pub fn new(op: FileIoOp, base_ptr: *mut u8, num_bytes: usize, fd: RawFd) -> IoTask {
        return IoTask {op, base_ptr, num_bytes, fd, completed: AtomicBool::new(false)}
    }
}

unsafe impl Send for IoTask {}
unsafe impl Sync for IoTask {}

pub struct IoUringThreadpool {
    sender: crossbeam_channel::Sender<Arc<IoTask>>,
    workers: Vec<thread::JoinHandle<()>>,
    // enabled: Atomic<bool>,
}

unsafe impl Sync for IoUringThreadpool {}

pub(crate) static IO_URING_THREAD_POOL_INST: LazyLock<IoUringThreadpool> = LazyLock::new(|| IoUringThreadpool::new(4));

impl IoUringThreadpool {
    const NUM_ENTRIES: u32 = 512;
    pub const BUFFER_ALIGNMENT: usize = 4096;

    fn new(num_workers: u32) -> IoUringThreadpool {
        let mut builder = IoUring::<squeue::Entry, cqueue::Entry>::builder();
        builder.setup_iopoll();
        // Add a similar argument to the worker thread as well to sleep when not busy
        builder.setup_sqpoll(50000);
        
        let (sender, receiver) = crossbeam_channel::unbounded::<Arc<IoTask>>();
        let mut workers = Vec::<thread::JoinHandle<()>>::new();
        let enabled = AtomicBool::new(false);

        for i in 0..num_workers as usize {
            let ring = builder
                .build(Self::NUM_ENTRIES)
                .expect("Failed to build IoUring instance");
            if i == 0 {
                builder.setup_attach_wq(ring.as_raw_fd());
            }
            let receiver_clone = receiver.clone();
            // let enabled_ref = &enabled;
            let worker = thread::spawn(move || {
                let mut uring_worker = UringWorker::new(receiver_clone, ring);
                uring_worker.thread_loop();
            });
            workers.push(worker);
        }
        
        IoUringThreadpool {
            sender: sender,
            workers: workers,
            // enabled: enabled,
        }
    }

    pub(crate) async fn submit_task(self: &Self, task: Arc<IoTask>) -> UringFuture {
        self.sender.send(task.clone()).expect("Failed to submit task through channel");
        UringFuture {
            task: task,
        }
    }
}

struct UringWorker {
    channel: crossbeam_channel::Receiver<Arc<IoTask>>,
    ring: io_uring::IoUring,
    // Assumption: There wont be more than 2^16 tasks on-the-fly at once
    op_counter: u16,
    completions_array: Vec<usize>,
    tasks: Vec<Option<Arc<IoTask>>>,
}

impl UringWorker {
    const CHUNK_SIZE: usize = 8192;

    fn new(channel: crossbeam_channel::Receiver<Arc<IoTask>>, ring: io_uring::IoUring) -> UringWorker {
        let mut completions_array = Vec::<usize>::new();
        completions_array.resize(1<<16, 0);

        let mut tasks = Vec::<Option<Arc<IoTask>>>::new();
        tasks.resize(1<<16, None);
        UringWorker { channel: channel, ring: ring, op_counter: 0, completions_array: completions_array, tasks: tasks }
    }

    fn thread_loop(self: &mut Self) {
        loop {
            // TODO(): Check enabled
            // Consume tasks from channel and submit them to the ring
            loop {
                let res = self.channel.try_recv();
                if res.is_err() {break;}
                let task = res.unwrap();
                let num_chunks = match task.op {
                    FileIoOp::FileRead => {
                        self.submit_reads(&task)
                    }
                    FileIoOp::FileWrite => {
                        self.submit_writes(&task)
                    }
                };
                self.tasks[self.op_counter as usize] = Some(task);
                self.completions_array[self.op_counter as usize] = num_chunks;
                self.op_counter = self.op_counter.wrapping_add(1);                
            }

            self.poll_completions();
        }
    }

    fn submit_reads(self: &mut Self, task: &IoTask) -> usize {
        let sq = &mut (self.ring.submission());
        let mut buf_ptr = task.base_ptr;
        let num_chunks = (task.num_bytes + Self::CHUNK_SIZE - 1) / Self::CHUNK_SIZE;
        // Most significant 16 bits are the op counter
        let user_data = (self.op_counter as u64)<<48;

        for i in 0..num_chunks {
            let read_op = opcode::Read::new(
                io_uring::types::Fd(task.fd),
                buf_ptr,
                Self::CHUNK_SIZE as _, // Logically, this should be the remaining number of bytes, but that fails...
            );

            let sqe = read_op
                .offset((i * Self::CHUNK_SIZE) as u64)
                .build()
                .user_data(user_data + i as u64);
            unsafe {
                sq.push(&sqe)
                    .expect("Failed to push to submission queue during read");
                buf_ptr = buf_ptr.add(Self::CHUNK_SIZE);
            }
        }
        sq.sync();
        num_chunks
    }

    fn submit_writes(self: &mut Self, task: &IoTask) -> usize {
        let sq = &mut (self.ring.submission());
        let mut buf_ptr = task.base_ptr;
        let num_chunks = (task.num_bytes + Self::CHUNK_SIZE - 1) / Self::CHUNK_SIZE;
        // Most significant 16 bits are the op counter
        let user_data = (self.op_counter as u64)<<48;

        for i in 0..num_chunks {
            let write_op = opcode::Write::new(
                io_uring::types::Fd(task.fd),
                buf_ptr,
                Self::CHUNK_SIZE as _, // Logically, this should be the remaining number of bytes, but that fails...
            );

            let sqe = write_op
                .offset((i * Self::CHUNK_SIZE) as u64)
                .build()
                .user_data(user_data + i as u64);
            unsafe {
                sq.push(&sqe)
                    .expect("Failed to push to submission queue during read");
                buf_ptr = buf_ptr.add(Self::CHUNK_SIZE);
            }
        }
        sq.sync();
        num_chunks
    }

    fn poll_completions(self: &mut Self) {
        let cq = &mut self.ring.completion();
        loop {
            cq.sync();
            match cq.next() {
                Some(cqe) => {
                    let errno = -cqe.result();
                    let err = std::io::Error::from_raw_os_error(errno);
                    assert!(
                        cqe.result() == Self::CHUNK_SIZE as i32,
                        "Read cqe result error: {err}"
                    );
                    let opcode = (cqe.user_data()>>48) as u16;
                    let remaining = &mut self.completions_array[opcode as usize];
                    *remaining -= 1;
                    if *remaining == 0 {
                        // TODO(): Should we call the waker here??
                        self.tasks[opcode as usize].as_ref().unwrap().completed.store(true, Ordering::Relaxed);
                    }
                }
                None => {
                    break;
                }
            }
        }
    }
}

pub struct UringFuture {
    task: Arc<IoTask>,
}

impl Future for UringFuture {
    type Output = ();

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<()> {
        use std::sync::atomic::Ordering;
        loop {
            match self.task.completed.load(Ordering::Relaxed) {
                false => {
                    return Poll::Pending;
                },
                true => {
                    cx.waker().wake_by_ref();
                    return Poll::Ready(());
                }
            }
        }
    }
}