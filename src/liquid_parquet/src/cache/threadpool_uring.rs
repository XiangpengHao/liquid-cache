use std::{os::fd::RawFd, pin::Pin, sync::{atomic::{fence, Ordering}, Arc, LazyLock, Mutex}, task::{Context, Poll, Waker}, thread};

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
    completed: AtomicBool,
    waker: Mutex<Option<Waker>>,
}

impl IoTask {
    pub(crate) fn new(op: FileIoOp, base_ptr: *mut u8, num_bytes: usize, fd: RawFd) -> IoTask {
        return IoTask {op, base_ptr, num_bytes, fd, completed: AtomicBool::new(false), waker: Mutex::<Option<Waker>>::new(None)}
    }

    pub(crate) fn set_waker(self: &Self, waker: Waker) {
        let mut guard = self.waker.lock().unwrap();
        *guard = Some(waker);
    }

    pub(crate) fn get_ptr(self: &Self) -> *mut u8 {
        self.base_ptr
    }
}

unsafe impl Send for IoTask {}
unsafe impl Sync for IoTask {}

pub struct IoUringThreadpool {
    sender: crossbeam_channel::Sender<Arc<IoTask>>,
    workers: Vec<thread::JoinHandle<()>>,
}

unsafe impl Sync for IoUringThreadpool {}

pub(crate) static IO_URING_THREAD_POOL_INST: LazyLock<IoUringThreadpool> = LazyLock::new(|| IoUringThreadpool::new(5));

static ENABLED: AtomicBool = AtomicBool::new(true);

impl IoUringThreadpool {
    const NUM_ENTRIES: u32 = 4096;
    pub const BUFFER_ALIGNMENT: usize = 4096;

    fn new(num_workers: u32) -> IoUringThreadpool {
        let mut builder = IoUring::<squeue::Entry, cqueue::Entry>::builder();
        builder.setup_iopoll();
        // Add a similar argument to the worker thread as well to sleep when not busy
        builder.setup_sqpoll(50000);
        
        let (sender, receiver) = crossbeam_channel::unbounded::<Arc<IoTask>>();
        let mut workers = Vec::<thread::JoinHandle<()>>::new();

        for i in 0..num_workers as usize {
            let ring = builder
                .build(Self::NUM_ENTRIES)
                .expect("Failed to build IoUring instance");
            if i == 0 {
                builder.setup_attach_wq(ring.as_raw_fd());
            }
            let receiver_clone = receiver.clone();
            let worker = thread::spawn(move || {
                let mut uring_worker = UringWorker::new(receiver_clone, ring);
                uring_worker.thread_loop();
            });
            workers.push(worker);
        }
        
        IoUringThreadpool {
            sender: sender,
            workers: workers,
        }
    }

    pub(crate) fn submit_task(self: &Self, task: Arc<IoTask>) {
        self.sender.send(task.clone()).expect("Failed to submit task through channel");
    }

}

impl Drop for IoUringThreadpool {
    fn drop(self: &mut Self) {
        ENABLED.store(false, Ordering::Relaxed);
        while let Some(worker) = self.workers.pop() {
            let _ = worker.join();
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
    const CHUNK_SIZE: usize = 8192 * 4;

    fn new(channel: crossbeam_channel::Receiver<Arc<IoTask>>, ring: io_uring::IoUring) -> UringWorker {
        let mut completions_array = Vec::<usize>::new();
        completions_array.resize(1<<16, 0);

        let mut tasks = Vec::<Option<Arc<IoTask>>>::new();
        tasks.resize(1<<16, None);
        UringWorker { channel: channel, ring: ring, op_counter: 0, completions_array: completions_array, tasks: tasks }
    }

    fn thread_loop(self: &mut Self) {
        loop {
            if !ENABLED.load(Ordering::Relaxed) {
                break;
            }
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
                    let opcode = (cqe.user_data()>>48) as usize;
                    let remaining = &mut self.completions_array[opcode];
                    *remaining -= 1;
                    if *remaining == 0 {
                        self.tasks[opcode].as_ref().unwrap().completed.store(true, Ordering::Relaxed);
                        let mut guard = self.tasks[opcode].as_ref().unwrap().waker.lock().unwrap();
                        if let Some(waker) = guard.take() {
                            waker.wake();
                        }
                    }
                },
                None => {
                    break;
                }
            }
        }
    }
}

enum UringState {
    Initialized,
    Submitted,
}

pub struct UringFuture {
    task: Arc<IoTask>,
    state: UringState,
}

impl UringFuture {
    pub fn new(task: Arc<IoTask>) -> UringFuture {
        return UringFuture { task: task, state: UringState::Initialized }
    }
}

impl Future for UringFuture {
    type Output = ();

    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<()> {
        match self.state {
            UringState::Initialized => {
                // Measure the io latency
                IO_URING_THREAD_POOL_INST.submit_task(self.task.clone());
                self.task.set_waker(cx.waker().clone());
                self.state = UringState::Submitted;
                return Poll::Pending;
            },
            UringState::Submitted => {
                match self.task.completed.load(Ordering::Relaxed) {
                    false => {
                        self.task.set_waker(cx.waker().clone());
                        return Poll::Pending;
                    },
                    true => {
                        return Poll::Ready(());
                    }
                }
            }
        }
    }
}