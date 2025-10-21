#![allow(missing_docs)]
use std::{
    alloc::Layout,
    fmt::Display,
    ops::Range,
    os::fd::RawFd,
    pin::Pin,
    str::FromStr,
    sync::{
        Arc, Mutex, OnceLock,
        atomic::{AtomicBool, Ordering},
    },
    task::{Context, Poll, Waker},
    thread,
};

use bytes::Bytes;
use io_uring::{IoUring, cqueue, opcode, squeue};

const BLOCK_ALIGN: usize = 4096;

/**
Represents an IO request to the uring worker thread
*/
pub trait IoTask: Send + Sync {
    #[inline]
    fn set_waker(&self, waker: Waker) {
        let mut guard = self.waker().lock().unwrap();
        *guard = Some(waker);
    }

    /**
    Get the waker associated with this IO request
     */
    fn waker(&self) -> &Mutex<Option<Waker>>;

    /**
    Converts the IO request to an IO uring submission queue entry
    */
    fn get_sqe(&self, user_data: u64) -> squeue::Entry;

    fn completed(&self) -> &AtomicBool;

    /**
    Wake the future that submitted this IO request
     */
    #[inline]
    fn notify_waker(&self) {
        self.completed().store(true, Ordering::Relaxed);
        let mut guard = self.waker().lock().unwrap();
        if let Some(waker) = guard.take() {
            waker.wake();
        }
    }

    fn debug_print(&self);
}

/**
Represents a request to read from a file
*/
#[allow(unused)]
pub struct FileReadTask {
    base_ptr: *mut u8,
    fd: RawFd,
    completed: AtomicBool,
    waker: Mutex<Option<Waker>>,
    range: Range<u64>,
    start_padding: usize,
    end_padding: usize,
}

impl FileReadTask {
    #[allow(unused)]
    pub fn new(range: Range<u64>, fd: RawFd) -> FileReadTask {
        let mut start_padding: usize = 0;
        let mut end_padding: usize = 0;
        if get_io_mode() == IoMode::Direct {
            // Padding must be applied to ensure that starting and ending addresses are block-aligned
            start_padding = range.start as usize & (BLOCK_ALIGN - 1);
            end_padding = if range.end as usize & (BLOCK_ALIGN - 1) == 0 {
                0
            } else {
                BLOCK_ALIGN - (range.end as usize & (BLOCK_ALIGN - 1))
            };
        }
        let layout = Layout::from_size_align(
            (range.end - range.start) as usize + start_padding + end_padding,
            4096,
        )
        .expect("Failed to create memory layout for disk read result");
        let base_ptr = unsafe { std::alloc::alloc(layout) };
        FileReadTask {
            base_ptr,
            fd,
            completed: AtomicBool::new(false),
            waker: Mutex::<Option<Waker>>::new(None),
            range,
            start_padding,
            end_padding,
        }
    }

    #[inline]
    pub fn ptr(&self) -> *const u8 {
        self.base_ptr as *const u8
    }

    /**
    Return a bytes object holding the result of the read operation
     */
    #[inline]
    pub fn get_bytes(&self) -> Bytes {
        let total_bytes =
            (self.range.end - self.range.start) as usize + self.start_padding + self.end_padding;
        unsafe {
            let vec = Vec::from_raw_parts(self.base_ptr, total_bytes, total_bytes);
            // Convert to vec in order to transfer ownership of underlying pointer
            let owned_slice: Box<[u8]> = vec.into_boxed_slice();
            // The below slice operation removes the padding. This is a no-op in case of buffered IO
            Bytes::from(owned_slice).slice(
                self.start_padding
                    ..(self.range.end as usize - self.range.start as usize + self.start_padding),
            )
        }
    }
}

impl IoTask for FileReadTask {
    #[inline]
    fn waker(&self) -> &Mutex<Option<Waker>> {
        &self.waker
    }

    #[inline]
    fn get_sqe(&self, user_data: u64) -> squeue::Entry {
        let num_bytes = (self.range.end - self.range.start) as usize;
        let num_bytes_aligned = num_bytes + self.start_padding + self.end_padding;
        let read_op = opcode::Read::new(
            io_uring::types::Fd(self.fd),
            self.base_ptr,
            num_bytes_aligned as u32,
        );

        read_op
            .offset(self.range.start - self.start_padding as u64)
            .build()
            .user_data(user_data)
    }

    #[inline]
    fn completed(&self) -> &AtomicBool {
        &self.completed
    }

    fn debug_print(&self) {
        println!(
            "Read op, range: ({}, {}), start_padding: {}, end_padding: {}",
            self.range.start, self.range.end, self.start_padding, self.end_padding
        );
    }
}

unsafe impl Send for FileReadTask {}
unsafe impl Sync for FileReadTask {}

/**
Represents a request to write to a file
*/
pub struct FileWriteTask {
    base_ptr: *const u8,
    num_bytes: usize,
    padding: usize,
    fd: RawFd,
    completed: AtomicBool,
    waker: Mutex<Option<Waker>>,
}

impl FileWriteTask {
    #[allow(dead_code)]
    pub(crate) fn new(base_ptr: *const u8, num_bytes: usize, fd: RawFd) -> FileWriteTask {
        let mut padding = 0;
        if get_io_mode() == IoMode::Direct && (num_bytes & 4095) > 0 {
            padding = 4096 - num_bytes % 4096;
        }
        FileWriteTask {
            base_ptr,
            num_bytes,
            padding,
            fd,
            completed: AtomicBool::new(false),
            waker: Mutex::<Option<Waker>>::new(None),
        }
    }
}

impl IoTask for FileWriteTask {
    #[inline]
    fn waker(&self) -> &Mutex<Option<Waker>> {
        &self.waker
    }

    #[inline]
    fn get_sqe(&self, user_data: u64) -> squeue::Entry {
        let num_bytes_aligned = self.num_bytes + self.padding;
        let write_op = opcode::Write::new(
            io_uring::types::Fd(self.fd),
            self.base_ptr,
            num_bytes_aligned as u32,
        );

        write_op.offset(0u64).build().user_data(user_data)
    }

    #[inline]
    fn completed(&self) -> &AtomicBool {
        &self.completed
    }

    fn debug_print(&self) {
        println!("Write op, num bytes: {}", self.num_bytes);
    }
}

unsafe impl Send for FileWriteTask {}
unsafe impl Sync for FileWriteTask {}

static ENABLED: AtomicBool = AtomicBool::new(true);

#[derive(Debug, Clone, PartialEq, Default)]
pub enum IoMode {
    Direct,
    #[default]
    Buffered,
}

impl Display for IoMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}",
            match self {
                IoMode::Buffered => "buffered",
                IoMode::Direct => "direct",
            }
        )
    }
}

impl FromStr for IoMode {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Ok(match s {
            "direct" => IoMode::Direct,
            "buffered" => IoMode::Buffered,
            _ => return Err(format!("Invalid IO mode: {s}")),
        })
    }
}

/**
 * Represents a pool of worker threads responsible for submitting IO requests to the
 * kernel via io-uring.
 */
pub struct IoUringThreadpool {
    sender: crossbeam_channel::Sender<Arc<dyn IoTask>>,
    worker: Option<thread::JoinHandle<()>>,
    io_type: IoMode,
}

unsafe impl Sync for IoUringThreadpool {}

pub(crate) static IO_URING_THREAD_POOL_INST: OnceLock<IoUringThreadpool> = OnceLock::new();

/**
 * Intializes the global io-uring threadpool. This function must be called before submitting any IO to the pool.
 */
pub fn initialize_uring_pool(io_mode: IoMode) {
    IO_URING_THREAD_POOL_INST.get_or_init(|| IoUringThreadpool::new(io_mode));
}

#[inline]
pub fn get_io_mode() -> IoMode {
    IO_URING_THREAD_POOL_INST
        .get()
        .expect("Uring threadpool not initialized")
        .io_mode()
}

impl IoUringThreadpool {
    const NUM_ENTRIES: u32 = 64;

    pub fn new(io_type: IoMode) -> IoUringThreadpool {
        let (sender, receiver) = crossbeam_channel::unbounded::<Arc<dyn IoTask>>();

        let mut builder = IoUring::<squeue::Entry, cqueue::Entry>::builder();
        if io_type == IoMode::Direct {
            // Polled IO is only supported for direct IO requests
            builder.setup_iopoll();
        }
        builder.setup_sqpoll(50000);
        let ring = builder
            .build(Self::NUM_ENTRIES)
            .expect("Failed to build IoUring instance");

        let receiver_clone = receiver.clone();
        let worker = thread::spawn(move || {
            let mut uring_worker = UringWorker::new(receiver_clone, ring);
            uring_worker.thread_loop();
        });

        IoUringThreadpool {
            sender,
            worker: Some(worker),
            io_type,
        }
    }

    #[inline]
    pub fn submit_task(&self, task: Arc<dyn IoTask>) {
        self.sender
            .send(task.clone())
            .expect("Failed to submit task through channel");
    }

    #[inline]
    pub fn io_mode(&self) -> IoMode {
        self.io_type.clone()
    }
}

impl Drop for IoUringThreadpool {
    fn drop(&mut self) {
        ENABLED.store(false, Ordering::Relaxed);
        let worker = self.worker.take();
        if let Some(w) = worker {
            let _ = w.join();
        }
    }
}

impl std::fmt::Debug for IoUringThreadpool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("IoUringThreadpool")
            .field("io_type", &self.io_type)
            .finish()
    }
}

/**
Represents a single worker thread. The worker thread busy loops through 3 phases:
* Receives new requests from the application via a crossbeam channel
* Converts the requests to submission queue entries and submits them to the ring
* Polls the ring for completions and notifies the application
*/
struct UringWorker {
    channel: crossbeam_channel::Receiver<Arc<dyn IoTask>>,
    ring: io_uring::IoUring,
    // Assumption: There wont be more than 2^16 tasks on-the-fly at once
    op_counter: u16,
    completions_array: Vec<usize>,
    submitted_tasks: Vec<Option<Arc<dyn IoTask>>>,
    inflight_requests: u32,
}

impl UringWorker {
    fn new(
        channel: crossbeam_channel::Receiver<Arc<dyn IoTask>>,
        ring: io_uring::IoUring,
    ) -> UringWorker {
        let completions_array = vec![0; 1 << 16];

        let mut tasks = Vec::<Option<Arc<dyn IoTask>>>::new();
        tasks.resize(IoUringThreadpool::NUM_ENTRIES as usize, None);
        UringWorker {
            channel,
            ring,
            op_counter: 0,
            completions_array,
            submitted_tasks: tasks,
            inflight_requests: 0,
        }
    }

    fn thread_loop(&mut self) {
        loop {
            if !ENABLED.load(Ordering::Relaxed) {
                break;
            }

            while self.inflight_requests < IoUringThreadpool::NUM_ENTRIES {
                let res = self.channel.try_recv();
                if res.is_err() {
                    break;
                }
                let task = res.unwrap();
                // Consume tasks from channel and submit them to the ring
                {
                    let sq = &mut (self.ring.submission());
                    let sqe = task.get_sqe((self.op_counter as u64) << 48);

                    unsafe {
                        sq.push(&sqe).expect("Failed to push to submission queue");
                    }
                    sq.sync();
                    self.submitted_tasks[self.op_counter as usize] = Some(task);
                    self.completions_array[self.op_counter as usize] = 1;
                    self.op_counter = (self.op_counter + 1) & 63;
                }
                self.ring.submit().expect("Failed to submit");

                self.inflight_requests += 1;
            }

            self.poll_completions();
        }
    }

    fn poll_completions(&mut self) {
        let cq = &mut self.ring.completion();
        loop {
            cq.sync();
            match cq.next() {
                Some(cqe) => {
                    self.inflight_requests -= 1;
                    let errno = -cqe.result();
                    let err = std::io::Error::from_raw_os_error(errno);
                    let opcode = (cqe.user_data() >> 48) as usize;
                    if cqe.result() < 0 {
                        // TODO(): Send an error to the requestor
                        self.submitted_tasks[opcode].as_ref().unwrap().debug_print();
                        panic!("Cqe indicates IO error: {}", err);
                    }

                    let remaining = &mut self.completions_array[opcode];
                    *remaining -= 1;
                    if *remaining == 0 {
                        self.submitted_tasks[opcode]
                            .as_ref()
                            .unwrap()
                            .notify_waker();
                    }
                }
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

pub struct UringFuture<T>
where
    T: IoTask,
{
    task: Arc<T>,
    state: UringState,
}

impl<T> UringFuture<T>
where
    T: IoTask,
{
    pub fn new(task: Arc<T>) -> UringFuture<T> {
        UringFuture {
            task,
            state: UringState::Initialized,
        }
    }
}

impl<T> Future for UringFuture<T>
where
    T: IoTask + 'static,
{
    type Output = ();

    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<()> {
        match self.state {
            UringState::Initialized => {
                let pool = IO_URING_THREAD_POOL_INST
                    .get()
                    .expect("Uring threadpool not initialized");
                pool.submit_task(self.task.clone());
                self.task.set_waker(cx.waker().clone());
                self.state = UringState::Submitted;
                Poll::Pending
            }
            UringState::Submitted => match self.task.completed().load(Ordering::Relaxed) {
                false => {
                    self.task.set_waker(cx.waker().clone());
                    Poll::Pending
                }
                true => Poll::Ready(()),
            },
        }
    }
}
