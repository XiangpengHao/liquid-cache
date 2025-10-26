use std::{
    alloc::Layout,
    collections::VecDeque,
    ops::Range,
    os::fd::RawFd,
    path::PathBuf,
    pin::Pin,
    sync::{
        Arc, Mutex, OnceLock,
        atomic::{AtomicBool, Ordering},
    },
    task::{Context, Poll, Waker},
    thread,
};

use bytes::Bytes;
use io_uring::{IoUring, cqueue, opcode, squeue};
use liquid_cache_common::IoMode;

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
    fn get_sqe(&self) -> squeue::Entry;

    fn completed(&self) -> &AtomicBool;

    /**
    Wake the future that submitted this IO request
     */
    #[inline]
    fn notify_waker(&self) {
        self.completed().store(true, Ordering::Release);
        let mut guard = self.waker().lock().unwrap();
        if let Some(waker) = guard.take() {
            waker.wake();
        }
    }

    fn process_completion(&self, cqe: &cqueue::Entry);
}

/**
Represents a request to read from a file
*/
pub struct FileReadTask {
    base_ptr: *mut u8,
    layout: Layout,
    fd: RawFd,
    completed: AtomicBool,
    waker: Mutex<Option<Waker>>,
    range: Range<u64>,
    start_padding: usize,
    end_padding: usize,
    error: Mutex<Option<std::io::Error>>,
}

impl FileReadTask {
    pub fn new(range: Range<u64>, fd: RawFd) -> FileReadTask {
        let mut start_padding: usize = 0;
        let mut end_padding: usize = 0;
        if get_io_mode() == IoMode::DirectIO {
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
            layout,
            fd,
            completed: AtomicBool::new(false),
            waker: Mutex::<Option<Waker>>::new(None),
            range,
            start_padding,
            end_padding,
            error: Mutex::new(None),
        }
    }

    /**
    Return a bytes object holding the result of the read operation
     */
    #[inline]
    pub fn get_result(&self) -> Result<Bytes, std::io::Error> {
        let mut err = self.error.lock().unwrap();
        if err.is_some() {
            unsafe {
                std::alloc::dealloc(self.base_ptr, self.layout);
            }
            return Err(err.take().unwrap());
        }
        let total_bytes =
            (self.range.end - self.range.start) as usize + self.start_padding + self.end_padding;
        unsafe {
            let vec = Vec::from_raw_parts(self.base_ptr, total_bytes, total_bytes);
            // Convert to vec in order to transfer ownership of underlying pointer
            let owned_slice: Box<[u8]> = vec.into_boxed_slice();
            // The below slice operation removes the padding. This is a no-op in case of buffered IO
            Ok(Bytes::from(owned_slice).slice(
                self.start_padding
                    ..(self.range.end as usize - self.range.start as usize + self.start_padding),
            ))
        }
    }
}

impl IoTask for FileReadTask {
    #[inline]
    fn waker(&self) -> &Mutex<Option<Waker>> {
        &self.waker
    }

    #[inline]
    fn get_sqe(&self) -> squeue::Entry {
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
    }

    #[inline]
    fn completed(&self) -> &AtomicBool {
        &self.completed
    }

    fn process_completion(&self, cqe: &cqueue::Entry) {
        if cqe.result() < 0 {
            let mut error = self.error.lock().unwrap();
            *error = Some(std::io::Error::from_raw_os_error(-cqe.result()));
        }
        self.notify_waker();
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
    error: Mutex<Option<std::io::Error>>,
}

impl FileWriteTask {
    pub fn new(base_ptr: *const u8, num_bytes: usize, fd: RawFd) -> FileWriteTask {
        let mut padding = 0;
        if get_io_mode() == IoMode::DirectIO && (num_bytes & 4095) > 0 {
            padding = 4096 - num_bytes % 4096;
        }
        FileWriteTask {
            base_ptr,
            num_bytes,
            padding,
            fd,
            completed: AtomicBool::new(false),
            waker: Mutex::<Option<Waker>>::new(None),
            error: Mutex::new(None),
        }
    }

    pub fn get_result(&self) -> Result<(), std::io::Error> {
        let mut err = self.error.lock().unwrap();
        if err.is_some() {
            return Err(err.take().unwrap());
        }
        Ok(())
    }
}

impl IoTask for FileWriteTask {
    #[inline]
    fn waker(&self) -> &Mutex<Option<Waker>> {
        &self.waker
    }

    #[inline]
    fn get_sqe(&self) -> squeue::Entry {
        let num_bytes_aligned = self.num_bytes + self.padding;
        let write_op = opcode::Write::new(
            io_uring::types::Fd(self.fd),
            self.base_ptr,
            num_bytes_aligned as u32,
        );

        write_op.offset(0u64).build()
    }

    #[inline]
    fn completed(&self) -> &AtomicBool {
        &self.completed
    }

    fn process_completion(&self, cqe: &cqueue::Entry) {
        if cqe.result() < 0 {
            let mut error = self.error.lock().unwrap();
            *error = Some(std::io::Error::from_raw_os_error(-cqe.result()));
        }
        self.notify_waker();
    }
}

unsafe impl Send for FileWriteTask {}
unsafe impl Sync for FileWriteTask {}

static ENABLED: AtomicBool = AtomicBool::new(true);

/// Represents a pool of worker threads responsible for submitting IO requests to the
/// kernel via io-uring.
struct IoUringThreadpool {
    sender: crossbeam_channel::Sender<Arc<dyn IoTask>>,
    worker: Option<thread::JoinHandle<()>>,
    io_type: IoMode,
}

unsafe impl Sync for IoUringThreadpool {}

static IO_URING_THREAD_POOL_INST: OnceLock<IoUringThreadpool> = OnceLock::new();

pub(crate) fn initialize_uring_pool(io_mode: IoMode) {
    IO_URING_THREAD_POOL_INST.get_or_init(|| IoUringThreadpool::new(io_mode));
}

#[inline]
fn get_io_mode() -> IoMode {
    IO_URING_THREAD_POOL_INST
        .get()
        .expect("Uring threadpool not initialized")
        .io_mode()
}

impl IoUringThreadpool {
    const NUM_ENTRIES: u32 = 64;

    fn new(io_type: IoMode) -> IoUringThreadpool {
        let (sender, receiver) = crossbeam_channel::unbounded::<Arc<dyn IoTask>>();

        let mut builder = IoUring::<squeue::Entry, cqueue::Entry>::builder();
        if io_type == IoMode::DirectIO {
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
    fn submit_task(&self, task: Arc<dyn IoTask>) {
        self.sender
            .send(task.clone())
            .expect("Failed to submit task through channel");
    }

    #[inline]
    fn io_mode(&self) -> IoMode {
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

/// Represents a single worker thread. The worker thread busy loops through 3 phases:
/// 1. Receives new requests from the application via a crossbeam channel
/// 2. Converts the requests to submission queue entries and submits them to the ring
/// 3. Polls the ring for completions and notifies the application
struct UringWorker {
    channel: crossbeam_channel::Receiver<Arc<dyn IoTask>>,
    ring: io_uring::IoUring,
    tokens: VecDeque<u16>,
    submitted_tasks: Vec<Option<Arc<dyn IoTask>>>,
}

impl UringWorker {
    fn new(
        channel: crossbeam_channel::Receiver<Arc<dyn IoTask>>,
        ring: io_uring::IoUring,
    ) -> UringWorker {
        let tokens = (0..IoUringThreadpool::NUM_ENTRIES as u16).collect();
        let mut tasks = Vec::<Option<Arc<dyn IoTask>>>::new();
        tasks.resize(IoUringThreadpool::NUM_ENTRIES as usize, None);
        UringWorker {
            channel,
            ring,
            tokens,
            submitted_tasks: tasks,
        }
    }

    fn thread_loop(&mut self) {
        loop {
            if !ENABLED.load(Ordering::Relaxed) {
                break;
            }

            while !self.tokens.is_empty() {
                let res = self.channel.try_recv();
                if res.is_err() {
                    break;
                }
                let token = self.tokens.pop_front().unwrap();
                let task = res.unwrap();
                // Consume tasks from channel and submit them to the ring
                {
                    let sq = &mut (self.ring.submission());
                    let sqe = task.get_sqe().user_data(token as u64);

                    unsafe {
                        sq.push(&sqe).expect("Failed to push to submission queue");
                    }
                    sq.sync();
                    self.submitted_tasks[token as usize] = Some(task);
                }
                self.ring.submit().expect("Failed to submit");
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
                    let token = cqe.user_data() as usize;
                    self.submitted_tasks[token]
                        .as_ref()
                        .unwrap()
                        .process_completion(&cqe);
                    self.tokens.push_back(token as u16);
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

pub(crate) async fn read_range_from_uring(
    path: PathBuf,
    mut range: Option<std::ops::Range<u64>>,
) -> Result<Bytes, std::io::Error> {
    use std::os::fd::AsRawFd;
    use std::{fs::OpenOptions, os::unix::fs::OpenOptionsExt as _};

    use crate::cache::io_uring::{FileReadTask, UringFuture, get_io_mode};
    use liquid_cache_common::IoMode;

    let flags = if get_io_mode() == IoMode::DirectIO {
        libc::O_DIRECT
    } else {
        0
    };
    let file = OpenOptions::new()
        .read(true)
        .custom_flags(flags)
        .open(path)
        .expect("failed to open file");

    if range.is_none() {
        range = Some(std::ops::Range::<u64> {
            start: 0,
            end: file.metadata()?.len(),
        });
    }

    let task = Arc::new(FileReadTask::new(range.unwrap(), file.as_raw_fd()));
    // UringFuture will be responsible for submitting and driving the future to completion
    let uring_fut = UringFuture::new(task.clone());
    uring_fut.await;
    task.get_result()
}

pub(crate) async fn write_to_uring(path: PathBuf, data: &Bytes) -> Result<(), std::io::Error> {
    use crate::cache::io_uring::{FileWriteTask, UringFuture, get_io_mode};
    use liquid_cache_common::IoMode;
    use std::os::fd::AsRawFd;
    use std::{fs::OpenOptions, os::unix::fs::OpenOptionsExt as _};

    let flags = if get_io_mode() == IoMode::DirectIO {
        libc::O_DIRECT
    } else {
        0
    };
    let file = OpenOptions::new()
        .create(true)
        .write(true)
        .custom_flags(flags)
        .open(path)
        .expect("failed to create file");

    let task = Arc::new(FileWriteTask::new(
        data.as_ptr(),
        data.len(),
        file.as_raw_fd(),
    ));
    // UringFuture will be responsible for submitting and driving the future to completion
    let uring_fut = UringFuture::new(task.clone());
    uring_fut.await;
    task.get_result()
}
