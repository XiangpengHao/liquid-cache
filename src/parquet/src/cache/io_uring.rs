use std::{
    alloc::Layout,
    any::Any,
    cell::RefCell,
    collections::VecDeque,
    ffi::CString,
    fs,
    future::Future,
    io,
    ops::Range,
    os::{
        fd::{AsRawFd, FromRawFd, RawFd},
        unix::ffi::OsStringExt,
    },
    path::PathBuf,
    pin::Pin,
    sync::{
        OnceLock,
        atomic::{AtomicBool, AtomicUsize, Ordering},
    },
    task::{Context, Poll},
    thread,
};

use bytes::Bytes;
use io_uring::{IoUring, cqueue, opcode, squeue};
use liquid_cache_common::IoMode;
use tokio::sync::oneshot;

const BLOCK_ALIGN: usize = 4096;

/// Represents an IO request to the uring worker thread
trait IoTask: Send + Any + std::fmt::Debug {
    /// Converts the IO request to an IO uring submission queue entry
    fn prepare_sqe(&mut self) -> squeue::Entry;

    /// Record the outcome of the completion queue entry
    fn complete(&mut self, cqe: &cqueue::Entry);

    /// Convert the boxed task to a boxed `Any` so callers can recover the original type.
    fn into_any(self: Box<Self>) -> Box<dyn Any>;
}

#[derive(Debug)]
struct FileOpenTask {
    path: CString,
    flags: i32,
    mode: libc::mode_t,
    fd: Option<RawFd>,
    error: Option<std::io::Error>,
}

impl FileOpenTask {
    fn build(
        path: PathBuf,
        flags: i32,
        mode: libc::mode_t,
    ) -> Result<FileOpenTask, std::io::Error> {
        let bytes = path.into_os_string().into_vec();
        let path = CString::new(bytes).map_err(|_| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "path contains interior null byte",
            )
        })?;
        Ok(FileOpenTask {
            path,
            flags,
            mode,
            fd: None,
            error: None,
        })
    }

    #[allow(clippy::new_ret_no_self)]
    fn new(
        path: PathBuf,
        flags: i32,
        mode: libc::mode_t,
    ) -> Result<FileOpenTaskFuture, std::io::Error> {
        let task = FileOpenTask::build(path, flags, mode)?;
        Ok(FileOpenTaskFuture(UringFuture::new(Box::new(task))))
    }

    fn into_result(mut self) -> Result<fs::File, std::io::Error> {
        if let Some(err) = self.error.take() {
            return Err(err);
        }
        let fd = self.fd.take().ok_or_else(|| {
            std::io::Error::other("open operation completed without returning file descriptor")
        })?;
        // SAFETY: `fd` has been received from the kernel for this task and is uniquely owned here.
        let file = unsafe { fs::File::from_raw_fd(fd) };
        Ok(file)
    }
}

impl IoTask for FileOpenTask {
    #[inline]
    fn prepare_sqe(&mut self) -> squeue::Entry {
        let open_op = opcode::OpenAt::new(io_uring::types::Fd(libc::AT_FDCWD), self.path.as_ptr())
            .flags(self.flags)
            .mode(self.mode);

        open_op.build()
    }

    #[inline]
    fn complete(&mut self, cqe: &cqueue::Entry) {
        let result = cqe.result();
        if result < 0 {
            self.error = Some(std::io::Error::from_raw_os_error(-result));
        } else {
            self.fd = Some(result);
        }
    }

    fn into_any(self: Box<Self>) -> Box<dyn Any> {
        self
    }
}

impl Drop for FileOpenTask {
    fn drop(&mut self) {
        if let Some(fd) = self.fd.take() {
            unsafe {
                libc::close(fd);
            }
        }
    }
}

struct FileOpenTaskFuture(UringFuture<FileOpenTask>);

impl Future for FileOpenTaskFuture {
    type Output = Result<fs::File, std::io::Error>;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        // SAFETY: We never move the inner future after pinning `self`.
        let inner = unsafe { self.map_unchecked_mut(|fut| &mut fut.0) };
        match inner.poll(cx) {
            Poll::Ready(task) => Poll::Ready(task.into_result()),
            Poll::Pending => Poll::Pending,
        }
    }
}

#[derive(Debug)]
struct FileReadTask {
    base_ptr: *mut u8,
    layout: Layout,
    file: fs::File,
    range: Range<u64>,
    start_padding: usize,
    end_padding: usize,
    error: Option<std::io::Error>,
}

impl FileReadTask {
    fn build(range: Range<u64>, file: fs::File) -> FileReadTask {
        let mut start_padding: usize = 0;
        let mut end_padding: usize = 0;
        if get_io_mode() == IoMode::UringDirect {
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
            file,
            range,
            start_padding,
            end_padding,
            error: None,
        }
    }

    #[allow(clippy::new_ret_no_self)]
    fn new(range: Range<u64>, file: fs::File) -> FileReadTaskFuture {
        let task = FileReadTask::build(range, file);
        FileReadTaskFuture(UringFuture::new(Box::new(task)))
    }

    /// Return a bytes object holding the result of the read operation
    #[inline]
    fn into_result(self: Box<Self>) -> Result<Bytes, std::io::Error> {
        let mut this = self;
        if let Some(err) = this.error.take() {
            unsafe {
                std::alloc::dealloc(this.base_ptr, this.layout);
            }
            this.base_ptr = std::ptr::null_mut();
            return Err(err);
        }
        let total_bytes =
            (this.range.end - this.range.start) as usize + this.start_padding + this.end_padding;
        let base_ptr = std::mem::replace(&mut this.base_ptr, std::ptr::null_mut());
        unsafe {
            let vec = Vec::from_raw_parts(base_ptr, total_bytes, total_bytes);
            // Convert to vec in order to transfer ownership of underlying pointer
            let owned_slice: Box<[u8]> = vec.into_boxed_slice();
            // The below slice operation removes the padding. This is a no-op in case of buffered IO
            Ok(Bytes::from(owned_slice).slice(
                this.start_padding
                    ..(this.range.end as usize - this.range.start as usize + this.start_padding),
            ))
        }
    }
}

impl IoTask for FileReadTask {
    #[inline]
    fn prepare_sqe(&mut self) -> squeue::Entry {
        let num_bytes = (self.range.end - self.range.start) as usize;
        let num_bytes_aligned = num_bytes + self.start_padding + self.end_padding;
        let read_op = opcode::Read::new(
            io_uring::types::Fd(self.file.as_raw_fd()),
            self.base_ptr,
            num_bytes_aligned as u32,
        );

        read_op
            .offset(self.range.start - self.start_padding as u64)
            .build()
    }

    #[inline]
    fn complete(&mut self, cqe: &cqueue::Entry) {
        if cqe.result() < 0 {
            self.error = Some(std::io::Error::from_raw_os_error(-cqe.result()));
        }
    }

    fn into_any(self: Box<Self>) -> Box<dyn Any> {
        self
    }
}

impl Drop for FileReadTask {
    fn drop(&mut self) {
        if !self.base_ptr.is_null() {
            unsafe {
                std::alloc::dealloc(self.base_ptr, self.layout);
            }
            self.base_ptr = std::ptr::null_mut();
        }
    }
}

unsafe impl Send for FileReadTask {}

struct FileReadTaskFuture(UringFuture<FileReadTask>);

impl Future for FileReadTaskFuture {
    type Output = Result<Bytes, std::io::Error>;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        // SAFETY: We never move the inner future after pinning `self`.
        let inner = unsafe { self.map_unchecked_mut(|fut| &mut fut.0) };
        match inner.poll(cx) {
            Poll::Ready(task) => Poll::Ready(task.into_result()),
            Poll::Pending => Poll::Pending,
        }
    }
}

/// Represents a request to write to a file
#[derive(Debug)]
pub struct FileWriteTask {
    base_ptr: *const u8,
    num_bytes: usize,
    padding: usize,
    fd: RawFd,
    error: Option<std::io::Error>,
}

impl FileWriteTask {
    fn build(base_ptr: *const u8, num_bytes: usize, fd: RawFd) -> FileWriteTask {
        let mut padding = 0;
        if get_io_mode() == IoMode::UringDirect && (num_bytes & 4095) > 0 {
            padding = 4096 - num_bytes % 4096;
        }
        FileWriteTask {
            base_ptr,
            num_bytes,
            padding,
            fd,
            error: None,
        }
    }

    #[allow(clippy::new_ret_no_self)]
    fn new(base_ptr: *const u8, num_bytes: usize, fd: RawFd) -> FileWriteTaskFuture {
        let task = FileWriteTask::build(base_ptr, num_bytes, fd);
        FileWriteTaskFuture(UringFuture::new(Box::new(task)))
    }

    fn into_result(self: Box<Self>) -> Result<(), std::io::Error> {
        let mut this = self;
        if let Some(err) = this.error.take() {
            return Err(err);
        }
        Ok(())
    }
}

impl IoTask for FileWriteTask {
    #[inline]
    fn prepare_sqe(&mut self) -> squeue::Entry {
        let num_bytes_aligned = self.num_bytes + self.padding;
        let write_op = opcode::Write::new(
            io_uring::types::Fd(self.fd),
            self.base_ptr,
            num_bytes_aligned as u32,
        );

        write_op.offset(0u64).build()
    }

    #[inline]
    fn complete(&mut self, cqe: &cqueue::Entry) {
        if cqe.result() < 0 {
            self.error = Some(std::io::Error::from_raw_os_error(-cqe.result()));
        }
    }

    fn into_any(self: Box<Self>) -> Box<dyn Any> {
        self
    }
}

impl Drop for FileWriteTask {
    fn drop(&mut self) {
        // Nothing to do; ownership of buffers is external to the task.
    }
}

unsafe impl Send for FileWriteTask {}

struct FileWriteTaskFuture(UringFuture<FileWriteTask>);

impl Future for FileWriteTaskFuture {
    type Output = Result<(), std::io::Error>;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        // SAFETY: We never move the inner future after pinning `self`.
        let inner = unsafe { self.map_unchecked_mut(|fut| &mut fut.0) };
        match inner.poll(cx) {
            Poll::Ready(task) => Poll::Ready(task.into_result()),
            Poll::Pending => Poll::Pending,
        }
    }
}

static IO_MODE: OnceLock<IoMode> = OnceLock::new();
static ENABLED: AtomicBool = AtomicBool::new(true);

struct Submission {
    task: Box<dyn IoTask>,
    completion_tx: oneshot::Sender<Box<dyn IoTask>>,
}

impl Submission {
    fn new(task: Box<dyn IoTask>, completion_tx: oneshot::Sender<Box<dyn IoTask>>) -> Submission {
        Submission {
            task,
            completion_tx,
        }
    }

    fn send_back(mut self, cqe: &cqueue::Entry) {
        self.task.complete(cqe);
        self.completion_tx
            .send(self.task)
            .expect("Failed to send task back to caller");
    }
}

struct JoinOnDropHandle<T>(Option<thread::JoinHandle<T>>);

impl<T> JoinOnDropHandle<T> {
    fn new(handle: thread::JoinHandle<T>) -> JoinOnDropHandle<T> {
        JoinOnDropHandle(Some(handle))
    }
}

impl<T> Drop for JoinOnDropHandle<T> {
    fn drop(&mut self) {
        if let Some(handle) = self.0.take() {
            let _ = handle.join();
        }
    }
}

/// Represents a pool of worker threads responsible for submitting IO requests to the
/// kernel via io-uring.
struct IoUringThreadpool {
    sender: crossbeam_channel::Sender<Submission>,
    _worker_guard: JoinOnDropHandle<()>,
    io_type: IoMode,
}

unsafe impl Sync for IoUringThreadpool {}

static IO_URING_THREAD_POOL_INST: OnceLock<IoUringThreadpool> = OnceLock::new();

pub(crate) fn initialize_uring_pool(io_mode: IoMode) {
    let current_mode = IO_MODE.get_or_init(|| io_mode);
    if *current_mode != io_mode {
        panic!(
            "io-uring runtime already initialized with mode {:?}, received {:?}",
            current_mode, io_mode
        );
    }

    if matches!(io_mode, IoMode::Uring | IoMode::UringDirect) {
        IO_URING_THREAD_POOL_INST.get_or_init(|| IoUringThreadpool::new(io_mode));
    }
}

#[inline]
fn get_io_mode() -> IoMode {
    *IO_MODE.get().expect("io-uring runtime not initialized")
}

impl IoUringThreadpool {
    const NUM_ENTRIES: u32 = 256;

    fn new(io_type: IoMode) -> IoUringThreadpool {
        let (sender, receiver) = crossbeam_channel::unbounded::<Submission>();

        let mut builder = IoUring::<squeue::Entry, cqueue::Entry>::builder();
        builder.setup_sqpoll(50000);
        let ring = builder
            .build(Self::NUM_ENTRIES)
            .expect("Failed to build IoUring instance");

        let worker = thread::Builder::new()
            .name("lc-io-worker".to_string())
            .spawn(move || {
                let mut uring_worker = UringWorker::new(receiver, ring);
                uring_worker.thread_loop();
            })
            .expect("Failed to spawn io-uring worker thread");

        IoUringThreadpool {
            sender,
            _worker_guard: JoinOnDropHandle::new(worker),
            io_type,
        }
    }

    #[inline]
    fn submit_task(&self, task: Box<dyn IoTask>, completion_tx: oneshot::Sender<Box<dyn IoTask>>) {
        self.sender
            .send(Submission::new(task, completion_tx))
            .expect("Failed to submit task through channel");
    }
}

impl Drop for IoUringThreadpool {
    fn drop(&mut self) {
        ENABLED.store(false, Ordering::Relaxed);
    }
}

impl std::fmt::Debug for IoUringThreadpool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("IoUringThreadpool")
            .field("io_type", &self.io_type)
            .finish()
    }
}

thread_local! {
    static BLOCKING_URING_RING: RefCell<Option<BlockingRing>> = const { RefCell::new(None) };
}

struct BlockingRing {
    ring: IoUring,
}

impl BlockingRing {
    fn new() -> io::Result<BlockingRing> {
        let ring = IoUring::builder().build(IoUringThreadpool::NUM_ENTRIES)?;
        Ok(BlockingRing { ring })
    }

    fn run_task<T>(&mut self, mut task: Box<T>) -> io::Result<Box<T>>
    where
        T: IoTask + 'static,
    {
        {
            let mut sq = self.ring.submission();
            let entry = task.prepare_sqe().user_data(0);
            unsafe {
                sq.push(&entry).expect("Failed to push to submission queue");
            }
            sq.sync();
        }

        self.ring.submit_and_wait(1)?;

        {
            let mut cq = self.ring.completion();
            cq.sync();
            let cqe = cq
                .next()
                .ok_or_else(|| io::Error::other("io-uring completion queue empty"))?;
            task.complete(&cqe);
        }

        Ok(task)
    }
}

fn with_blocking_ring<F, R>(f: F) -> io::Result<R>
where
    F: FnOnce(&mut BlockingRing) -> io::Result<R>,
{
    BLOCKING_URING_RING.with(|cell| {
        let mut borrowed = cell.borrow_mut();
        if borrowed.is_none() {
            *borrowed = Some(BlockingRing::new()?);
        }
        let ring = borrowed
            .as_mut()
            .expect("BlockingRing missing after initialization");
        f(ring)
    })
}

fn run_blocking_task<T>(task: Box<T>) -> io::Result<Box<T>>
where
    T: IoTask + 'static,
{
    with_blocking_ring(move |ring| ring.run_task(task))
}

pub(crate) fn read_range_from_blocking_uring(
    path: PathBuf,
    range: Option<Range<u64>>,
) -> Result<Bytes, std::io::Error> {
    let mut flags = libc::O_RDONLY | libc::O_CLOEXEC;
    if get_io_mode() == IoMode::UringDirect {
        flags |= libc::O_DIRECT;
    }

    let open_task = FileOpenTask::build(path, flags, 0)?;
    let file = run_blocking_task(Box::new(open_task))?.into_result()?;
    let effective_range = if let Some(range) = range {
        range
    } else {
        let len = file.metadata()?.len();
        0..len
    };

    let read_task = FileReadTask::build(effective_range, file);
    run_blocking_task(Box::new(read_task))?.into_result()
}

pub(crate) fn write_to_blocking_uring(path: PathBuf, data: &Bytes) -> Result<(), std::io::Error> {
    use std::{fs::OpenOptions, os::unix::fs::OpenOptionsExt as _};

    let flags = if get_io_mode() == IoMode::UringDirect {
        libc::O_DIRECT
    } else {
        0
    };

    let file = OpenOptions::new()
        .create(true)
        .write(true)
        .custom_flags(flags)
        .open(path)?;

    let write_task = FileWriteTask::build(data.as_ptr(), data.len(), file.as_raw_fd());
    run_blocking_task(Box::new(write_task))?.into_result()
}

/// Represents a single worker thread. The worker thread busy loops through 3 phases:
/// 1. Receives new requests from the application via a crossbeam channel
/// 2. Converts the requests to submission queue entries and submits them to the ring
/// 3. Polls the ring for completions and notifies the application
struct UringWorker {
    receiver: crossbeam_channel::Receiver<Submission>,
    ring: io_uring::IoUring,
    tokens: VecDeque<u16>,
    submitted_tasks: Vec<Option<Submission>>,
    io_performed: AtomicUsize,
}

impl UringWorker {
    #[allow(clippy::new_ret_no_self)]
    fn new(
        channel: crossbeam_channel::Receiver<Submission>,
        ring: io_uring::IoUring,
    ) -> UringWorker {
        let tokens = (0..IoUringThreadpool::NUM_ENTRIES as u16).collect();
        let mut tasks = Vec::with_capacity(IoUringThreadpool::NUM_ENTRIES as usize);
        tasks.resize_with(IoUringThreadpool::NUM_ENTRIES as usize, || None);
        UringWorker {
            receiver: channel,
            ring,
            tokens,
            submitted_tasks: tasks,
            io_performed: AtomicUsize::new(0),
        }
    }

    fn thread_loop(&mut self) {
        loop {
            if !ENABLED.load(Ordering::Relaxed) {
                break;
            }

            self.drain_submissions();
            self.poll_completions();
        }
    }

    #[inline(never)]
    fn drain_submissions(&mut self) {
        let mut need_submit = false;
        while !self.receiver.is_empty() && !self.tokens.is_empty() {
            let mut submission = self.receiver.recv().unwrap();
            let token = self.tokens.pop_front().unwrap();
            // Consume tasks from channel and submit them to the ring
            {
                let sq = &mut (self.ring.submission());
                let task = submission.task.as_mut();
                let sqe = task.prepare_sqe().user_data(token as u64);
                unsafe {
                    sq.push(&sqe).expect("Failed to push to submission queue");
                }
                sq.sync();
            }
            self.submitted_tasks[token as usize] = Some(submission);
            need_submit = true;
        }
        if need_submit {
            self.ring.submit().expect("Failed to submit");
        }
    }

    #[inline(never)]
    fn poll_completions(&mut self) {
        let cq = &mut self.ring.completion();
        loop {
            cq.sync();
            match cq.next() {
                Some(cqe) => {
                    let token = cqe.user_data() as usize;
                    let submission = self.submitted_tasks[token]
                        .take()
                        .expect("Task not found in submitted tasks");
                    submission.send_back(&cqe);
                    self.tokens.push_back(token as u16);
                    self.io_performed.fetch_add(1, Ordering::Relaxed);
                }
                None => {
                    break;
                }
            }
        }
    }
}

enum UringState<T> {
    Created(Box<T>),
    Submitted(oneshot::Receiver<Box<dyn IoTask>>),
    Undecided, // used to please the borrow checker
}

struct UringFuture<T>
where
    T: IoTask,
{
    state: UringState<T>,
}

impl<T> UringFuture<T>
where
    T: IoTask + 'static,
{
    pub fn new(task: Box<T>) -> UringFuture<T> {
        UringFuture {
            state: UringState::Created(task),
        }
    }
}

impl<T> Future for UringFuture<T>
where
    T: IoTask + 'static,
{
    type Output = Box<T>;

    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        loop {
            let state = std::mem::replace(&mut self.state, UringState::Undecided);
            match state {
                UringState::Created(task) => {
                    let pool = IO_URING_THREAD_POOL_INST
                        .get()
                        .expect("Uring threadpool not initialized");
                    let (tx, rx) = oneshot::channel::<Box<dyn IoTask>>();
                    let boxed_task: Box<dyn IoTask> = task;
                    pool.submit_task(boxed_task, tx);
                    self.state = UringState::Submitted(rx);
                }
                UringState::Submitted(mut receiver) => match Pin::new(&mut receiver).poll(cx) {
                    Poll::Ready(Ok(task)) => {
                        let typed_task = task
                            .into_any()
                            .downcast::<T>()
                            .expect("io task downcast failure");
                        return Poll::Ready(typed_task);
                    }
                    Poll::Ready(Err(_)) => {
                        panic!("io-uring worker dropped completion channel")
                    }
                    Poll::Pending => {
                        self.state = UringState::Submitted(receiver);
                        return Poll::Pending;
                    }
                },
                UringState::Undecided => unreachable!("state cannot be undecided during poll"),
            }
        }
    }
}

pub(crate) async fn read_range_from_uring(
    path: PathBuf,
    range: Option<std::ops::Range<u64>>,
) -> Result<Bytes, std::io::Error> {
    use crate::cache::io_uring::{FileOpenTask, FileReadTask, get_io_mode};
    use liquid_cache_common::IoMode;

    let mut flags = libc::O_RDONLY | libc::O_CLOEXEC;
    if get_io_mode() == IoMode::UringDirect {
        flags |= libc::O_DIRECT;
    }

    let open_future = FileOpenTask::new(path, flags, 0)?;
    let file = open_future.await?;

    let effective_range = if let Some(range) = range {
        range
    } else {
        let len = file.metadata()?.len();
        0..len
    };

    FileReadTask::new(effective_range, file).await
}

pub(crate) async fn write_to_uring(path: PathBuf, data: &Bytes) -> Result<(), std::io::Error> {
    use crate::cache::io_uring::{FileWriteTask, get_io_mode};
    use liquid_cache_common::IoMode;
    use std::os::fd::AsRawFd;
    use std::{fs::OpenOptions, os::unix::fs::OpenOptionsExt as _};

    let flags = if get_io_mode() == IoMode::UringDirect {
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

    FileWriteTask::new(data.as_ptr(), data.len(), file.as_raw_fd()).await
}
