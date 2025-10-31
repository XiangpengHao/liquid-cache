use std::{
    collections::VecDeque,
    fs::OpenOptions,
    future::Future,
    ops::Range,
    os::fd::AsRawFd,
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
use io_uring::{IoUring, cqueue, squeue};
use liquid_cache_common::IoMode;
use tokio::sync::oneshot;

use super::tasks::{FileOpenTask, FileReadTask, FileWriteTask, IoTask};

pub(crate) const URING_NUM_ENTRIES: u32 = 256;

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
pub(crate) fn get_io_mode() -> IoMode {
    *IO_MODE.get().expect("io-uring runtime not initialized")
}

impl IoUringThreadpool {
    fn new(io_type: IoMode) -> IoUringThreadpool {
        let (sender, receiver) = crossbeam_channel::unbounded::<Submission>();

        let builder = IoUring::<squeue::Entry, cqueue::Entry>::builder();
        let ring = builder
            .build(URING_NUM_ENTRIES)
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

struct UringWorker {
    receiver: crossbeam_channel::Receiver<Submission>,
    ring: IoUring,
    tokens: VecDeque<u16>,
    submitted_tasks: Vec<Option<Submission>>,
    io_performed: AtomicUsize,
}

impl UringWorker {
    #[allow(clippy::new_ret_no_self)]
    fn new(channel: crossbeam_channel::Receiver<Submission>, ring: IoUring) -> UringWorker {
        let tokens = (0..URING_NUM_ENTRIES as u16).collect();
        let mut tasks = Vec::with_capacity(URING_NUM_ENTRIES as usize);
        tasks.resize_with(URING_NUM_ENTRIES as usize, || None);
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
            {
                let sq = &mut self.ring.submission();
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
                None => break,
            }
        }
    }
}

enum UringState<T>
where
    T: IoTask + 'static,
{
    Undecided,
    Created(Box<T>),
    Submitted(oneshot::Receiver<Box<dyn IoTask>>),
}

pub(crate) struct UringFuture<T>
where
    T: IoTask + 'static,
{
    state: UringState<T>,
}

impl<T> UringFuture<T>
where
    T: IoTask + 'static,
{
    fn new(task: Box<T>) -> UringFuture<T> {
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

fn submit_async_task<T>(task: T) -> UringFuture<T>
where
    T: IoTask + 'static,
{
    UringFuture::new(Box::new(task))
}

pub(crate) async fn read(
    path: PathBuf,
    range: Option<Range<u64>>,
    direct_io: bool,
) -> Result<Bytes, std::io::Error> {
    let mut flags = libc::O_RDONLY | libc::O_CLOEXEC;
    if direct_io {
        flags |= libc::O_DIRECT;
    }

    let open_task = FileOpenTask::build(path, flags, 0)?;
    let file = submit_async_task(open_task).await.into_result()?;

    let effective_range = if let Some(range) = range {
        range
    } else {
        let len = file.metadata()?.len();
        0..len
    };

    let read_task = FileReadTask::build(effective_range, file, direct_io);
    submit_async_task(read_task).await.into_result()
}

pub(crate) async fn write(path: PathBuf, data: &Bytes) -> Result<(), std::io::Error> {
    use std::os::unix::fs::OpenOptionsExt as _;

    let direct = matches!(get_io_mode(), IoMode::UringDirect);
    let flags = if direct { libc::O_DIRECT } else { 0 };
    let file = OpenOptions::new()
        .create(true)
        .write(true)
        .custom_flags(flags)
        .open(path)
        .expect("failed to create file");

    let write_task = FileWriteTask::build(data.as_ptr(), data.len(), file.as_raw_fd(), direct);
    submit_async_task(write_task).await.into_result()
}
