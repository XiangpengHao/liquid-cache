use std::{
    collections::VecDeque, fs::OpenOptions, future::Future, io, ops::Range, os::{fd::AsRawFd, unix::fs::OpenOptionsExt}, path::PathBuf, pin::Pin, sync::{
        OnceLock,
        atomic::{AtomicBool, AtomicUsize, Ordering},
    }, task::{Context, Poll}, thread
};

use bytes::Bytes;
use io_uring::{IoUring, cqueue, squeue};
use liquid_cache_common::{IoMode, memory::pool::FixedBufferPool};
use tokio::sync::oneshot;

use crate::io::io_uring::tasks::FixedFileReadTask;

use super::tasks::{FileOpenTask, FileReadTask, FileWriteTask, IoTask};

pub(crate) const URING_NUM_ENTRIES: u32 = 256;

static ENABLED: AtomicBool = AtomicBool::new(true);

struct Submission {
    task: Box<dyn IoTask>,
    completion_tx: oneshot::Sender<Box<dyn IoTask>>,
    pending_completions: usize,   // No. of pending completions. Will be populated later by the uring worker
    completions: Vec<cqueue::Entry>,
}

impl Submission {
    fn new(task: Box<dyn IoTask>, completion_tx: oneshot::Sender<Box<dyn IoTask>>) -> Submission {
        Submission {
            task,
            completion_tx,
            pending_completions: 0,
            completions: Vec::new(),
        }
    }

    fn send_back(mut self) {
        self.task.complete(self.completions.iter().collect());
        self.completion_tx
            .send(self.task)
            .expect("Failed to send task back to caller");
    }

    #[inline]
    fn set_completions(&mut self, count: usize) {
        self.pending_completions = count;
    }

    #[inline]
    fn reduce_completions(&mut self) {
        self.pending_completions -= 1;
    }

    #[inline]
    fn push_completion(&mut self, cqe: cqueue::Entry) {
        self.completions.push(cqe);
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

pub(crate) fn initialize_uring_pool(io_mode: IoMode, register_buffers: bool) {
    if matches!(io_mode, IoMode::Uring | IoMode::UringDirect) {
        IO_URING_THREAD_POOL_INST.get_or_init(|| IoUringThreadpool::new(io_mode, register_buffers));
    }
    if matches!(io_mode, IoMode::UringBlocking) {
        super::multi_blocking_uring::initialize_blocking_rings();
    }
}

impl IoUringThreadpool {
    fn new(io_type: IoMode, register_buffers: bool) -> IoUringThreadpool {
        let (sender, receiver) = crossbeam_channel::unbounded::<Submission>();

        let mut builder = IoUring::<squeue::Entry, cqueue::Entry>::builder();
        let ring = builder
            .setup_iopoll()
            // .setup_sqpoll(50000)
            .build(URING_NUM_ENTRIES)
            .expect("Failed to build IoUring instance");

        if register_buffers {
            let res = FixedBufferPool::register_buffers_with_ring(&ring);
            if res.is_err() {
                log::error!("Failed to register buffers with io-uring ring: {:?}", res);
            }
        }

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
    /**
     * When using fixed buffers, a single task can produce multiple submission queue entries.
     * It is possible that we aren't able to submit all of them at one go. Hold them in an 
     * intermediate queue in that case
     */
    queued_entries: VecDeque<squeue::Entry>,
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
            queued_entries: VecDeque::with_capacity(URING_NUM_ENTRIES as usize),
        }
    }

    fn thread_loop(&mut self) {
        loop {
            if !ENABLED.load(Ordering::Relaxed) {
                break;
            }

            self.drain_intermediate_queue();
            self.drain_submissions();
            self.poll_completions();
        }
    }

    fn drain_intermediate_queue(&mut self) {
        let mut need_submit = false;
        {
            let sq = &mut self.ring.submission();            
            while !sq.is_full() && !self.queued_entries.is_empty() {
                let sqe = self.queued_entries.pop_front().unwrap();
                unsafe {
                    sq.push(&sqe).expect("Failed to push to submission queue");
                }
                sq.sync();
                need_submit = true;
            }
        }
        if need_submit {
            self.ring.submit().expect("Failed to submit");
        }
    }

    #[inline(never)]
    fn drain_submissions(&mut self) {
        let mut need_submit = false;
        while !self.receiver.is_empty() && !self.tokens.is_empty() {
            let sq = &mut self.ring.submission();
            sq.sync();
            if sq.is_full() {
                // A single token might have multiple associated sqes. Free token doesn't always imply that we have free submission slots
                break;
            }

            let token = self.tokens.pop_front().unwrap();
            let mut submission = self.receiver.recv().unwrap();
            let task = submission.task.as_mut();
            let mut sqes = task.prepare_sqe();
            submission.set_completions(sqes.len());
            let mut tasks_submitted = 0;
            
            for sqe in sqes.iter_mut() {
                let res = unsafe {
                    sq.push(&sqe.clone().user_data(token as u64))
                };
                if res.is_err() {
                    break;
                }
                tasks_submitted += 1;
                sq.sync();
            }
            for i in tasks_submitted..sqes.len() {
                self.queued_entries.push_back(sqes[i].clone().user_data(token as u64));
            }
            self.submitted_tasks[token as usize] = Some(submission);
            need_submit = true;
        }
        let need_poll = self.tokens.len() < URING_NUM_ENTRIES as usize;
        if need_submit || need_poll {
            loop {
                match self.ring.submit() {
                    Ok(_num_entries) => {
                        break;
                    }
                    Err(e) => {
                        if e.kind() == io::ErrorKind::Interrupted {
                            continue;
                        }
                        panic!("Failed to submit: {}", e.to_string());
                    }
                }
            }
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
                    let pending_completions = self.submitted_tasks[token]
                        .as_ref()
                        .expect("Task not found in submitted tasks")
                        .pending_completions;
                    
                    if pending_completions == 1 {
                        let mut submission = self.submitted_tasks[token]
                            .take()
                            .expect("Task not found in submitted tasks");
                        submission.push_completion(cqe);
                        submission.send_back();
                        self.tokens.push_back(token as u16);
                        self.io_performed.fetch_add(1, Ordering::Relaxed);
                    } else {
                        let submission = self.submitted_tasks[token]
                            .as_mut()
                            .expect("Task not found in submitted tasks");
                        submission.reduce_completions();
                        submission.push_completion(cqe);
                    }
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
    use_fixed_buffers: bool,
) -> Result<Bytes, std::io::Error> {
    // Perform open operations in a blocking manner as they are not compatible with a io_uring instance that uses polled mode IO
    let file = OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_DIRECT)
        .open(path)
        .expect("failed to create file");

    let effective_range = if let Some(range) = range {
        range
    } else {
        let len = file.metadata()?.len();
        0..len
    };

    if use_fixed_buffers {
        let read_task = FixedFileReadTask::build(effective_range.clone(), &file, direct_io);
        // Fall back to normal read if fixed buffers are not available
        if read_task.is_ok() {
            return submit_async_task(read_task.unwrap()).await.into_result()
        }
    }
    let read_task = FileReadTask::build(effective_range, file, direct_io);
    return submit_async_task(read_task).await.into_result()
}

pub(crate) async fn write(
    path: PathBuf,
    data: &Bytes,
    direct_io: bool,
    use_fixed_buffers: bool
) -> Result<(), std::io::Error> {
    let file = OpenOptions::new()
        .create(true)
        .truncate(true)
        .write(true)
        .custom_flags(libc::O_DIRECT)
        .open(path)
        .expect("failed to create file");

    let write_task = FileWriteTask::build(data.clone(), file.as_raw_fd(), direct_io, use_fixed_buffers);
    submit_async_task(write_task).await.into_result()
}
