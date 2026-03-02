use std::{cell::RefCell, collections::VecDeque, fs::OpenOptions, ops::Range, os::{fd::AsRawFd as _, unix::fs::OpenOptionsExt}, path::PathBuf, pin::Pin, rc::Rc, sync::atomic::{AtomicBool, Ordering}, task::{Context, Poll, Waker}, thread::{self, JoinHandle}, time::{Duration, Instant}};

use async_executor::LocalExecutor;
use bytes::Bytes;
use futures::Future;
use io_uring::{EnterFlags, IoUring, cqueue, squeue};
use rand::Rng;
use tokio::sync::oneshot;

use crate::io::io_uring::tasks::{FileReadTask, FileWriteTask, FixedFileReadTask, IoTask};

const URING_NUM_ENTRIES: u32 = 256;

const MAX_CONCURRENT_TASKS: u32 = 128;

type ExecutorTask = Pin<Box<dyn Future<Output = ()> + Send>>;

/// A dedicated runtime for io_uring, in which the worker threads are responsible for submitting IO and polling for completions.
/// Each worker thread has its own ring, and an executor which is responsible for scheduling.
pub struct UringExecutor {
    workers: Vec<JoinHandle<()>>,
    /// One sender per worker; tasks are submitted to a worker's dedicated channel.
    senders: Vec<crossbeam_channel::Sender<ExecutorTask>>,
}

impl UringExecutor {
    /// Spawn worker threads; each worker has its own channel to receive tasks.
    pub fn new(num_threads: usize) -> UringExecutor {
        let mut workers = Vec::new();
        let mut senders = Vec::with_capacity(num_threads);
        for i in 0..num_threads {
            let (sender, receiver) = crossbeam_channel::unbounded::<ExecutorTask>();
            senders.push(sender);
            let worker = thread::Builder::new()
                .name(std::format!("lc-io-worker-{}", i))
                .spawn(move || {
                    worker_main_loop(receiver);
                })
                .expect("Failed to spawn IO runtime worker");
            workers.push(worker);
        }
        UringExecutor {
            workers,
            senders,
        }
    }

    /// Spawns a task in the uring runtime by sending it to a randomly chosen worker's channel.
    /// The result is received through a oneshot channel.
    pub fn spawn<F: Future + Send + 'static>(self: &mut Self, future: F) -> oneshot::Receiver<F::Output>
    where
        F::Output: Send + 'static,
    {
        let (sender, receiver) = oneshot::channel();
        let f = async move {
            let output = future.await;
            let _res = sender.send(output);
            if !_res.is_ok() {
                panic!("Failed to send task result back");
            }
        };
        let task = Box::pin(f);
        let idx = rand::rng().random_range(0..self.senders.len());
        self.senders[idx]
            .send(task)
            .expect("UringExecutor failed to send task");
        receiver
    }

    /// Spawn a batch of tasks on the io_uring runtime, balancing across workers (round-robin).
    pub fn spawn_many<F: Future + Send + 'static>(self: &mut Self, futures: &mut Vec<F>) -> crossbeam_channel::Receiver<F::Output>
    where
        F::Output: Send + 'static,
    {
        let (sender, receiver) = crossbeam_channel::bounded::<F::Output>(futures.len());
        let num_workers = self.senders.len();
        for (i, f) in futures.drain(..).enumerate() {
            let sender_clone = sender.clone();
            let f = Box::pin(f);
            let task = async move {
                let output = f.await;
                sender_clone.send(output).expect("Failed to send back result");
            };
            let idx = i % num_workers;
            self.senders[idx]
                .send(Box::pin(task))
                .expect("UringExecutor failed to send task");
        }
        receiver
    }

    /// Spawns a task on the io_uring runtime and blocks on it
    pub fn run_to_completion<F: Future + Send + 'static>(self: &mut Self, future: F) -> F::Output
    where
        F::Output: Send + 'static,
    {
        let receiver = self.spawn(future);
        receiver.blocking_recv().expect("Failed to receive result")
    }
}

thread_local! {
    static LOCAL_WORKER: RefCell<RuntimeWorker> = RefCell::new(RuntimeWorker::new());
}

const URING_BATCH_SIZE: u32 = 8;

const URING_SYSCALL_INTERVAL_US: u64 = 5;

const RUNTIME_TASK_BATCH_SIZE: u32 = 4;

struct RuntimeWorker {
    ring: io_uring::IoUring,
    submitted_tasks: Vec<Option<AsyncTask>>,
    /**
     * When using fixed buffers, a single task can produce multiple submission queue entries.
     * It is possible that we aren't able to submit all of them at one go. Hold them in an 
     * intermediate queue in that case
     */
    queued_entries: VecDeque<squeue::Entry>,
    last_syscall: Instant,
    tokens: VecDeque<u16>,
    io_performed: u64,
    queued_submissions: u64,
}

impl RuntimeWorker {
    pub fn new() -> RuntimeWorker {
        let mut builder = IoUring::<squeue::Entry, cqueue::Entry>::builder();
        let ring = builder
            .setup_single_issuer()      // Only the worker thread will issue IO and poll completions
            .setup_defer_taskrun()
            .build(URING_NUM_ENTRIES)
            .expect("Failed to build IoUring instance");
        let mut tokens = VecDeque::<u16>::with_capacity(MAX_CONCURRENT_TASKS as usize);
        let mut inflight_tasks = Vec::<Option<AsyncTask>>::with_capacity(MAX_CONCURRENT_TASKS as usize);
        for i in 0..MAX_CONCURRENT_TASKS {
            tokens.push_back(i as u16);
            inflight_tasks.push(None);
        }
        
        RuntimeWorker {
            ring, 
            submitted_tasks: inflight_tasks,
            tokens,
            queued_entries: VecDeque::with_capacity(URING_NUM_ENTRIES as usize),
            last_syscall: Instant::now(),
            io_performed: 0,
            queued_submissions: 0,
        }
    }

    #[inline]
    fn need_syscall(self: &Self) -> bool {
        let time_from_last_submit = self.last_syscall.elapsed();
        let is_batch_full = self.queued_entries.len() >= URING_BATCH_SIZE as usize;
        is_batch_full || time_from_last_submit > Duration::from_micros(URING_SYSCALL_INTERVAL_US)
    }

    fn poll_completions(self: &mut Self) {
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
                    
                    let mut submission = self.submitted_tasks[token]
                        .take()
                        .expect("Task not found in submitted tasks");
                    submission.push_completion(cqe);
                    if pending_completions == 1 {
                        submission.complete();
                        self.tokens.push_back(token as u16);
                        self.io_performed += 1;
                    } else {
                        submission.reduce_completions();
                    }
                }
                None => break,
            }
        }
    }

    fn drain_intermediate_queue(&mut self) {
        {
            let sq = &mut self.ring.submission();           
            while !sq.is_full() && !self.queued_entries.is_empty() {
                let sqe = self.queued_entries.pop_front().unwrap();
                unsafe {
                    sq.push(&sqe).expect("Failed to push to submission queue");
                }
                sq.sync();
                self.queued_submissions += 1;
            }
        }
    }

    fn submit_task(self: &mut Self, mut task: AsyncTask) {
        let token = self.tokens.pop_front().expect("No more tokens");
        let sq = &mut self.ring.submission();
        let sqes = task.inner.borrow_mut().prepare_sqe();
        task.set_completions(sqes.len());
        self.submitted_tasks[token as usize] = Some(task);
        let mut sqes_submitted = 0;

        for sqe in sqes.iter() {
            let res = unsafe {
                sq.push(&sqe.clone().user_data(token as u64))
            };
            if res.is_err() {
                // submission queue is full
                break;
            }
            sqes_submitted += 1;
            self.queued_submissions += 1;
            sq.sync();
        }
        for i in sqes_submitted..sqes.len() {
            self.queued_entries.push_back(sqes[i].clone().user_data(token as u64));
        }
    }

    pub fn add_task(task: AsyncTask) {
        LOCAL_WORKER.with(|worker| {
            let mut worker = worker.borrow_mut();
            worker.submit_task(task);
        });
    }
}

fn worker_main_loop(receiver: crossbeam_channel::Receiver<ExecutorTask>) {
    let executor = LocalExecutor::new();
    loop {
        let mut tasks_submitted = 0;
        // Need some form of admission control here
        while tasks_submitted < RUNTIME_TASK_BATCH_SIZE && !receiver.is_empty() {
            let task = receiver.try_recv();
            if task.is_err() {
                continue;
            }
            executor.spawn(task.unwrap()).detach();
            tasks_submitted += 1;
        }
        // Can we batch the ticks?
        let _task_found = executor.try_tick();
        LOCAL_WORKER.with(|worker| {
            let mut worker = worker.borrow_mut();
            worker.drain_intermediate_queue();
            if worker.need_syscall() {
                let mut flags = EnterFlags::empty();
                flags.insert(EnterFlags::GETEVENTS);
                loop {
                    let res = unsafe {
                        worker.ring.submitter().enter::<libc::sigset_t>(
                            worker.queued_submissions as u32, 
                            0, 
                            flags.bits(), 
                            None
                        )
                    };
                    match res {
                        Ok(_num_entries) => {
                            break;
                        }
                        Err(e) => {
                            if e.kind() == std::io::ErrorKind::Interrupted {
                                continue;
                            }
                            panic!("Failed to submit: {}", e.to_string());
                        }
                    }
                }
                worker.queued_submissions = 0;
                worker.last_syscall = Instant::now();
            }
            // else if !task_found && worker.tokens.len() < MAX_CONCURRENT_TASKS as usize {
            //     worker.ring.submit_and_wait(1).expect("Failed to submit");
            // }
            worker.poll_completions();
        });
    }
}

struct AsyncTask {
    // Note: Should change this to Arc in case of a work-stealing scheduler
    pub inner: Rc<RefCell<dyn IoTask>>,
    waker: Waker,
    completed: *mut AtomicBool,
    pending_completions: usize,   // No. of pending completions. Will be populated later by the uring worker
    completions: Vec<cqueue::Entry>,
}

impl AsyncTask {
    #[inline]
    fn complete(self) {
        self.inner.borrow_mut().complete(self.completions.iter().collect());
        unsafe {
            (*self.completed).store(true, Ordering::Relaxed);
        }
        self.waker.wake();
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

enum UringState
{
    Undecided,
    Created,
    Submitted,
}

pub(crate) struct UringFuture<T>
where
    T: IoTask + 'static,
{
    state: UringState,
    task: Rc<RefCell<T>>,
    completed: AtomicBool,
}

unsafe impl<T> Send for UringFuture<T>
where T: IoTask + 'static, {}

impl<T> UringFuture<T>
where
    T: IoTask + 'static,
{
    fn new(task: Rc<RefCell<T>>) -> UringFuture<T> {
        UringFuture {
            state: UringState::Created,
            task: task,
            completed: AtomicBool::new(false),
        }
    }
}

impl<T> Future for UringFuture<T>
where
    T: IoTask + 'static,
{
    type Output = Rc<RefCell<T>>;

    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        loop {
            let state = std::mem::replace(&mut self.state, UringState::Undecided);
            match state {
                UringState::Created => {
                    let async_task = AsyncTask {
                        inner: self.task.clone(),
                        waker: cx.waker().clone(),
                        completed: &mut self.completed,
                        pending_completions: 0,
                        completions: Vec::new(),
                    };
                    RuntimeWorker::add_task(async_task);
                    self.state = UringState::Submitted;
                }
                UringState::Submitted => match self.completed.load(Ordering::Relaxed) {
                    true => {
                        return Poll::Ready(self.task.clone());
                    }
                    false => {
                        self.state = UringState::Submitted;
                        return Poll::Pending;
                    }
                }
                UringState::Undecided => unreachable!("state cannot be undecided during poll"),
            }
        }
    }
}

fn submit_async_task<T>(task: T) -> UringFuture<T>
where
    T: IoTask + 'static,
{
    UringFuture::new(Rc::new(RefCell::new(task)))
}

pub(crate) async fn read(
    path: PathBuf,
    range: Option<Range<u64>>,
    direct_io: bool,
) -> Result<Bytes, std::io::Error> {
    let file = OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_DIRECT)
        .open(path)
        .expect("failed to open file");

    let effective_range = if let Some(range) = range {
        range
    } else {
        let len = file.metadata()?.len();
        0..len
    };

    {
        let read_task = FixedFileReadTask::build(effective_range.clone(), &file, direct_io);
        if read_task.is_ok() {
            let rc = submit_async_task(read_task.unwrap()).await;
            return match Rc::try_unwrap(rc) {
                Ok(cell) => FixedFileReadTask::into_result(Box::new(cell.into_inner())),
                Err(rc) => rc.borrow_mut().get_result(),
            };
        }
    }
    // Fall back to normal read if fixed buffers are not available
    let read_task = FileReadTask::build(effective_range, file, direct_io);
    submit_async_task(read_task).await.borrow_mut().get_result()
}

pub(crate) async fn write(path: PathBuf, data: &Bytes) -> Result<(), std::io::Error> {
    let file = OpenOptions::new()
        .create(true)
        .truncate(true)
        .write(true)
        .open(path)
        .expect("failed to create file");

    let write_task = FileWriteTask::build(data.clone(), file.as_raw_fd(), true, false);
    submit_async_task(write_task).await.borrow_mut().get_result()
}