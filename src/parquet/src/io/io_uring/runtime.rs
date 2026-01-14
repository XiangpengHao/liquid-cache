use std::{cell::RefCell, collections::VecDeque, fs::OpenOptions, ops::Range, os::fd::AsRawFd as _, path::PathBuf, pin::Pin, rc::Rc, sync::atomic::{AtomicBool, Ordering}, task::{Context, Poll, Waker}, thread::{self, JoinHandle}};

use async_executor::LocalExecutor;
use bytes::Bytes;
use futures::Future;
use io_uring::{IoUring, squeue, cqueue};
use tokio::sync::oneshot;

use crate::io::io_uring::tasks::{FileOpenTask, FileReadTask, FileWriteTask, IoTask};

const URING_NUM_ENTRIES: u32 = 128;

const MAX_CONCURRENT_TASKS: u32 = 128;

type ExecutorTask = Pin<Box<dyn Future<Output = ()> + Send>>;

pub struct UringExecutor {
    workers: Vec<JoinHandle<()>>,
    sender: crossbeam_channel::Sender<ExecutorTask>,
}

impl UringExecutor {
    /// Spawn worker threads and initialize channel to receive tasks
    pub fn new(num_threads: usize) -> UringExecutor {
        let mut workers = Vec::new();
        let (sender, receiver) = crossbeam_channel::unbounded::<ExecutorTask>();
        for i in 0..num_threads {
            let receiver_clone = receiver.clone();
            let worker = thread::Builder::new()
                .name(std::format!("lc-io-worker-{}", i))
                .spawn(move || {
                    worker_main_loop(receiver_clone);
                })
                .expect("Failed to spawn IO runtime worker");
            workers.push(worker);
        }
        UringExecutor {
            workers,
            sender,
        }
    }

    /// Spawns a task in the uring runtime by sending it through a crossbeam channel.
    /// The result is received through a oneshot channel
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
        self.sender.send(task).expect("UringExecutor failed to send task");
        receiver
    }

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

struct RuntimeWorker {
    ring: io_uring::IoUring,
    inflight_tasks: Vec<Option<AsyncTask>>,
    tokens: VecDeque<u16>,
    need_submit: bool,
    io_performed: u64,
}

impl RuntimeWorker {
    pub fn new() -> RuntimeWorker {
        let builder = IoUring::<squeue::Entry, cqueue::Entry>::builder();
        let ring = builder
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
            inflight_tasks,
            tokens,
            need_submit: false,
            io_performed: 0,
        }
    }

    fn poll_completions(self: &mut Self) {
        let cq = &mut self.ring.completion();
        loop {
            cq.sync();
            match cq.next() {
                Some(cqe) => {
                    let token = cqe.user_data() as usize;
                    let task = self.inflight_tasks[token]
                        .take()
                        .expect("Task not found in submitted tasks");
                    task.inner.borrow_mut().complete(&cqe);
                    unsafe { (*task.completed).store(true, Ordering::Relaxed); }
                    task.waker.wake();
                    self.tokens.push_back(token as u16);
                    self.io_performed += 1;
                }
                None => break,
            }
        }
    }

    fn submit_task(self: &mut Self, task: AsyncTask) {
        let token = self.tokens.pop_front().expect("No more tokens");
        let sq = &mut self.ring.submission();
        let sqe = task.inner.borrow_mut().prepare_sqe().user_data(token as u64);
        unsafe {
            sq.push(&sqe).expect("Failed to push to submission queue");
        }
        sq.sync();
        self.inflight_tasks[token as usize] = Some(task);
        self.need_submit = true;
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
        while !receiver.is_empty() {
            let task = receiver.recv()
                .expect("Failed to receive task");
            executor.spawn(task).detach();
        }
        let task_found = executor.try_tick();
        LOCAL_WORKER.with(|worker| {
            let mut worker = worker.borrow_mut();
            if worker.need_submit {
                worker.ring.submit().expect("Failed to submit");
                worker.need_submit = false;
            } else if !task_found && worker.tokens.len() < MAX_CONCURRENT_TASKS as usize {
                worker.ring.submit_and_wait(1).expect("Failed to submit");
            }
            worker.poll_completions();
        });
    }
}

struct AsyncTask {
    // Note: Should change this to Arc in case of a work-stealing scheduler
    pub inner: Rc<RefCell<dyn IoTask>>,
    pub waker: Waker,
    pub completed: *mut AtomicBool,
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
    let open_task = FileOpenTask::build(path, direct_io)?;
    let file = submit_async_task(open_task).await.borrow_mut().get_result()?;

    let effective_range = if let Some(range) = range {
        range
    } else {
        let len = file.metadata()?.len();
        0..len
    };

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

    let write_task = FileWriteTask::build(data.clone(), file.as_raw_fd());
    submit_async_task(write_task).await.borrow_mut().get_result()
}