
use std::{
    cell::{Cell, RefCell},
    collections::VecDeque,
    fs::OpenOptions,
    ops::Range,
    os::{fd::AsRawFd as _, unix::fs::OpenOptionsExt},
    path::PathBuf,
    pin::Pin,
    rc::Rc,
    sync::{
        Arc, Mutex, OnceLock, atomic::{AtomicBool, Ordering},
    },
    task::{Context, Poll, Waker},
    thread::{self, JoinHandle},
    time::{Duration, Instant},
};

use async_task::Runnable;
use bytes::Bytes;
use futures::Future;
use io_uring::{EnterFlags, IoUring, cqueue, squeue};
use liquid_cache_common::memory::pool::FixedBufferPool;
use rand::Rng;
use tokio::sync::oneshot;

use super::tasks::{FileReadTask, FileWriteTask, FixedFileReadTask, IoTask};

#[usdt::provider]
mod ws_uring_runtime {
    fn io_submission(id: u64) {}
    fn io_completion(id: u64) {}
}

fn ensure_uring_trace_registered() -> bool {
    static REGISTERED: OnceLock<bool> = OnceLock::new();
    *REGISTERED.get_or_init(|| match usdt::register_probes() {
        Ok(()) => true,
        Err(err) => {
            log::debug!("failed to register work-stealing io_uring USDT probes: {err}");
            false
        }
    })
}

type ExecutorTask = Pin<Box<dyn Future<Output = ()> + Send>>;

const URING_NUM_ENTRIES: u32 = 256;
const MAX_CONCURRENT_IO: u32 = 128;
const URING_BATCH_SIZE: u32 = 8;
const URING_SYSCALL_INTERVAL_US: u64 = 5;
const MAX_ACTIVE_TASKS_PER_THREAD: u32 = 5;

pub struct WorkStealingUringRuntime {
    _workers: Vec<JoinHandle<()>>,
    sender: crossbeam_channel::Sender<ExecutorTask>,
}

impl WorkStealingUringRuntime {
    /// Spawn `num_threads` worker threads, each with its own io_uring ring.
    pub fn new(num_threads: usize) -> Self {
        let (sender, receiver) = crossbeam_channel::unbounded();

        let mut workers = Vec::new();
        for i in 0..num_threads {
            let receiver_clone = receiver.clone();
            let worker = thread::Builder::new()
                .name(format!("ws-io-worker-{}", i))
                .spawn(move || worker_main_loop(receiver_clone))
                .expect("Failed to spawn worker");
            workers.push(worker);
        }

        WorkStealingUringRuntime {
            _workers: workers,
            sender
        }
    }

    /// Spawn a future on the runtime; the result is returned through a oneshot channel.
    pub fn spawn<F: Future + Send + 'static>(
        &self,
        future: F,
    ) -> oneshot::Receiver<F::Output>
    where
        F::Output: Send + 'static,
    {
        let (tx, rx) = oneshot::channel();

        let wrapped_fut = async move {
            let output = future.await;
            let _ = tx.send(output);
        };
        self.sender.send(Box::pin(wrapped_fut)).expect("Failed to send task");
        rx
    }

    /// Spawn a batch of futures, returning results via a crossbeam channel.
    pub fn spawn_many<F: Future + Send + 'static>(
        &self,
        futures: &mut Vec<F>,
    ) -> crossbeam_channel::Receiver<F::Output>
    where
        F::Output: Send + 'static,
    {
        let (tx, rx) = crossbeam_channel::bounded::<F::Output>(futures.len());
        for f in futures.drain(..) {
            let tx = tx.clone();
            let wrapped_fut = async move {
                let output = f.await;
                tx.send(output).expect("Failed to send result");
            };
            self.sender.send(Box::pin(wrapped_fut)).expect("Failed to send task");
        }
        rx
    }

    /// Spawn a future and block the caller until it completes.
    pub fn run_to_completion<F: Future + Send + 'static>(
        &self,
        future: F,
    ) -> F::Output
    where
        F::Output: Send + 'static,
    {
        let receiver = self.spawn(future);
        receiver.blocking_recv().expect("Failed to receive result")
    }
}

struct IoDriver {
    ring: IoUring,
    submitted_tasks: Vec<Option<AsyncIoTask>>,
    queued_entries: VecDeque<squeue::Entry>,
    last_syscall: Instant,
    tokens: VecDeque<u16>,
    io_performed: u64,
    queued_submissions: u64,
    fixed_buffers_available: bool,
}

impl IoDriver {
    fn new() -> IoDriver {
        let ring = IoUring::<squeue::Entry, cqueue::Entry>::builder()
            .setup_single_issuer()
            .setup_defer_taskrun()
            .build(URING_NUM_ENTRIES)
            .expect("Failed to build IoUring instance");

        let fixed_buffers_available =
            FixedBufferPool::register_buffers_with_ring(&ring).is_ok();

        let mut tokens = VecDeque::with_capacity(MAX_CONCURRENT_IO as usize);
        let mut submitted_tasks = Vec::with_capacity(MAX_CONCURRENT_IO as usize);
        for i in 0..MAX_CONCURRENT_IO {
            tokens.push_back(i as u16);
            submitted_tasks.push(None);
        }

        IoDriver {
            ring,
            submitted_tasks,
            tokens,
            queued_entries: VecDeque::with_capacity(URING_NUM_ENTRIES as usize),
            last_syscall: Instant::now(),
            io_performed: 0,
            queued_submissions: 0,
            fixed_buffers_available,
        }
    }

    #[inline]
    fn need_syscall(&self) -> bool {
        let is_batch_full = self.queued_entries.len() >= URING_BATCH_SIZE as usize;
        is_batch_full
            || self.last_syscall.elapsed() > Duration::from_micros(URING_SYSCALL_INTERVAL_US)
    }

    fn poll_completions(&mut self) {
        let cq = &mut self.ring.completion();
        loop {
            cq.sync();
            match cq.next() {
                Some(cqe) => {
                    let token = cqe.user_data() as usize;
                    let pending = self.submitted_tasks[token]
                        .as_ref()
                        .expect("Task not found in submitted tasks")
                        .pending_completions;
                    if pending == 1 {
                        let mut task = self.submitted_tasks[token]
                            .take()
                            .expect("Task not found in submitted tasks");
                        task.push_completion(cqe);
                        task.complete();
                        self.tokens.push_back(token as u16);
                        self.io_performed += 1;
                    } else {
                        let task = self.submitted_tasks[token]
                            .as_mut()
                            .expect("Task not found in submitted tasks");
                        task.push_completion(cqe);
                        task.reduce_completions();
                    }
                }
                None => break,
            }
        }
    }

    fn drain_intermediate_queue(&mut self) {
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

    fn submit_task(&mut self, mut task: AsyncIoTask) {
        let token = self.tokens.pop_front().expect("No more IO tokens");
        let sq = &mut self.ring.submission();
        let sqes = task.inner.lock().unwrap().prepare_sqe();
        let num_sqes = sqes.len();
        task.set_completions(num_sqes);
        self.submitted_tasks[token as usize] = Some(task);
        let mut sqes_submitted = 0;

        for sqe in sqes.iter() {
            let res = unsafe { sq.push(&sqe.clone().user_data(token as u64)) };
            if res.is_err() {
                break;
            }
            sqes_submitted += 1;
            self.queued_submissions += 1;
            sq.sync();
        }
        for i in sqes_submitted..sqes.len() {
            self.queued_entries
                .push_back(sqes[i].clone().user_data(token as u64));
        }
    }

    fn add_task(task: AsyncIoTask) {
        IO_REACTOR.with(|reactor| {
            reactor.borrow_mut().submit_task(task);
        });
    }
}

fn worker_main_loop(receiver: crossbeam_channel::Receiver<ExecutorTask>) {
    EXECUTOR.with(|worker| {
        let mut worker = worker.borrow_mut();
        worker.set_receiver(receiver);
    });
    loop {
        EXECUTOR.with(|worker| {
            let worker = &mut worker.borrow_mut();
            worker.try_tick();
        });
        IO_REACTOR.with(|reactor| {
            let reactor = &mut reactor.borrow_mut();
            reactor.drain_intermediate_queue();
            if reactor.need_syscall() {
                let mut flags = EnterFlags::empty();
                flags.insert(EnterFlags::GETEVENTS);
                loop {
                    let res = unsafe {
                        reactor.ring.submitter().enter::<libc::sigset_t>(
                            reactor.queued_submissions as u32,
                            0,
                            flags.bits(),
                            None,
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
                reactor.queued_submissions = 0;
                reactor.last_syscall = Instant::now();
            }
            reactor.poll_completions();
        });
    }
}

thread_local! {
    static EXECUTOR: RefCell<RuntimeWorker> = RefCell::new(RuntimeWorker::new());
    static IO_REACTOR: RefCell<IoDriver> = RefCell::new(IoDriver::new());
}

struct RuntimeWorker {
    task_receiver: Option<crossbeam_channel::Receiver<ExecutorTask>>,
    active_tasks: Rc<Cell<u32>>,
    local: Rc<RefCell<VecDeque<Runnable>>>,
}

impl RuntimeWorker {
    fn new() -> RuntimeWorker {
        RuntimeWorker {
            task_receiver: None,
            active_tasks: Rc::new(Cell::new(0)),
            local: Rc::new(RefCell::new(VecDeque::new())),
        }
    }

    fn set_receiver(&mut self, receiver: crossbeam_channel::Receiver<ExecutorTask>) {
        self.task_receiver = Some(receiver);
    }
    
    fn try_tick(&mut self) {
        let mut runnable = self.local.borrow_mut().pop_front();
        if runnable.is_none() && self.active_tasks.get() < MAX_ACTIVE_TASKS_PER_THREAD {
            if let Ok(future) = self.task_receiver.as_mut().unwrap().try_recv() {
                self.active_tasks.set(self.active_tasks.get().saturating_add(1));
                let active_tasks = Rc::clone(&self.active_tasks);
                let local_clone = Rc::clone(&self.local);
                let wrapped = async move {
                    future.await;
                    active_tasks.set(active_tasks.get().saturating_sub(1));
                };
                let schedule = move |r: Runnable| {
                    local_clone.borrow_mut().push_back(r);
                };
                let (r, task) = unsafe { async_task::spawn_unchecked(wrapped, schedule) };
                // Dropping `Task` would cancel the future and drop the oneshot sender (RecvError).
                task.detach();
                runnable = Some(r);
            }
        }
        if let Some(r) = runnable {
            r.run();
        }
    }
}


/// Thread-safe wrapper around an `IoTask`. Unlike the local runtime's version
/// which uses `Rc<RefCell<..>>`, this uses `Arc<Mutex<..>>` so that the task
/// can be submitted on one thread and completed/read on another.
struct AsyncIoTask {
    inner: Arc<Mutex<dyn IoTask>>,
    waker: Waker,
    completed: Arc<AtomicBool>,
    pending_completions: usize,
    completions: Vec<cqueue::Entry>,
}

unsafe impl Send for AsyncIoTask {}

impl AsyncIoTask {
    #[inline]
    fn complete(self) {
        self.inner
            .lock()
            .unwrap()
            .complete(self.completions.iter().collect());
        self.completed.store(true, Ordering::Release);
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

enum UringState {
    Undecided,
    Created,
    Submitted,
}

pub(crate) struct UringFuture<T>
where
    T: IoTask + 'static,
{
    state: UringState,
    task: Arc<Mutex<T>>,
    completed: Arc<AtomicBool>,
    id: u64,
}

unsafe impl<T> Send for UringFuture<T> where T: IoTask + 'static {}

impl<T> UringFuture<T>
where
    T: IoTask + 'static,
{
    fn new(task: Arc<Mutex<T>>) -> Self {
        UringFuture {
            state: UringState::Created,
            task,
            completed: Arc::new(AtomicBool::new(false)),
            id: rand::rng().random(),
        }
    }
}

impl<T> Future for UringFuture<T>
where
    T: IoTask + 'static,
{
    type Output = Arc<Mutex<T>>;

    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        loop {
            let state = std::mem::replace(&mut self.state, UringState::Undecided);
            match state {
                UringState::Created => {
                    let async_task = AsyncIoTask {
                        inner: self.task.clone(),
                        waker: cx.waker().clone(),
                        completed: self.completed.clone(),
                        pending_completions: 0,
                        completions: Vec::new(),
                    };
                    IoDriver::add_task(async_task);
                    if ensure_uring_trace_registered() {
                        ws_uring_runtime::io_submission!(|| self.id);
                    }
                    self.state = UringState::Submitted;
                }
                UringState::Submitted => {
                    if self.completed.load(Ordering::Acquire) {
                        if ensure_uring_trace_registered() {
                            ws_uring_runtime::io_completion!(|| self.id);
                        }
                        return Poll::Ready(self.task.clone());
                    }
                    self.state = UringState::Submitted;
                    return Poll::Pending;
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
    UringFuture::new(Arc::new(Mutex::new(task)))
}

pub(crate) async fn read(
    path: PathBuf,
    range: Option<Range<u64>>,
) -> Result<Bytes, std::io::Error> {
    let direct_io = IO_REACTOR.with(|w| w.borrow().fixed_buffers_available);

    let mut opts = OpenOptions::new();
    opts.read(true);
    if direct_io {
        opts.custom_flags(libc::O_DIRECT);
    }
    let file = opts.open(&path).expect("failed to open file");

    let effective_range = if let Some(range) = range {
        range
    } else {
        let len = file.metadata()?.len();
        0..len
    };

    if direct_io {
        let read_task = FixedFileReadTask::build(effective_range.clone(), &file, true);
        if let Ok(task) = read_task {
            let arc = submit_async_task(task).await;
            return match Arc::try_unwrap(arc) {
                Ok(mutex) => {
                    FixedFileReadTask::into_result(Box::new(mutex.into_inner().unwrap()))
                }
                Err(arc) => arc.lock().unwrap().get_result(),
            };
        }
    }

    let read_task = FileReadTask::build(effective_range, file, direct_io);
    let arc = submit_async_task(read_task).await;
    match Arc::try_unwrap(arc) {
        Ok(mutex) => FileReadTask::into_result(Box::new(mutex.into_inner().unwrap())),
        Err(arc) => arc.lock().unwrap().get_result(),
    }
}

pub(crate) async fn write(path: PathBuf, data: &Bytes) -> Result<(), std::io::Error> {
    let file = OpenOptions::new()
        .create(true)
        .truncate(true)
        .write(true)
        .open(&path)
        .expect("failed to create file");

    let write_task = FileWriteTask::build(data.clone(), file.as_raw_fd(), false, false);
    let arc = submit_async_task(write_task).await;
    match Arc::try_unwrap(arc) {
        Ok(mutex) => mutex.into_inner().unwrap().get_result(),
        Err(arc) => arc.lock().unwrap().get_result(),
    }
}
