use std::{
    fs::OpenOptions,
    future::Future,
    ops::Range,
    os::fd::AsRawFd,
    path::PathBuf,
    pin::Pin,
    sync::{Arc, Mutex, OnceLock, TryLockError},
    task::{Context, Poll},
};

use ahash::AHashSet;
use bytes::Bytes;
use crossbeam_queue::ArrayQueue;
use io_uring::{IoUring, cqueue};

use super::tasks::{FileOpenTask, FileReadTask, FileWriteTask, IoTask};
use super::thread_pool_uring::URING_NUM_ENTRIES;

struct CompletionStore {
    entries: Vec<Mutex<Option<cqueue::Entry>>>,
}

impl CompletionStore {
    fn new() -> CompletionStore {
        let mut entries = Vec::with_capacity(URING_NUM_ENTRIES as usize);
        entries.resize_with(URING_NUM_ENTRIES as usize, || Mutex::new(None));
        CompletionStore { entries }
    }

    #[inline]
    fn insert(&self, token: u16, cqe: cqueue::Entry) {
        let idx = token as usize;
        let mut guard = self.entries[idx]
            .lock()
            .expect("completion entry mutex poisoned");
        let replaced = guard.replace(cqe);
        debug_assert!(
            replaced.is_none(),
            "completion map already contained token {token}"
        );
    }

    #[inline]
    fn take(&self, token: u16) -> Option<cqueue::Entry> {
        let idx = token as usize;
        let mut guard = self.entries[idx]
            .lock()
            .expect("completion entry mutex poisoned");
        guard.take()
    }
}

struct SharedRing {
    ring: Mutex<SharedRingInner>,
    completions: CompletionStore,
    free_tokens: Arc<ArrayQueue<u16>>,
}

impl SharedRing {
    fn new() -> SharedRing {
        let ring = IoUring::builder()
            .build(URING_NUM_ENTRIES)
            .expect("Failed to build shared io-uring instance");
        let free_tokens = Arc::new(ArrayQueue::new(URING_NUM_ENTRIES as usize));
        for idx in 0..URING_NUM_ENTRIES {
            free_tokens
                .push(idx as u16)
                .expect("token queue capacity exceeded during initialization");
        }
        SharedRing {
            ring: Mutex::new(SharedRingInner::from_ring(ring)),
            completions: CompletionStore::new(),
            free_tokens,
        }
    }

    fn acquire_token(&self) -> Option<u16> {
        if let Some(token) = self.free_tokens.pop() {
            return Some(token);
        }

        let mut guard = self.ring.lock().expect("shared io-uring mutex poisoned");
        guard.drain_completions(&self.completions, &self.free_tokens);
        drop(guard);

        self.free_tokens.pop()
    }

    fn submit_task(&self, task: &mut dyn IoTask) -> Option<u16> {
        let token = self.acquire_token()?;

        let mut guard = self.ring.lock().expect("shared io-uring mutex poisoned");
        guard.submit_task(task, token);

        Some(token)
    }

    fn take_completion(&self, token: u16) -> Option<cqueue::Entry> {
        if let Some(cqe) = self.completions.take(token) {
            self.free_tokens
                .push(token)
                .expect("token queue capacity exceeded while releasing completion");
            return Some(cqe);
        }

        let mut guard = match self.ring.try_lock() {
            Ok(guard) => guard,
            Err(TryLockError::WouldBlock) => {
                return None;
            }
            Err(TryLockError::Poisoned(_)) => {
                panic!("shared io-uring mutex poisoned");
            }
        };
        guard.drain_completions(&self.completions, &self.free_tokens);
        drop(guard);

        if let Some(cqe) = self.completions.take(token) {
            self.free_tokens
                .push(token)
                .expect("token queue capacity exceeded while releasing completion");
            Some(cqe)
        } else {
            None
        }
    }

    fn drain_completions(&self) {
        let mut guard = self.ring.lock().expect("shared io-uring mutex poisoned");
        guard.drain_completions(&self.completions, &self.free_tokens);
    }

    fn abandon_submission(&self, token: u16) {
        if self.completions.take(token).is_some() {
            self.free_tokens
                .push(token)
                .expect("token queue capacity exceeded while abandoning completion");
            return;
        }

        if let Ok(mut guard) = self.ring.lock() {
            guard.mark_abandoned(token);
        }
    }
}

struct SharedRingInner {
    ring: IoUring,
    abandoned: AHashSet<u16>,
}

impl SharedRingInner {
    fn from_ring(ring: IoUring) -> SharedRingInner {
        SharedRingInner {
            ring,
            abandoned: AHashSet::with_capacity(URING_NUM_ENTRIES as usize),
        }
    }

    fn submit_task(&mut self, task: &mut dyn IoTask, token: u16) {
        {
            let mut sq = self.ring.submission();
            let entry = task.prepare_sqe().user_data(token as u64);
            unsafe {
                sq.push(&entry)
                    .expect("Failed to push entry to io-uring submission queue");
            }
            sq.sync();
        }

        self.flush_submission_queue();
    }

    fn flush_submission_queue(&mut self) {
        loop {
            match self.ring.submit() {
                Ok(_) => break,
                Err(err) => {
                    if err.raw_os_error() == Some(libc::EINTR) {
                        continue;
                    }
                    panic!("Failed to submit to io-uring: {err}");
                }
            }
        }
    }

    fn drain_completions(
        &mut self,
        completions: &CompletionStore,
        free_tokens: &Arc<ArrayQueue<u16>>,
    ) {
        let mut cq = self.ring.completion();
        cq.sync();
        for cqe in cq {
            let raw_token = cqe.user_data();
            debug_assert!(
                raw_token < URING_NUM_ENTRIES as u64,
                "completion token {raw_token} exceeds ring capacity"
            );
            let token = raw_token as u16;
            if self.abandoned.remove(&token) {
                free_tokens
                    .push(token)
                    .expect("token queue capacity exceeded while reclaiming abandoned submission");
            } else {
                completions.insert(token, cqe);
            }
        }
    }

    #[inline]
    fn mark_abandoned(&mut self, token: u16) {
        self.abandoned.insert(token);
    }
}

static SHARED_RING: OnceLock<SharedRing> = OnceLock::new();

fn shared_ring() -> &'static SharedRing {
    SHARED_RING.get_or_init(SharedRing::new)
}

enum State<T>
where
    T: IoTask + 'static,
{
    Created(Box<T>),
    Pending { token: u16, task: Box<T> },
    Invalid,
}

pub(crate) struct SharedUringFuture<T>
where
    T: IoTask + 'static,
{
    state: State<T>,
}

impl<T> SharedUringFuture<T>
where
    T: IoTask + 'static,
{
    fn new(task: T) -> SharedUringFuture<T> {
        SharedUringFuture {
            state: State::Created(Box::new(task)),
        }
    }
}

impl<T> Future for SharedUringFuture<T>
where
    T: IoTask + 'static,
{
    type Output = Box<T>;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let this = self.get_mut();
        let ring = shared_ring();

        let state = std::mem::replace(&mut this.state, State::Invalid);

        match state {
            State::Pending { token, mut task } => {
                if let Some(cqe) = ring.take_completion(token) {
                    task.complete(&cqe);
                    return Poll::Ready(task);
                }
                // Not ready yet, restore state
                this.state = State::Pending { token, task };
            }
            State::Created(mut task) => {
                if let Some(token) = ring.submit_task(task.as_mut()) {
                    this.state = State::Pending { token, task };
                    // Ensure the runtime keeps polling so completions are drained without a background thread.
                    cx.waker().wake_by_ref();
                    return Poll::Pending;
                }
                // Submission failed, restore state, and drain completions
                ring.drain_completions();
                this.state = State::Created(task);
            }
            State::Invalid => {
                panic!("poll called on invalid future state");
            }
        }

        // No new data yet; reschedule so another poll can drive the ring forward.
        cx.waker().wake_by_ref();
        Poll::Pending
    }
}

impl<T> Drop for SharedUringFuture<T>
where
    T: IoTask + 'static,
{
    fn drop(&mut self) {
        if let State::Pending { token, .. } = self.state {
            // Attempt to drop any completion state, or mark the submission as abandoned.
            let ring = shared_ring();
            ring.abandon_submission(token);
        }
    }
}

fn submit_async_task<T>(task: T) -> SharedUringFuture<T>
where
    T: IoTask + 'static,
{
    SharedUringFuture::new(task)
}

pub(crate) async fn read(
    path: PathBuf,
    range: Option<Range<u64>>,
    direct_io: bool,
) -> Result<Bytes, std::io::Error> {
    let open_task = FileOpenTask::build(path, direct_io)?;
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
    let file = OpenOptions::new()
        .create(true)
        .truncate(true)
        .write(true)
        .open(path)
        .expect("failed to create file");

    let write_task = FileWriteTask::build(data.clone(), file.as_raw_fd());
    submit_async_task(write_task).await.into_result()
}
