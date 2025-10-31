use std::{
    collections::VecDeque,
    fs::OpenOptions,
    future::Future,
    ops::Range,
    os::fd::AsRawFd,
    path::PathBuf,
    pin::Pin,
    sync::{Mutex, MutexGuard, OnceLock},
    task::{Context, Poll},
};

use ahash::{AHashMap, AHashSet};
use bytes::Bytes;
use io_uring::{IoUring, cqueue};

use super::tasks::{FileOpenTask, FileReadTask, FileWriteTask, IoTask};
use super::thread_pool_uring::URING_NUM_ENTRIES;

struct SharedRing {
    inner: Mutex<SharedRingInner>,
}

impl SharedRing {
    fn new() -> SharedRing {
        let ring = IoUring::builder()
            .build(URING_NUM_ENTRIES)
            .expect("Failed to build shared io-uring instance");
        SharedRing {
            inner: Mutex::new(SharedRingInner::from_ring(ring)),
        }
    }

    #[inline]
    fn lock(&self) -> MutexGuard<'_, SharedRingInner> {
        self.inner.lock().expect("shared io-uring mutex poisoned")
    }
}

struct SharedRingInner {
    ring: IoUring,
    free_tokens: VecDeque<u16>,
    completions: AHashMap<u16, cqueue::Entry>,
    abandoned: AHashSet<u16>,
}

impl SharedRingInner {
    fn from_ring(ring: IoUring) -> SharedRingInner {
        let free_tokens = (0..URING_NUM_ENTRIES)
            .map(|idx| idx as u16)
            .collect::<VecDeque<_>>();
        SharedRingInner {
            ring,
            free_tokens,
            completions: AHashMap::with_capacity(URING_NUM_ENTRIES as usize),
            abandoned: AHashSet::with_capacity(URING_NUM_ENTRIES as usize),
        }
    }

    #[inline]
    fn acquire_token(&mut self) -> Option<u16> {
        if let Some(token) = self.free_tokens.pop_front() {
            Some(token)
        } else {
            self.drain_completions();
            self.free_tokens.pop_front()
        }
    }

    fn submit_task(&mut self, task: &mut dyn IoTask) -> Option<u16> {
        let token = self.acquire_token()?;

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

        Some(token)
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

    fn drain_completions(&mut self) {
        let mut cq = self.ring.completion();
        cq.sync();
        while let Some(cqe) = cq.next() {
            let raw_token = cqe.user_data();
            debug_assert!(
                raw_token < URING_NUM_ENTRIES as u64,
                "completion token {raw_token} exceeds ring capacity"
            );
            let token = raw_token as u16;
            if self.abandoned.remove(&token) {
                self.free_tokens.push_back(token);
            } else {
                let replaced = self.completions.insert(token, cqe);
                debug_assert!(
                    replaced.is_none(),
                    "completion map already contained token {token}"
                );
            }
        }
    }

    #[inline]
    fn take_completion(&mut self, token: u16) -> Option<cqueue::Entry> {
        self.completions.remove(&token).map(|entry| {
            self.free_tokens.push_back(token);
            entry
        })
    }

    #[inline]
    fn abandon_submission(&mut self, token: u16) {
        if self.completions.remove(&token).is_some() {
            self.free_tokens.push_back(token);
        } else {
            self.abandoned.insert(token);
        }
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
        let mut guard = ring.lock();

        let state = std::mem::replace(&mut this.state, State::Invalid);

        match state {
            State::Pending { token, mut task } => {
                if let Some(cqe) = guard.take_completion(token) {
                    drop(guard);
                    task.complete(&cqe);
                    return Poll::Ready(task);
                } else {
                    guard.drain_completions();
                }
                // Not ready yet, restore state
                this.state = State::Pending { token, task };
            }
            State::Created(mut task) => {
                if let Some(token) = guard.submit_task(task.as_mut()) {
                    this.state = State::Pending { token, task };
                    drop(guard);
                    // Ensure the runtime keeps polling so completions are drained without a background thread.
                    cx.waker().wake_by_ref();
                    return Poll::Pending;
                }
                // Submission failed, restore state, and drain completions
                guard.drain_completions();
                this.state = State::Created(task);
            }
            State::Invalid => {
                panic!("poll called on invalid future state");
            }
        }

        drop(guard);
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
            if let Ok(mut guard) = ring.inner.lock() {
                guard.abandon_submission(token);
            }
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
        .write(true)
        .open(path)
        .expect("failed to create file");

    let write_task = FileWriteTask::build(data.as_ptr(), data.len(), file.as_raw_fd());
    submit_async_task(write_task).await.into_result()
}
