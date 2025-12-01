use std::{
    fs::OpenOptions,
    future::Future,
    io,
    ops::Range,
    os::fd::AsRawFd,
    path::PathBuf,
    pin::Pin,
    sync::OnceLock,
    task::{Context, Poll},
    thread,
};

use bytes::Bytes;
use crossbeam_queue::SegQueue;
use io_uring::{IoUring, cqueue};

use super::{
    tasks::{FileOpenTask, FileReadTask, FileWriteTask, IoTask},
    thread_pool_uring::URING_NUM_ENTRIES,
};

const SQPOLL_IDLE_MS: u32 = 10;

struct AsyncRing {
    ring: Box<IoUring>,
}

impl AsyncRing {
    fn attached_to(shared_fd: libc::c_int) -> io::Result<AsyncRing> {
        let mut builder = IoUring::builder();
        builder.setup_attach_wq(shared_fd);
        let ring = builder.build(URING_NUM_ENTRIES)?;
        Ok(AsyncRing {
            ring: Box::new(ring),
        })
    }

    fn submit_task(&mut self, task: &mut dyn IoTask) {
        {
            let mut sq = self.ring.submission();
            let entry = task.prepare_sqe()[0].clone().user_data(0);
            unsafe {
                sq.push(&entry)
                    .expect("failed to push entry to io-uring submission queue");
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
                    panic!("failed to submit io-uring request: {err}");
                }
            }
        }
    }

    fn take_completion(&mut self) -> Option<cqueue::Entry> {
        let mut cq = self.ring.completion();
        cq.sync();
        cq.next()
    }
}

struct AsyncRingManager {
    shared_ring: IoUring,
    available_rings: SegQueue<AsyncRing>,
}

impl AsyncRingManager {
    fn new(num_rings: usize) -> io::Result<AsyncRingManager> {
        let mut shared_builder = IoUring::builder();
        shared_builder.setup_sqpoll(SQPOLL_IDLE_MS);
        let shared_ring = shared_builder.build(URING_NUM_ENTRIES)?;
        let shared_fd = shared_ring.as_raw_fd();

        let queue = SegQueue::new();
        for _ in 0..num_rings {
            queue.push(AsyncRing::attached_to(shared_fd).unwrap_or_else(|err| {
                panic!("failed to attach io-uring instance to shared work queue: {err}")
            }));
        }

        Ok(AsyncRingManager {
            shared_ring,
            available_rings: queue,
        })
    }

    fn acquire_ring(&self) -> AsyncRing {
        if let Some(ring) = self.available_rings.pop() {
            return ring;
        }

        AsyncRing::attached_to(self.shared_ring.as_raw_fd()).unwrap_or_else(|err| {
            panic!("failed to attach io-uring instance to shared work queue: {err}")
        })
    }

    fn release_ring(&self, ring: AsyncRing) {
        self.available_rings.push(ring);
    }

    fn lease(&'static self) -> AsyncRingLease<'static> {
        let ring = self.acquire_ring();
        AsyncRingLease {
            manager: self,
            ring: Some(ring),
        }
    }
}

struct AsyncRingLease<'a> {
    manager: &'a AsyncRingManager,
    ring: Option<AsyncRing>,
}

impl<'a> AsyncRingLease<'a> {
    fn as_mut(&mut self) -> &mut AsyncRing {
        self.ring
            .as_mut()
            .expect("async io-uring lease is missing ring instance")
    }

    fn abandon(&mut self) {
        self.ring.take();
    }
}

impl<'a> Drop for AsyncRingLease<'a> {
    fn drop(&mut self) {
        if let Some(ring) = self.ring.take() {
            self.manager.release_ring(ring);
        }
    }
}

static ASYNC_RING_MANAGER: OnceLock<AsyncRingManager> = OnceLock::new();

fn async_ring_manager() -> &'static AsyncRingManager {
    ASYNC_RING_MANAGER.get_or_init(|| {
        let ring_count = thread::available_parallelism()
            .map(|p| p.get())
            .unwrap_or(1);
        AsyncRingManager::new(ring_count)
            .unwrap_or_else(|err| panic!("failed to initialize async io-uring rings: {err}"))
    })
}

enum State<T>
where
    T: IoTask + 'static,
{
    Created(Box<T>),
    Pending {
        ring: AsyncRingLease<'static>,
        task: Box<T>,
    },
    Invalid,
}

pub(crate) struct MultiAsyncUringFuture<T>
where
    T: IoTask + 'static,
{
    state: State<T>,
}

impl<T> MultiAsyncUringFuture<T>
where
    T: IoTask + 'static,
{
    fn new(task: T) -> MultiAsyncUringFuture<T> {
        MultiAsyncUringFuture {
            state: State::Created(Box::new(task)),
        }
    }
}

impl<T> Future for MultiAsyncUringFuture<T>
where
    T: IoTask + 'static,
{
    type Output = Box<T>;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let this = self.get_mut();
        let state = std::mem::replace(&mut this.state, State::Invalid);

        match state {
            State::Created(mut task) => {
                let manager = async_ring_manager();
                let mut ring = manager.lease();
                ring.as_mut().submit_task(task.as_mut());
                this.state = State::Pending { ring, task };
                cx.waker().wake_by_ref();
                Poll::Pending
            }
            State::Pending { mut ring, mut task } => {
                if let Some(cqe) = ring.as_mut().take_completion() {
                    task.complete(vec![&cqe]);
                    return Poll::Ready(task);
                }
                this.state = State::Pending { ring, task };
                cx.waker().wake_by_ref();
                Poll::Pending
            }
            State::Invalid => panic!("poll called on invalid future state"),
        }
    }
}

impl<T> Drop for MultiAsyncUringFuture<T>
where
    T: IoTask + 'static,
{
    fn drop(&mut self) {
        if let State::Pending { ring, .. } = &mut self.state {
            ring.abandon();
        }
    }
}

fn submit_async_task<T>(task: T) -> MultiAsyncUringFuture<T>
where
    T: IoTask + 'static,
{
    MultiAsyncUringFuture::new(task)
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

    let write_task = FileWriteTask::build(data.clone(), file.as_raw_fd(), false);
    submit_async_task(write_task).await.into_result()
}
