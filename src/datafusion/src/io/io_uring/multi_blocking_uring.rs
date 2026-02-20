use std::{io, ops::Range, os::fd::AsRawFd, path::PathBuf, sync::OnceLock};

use bytes::Bytes;
use crossbeam_queue::SegQueue;
use io_uring::IoUring;

use super::{
    tasks::{FileOpenTask, FileReadTask, FileWriteTask, IoTask},
    thread_pool_uring::URING_NUM_ENTRIES,
};

const SQPOLL_IDLE_MS: u32 = 10;

struct BlockingRing {
    ring: IoUring,
}

impl BlockingRing {
    fn attached_to(shared_fd: libc::c_int) -> io::Result<BlockingRing> {
        let mut builder = IoUring::builder();
        builder.setup_attach_wq(shared_fd);
        let ring = builder.build(URING_NUM_ENTRIES)?;
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

struct BlockingRingManager {
    shared_ring: IoUring,
    available_rings: SegQueue<BlockingRing>,
}

impl BlockingRingManager {
    fn new(num_rings: usize) -> io::Result<BlockingRingManager> {
        let mut shared_builder = IoUring::builder();
        shared_builder.setup_sqpoll(SQPOLL_IDLE_MS);
        let shared_ring = shared_builder.build(URING_NUM_ENTRIES)?;
        let shared_fd = shared_ring.as_raw_fd();

        let queue = SegQueue::new();
        for _ in 0..num_rings {
            queue.push(BlockingRing::attached_to(shared_fd)?);
        }

        Ok(BlockingRingManager {
            shared_ring,
            available_rings: queue,
        })
    }

    fn acquire_ring(&self) -> io::Result<BlockingRing> {
        if let Some(ring) = self.available_rings.pop() {
            return Ok(ring);
        }

        BlockingRing::attached_to(self.shared_ring.as_raw_fd())
    }

    fn release_ring(&self, ring: BlockingRing) {
        self.available_rings.push(ring);
    }

    fn lease(&self) -> io::Result<BlockingRingLease<'_>> {
        let ring = self.acquire_ring()?;
        Ok(BlockingRingLease {
            manager: self,
            ring: Some(ring),
        })
    }
}

struct BlockingRingLease<'a> {
    manager: &'a BlockingRingManager,
    ring: Option<BlockingRing>,
}

impl<'a> BlockingRingLease<'a> {
    fn as_mut(&mut self) -> &mut BlockingRing {
        self.ring
            .as_mut()
            .expect("blocking ring lease missing ring instance")
    }
}

impl<'a> Drop for BlockingRingLease<'a> {
    fn drop(&mut self) {
        if let Some(ring) = self.ring.take() {
            self.manager.release_ring(ring);
        }
    }
}

static BLOCKING_RING_MANAGER: OnceLock<BlockingRingManager> = OnceLock::new();

pub(crate) fn initialize_blocking_rings() {
    BLOCKING_RING_MANAGER.get_or_init(|| {
        let ring_count = std::thread::available_parallelism()
            .map(|p| p.get())
            .unwrap_or(1);
        BlockingRingManager::new(ring_count)
            .unwrap_or_else(|err| panic!("failed to initialize blocking io-uring rings: {err}"))
    });
}

fn blocking_ring_manager() -> io::Result<&'static BlockingRingManager> {
    BLOCKING_RING_MANAGER
        .get()
        .ok_or_else(|| io::Error::other("blocking io-uring rings have not been initialized"))
}

fn with_blocking_ring<F, R>(f: F) -> io::Result<R>
where
    F: FnOnce(&mut BlockingRing) -> io::Result<R>,
{
    let manager = blocking_ring_manager()?;
    let mut lease = manager.lease()?;
    f(lease.as_mut())
}

fn run_blocking_task<T>(task: Box<T>) -> io::Result<Box<T>>
where
    T: IoTask + 'static,
{
    with_blocking_ring(move |ring| ring.run_task(task))
}

pub(crate) fn read(
    path: PathBuf,
    range: Option<Range<u64>>,
    direct_io: bool,
) -> Result<Bytes, std::io::Error> {
    let open_task = FileOpenTask::build(path, direct_io)?;
    let file = run_blocking_task(Box::new(open_task))?.into_result()?;
    let effective_range = if let Some(range) = range {
        range
    } else {
        let len = file.metadata()?.len();
        0..len
    };

    let read_task = FileReadTask::build(effective_range, file, direct_io);
    run_blocking_task(Box::new(read_task))?.into_result()
}

pub(crate) fn write(path: PathBuf, data: &Bytes) -> Result<(), std::io::Error> {
    use std::fs::OpenOptions;

    let file = OpenOptions::new()
        .create(true)
        .truncate(true)
        .write(true)
        .open(path)?;
    let write_task = FileWriteTask::build(data.clone(), file.as_raw_fd());
    run_blocking_task(Box::new(write_task))?.into_result()
}
