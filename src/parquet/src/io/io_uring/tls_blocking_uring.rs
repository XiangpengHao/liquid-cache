use std::{cell::RefCell, io, ops::Range, path::PathBuf};

use bytes::Bytes;
use io_uring::IoUring;
use liquid_cache_common::IoMode;

use super::{
    tasks::{FileOpenTask, FileReadTask, FileWriteTask, IoTask},
    thread_pool_uring::{URING_NUM_ENTRIES, get_io_mode},
};

thread_local! {
    static BLOCKING_URING_RING: RefCell<Option<BlockingRing>> = const { RefCell::new(None) };
}

struct BlockingRing {
    ring: IoUring,
}

impl BlockingRing {
    fn new() -> io::Result<BlockingRing> {
        let ring = IoUring::builder().build(URING_NUM_ENTRIES)?;
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

pub(crate) fn read(
    path: PathBuf,
    range: Option<Range<u64>>,
    direct_io: bool,
) -> Result<Bytes, std::io::Error> {
    let mut flags = libc::O_RDONLY | libc::O_CLOEXEC;
    if direct_io {
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

    let read_task = FileReadTask::build(effective_range, file, direct_io);
    run_blocking_task(Box::new(read_task))?.into_result()
}

pub(crate) fn write(path: PathBuf, data: &Bytes) -> Result<(), std::io::Error> {
    use std::os::fd::AsRawFd;
    use std::{fs::OpenOptions, os::unix::fs::OpenOptionsExt as _};

    let direct = matches!(get_io_mode(), IoMode::UringDirect);
    let flags = if direct { libc::O_DIRECT } else { 0 };

    let file = OpenOptions::new()
        .create(true)
        .write(true)
        .custom_flags(flags)
        .open(path)?;

    let write_task = FileWriteTask::build(data.as_ptr(), data.len(), file.as_raw_fd(), direct);
    run_blocking_task(Box::new(write_task))?.into_result()
}
