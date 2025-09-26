use std::cell::RefCell;
use std::collections::{HashMap, VecDeque};
use std::ffi::CString;
use std::fmt;
use std::future::Future;
use std::io;
use std::mem::ManuallyDrop;
use std::ops::{Deref, DerefMut};
use std::os::unix::ffi::OsStrExt;
use std::os::unix::io::RawFd;
use std::path::Path;
use std::pin::Pin;
use std::ptr;
use std::rc::Rc;
use std::sync::atomic::{AtomicU32, AtomicUsize, Ordering};
use std::task::{Context, Poll, RawWaker, RawWakerVTable, Waker};

use io_uring::{IoUring, opcode, types};
use libc;

use crate::sync::Mutex;

pub(crate) const DEFAULT_RING_ENTRIES: u32 = 32;

/// In an ideal world, we'd never need to care about pooling IoUring instances,
/// because OS will automatically do it for us, we just create as needed.
/// However, in practice, creating IoUring consumes user's memory lock,
/// which by default has a very low 8192KB limit on Linux.
///
/// One way to workaround this is to ask all our dependent to set unlimited memory lock,
/// which is not always possible, and ugly design.
///
/// Instead, we pool IoUring instances within the storage, taking care of the work
/// that OS should have done for us.
pub(crate) struct IoUringPool {
    rings: Mutex<VecDeque<IoUring>>,
    entries: u32,
    created: AtomicUsize,
}

impl IoUringPool {
    pub(crate) fn new(entries: u32) -> Self {
        Self {
            rings: Mutex::new(VecDeque::new()),
            entries,
            created: AtomicUsize::new(0),
        }
    }

    fn acquire(&self) -> PooledRing<'_> {
        if let Some(ring) = self.try_take_ring() {
            return PooledRing::new(ring, self);
        }

        let ring = IoUring::new(self.entries).expect("io_uring init failed");
        self.created.fetch_add(1, Ordering::Relaxed);
        PooledRing::new(ring, self)
    }

    fn try_take_ring(&self) -> Option<IoUring> {
        let mut guard = self.rings.lock().unwrap();
        guard.pop_front()
    }

    fn release(&self, ring: IoUring) {
        let mut guard = self.rings.lock().unwrap();
        guard.push_back(ring);
    }

    #[cfg(test)]
    fn created_count(&self) -> usize {
        self.created.load(Ordering::SeqCst)
    }

    #[cfg(test)]
    fn available_for_tests(&self) -> usize {
        let guard = self.rings.lock().unwrap();
        guard.len()
    }

    #[cfg(test)]
    fn reset_for_tests(&self) {
        let mut guard = self.rings.lock().unwrap();
        guard.clear();
        self.created.store(0, Ordering::SeqCst);
    }
}

impl fmt::Debug for IoUringPool {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let available = match self.rings.lock() {
            Ok(rings) => rings.len(),
            Err(_) => 0,
        };
        f.debug_struct("IoUringPool")
            .field("entries", &self.entries)
            .field("created", &self.created.load(Ordering::SeqCst))
            .field("available", &available)
            .finish()
    }
}

struct PooledRing<'pool> {
    ring: ManuallyDrop<IoUring>,
    pool: &'pool IoUringPool,
}

impl<'pool> PooledRing<'pool> {
    fn new(ring: IoUring, pool: &'pool IoUringPool) -> Self {
        Self {
            ring: ManuallyDrop::new(ring),
            pool,
        }
    }
}

impl Deref for PooledRing<'_> {
    type Target = IoUring;

    fn deref(&self) -> &Self::Target {
        &self.ring
    }
}

impl DerefMut for PooledRing<'_> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.ring
    }
}

impl Drop for PooledRing<'_> {
    fn drop(&mut self) {
        // Move the ring out without dropping it so we can return it to the pool.
        let ring_ptr: *mut IoUring = &mut *self.ring;
        let ring = unsafe { ptr::read(ring_ptr) };
        self.pool.release(ring);
    }
}

struct RingState<'pool> {
    ring: PooledRing<'pool>,
    completions: HashMap<UringID, i32>,
}

impl<'pool> RingState<'pool> {
    fn new(pool: &'pool IoUringPool) -> Self {
        Self {
            ring: pool.acquire(),
            completions: HashMap::new(),
        }
    }
}

impl Drop for RingState<'_> {
    fn drop(&mut self) {
        self.completions.clear();
        {
            let mut cq = self.ring.completion();
            for _ in &mut cq {}
        }
    }
}

static NEXT_TASK_ID: AtomicU32 = AtomicU32::new(1);
static NEXT_OP_ID: AtomicU32 = AtomicU32::new(1);

fn submit_if_needed(state: &mut RingState) {
    let _ = state.ring.submit();
}

fn next_task_id() -> TaskId {
    TaskId(NEXT_TASK_ID.fetch_add(1, Ordering::Relaxed))
}

fn next_op_id() -> u32 {
    NEXT_OP_ID.fetch_add(1, Ordering::Relaxed)
}

#[derive(Eq, PartialEq, Hash, Copy, Clone)]
struct TaskId(u32);

impl TaskId {
    fn create_new_uring_id(&self) -> UringID {
        UringID {
            task_id: *self,
            op_id: next_op_id(),
        }
    }
}

#[derive(Eq, PartialEq, Hash, Clone)]
struct UringID {
    task_id: TaskId,
    op_id: u32,
}

impl UringID {
    fn from_uring_user_data(data: u64) -> Self {
        let task_id = (data >> 32) as u32;
        let op_id = data as u32;
        Self {
            task_id: TaskId(task_id),
            op_id,
        }
    }

    fn to_uring_user_data(&self) -> u64 {
        (self.task_id.0 as u64) << 32 | self.op_id as u64
    }
}

pub struct File {
    fd: RawFd,
}

impl Drop for File {
    fn drop(&mut self) {
        unsafe {
            let _ = libc::close(self.fd);
        }
    }
}

impl File {
    pub fn create<P: AsRef<Path>>(path: P) -> CreateFile {
        CreateFile::new(path.as_ref())
    }

    pub fn write_all<'a>(&'a mut self, buf: &'a [u8]) -> WriteAll<'a> {
        WriteAll::new(self, buf)
    }
}

pub struct CreateFile {
    c_path: CString,
    in_flight: Option<UringID>,
}

impl CreateFile {
    fn new(path: &Path) -> Self {
        let bytes = path.as_os_str().as_bytes();
        let c_path = CString::new(bytes.to_vec()).expect("path contains NUL byte");
        Self {
            c_path,
            in_flight: None,
        }
    }
}

impl Future for CreateFile {
    type Output = io::Result<File>;

    fn poll(self: Pin<&mut Self>, ctx: &mut Context<'_>) -> Poll<Self::Output> {
        let me = unsafe { self.get_unchecked_mut() };
        let task = unsafe { TaskContext::from_waker_data(ctx.waker().data()) };

        let res = task.with_ring(|state| {
            match me.in_flight {
                Some(ref uring_id) => {
                    if let Some(r) = state.completions.remove(uring_id) {
                        return Some(r);
                    }
                }
                None => {
                    let new_uring_id = task.id.create_new_uring_id();
                    let flags = libc::O_CREAT | libc::O_WRONLY | libc::O_TRUNC;
                    let mode = 0o644u32;
                    let open_e = opcode::OpenAt::new(types::Fd(libc::AT_FDCWD), me.c_path.as_ptr())
                        .flags(flags)
                        .mode(mode)
                        .build()
                        .user_data(new_uring_id.to_uring_user_data());
                    let mut sq = state.ring.submission();
                    let ok = unsafe { sq.push(&open_e).is_ok() };
                    drop(sq);
                    if ok {
                        me.in_flight = Some(new_uring_id);
                        submit_if_needed(state);
                    }
                }
            }

            None
        });

        if let Some(r) = res {
            if r >= 0 {
                return Poll::Ready(Ok(File { fd: r as RawFd }));
            } else {
                return Poll::Ready(Err(io::Error::from_raw_os_error(-r)));
            }
        }

        Poll::Pending
    }
}

pub struct WriteAll<'a> {
    file_fd: RawFd,
    buf: &'a [u8],
    written: usize,
    in_flight: Option<UringID>,
}

impl<'a> WriteAll<'a> {
    pub fn new(file: &'a mut File, buf: &'a [u8]) -> Self {
        Self {
            file_fd: file.fd,
            buf,
            written: 0,
            in_flight: None,
        }
    }
}

impl Future for WriteAll<'_> {
    type Output = io::Result<()>;

    fn poll(self: Pin<&mut Self>, ctx: &mut Context<'_>) -> Poll<Self::Output> {
        let me = unsafe { self.get_unchecked_mut() };

        let task = unsafe { TaskContext::from_waker_data(ctx.waker().data()) };

        loop {
            if let Some(ref uring_id) = me.in_flight {
                let completed = task.with_ring(|state| state.completions.remove(uring_id));
                if let Some(r) = completed {
                    me.in_flight = None;
                    if r < 0 {
                        return Poll::Ready(Err(io::Error::from_raw_os_error(-r)));
                    }
                    let n = r as usize;
                    if n == 0 {
                        return Poll::Ready(Err(io::ErrorKind::WriteZero.into()));
                    }
                    me.written += n;
                    if me.written >= me.buf.len() {
                        return Poll::Ready(Ok(()));
                    }
                } else {
                    return Poll::Pending;
                }
            }

            let remaining = &me.buf[me.written..];
            if remaining.is_empty() {
                return Poll::Ready(Ok(()));
            }

            let new_uring_id = task.id.create_new_uring_id();
            let ptr = remaining.as_ptr();
            let len = remaining.len();
            let write_e = opcode::Write::new(types::Fd(me.file_fd), ptr, len as _)
                .offset(u64::MAX) // u64::MAX is -1, so it will use the current file position
                .build()
                .user_data(new_uring_id.to_uring_user_data());

            let submitted = task.with_ring(|state| {
                let mut sq = state.ring.submission();
                let ok = unsafe { sq.push(&write_e).is_ok() };
                drop(sq);
                if ok {
                    submit_if_needed(state);
                    Some(())
                } else {
                    None
                }
            });

            if submitted.is_some() {
                me.in_flight = Some(new_uring_id);
                continue;
            } else {
                return Poll::Pending;
            }
        }
    }
}

#[allow(unused)]
pub fn block_on<F: Future>(mut task: F) -> F::Output {
    let pool = IoUringPool::new(DEFAULT_RING_ENTRIES);
    let mut executor = Executor::new(&pool);
    executor.spawn(task);
    let mut outputs = executor.join();
    outputs.pop().expect("no output from task")
}

struct Task<'pool, F: Future> {
    id: TaskId,
    ring: Rc<RefCell<RingState<'pool>>>,
    future: Pin<Box<F>>,
}

impl<'pool, F: Future> Task<'pool, F> {
    fn context(&self) -> TaskContext<'pool> {
        TaskContext {
            id: self.id,
            ring: self.ring.clone(),
        }
    }
}

struct TaskContext<'pool> {
    id: TaskId,
    ring: Rc<RefCell<RingState<'pool>>>,
}

impl<'pool> TaskContext<'pool> {
    unsafe fn from_waker_data<'a>(data: *const ()) -> &'a TaskContext<'pool> {
        unsafe { &*(data as *const TaskContext<'pool>) }
    }

    fn to_waker_data(&self) -> *const () {
        self as *const Self as *const ()
    }

    fn with_ring<T>(&self, f: impl FnOnce(&mut RingState<'pool>) -> T) -> T {
        let mut guard = self.ring.borrow_mut();
        let ring_state = guard.deref_mut();
        f(ring_state)
    }
}

pub struct Executor<'pool, F: Future> {
    ring: Rc<RefCell<RingState<'pool>>>,
    pending_tasks: HashMap<TaskId, Task<'pool, F>>,
    ready_queue: VecDeque<Task<'pool, F>>,
}

impl<'pool, F: Future> Executor<'pool, F> {
    pub fn new(pool: &'pool IoUringPool) -> Self {
        Executor {
            ring: Rc::new(RefCell::new(RingState::new(pool))),
            pending_tasks: HashMap::new(),
            ready_queue: VecDeque::new(),
        }
    }

    pub fn spawn(&mut self, task: F) {
        let ring = Rc::clone(&self.ring);
        self.ready_queue.push_back(Task {
            id: next_task_id(),
            ring,
            future: Box::pin(task),
        });
    }

    pub fn join(&mut self) -> Vec<F::Output> {
        let mut outputs: Vec<F::Output> = Vec::new();

        loop {
            // Step 1: poll ready tasks
            while let Some(mut task) = self.ready_queue.pop_front() {
                let task_id = task.id;
                let context = task.context();
                let waker = WakerWrapper::new(&context);
                let mut context = Context::from_waker(&waker.waker);
                match task.future.as_mut().poll(&mut context) {
                    Poll::Ready(out) => outputs.push(out),
                    Poll::Pending => {
                        self.pending_tasks.insert(task_id, task);
                    }
                }
            }

            // Step 2: poll io_uring completions
            {
                let mut guard = self.ring.borrow_mut();
                let ring_state = guard.deref_mut();
                for cqe in ring_state.ring.completion() {
                    let id = UringID::from_uring_user_data(cqe.user_data());
                    let task = self.pending_tasks.remove(&id.task_id).unwrap();
                    self.ready_queue.push_back(task);
                    ring_state.completions.insert(id, cqe.result());
                }
            }

            if self.ready_queue.is_empty() && self.pending_tasks.is_empty() {
                break;
            }
        }

        outputs
    }
}

#[inline]
fn raw_waker_from_task_id<'pool>(task: &TaskContext<'pool>) -> RawWaker {
    unsafe fn clone(data: *const ()) -> RawWaker {
        RawWaker::new(data, &TASK_WAKER_VTABLE)
    }
    unsafe fn wake(_: *const ()) {}
    unsafe fn wake_by_ref(_: *const ()) {}
    unsafe fn drop(_: *const ()) {}
    static TASK_WAKER_VTABLE: RawWakerVTable = RawWakerVTable::new(clone, wake, wake_by_ref, drop);
    let data_ptr = task.to_waker_data(); // Only safe if RawWaker does not outlive the `TaskContext`
    RawWaker::new(data_ptr, &TASK_WAKER_VTABLE)
}

struct WakerWrapper<'pool, 'a> {
    waker: Waker,
    _context: &'a TaskContext<'pool>, // Ensure `Waker` lifetime bound to context
}

impl<'pool, 'a> WakerWrapper<'pool, 'a> {
    fn new(task: &'a TaskContext<'pool>) -> Self {
        let waker = unsafe { Waker::from_raw(raw_waker_from_task_id(task)) };
        Self {
            waker,
            _context: task,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::io::Read;
    use std::path::PathBuf;

    #[test]
    fn create_and_write_all_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("io_uring_test.bin");
        let data = b"hello-io-uring".to_vec();

        block_on(async {
            let mut f = File::create(&path).await.unwrap();
            f.write_all(&data).await.unwrap();
        });

        let mut read_back = Vec::new();
        fs::File::open(&path)
            .unwrap()
            .read_to_end(&mut read_back)
            .unwrap();
        assert_eq!(read_back, data);
    }

    #[test]
    fn write_empty_ok() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("empty.bin");
        block_on(async {
            let mut f = File::create(&path).await.unwrap();
            f.write_all(&[]).await.unwrap();
        });
        let metadata = fs::metadata(&path).unwrap();
        assert_eq!(metadata.len(), 0);
    }

    #[test]
    fn create_invalid_path_errors() {
        let dir = tempfile::tempdir().unwrap();
        let mut bogus = PathBuf::from(dir.path());
        bogus.push("nonexistent-dir");
        bogus.push("file");

        block_on(async {
            let res = File::create(&bogus).await;
            assert!(res.is_err());
        });
    }

    #[test]
    fn multiple_concurrent_file_writes() {
        let dir = tempfile::tempdir().unwrap();
        let num_tasks = 8usize;

        let mut expected: Vec<(PathBuf, Vec<u8>)> = Vec::new();
        let pool = IoUringPool::new(DEFAULT_RING_ENTRIES);
        let mut executor = Executor::new(&pool);

        for i in 0..num_tasks {
            let path = dir.path().join(format!("f{i}.bin"));
            let mut data = Vec::new();
            data.extend_from_slice(format!("hello-{i}-").as_bytes());
            data.extend(std::iter::repeat_n(b'a' + (i as u8 % 26), 10 + i));

            expected.push((path.clone(), data.clone()));

            executor.spawn(async move {
                let mut f = File::create(&path).await.unwrap();
                f.write_all(&data).await.unwrap();
                i
            });
        }

        let mut outputs = executor.join();
        assert_eq!(outputs.len(), num_tasks);
        outputs.sort_unstable();
        assert_eq!(outputs, (0..num_tasks).collect::<Vec<_>>());

        for (path, expected_data) in expected {
            let actual = fs::read(&path).unwrap();
            assert_eq!(actual, expected_data);
        }
    }

    #[test]
    fn multiple_writes_per_task() {
        let dir = tempfile::tempdir().unwrap();
        let num_tasks = 4usize;

        let mut expected: Vec<(PathBuf, Vec<u8>)> = Vec::new();
        let pool = IoUringPool::new(DEFAULT_RING_ENTRIES);
        let mut executor = Executor::new(&pool);

        for i in 0..num_tasks {
            let path = dir.path().join(format!("mw-{i}.bin"));
            let part1 = vec![b'X' + (i as u8); 3 + i];
            let part2 = vec![b'0' + (i as u8); 7 + 2 * i];

            // With std-like behavior, second write appends after the first.
            let mut all = Vec::new();
            all.extend_from_slice(&part1);
            all.extend_from_slice(&part2);
            expected.push((path.clone(), all));

            executor.spawn(async move {
                let mut f = File::create(&path).await.unwrap();
                f.write_all(&part1).await.unwrap();
                f.write_all(&part2).await.unwrap();
                path
            });
        }

        let paths = executor.join();
        assert_eq!(paths.len(), num_tasks);

        for (path, expected_data) in expected {
            let actual = fs::read(&path).unwrap();
            assert_eq!(actual, expected_data);
        }
    }

    #[test]
    fn executor_reuses_io_uring_instances() {
        let pool = IoUringPool::new(DEFAULT_RING_ENTRIES);
        pool.reset_for_tests();
        assert_eq!(pool.created_count(), 0);
        assert_eq!(pool.available_for_tests(), 0);

        {
            let mut executor = Executor::new(&pool);
            executor.spawn(async {});
            executor.join();
            assert_eq!(pool.created_count(), 1);
            assert_eq!(pool.available_for_tests(), 0);
        }

        assert_eq!(pool.available_for_tests(), 1);
        assert_eq!(pool.created_count(), 1);

        {
            let mut executor = Executor::new(&pool);
            executor.spawn(async {});
            executor.join();
            assert_eq!(pool.created_count(), 1);
            assert_eq!(pool.available_for_tests(), 0);
        }

        assert_eq!(pool.available_for_tests(), 1);
    }
}
