use std::collections::{HashMap, VecDeque};
use std::ffi::CString;
use std::future::Future;
use std::io;
use std::os::unix::ffi::OsStrExt;
use std::os::unix::io::RawFd;
use std::path::Path;
use std::pin::Pin;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};
use std::task::{Context, Poll, RawWaker, RawWakerVTable, Waker};

use io_uring::{IoUring, opcode, types};
use libc;

struct RingState {
    ring: IoUring,
    completions: HashMap<u64, i32>,
}

static RING: OnceLock<Mutex<RingState>> = OnceLock::new();
static NEXT_TOKEN: AtomicU64 = AtomicU64::new(1);

fn with_ring<T>(f: impl FnOnce(&mut RingState) -> T) -> T {
    let mutex = RING.get_or_init(|| {
        let ring = IoUring::new(256).expect("io_uring init failed");
        Mutex::new(RingState {
            ring,
            completions: HashMap::new(),
        })
    });
    let mut guard = mutex.lock().expect("ring mutex poisoned");
    f(&mut guard)
}

fn drain_completions(state: &mut RingState) {
    for cqe in state.ring.completion() {
        let token = cqe.user_data();
        state.completions.insert(token, cqe.result());
    }
}

#[inline]
fn take_completion(state: &mut RingState, token: u64) -> Option<i32> {
    if let Some(r) = state.completions.remove(&token) {
        return Some(r);
    }
    drain_completions(state);
    state.completions.remove(&token)
}

fn submit_if_needed(state: &mut RingState) {
    let _ = state.ring.submit();
}

fn next_token() -> u64 {
    NEXT_TOKEN.fetch_add(1, Ordering::Relaxed)
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

pub struct CreateFile {
    c_path: CString,
    token: u64,
    submitted: bool,
}

impl CreateFile {
    fn new(path: &Path) -> Self {
        let bytes = path.as_os_str().as_bytes();
        let c_path = CString::new(bytes.to_vec()).expect("path contains NUL byte");
        Self {
            c_path,
            token: next_token(),
            submitted: false,
        }
    }
}

impl Future for CreateFile {
    type Output = io::Result<File>;

    fn poll(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<Self::Output> {
        let me = unsafe { self.get_unchecked_mut() };

        let res = with_ring(|state| {
            if let Some(r) = take_completion(state, me.token) {
                return Some(r);
            }
            if !me.submitted {
                let flags = libc::O_CREAT | libc::O_WRONLY | libc::O_TRUNC;
                let mode = 0o644u32;
                let open_e = opcode::OpenAt::new(types::Fd(libc::AT_FDCWD), me.c_path.as_ptr())
                    .flags(flags)
                    .mode(mode)
                    .build()
                    .user_data(me.token);
                let mut sq = state.ring.submission();
                let ok = unsafe { sq.push(&open_e).is_ok() };
                drop(sq);
                if ok {
                    me.submitted = true;
                    submit_if_needed(state);
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
    offset: u64,
    token: u64,
    in_flight: bool,
}

impl<'a> WriteAll<'a> {
    pub fn new(file: &'a mut File, buf: &'a [u8]) -> Self {
        Self {
            file_fd: file.fd,
            buf,
            written: 0,
            offset: 0,
            token: next_token(),
            in_flight: false,
        }
    }
}

impl Future for WriteAll<'_> {
    type Output = io::Result<()>;

    fn poll(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<Self::Output> {
        let me = unsafe { self.get_unchecked_mut() };

        loop {
            if me.in_flight {
                let completed = with_ring(|state| take_completion(state, me.token));
                if let Some(r) = completed {
                    me.in_flight = false;
                    if r < 0 {
                        return Poll::Ready(Err(io::Error::from_raw_os_error(-r)));
                    }
                    let n = r as usize;
                    if n == 0 {
                        return Poll::Ready(Err(io::ErrorKind::WriteZero.into()));
                    }
                    let expected_len = me.buf.len() - me.written;
                    if n != expected_len {
                        return Poll::Ready(Err(io::Error::other("short write to file")));
                    }
                    me.written += n;
                    me.offset += n as u64;
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

            let ptr = remaining.as_ptr();
            let len = remaining.len();
            let write_e = opcode::Write::new(types::Fd(me.file_fd), ptr, len as _)
                .offset(me.offset)
                .build()
                .user_data(me.token);

            let submitted = with_ring(|state| {
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
                me.in_flight = true;
                continue;
            } else {
                return Poll::Pending;
            }
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

#[allow(unused)]
pub fn block_on<F: Future>(mut task: F) -> F::Output {
    let waker = dummy_waker();
    let mut context = Context::from_waker(&waker);

    loop {
        let pinned_task = unsafe { Pin::new_unchecked(&mut task) };
        if let Poll::Ready(out) = pinned_task.poll(&mut context) {
            return out;
        }
    }
}

pub struct Executor<F: Future> {
    task_queue: VecDeque<Pin<Box<F>>>,
}

impl<F: Future> Default for Executor<F> {
    fn default() -> Self {
        Self::new()
    }
}

impl<F: Future> Executor<F> {
    pub fn new() -> Self {
        Executor {
            task_queue: VecDeque::new(),
        }
    }

    pub fn spawn(&mut self, task: F) {
        self.task_queue.push_back(Box::pin(task));
    }

    pub fn join(&mut self) -> Vec<F::Output> {
        let waker = dummy_waker();
        let mut context = Context::from_waker(&waker);
        let mut outputs: Vec<F::Output> = Vec::new();

        while let Some(mut task) = self.task_queue.pop_front() {
            match task.as_mut().poll(&mut context) {
                Poll::Ready(out) => {
                    outputs.push(out);
                }
                Poll::Pending => {
                    self.task_queue.push_back(task);
                }
            }

            if self.task_queue.is_empty() {
                break;
            }
        }

        while !self.task_queue.is_empty() {
            if let Some(mut task) = self.task_queue.pop_front() {
                match task.as_mut().poll(&mut context) {
                    Poll::Ready(out) => outputs.push(out),
                    Poll::Pending => self.task_queue.push_back(task),
                }
            }
        }

        outputs
    }
}

fn dummy_raw_waker() -> RawWaker {
    fn no_op(_: *const ()) {}
    fn clone(_: *const ()) -> RawWaker {
        dummy_raw_waker()
    }
    let vtable = &RawWakerVTable::new(clone, no_op, no_op, no_op);
    RawWaker::new(std::ptr::null(), vtable)
}

fn dummy_waker() -> Waker {
    unsafe { Waker::from_raw(dummy_raw_waker()) }
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
}
