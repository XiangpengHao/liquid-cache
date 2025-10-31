use std::{
    alloc::{self, Layout},
    any::Any,
    ffi::CString,
    fs,
    ops::Range,
    os::{
        fd::{AsRawFd, FromRawFd, RawFd},
        unix::ffi::OsStringExt,
    },
    path::PathBuf,
};

use bytes::Bytes;
use io_uring::{cqueue, opcode, squeue};

pub(crate) const BLOCK_ALIGN: usize = 4096;

/// Represents an IO request to the uring worker thread.
pub(crate) trait IoTask: Send + Any + std::fmt::Debug {
    /// Convert the request to an io-uring submission queue entry.
    fn prepare_sqe(&mut self) -> squeue::Entry;

    /// Record the outcome of the completion queue entry.
    fn complete(&mut self, cqe: &cqueue::Entry);

    /// Convert the boxed task to a boxed `Any` so callers can recover the original type.
    fn into_any(self: Box<Self>) -> Box<dyn Any>;
}

#[derive(Debug)]
pub(crate) struct FileOpenTask {
    path: CString,
    direct_io: bool,
    fd: Option<RawFd>,
    error: Option<std::io::Error>,
}

impl FileOpenTask {
    pub(crate) fn build(path: PathBuf, direct_io: bool) -> Result<FileOpenTask, std::io::Error> {
        let bytes = path.into_os_string().into_vec();
        let path = CString::new(bytes).map_err(|_| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "path contains interior null byte",
            )
        })?;
        Ok(FileOpenTask {
            path,
            direct_io,
            fd: None,
            error: None,
        })
    }

    pub(crate) fn into_result(mut self) -> Result<fs::File, std::io::Error> {
        if let Some(err) = self.error.take() {
            return Err(err);
        }
        let fd = self.fd.take().ok_or_else(|| {
            std::io::Error::other("open operation completed without returning file descriptor")
        })?;
        // SAFETY: `fd` has been received from the kernel for this task and is uniquely owned here.
        let file = unsafe { fs::File::from_raw_fd(fd) };
        Ok(file)
    }
}

impl IoTask for FileOpenTask {
    #[inline]
    fn prepare_sqe(&mut self) -> squeue::Entry {
        let mut flags = libc::O_RDONLY | libc::O_CLOEXEC;
        if self.direct_io {
            flags |= libc::O_DIRECT;
        }
        let open_op = opcode::OpenAt::new(io_uring::types::Fd(libc::AT_FDCWD), self.path.as_ptr())
            .flags(flags);

        open_op.build()
    }

    #[inline]
    fn complete(&mut self, cqe: &cqueue::Entry) {
        let result = cqe.result();
        if result < 0 {
            self.error = Some(std::io::Error::from_raw_os_error(-result));
        } else {
            self.fd = Some(result);
        }
    }

    fn into_any(self: Box<Self>) -> Box<dyn Any> {
        self
    }
}

impl Drop for FileOpenTask {
    fn drop(&mut self) {
        if let Some(fd) = self.fd.take() {
            unsafe {
                libc::close(fd);
            }
        }
    }
}

#[derive(Debug)]
pub(crate) struct FileReadTask {
    base_ptr: *mut u8,
    layout: Layout,
    file: fs::File,
    range: Range<u64>,
    direct_io: bool,
    error: Option<std::io::Error>,
}

impl FileReadTask {
    #[inline]
    fn padding(&self) -> (usize, usize) {
        if self.direct_io {
            let start_padding = self.range.start as usize & (BLOCK_ALIGN - 1);
            let end_mod = self.range.end as usize & (BLOCK_ALIGN - 1);
            let end_padding = if end_mod == 0 {
                0
            } else {
                BLOCK_ALIGN - end_mod
            };
            (start_padding, end_padding)
        } else {
            (0, 0)
        }
    }

    pub(crate) fn build(range: Range<u64>, file: fs::File, direct_io: bool) -> FileReadTask {
        let (start_padding, end_padding) = if direct_io {
            let start_padding = range.start as usize & (BLOCK_ALIGN - 1);
            let end_mod = range.end as usize & (BLOCK_ALIGN - 1);
            let end_padding = if end_mod == 0 {
                0
            } else {
                BLOCK_ALIGN - end_mod
            };
            (start_padding, end_padding)
        } else {
            (0, 0)
        };
        let layout = Layout::from_size_align(
            (range.end - range.start) as usize + start_padding + end_padding,
            BLOCK_ALIGN,
        )
        .expect("Failed to create memory layout for disk read result");
        let base_ptr = unsafe { alloc::alloc(layout) };
        FileReadTask {
            base_ptr,
            layout,
            file,
            range,
            direct_io,
            error: None,
        }
    }

    /// Return a bytes object holding the result of the read operation.
    #[inline]
    pub(crate) fn into_result(self: Box<Self>) -> Result<Bytes, std::io::Error> {
        let mut this = self;
        if let Some(err) = this.error.take() {
            unsafe {
                alloc::dealloc(this.base_ptr, this.layout);
            }
            this.base_ptr = std::ptr::null_mut();
            return Err(err);
        }
        let (start_padding, end_padding) = this.padding();
        let total_bytes =
            (this.range.end - this.range.start) as usize + start_padding + end_padding;
        let base_ptr = std::mem::replace(&mut this.base_ptr, std::ptr::null_mut());
        unsafe {
            let vec = Vec::from_raw_parts(base_ptr, total_bytes, total_bytes);
            // Convert to vec in order to transfer ownership of underlying pointer.
            let owned_slice: Box<[u8]> = vec.into_boxed_slice();
            // The below slice operation removes the padding. This is a no-op in case of buffered IO.
            Ok(Bytes::from(owned_slice).slice(
                start_padding
                    ..(this.range.end as usize - this.range.start as usize + start_padding),
            ))
        }
    }
}

impl IoTask for FileReadTask {
    #[inline]
    fn prepare_sqe(&mut self) -> squeue::Entry {
        let num_bytes = (self.range.end - self.range.start) as usize;
        let (start_padding, end_padding) = self.padding();
        let num_bytes_aligned = num_bytes + start_padding + end_padding;
        let read_op = opcode::Read::new(
            io_uring::types::Fd(self.file.as_raw_fd()),
            self.base_ptr,
            num_bytes_aligned as u32,
        );

        read_op
            .offset(self.range.start - start_padding as u64)
            .build()
    }

    #[inline]
    fn complete(&mut self, cqe: &cqueue::Entry) {
        if cqe.result() < 0 {
            self.error = Some(std::io::Error::from_raw_os_error(-cqe.result()));
        }
    }

    fn into_any(self: Box<Self>) -> Box<dyn Any> {
        self
    }
}

impl Drop for FileReadTask {
    fn drop(&mut self) {
        if !self.base_ptr.is_null() {
            unsafe {
                alloc::dealloc(self.base_ptr, self.layout);
            }
            self.base_ptr = std::ptr::null_mut();
        }
    }
}

unsafe impl Send for FileReadTask {}

#[derive(Debug)]
pub(crate) struct FileWriteTask {
    base_ptr: *const u8,
    num_bytes: usize,
    fd: RawFd,
    error: Option<std::io::Error>,
}

impl FileWriteTask {
    pub(crate) fn build(base_ptr: *const u8, num_bytes: usize, fd: RawFd) -> FileWriteTask {
        FileWriteTask {
            base_ptr,
            num_bytes,
            fd,
            error: None,
        }
    }

    pub(crate) fn into_result(self: Box<Self>) -> Result<(), std::io::Error> {
        let mut this = self;
        if let Some(err) = this.error.take() {
            return Err(err);
        }
        Ok(())
    }
}

impl IoTask for FileWriteTask {
    #[inline]
    fn prepare_sqe(&mut self) -> squeue::Entry {
        let write_op = opcode::Write::new(
            io_uring::types::Fd(self.fd),
            self.base_ptr,
            self.num_bytes as u32,
        );

        write_op.offset(0u64).build()
    }

    #[inline]
    fn complete(&mut self, cqe: &cqueue::Entry) {
        if cqe.result() < 0 {
            self.error = Some(std::io::Error::from_raw_os_error(-cqe.result()));
        }
    }

    fn into_any(self: Box<Self>) -> Box<dyn Any> {
        self
    }
}

impl Drop for FileWriteTask {
    fn drop(&mut self) {
        // Nothing to do; ownership of buffers is external to the task.
    }
}

unsafe impl Send for FileWriteTask {}
