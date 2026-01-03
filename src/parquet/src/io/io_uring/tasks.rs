use std::{
    any::Any,
    ffi::CString,
    fs, mem,
    ops::Range,
    os::{
        fd::{AsRawFd, FromRawFd, RawFd},
        unix::ffi::OsStringExt,
    },
    path::PathBuf,
};

use bytes::Bytes;
use io_uring::{cqueue, opcode, squeue};
use liquid_cache_common::memory::pool::{FixedBufferAllocation, FixedBufferPool};

pub(crate) const BLOCK_ALIGN: usize = 4096;

/// Represents an IO request to the uring worker thread.
pub(crate) trait IoTask: Send + Any + std::fmt::Debug {
    /// Convert the request to an io-uring submission queue entry.
    fn prepare_sqe(&mut self) -> Vec<squeue::Entry>;

    // TODO(): Can we pass completion queue entries on the stack?
    /// Record the outcome of the completion queue entry.
    fn complete(&mut self, cqe: Vec<&cqueue::Entry>);

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
    fn prepare_sqe(&mut self) -> Vec<squeue::Entry> {
        let mut flags = libc::O_RDONLY | libc::O_CLOEXEC;
        if self.direct_io {
            flags |= libc::O_DIRECT;
        }
        let open_op = opcode::OpenAt::new(io_uring::types::Fd(libc::AT_FDCWD), self.path.as_ptr())
            .flags(flags);

        vec![open_op.build()]
    }

    #[inline]
    fn complete(&mut self, cqe: Vec<&cqueue::Entry>) {
        debug_assert_eq!(cqe.len(), 1, "Should receive a single completion for a file open task");
        let result = cqe[0].result();
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
    buffer: Vec<u8>,
    aligned_offset: usize,
    file: fs::File,
    range: Range<u64>,
    direct_io: bool,
    error: Option<std::io::Error>,
}

impl FileReadTask {
    #[inline]
    fn compute_padding(range: &Range<u64>, direct_io: bool) -> (usize, usize) {
        if direct_io {
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
        }
    }

    #[inline]
    fn padding(&self) -> (usize, usize) {
        Self::compute_padding(&self.range, self.direct_io)
    }

    pub(crate) fn build(range: Range<u64>, file: fs::File, direct_io: bool) -> FileReadTask {
        let (start_padding, end_padding) = Self::compute_padding(&range, direct_io);
        let requested_bytes = (range.end - range.start) as usize;
        let num_bytes_aligned = requested_bytes + start_padding + end_padding;

        let (buffer, aligned_offset) = if direct_io {
            let buffer = vec![0u8; num_bytes_aligned + BLOCK_ALIGN];
            let base_addr = buffer.as_ptr() as usize;
            let aligned_addr = (base_addr + (BLOCK_ALIGN - 1)) & !(BLOCK_ALIGN - 1);
            let offset = aligned_addr - base_addr;
            debug_assert!(offset < BLOCK_ALIGN);
            debug_assert!(offset + num_bytes_aligned <= buffer.len());
            (buffer, offset)
        } else {
            (vec![0u8; num_bytes_aligned], 0)
        };

        FileReadTask {
            buffer,
            aligned_offset,
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
            return Err(err);
        }

        let (start_padding, _) = this.padding();
        let range_len = (this.range.end - this.range.start) as usize;
        let data_start = this.aligned_offset + start_padding;
        let data_end = data_start + range_len;

        let buffer = mem::take(&mut this.buffer);
        let bytes = Bytes::from(buffer);

        Ok(bytes.slice(data_start..data_end))
    }
}

impl IoTask for FileReadTask {
    #[inline]
    fn prepare_sqe(&mut self) -> Vec<squeue::Entry> {
        let num_bytes = (self.range.end - self.range.start) as usize;
        let (start_padding, end_padding) = self.padding();
        let num_bytes_aligned = num_bytes + start_padding + end_padding;
        let buffer_start = self.aligned_offset;
        let buffer_end = buffer_start + num_bytes_aligned;
        let slice = &mut self.buffer[buffer_start..buffer_end];

        let read_op = opcode::Read::new(
            io_uring::types::Fd(self.file.as_raw_fd()),
            slice.as_mut_ptr(),
            num_bytes_aligned as u32,
        );

        vec![read_op
            .offset(self.range.start - start_padding as u64)
            .build()]
    }

    #[inline]
    fn complete(&mut self, cqe: Vec<&cqueue::Entry>) {
        debug_assert_eq!(cqe.len(), 1, "Should receive a single completion for a FileRead task");
        let result = cqe[0].result();
        if result < 0 {
            self.error = Some(std::io::Error::from_raw_os_error(-result));
        }
    }

    fn into_any(self: Box<Self>) -> Box<dyn Any> {
        self
    }
}

#[derive(Debug)]
pub(crate) struct FixedFileReadTask {
    fixed_buffer: FixedBufferAllocation,
    file: RawFd,
    range: Range<u64>,
    direct_io: bool,
    error: Option<std::io::Error>,
}

impl FixedFileReadTask {
    #[inline]
    fn compute_padding(range: &Range<u64>, direct_io: bool) -> (usize, usize) {
        if direct_io {
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
        }
    }

    #[inline]
    fn padding(&self) -> (usize, usize) {
        Self::compute_padding(&self.range, self.direct_io)
    }

    pub(crate) fn build(range: Range<u64>, file: &fs::File, direct_io: bool) -> Result<FixedFileReadTask, std::io::Error> {
        let (start_padding, end_padding) = Self::compute_padding(&range, direct_io);
        let requested_bytes = (range.end - range.start) as usize;
        let num_bytes_aligned = requested_bytes + start_padding + end_padding;

        // Fixed buffers are aligned to the block size. Don't worry about alignment here
        let ptr = FixedBufferPool::malloc(num_bytes_aligned);
        if ptr.is_null() {
            return Err(std::io::Error::from(std::io::ErrorKind::OutOfMemory));
        }
        let alloc = FixedBufferAllocation {ptr, size: num_bytes_aligned};

        Ok(FixedFileReadTask {
            fixed_buffer: alloc,
            file: file.as_raw_fd(),
            range,
            direct_io,
            error: None,
        })
    }

    /// Return a bytes object holding the result of the read operation.
    #[inline]
    pub(crate) fn into_result(self: Box<Self>) -> Result<Bytes, std::io::Error> {
        let mut this = self;
        if let Some(err) = this.error.take() {
            return Err(err);
        }

        let (start_padding, _) = this.padding();
        let range_len = (this.range.end - this.range.start) as usize;
        let data_end = start_padding + range_len;
        let bytes = Bytes::from_owner(this.fixed_buffer);

        Ok(bytes.slice(start_padding..data_end))
    }
}

impl IoTask for FixedFileReadTask {
    #[inline]
    fn prepare_sqe(&mut self) -> Vec<squeue::Entry> {        
        let buffers = FixedBufferPool::get_fixed_buffers(&self.fixed_buffer);
        let mut sqes = Vec::<squeue::Entry>::new();
        let (start_padding, _) = self.padding();
        let mut file_offset = self.range.start - start_padding as u64;
        for buffer in buffers {
            let sqe = opcode::ReadFixed::new(
                io_uring::types::Fd(self.file),
                buffer.ptr,
                buffer.bytes as u32,
                buffer.buf_id as u16)
                .offset(file_offset).build();
                file_offset += buffer.bytes as u64;
            sqes.push(sqe);
        }
        sqes
    }

    #[inline]
    fn complete(&mut self, cqes: Vec<&cqueue::Entry>) {
        for cqe in cqes.iter().as_ref() {
            if cqe.result() < 0 {
                self.error = Some(std::io::Error::from_raw_os_error(-cqe.result()));
            }
        }
    }

    fn into_any(self: Box<Self>) -> Box<dyn Any> {
        self
    }
}

#[derive(Debug)]
pub(crate) struct FileWriteTask {
    data: Bytes,
    fd: RawFd,
    error: Option<std::io::Error>,
}

impl FileWriteTask {
    pub(crate) fn build(data: Bytes, fd: RawFd, use_fixed_buffers: bool) -> FileWriteTask {
        FileWriteTask {
            data,
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
    fn prepare_sqe(&mut self) -> Vec<squeue::Entry> {
        let write_op = opcode::Write::new(
            io_uring::types::Fd(self.fd),
            self.data.as_ptr(),
            self.data.len() as u32,
        );

       vec![write_op.offset(0u64).build()]
    }

    #[inline]
    fn complete(&mut self, cqes: Vec<&cqueue::Entry>) {
        debug_assert_eq!(cqes.len(), 1, "Should receive a single completion for a FileRead task");
        let result = cqes[0].result();
        if result < 0 {
            self.error = Some(std::io::Error::from_raw_os_error(-result));
        }
    }

    fn into_any(self: Box<Self>) -> Box<dyn Any> {
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::ErrorKind;
    use tempfile::NamedTempFile;

    fn temp_file() -> fs::File {
        NamedTempFile::new()
            .expect("failed to create temp file")
            .reopen()
            .expect("failed to reopen temp file")
    }

    #[test]
    fn build_direct_io_aligns_buffer() {
        let range = 123u64..5321u64;
        let file = temp_file();

        let task = FileReadTask::build(range.clone(), file, true);
        let (start_padding, end_padding) = FileReadTask::compute_padding(&range, true);
        let requested = (range.end - range.start) as usize;
        let aligned_len = requested + start_padding + end_padding;

        assert!(task.aligned_offset < task.buffer.len());
        assert!(task.aligned_offset + aligned_len <= task.buffer.len());

        let aligned_ptr = unsafe { task.buffer.as_ptr().add(task.aligned_offset) } as usize;
        assert_eq!(aligned_ptr % BLOCK_ALIGN, 0);
    }

    #[test]
    fn build_buffered_read_has_no_padding() {
        let range = 10u64..1024u64;
        let file = temp_file();

        let task = FileReadTask::build(range.clone(), file, false);
        assert_eq!(task.aligned_offset, 0);
        assert_eq!(task.buffer.len(), (range.end - range.start) as usize);
    }

    #[test]
    fn into_result_trims_padding() {
        let range = 377u64..4999u64;
        let file = temp_file();
        let mut task = FileReadTask::build(range.clone(), file, true);

        let (start_padding, end_padding) = FileReadTask::compute_padding(&range, true);
        let requested = (range.end - range.start) as usize;
        let buffer_start = task.aligned_offset;
        let buffer_end = buffer_start + start_padding + requested + end_padding;

        let mut expected = Vec::with_capacity(requested);
        expected.extend((0..requested).map(|idx| (idx % 251) as u8));

        {
            let slice = &mut task.buffer[buffer_start..buffer_end];
            for byte in &mut slice[..start_padding] {
                *byte = 0xAA;
            }
            for (dst, value) in slice[start_padding..start_padding + requested]
                .iter_mut()
                .zip(expected.iter())
            {
                *dst = *value;
            }
            for byte in &mut slice[start_padding + requested..] {
                *byte = 0xBB;
            }
        }

        let bytes = FileReadTask::into_result(Box::new(task))
            .expect("expected successful conversion to Bytes");
        assert_eq!(bytes.len(), requested);
        assert_eq!(bytes.to_vec(), expected);
    }

    #[test]
    fn into_result_propagates_error() {
        let range = 0u64..128u64;
        let file = temp_file();
        let mut task = FileReadTask::build(range, file, false);
        task.error = Some(std::io::Error::from(ErrorKind::Other));

        let err = FileReadTask::into_result(Box::new(task)).expect_err("expected error");
        assert_eq!(err.kind(), ErrorKind::Other);
    }
}
