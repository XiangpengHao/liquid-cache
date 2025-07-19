use bytes::Bytes;
use io_uring::types::Fixed;
use io_uring::{self, opcode};
use num_traits::AsPrimitive;
use std::alloc::Layout;
use std::cell::RefCell;
use std::os::fd::RawFd;
use std::usize;

pub struct IoUringInstance {
    inner: io_uring::IoUring,
    #[allow(unused)]
    registered_files: Vec<RawFd>,
}

impl IoUringInstance {
    const NUM_ENTRIES: u32 = 512;
    const CHUNK_SIZE: usize = 8192;
    const BUFFER_ALIGNMENT: usize = 4096;

    pub fn new() -> IoUringInstance {
        let mut builder = io_uring::IoUring::builder();
        builder.setup_iopoll();
        builder.setup_sqpoll(50000);
        let ring = builder
            .build(Self::NUM_ENTRIES)
            .expect("Failed to build IoUring instance");
        // let buffers = Self::register_buffers(&ring);
        let registered_files = vec![-1];
        ring.submitter()
            .register_files(registered_files.as_slice())
            .expect("Failed to create file registry");
        IoUringInstance {
            inner: ring,
            // registered_buffers: buffers,
            registered_files: registered_files,
        }
    }

    #[inline]
    fn register_fd(self: &mut Self, fd: RawFd) {
        self.inner
            .submitter()
            .register_files_update(0, &[fd])
            .expect("Failed to register file");
    }

    pub(crate) fn read_blocking(self: &mut Self, fd: RawFd, nbytes: usize) -> Bytes {
        self.register_fd(fd);
        // TODO(): How to handle overflow???
        let num_chunks = (nbytes + Self::CHUNK_SIZE - 1) / Self::CHUNK_SIZE;
        assert!(
            num_chunks < Self::NUM_ENTRIES as usize,
            "More chunks than sqe entries"
        );

        let layout = Layout::from_size_align(nbytes, Self::BUFFER_ALIGNMENT)
            .expect("Failed to create memory layout for disk read result");
        let base_ptr = unsafe { std::alloc::alloc(layout) };

        #[allow(unused_mut)]
        let mut inner_mut = &mut self.inner;
        {
            #[allow(unused_mut)]
            let mut sq = &mut (inner_mut.submission());
            let mut buf_ptr = base_ptr;
            for i in 0..num_chunks {
                let read_op = opcode::Read::new(
                    Fixed(0),
                    buf_ptr,
                    Self::CHUNK_SIZE as _, // Logically, this should be the remaining number of bytes, but that fails...
                );

                let sqe = read_op
                    .offset((i * Self::CHUNK_SIZE).as_())
                    .build()
                    .user_data(i.as_());
                // TODO(): Assess if we need to sync the sq in between?
                unsafe {
                    sq.push(&sqe)
                        .expect("Failed to push to submission queue during read");
                    buf_ptr = buf_ptr.add(Self::CHUNK_SIZE);
                }
            }
            sq.sync();
        }
        // Since we are using kernel thread for polling, this might or might not result in a syscall. It will internally check if the kernel thread needs to be woken up
        inner_mut.submit_and_wait(0 /* polled io */).unwrap();

        #[allow(unused_mut)]
        let mut cq = &mut inner_mut.completion();
        let mut num_completions_recvd = 0;
        loop {
            cq.sync();
            match cq.next() {
                Some(cqe) => {
                    let chunk_idx = cqe.user_data() as usize;
                    let chunk_start = chunk_idx as usize * Self::CHUNK_SIZE;
                    let chunk_end = nbytes.min(chunk_start + Self::CHUNK_SIZE);
                    let errno = -cqe.result();
                    let err = std::io::Error::from_raw_os_error(errno);
                    assert!(
                        cqe.result() == (chunk_end - chunk_start) as i32,
                        "Read cqe result error: {err}"
                    );
                    // let result_slice = &mut result[chunk_start..chunk_end];
                    // let iovec = self.registered_buffers[chunk_idx];
                    // let fixed_buf_slice = unsafe {
                    //     std::slice::from_raw_parts(
                    //         iovec.iov_base as *const u8,
                    //         chunk_end - chunk_start,
                    //     )
                    // };
                    // result_slice.copy_from_slice(fixed_buf_slice);
                    num_completions_recvd += 1;
                    if num_completions_recvd == num_chunks {
                        break;
                    }
                }
                None => {
                    continue;
                }
            }
        }
        // TODO(): Discuss if we should pad the vector to 64 bytes
        Bytes::from(unsafe { Vec::from_raw_parts(base_ptr as *mut u8, nbytes, nbytes) })
    }

    pub(crate) fn write_blocking(self: &mut Self, fd: RawFd, buffer: &Vec<u8>) {
        self.register_fd(fd);
        let nbytes = buffer.len();
        let num_chunks = (nbytes + Self::CHUNK_SIZE - 1) / Self::CHUNK_SIZE;
        assert!(
            num_chunks < Self::NUM_ENTRIES as usize,
            "More chunks than sqe entries"
        );

        let base_ptr = buffer.as_ptr();
        #[allow(unused_mut)]
        let mut inner_mut = &mut self.inner;
        {
            #[allow(unused_mut)]
            let mut sq = &mut (inner_mut.submission());
            let mut buf_ptr = base_ptr;
            for i in 0..num_chunks {
                // Same issue as read above...
                let write_op = opcode::Write::new(
                    Fixed(0),
                    buf_ptr,
                    Self::CHUNK_SIZE as u32,
                );

                let sqe = write_op
                    .offset((i * Self::CHUNK_SIZE).as_())
                    .build()
                    .user_data(i.as_());
                unsafe {
                    sq.push(&sqe)
                        .expect("Failed to push to submission queue during write");
                    buf_ptr = buf_ptr.add(Self::CHUNK_SIZE);
                }
            }
            sq.sync();
        }
        // Since we are using kernel thread for polling, this might or might not result in a syscall. It will internally check if the kernel thread needs to be woken up
        inner_mut.submit_and_wait(0 /* polled io */).unwrap();

        #[allow(unused_mut)]
        let mut cq = &mut inner_mut.completion();
        let mut num_completions_recvd = 0;
        loop {
            cq.sync();
            match cq.next() {
                Some(cqe) => {
                    let errno = -cqe.result();
                    let err = std::io::Error::from_raw_os_error(errno);
                    assert!(
                        cqe.result() == Self::CHUNK_SIZE as i32,
                        "Write cqe result error: {err}"
                    );
                    num_completions_recvd += 1;
                    if num_completions_recvd == num_chunks {
                        break;
                    }
                }
                None => {
                    continue;
                }
            }
        };
    }
}

// TODO(): Share kernel polling thread between user threads
thread_local! {
    pub(crate) static INST: RefCell<IoUringInstance> = RefCell::new(IoUringInstance::new());
}
