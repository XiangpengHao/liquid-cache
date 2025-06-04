use bytes::Bytes;
use io_uring::types::Fixed;
/**
 * Blocking io uring:
 * - Add a thread local io_uring instance
 * - Pre-register buffers for submission (these should also be thread-local)
 * - Submit the reads/writes
 * - Poll the completion queue
 */
use io_uring::{self, opcode};
use libc::{c_void, iovec};
use num_traits::AsPrimitive;
use std::alloc::Layout;
use std::cell::RefCell;
use std::os::fd::RawFd;

pub struct IoUringInstance {
    inner: io_uring::IoUring,
    registered_buffers: Vec<iovec>,
    #[allow(unused)]
    registered_files: Vec<RawFd>,
}

impl IoUringInstance {
    const NUM_ENTRIES: u32 = 256;
    const CHUNK_SIZE: usize = 4096;
    const BUFFER_ALIGNMENT: usize = 512;

    pub fn new() -> IoUringInstance {
        let mut builder = io_uring::IoUring::builder();
        builder.setup_iopoll();
        builder.setup_sqpoll(50000);
        let ring = builder
            .build(Self::NUM_ENTRIES)
            .expect("Failed to build IoUring instance");
        let buffers = Self::register_buffers(&ring);
        let registered_files = Vec::<RawFd>::new();
        ring.submitter()
            .register_files(registered_files.as_slice())
            .expect("Failed to create file registry");
        IoUringInstance {
            inner: ring,
            registered_buffers: buffers,
            registered_files: registered_files,
        }
    }

    fn register_buffers(ring: &io_uring::IoUring) -> Vec<iovec> {
        let layout = Layout::from_size_align(
            (Self::NUM_ENTRIES as usize) * Self::CHUNK_SIZE,
            Self::BUFFER_ALIGNMENT,
        )
        .expect("Failed to create memory layout for pre-registered buffers");
        let mut vectors = Vec::<iovec>::new();
        unsafe {
            let ptr = std::alloc::alloc(layout);
            for pos in 0..Self::NUM_ENTRIES {
                let iov = iovec {
                    iov_base: (ptr.wrapping_add((pos as usize) * Self::CHUNK_SIZE)) as *mut c_void,
                    iov_len: Self::CHUNK_SIZE,
                };
                vectors.push(iov);
            }
            ring.submitter()
                .register_buffers(vectors.as_slice())
                .expect("Failed to pre-register buffers");
        }
        vectors
    }

    #[inline]
    pub(crate) fn register_fd(self: &Self, fd: RawFd) {
        self.inner
            .submitter()
            .register_files_update(0, &[fd])
            .expect("Failed to register file");
    }

    pub(crate) fn read_blocking(self: &mut Self, fd: RawFd, nbytes: usize) -> Bytes {
        // TODO(): How to handle overflow???
        let num_chunks = (nbytes + Self::CHUNK_SIZE - 1) / Self::CHUNK_SIZE;
        #[allow(unused_mut)]
        let mut inner_mut = &mut self.inner;
        {
            #[allow(unused_mut)]
            let mut sq = &mut (inner_mut.submission());
            for i in 0..num_chunks {
                let read_op = opcode::ReadFixed::new(
                    Fixed(fd.as_()),
                    self.registered_buffers.as_mut_ptr() as *mut u8,
                    Self::CHUNK_SIZE.as_(),
                    i.as_(),
                );
                let sqe = read_op
                    .offset((i * Self::CHUNK_SIZE).as_())
                    .build()
                    .user_data(i.as_());
                /*
                 * TODO(): Assess the following:
                 * - Do we need to set IOSQE_FIXED_FILE?
                 * - Do we need to sync the sq in between?
                 */
                unsafe {
                    sq.push(&sqe).expect("Failed to push to submission queue during read");
                }
            }
            sq.sync();
        }
        // Since we are using kernel thread for polling, this might or might not result in a syscall. It will internally check if the kernel thread needs to be woken up
        inner_mut.submit_and_wait(0 /* polled io */).unwrap();
        let layout = Layout::from_size_align(nbytes, 64)
            .expect("Failed to create memory layout for disk read result");
        let ptr = unsafe { std::alloc::alloc(layout) };
        // TODO(): Ask if we should pad the vector to 64 bytes
        #[allow(unused_mut)]
        let mut result = unsafe { Vec::from_raw_parts(ptr as *mut u8, nbytes, nbytes) };

        #[allow(unused_mut)]
        let mut cq = &mut inner_mut.completion();
        let mut num_completions_recvd = 0;
        loop {
            cq.sync();
            match cq.next() {
                Some(cqe) => {
                    // TODO(): Process cqe.result
                    let chunk_idx = cqe.user_data() as usize;
                    let chunk_start = chunk_idx as usize * Self::CHUNK_SIZE;
                    let chunk_end = nbytes.min((chunk_idx as usize + 1) * Self::CHUNK_SIZE);
                    let result_slice = &mut result[chunk_start..chunk_end];
                    let iovec = self.registered_buffers[chunk_idx];
                    let fixed_buf_slice =
                        unsafe { std::slice::from_raw_parts(iovec.iov_base as * const u8, iovec.iov_len) };
                    result_slice.copy_from_slice(fixed_buf_slice);
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
        Bytes::from(result)
    }
}

// TODO(): Share kernel polling thread between user threads
thread_local! {
    pub(crate) static INST: RefCell<IoUringInstance> = RefCell::new(IoUringInstance::new());
}
