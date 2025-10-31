#[cfg(target_os = "linux")]
mod io_uring;

mod io;

pub(crate) use io::ParquetIoContext;
