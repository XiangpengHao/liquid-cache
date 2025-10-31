use bytes::Bytes;
use liquid_cache_common::IoMode;
use std::path::PathBuf;
use std::{
    io::{Read, Seek, Write},
    ops::Range,
};
use tokio::io::{AsyncReadExt, AsyncSeekExt, AsyncWriteExt};

pub(super) async fn read(
    io_mode: IoMode,
    path: PathBuf,
    range: Option<Range<u64>>,
) -> Result<Bytes, std::io::Error> {
    match io_mode {
        IoMode::Uring | IoMode::UringDirect => {
            return read_uring(path, range).await;
        }
        IoMode::UringBlocking => {
            return read_blocking_uring(path, range).await;
        }
        IoMode::StdSpawnBlocking => {
            return read_spawn_blocking(path, range).await;
        }
        IoMode::StdBlocking => {
            return read_blocking(path, range);
        }
        IoMode::TokioIO => {
            return read_tokio(path, range).await;
        }
    }
}

pub(super) async fn write_file(
    io_mode: IoMode,
    path: PathBuf,
    data: Bytes,
) -> Result<(), std::io::Error> {
    match io_mode {
        IoMode::Uring | IoMode::UringDirect => write_file_uring(path, data).await,
        IoMode::UringBlocking => write_file_blocking_uring(path, data).await,
        IoMode::StdSpawnBlocking => write_file_spawn_blocking(path, data).await,
        IoMode::StdBlocking => write_file_blocking(path, data),
        IoMode::TokioIO => write_file_tokio(path, data).await,
    }
}

fn read_blocking_impl(path: PathBuf, range: Option<Range<u64>>) -> Result<Bytes, std::io::Error> {
    match range {
        Some(range) => {
            let mut file = std::fs::File::open(path)?;
            let len = (range.end - range.start) as usize;
            let mut bytes = vec![0u8; len];
            file.seek(std::io::SeekFrom::Start(range.start))?;
            file.read_exact(&mut bytes)?;
            Ok(Bytes::from(bytes))
        }
        None => {
            let bytes = std::fs::read(path)?;
            Ok(Bytes::from(bytes))
        }
    }
}

fn write_file_blocking_impl(path: PathBuf, data: Bytes) -> Result<(), std::io::Error> {
    let mut file = std::fs::File::create(path)?;
    file.write_all(data.as_ref())?;
    Ok(())
}

fn read_blocking(path: PathBuf, range: Option<Range<u64>>) -> Result<Bytes, std::io::Error> {
    read_blocking_impl(path, range)
}

fn write_file_blocking(path: PathBuf, data: Bytes) -> Result<(), std::io::Error> {
    write_file_blocking_impl(path, data)
}

async fn read_spawn_blocking(
    path: PathBuf,
    range: Option<Range<u64>>,
) -> Result<Bytes, std::io::Error> {
    maybe_spawn_blocking(move || read_blocking_impl(path, range)).await
}

async fn write_file_spawn_blocking(path: PathBuf, data: Bytes) -> Result<(), std::io::Error> {
    maybe_spawn_blocking(move || write_file_blocking_impl(path, data)).await
}

async fn read_tokio(path: PathBuf, range: Option<Range<u64>>) -> Result<Bytes, std::io::Error> {
    let mut file = tokio::fs::File::open(path).await?;
    match range {
        Some(range) => {
            let len = (range.end - range.start) as usize;
            let mut bytes = vec![0u8; len];
            file.seek(tokio::io::SeekFrom::Start(range.start)).await?;
            file.read_exact(&mut bytes).await?;
            Ok(Bytes::from(bytes))
        }
        None => {
            let mut bytes = Vec::new();
            file.read_to_end(&mut bytes).await?;
            Ok(Bytes::from(bytes))
        }
    }
}

async fn write_file_tokio(path: PathBuf, data: Bytes) -> Result<(), std::io::Error> {
    let mut file = tokio::fs::File::create(path).await?;
    file.write_all(data.as_ref()).await?;
    Ok(())
}

#[cfg(target_os = "linux")]
async fn read_uring(path: PathBuf, range: Option<Range<u64>>) -> Result<Bytes, std::io::Error> {
    crate::io::io_uring::thread_pool_uring::read_range(path, range).await
}

#[cfg(not(target_os = "linux"))]
async fn read_uring(_path: PathBuf, _range: Option<Range<u64>>) -> Result<Bytes, std::io::Error> {
    panic!("io_uring modes are only supported on Linux");
}

#[cfg(target_os = "linux")]
async fn read_blocking_uring(
    path: PathBuf,
    range: Option<Range<u64>>,
) -> Result<Bytes, std::io::Error> {
    crate::io::io_uring::tls_blocking_uring::read_range(path, range)
}

#[cfg(not(target_os = "linux"))]
async fn read_blocking_uring(
    _path: PathBuf,
    _range: Option<Range<u64>>,
) -> Result<Bytes, std::io::Error> {
    panic!("io_uring modes are only supported on Linux");
}

#[cfg(target_os = "linux")]
async fn write_file_uring(path: PathBuf, data: Bytes) -> Result<(), std::io::Error> {
    crate::io::io_uring::thread_pool_uring::write(path, &data).await
}

#[cfg(not(target_os = "linux"))]
async fn write_file_uring(_path: PathBuf, _data: Bytes) -> Result<(), std::io::Error> {
    panic!("io_uring modes are only supported on Linux");
}

#[cfg(target_os = "linux")]
async fn write_file_blocking_uring(path: PathBuf, data: Bytes) -> Result<(), std::io::Error> {
    crate::io::io_uring::tls_blocking_uring::write(path, &data)
}

#[cfg(not(target_os = "linux"))]
async fn write_file_blocking_uring(_path: PathBuf, _data: Bytes) -> Result<(), std::io::Error> {
    panic!("io_uring modes are only supported on Linux");
}

async fn maybe_spawn_blocking<F, T>(f: F) -> Result<T, std::io::Error>
where
    F: FnOnce() -> Result<T, std::io::Error> + Send + 'static,
    T: Send + 'static,
{
    match tokio::runtime::Handle::try_current() {
        Ok(runtime) => match runtime.spawn_blocking(f).await {
            Ok(result) => result,
            Err(err) => Err(std::io::Error::other(err)),
        },
        Err(_) => f(),
    }
}
