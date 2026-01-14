#![cfg(target_os = "linux")]

use crate::io::io_uring::runtime::{self, UringExecutor};

use super::{
    initialize_uring_pool, multi_async_uring, multi_blocking_uring, single_uring, thread_pool_uring,
};
use bytes::Bytes;
use futures::{FutureExt, future::BoxFuture};
use liquid_cache_common::IoMode;
use std::{
    fs::{self, File},
    io::Write,
    ops::Range,
    path::{Path, PathBuf},
    sync::Once,
};
use tempfile::tempdir;

type IoResult<T> = Result<T, std::io::Error>;
type IoFuture<T> = BoxFuture<'static, IoResult<T>>;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum BackendKind {
    Shared,
    MultiAsync,
    MultiBlocking,
    ThreadPool,
}

impl BackendKind {
    const ALL: &'static [BackendKind] = &[
        BackendKind::Shared,
        BackendKind::MultiAsync,
        BackendKind::MultiBlocking,
        BackendKind::ThreadPool,
    ];

    fn name(self) -> &'static str {
        match self {
            BackendKind::Shared => "shared",
            BackendKind::MultiAsync => "multi_async",
            BackendKind::MultiBlocking => "multi_blocking",
            BackendKind::ThreadPool => "thread_pool",
        }
    }

    fn prepare(self, direct_io: bool) {
        static THREAD_POOL_INIT: Once = Once::new();
        static BLOCKING_INIT: Once = Once::new();

        match self {
            BackendKind::ThreadPool => {
                THREAD_POOL_INIT.call_once(|| {
                    let mode = if direct_io {
                        IoMode::UringDirect
                    } else {
                        IoMode::Uring
                    };
                    initialize_uring_pool(mode);
                });
            }
            BackendKind::MultiBlocking => {
                BLOCKING_INIT.call_once(|| {
                    multi_blocking_uring::initialize_blocking_rings();
                });
            }
            BackendKind::Shared | BackendKind::MultiAsync => {}
        }
    }

    fn read_future(
        self,
        path: PathBuf,
        range: Option<Range<u64>>,
        direct_io: bool,
    ) -> IoFuture<Bytes> {
        match self {
            BackendKind::Shared => single_uring::read(path, range, direct_io).boxed(),
            BackendKind::MultiAsync => multi_async_uring::read(path, range, direct_io).boxed(),
            BackendKind::MultiBlocking => {
                async move { multi_blocking_uring::read(path, range, direct_io) }.boxed()
            }
            BackendKind::ThreadPool => thread_pool_uring::read(path, range, direct_io).boxed(),
        }
    }

    fn write_future(self, path: PathBuf, data: Bytes) -> IoFuture<()> {
        match self {
            BackendKind::Shared => async move { single_uring::write(path, &data).await }.boxed(),
            BackendKind::MultiAsync => {
                async move { multi_async_uring::write(path, &data).await }.boxed()
            }
            BackendKind::MultiBlocking => {
                async move { multi_blocking_uring::write(path, &data) }.boxed()
            }
            BackendKind::ThreadPool => {
                async move { thread_pool_uring::write(path, &data).await }.boxed()
            }
        }
    }
}

fn block_on_io<T>(future: impl futures::Future<Output = T>) -> T {
    futures::executor::block_on(future)
}

fn write_file(path: &Path, data: &[u8]) {
    let mut file = File::create(path).expect("failed to create temp file");
    file.write_all(data).expect("failed to write data");
    file.sync_all().expect("failed to flush file");
}

fn seed_file(data: &[u8]) -> (tempfile::TempDir, PathBuf) {
    let tmp = tempdir().expect("failed to create tempdir");
    let path = tmp.path().join("payload.bin");
    write_file(&path, data);
    (tmp, path)
}

#[test]
fn read_write_roundtrip_all_backends() {
    let original: Vec<u8> = (0..128).map(|i| (i as u8).wrapping_mul(3)).collect();

    for backend in BackendKind::ALL {
        backend.prepare(false);

        let (tmpdir, path) = seed_file(&original);
        let read_bytes = block_on_io(backend.read_future(path.clone(), None, false))
            .unwrap_or_else(|err| panic!("backend {} read failed: {err}", backend.name()));
        assert_eq!(
            read_bytes.as_ref(),
            original.as_slice(),
            "backend {} returned unexpected payload",
            backend.name()
        );

        let name_len = backend.name().len() as u8;
        let new_payload: Vec<u8> = (0..64).map(|i| (i as u8).wrapping_add(name_len)).collect();
        let bytes = Bytes::from(new_payload.clone());
        block_on_io(backend.write_future(path.clone(), bytes.clone()))
            .unwrap_or_else(|err| panic!("backend {} write failed: {err}", backend.name()));

        let on_disk = fs::read(&path).expect("failed to read updated file");
        assert_eq!(
            on_disk,
            new_payload,
            "backend {} wrote unexpected data",
            backend.name()
        );

        drop(tmpdir);
    }
}

/// Non-blocking uring requires a dedicated runtime
#[test]
fn read_write_roundtrip_non_blocking_uring() {
    let original: Vec<u8> = (0..128).map(|i| (i as u8).wrapping_mul(3)).collect();
    let mut executor = UringExecutor::new(1);

    let (tmpdir, path) = seed_file(&original);
    let path_clone = path.clone();
    let read_bytes = executor.run_to_completion(async move {
        runtime::read(path_clone, None, false).await
    }).unwrap_or_else(|err| panic!("read failed: {err}"));
    assert_eq!(
        read_bytes.as_ref(),
        original.as_slice(),
        "read returned unexpected payload",
    );

    let new_payload: Vec<u8> = (0..64).map(|i| (i as u8).wrapping_add(1)).collect();
    let bytes = Bytes::from(new_payload.clone());
    let path_clone = path.clone();
    executor.run_to_completion(async move {
        runtime::write(path_clone, &bytes.clone()).await
    }).unwrap_or_else(|err| panic!("write failed: {err}"));

    let on_disk = fs::read(&path).expect("failed to read updated file");
    assert_eq!(
        on_disk,
        new_payload,
        "wrote unexpected data",
    );

    drop(tmpdir);
}
