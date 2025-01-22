/// Ordered locks for Arrow cache.
///
/// We use Rust type system to statically ensure that every thread acquires the locks in the specified order.
/// This eliminates deadlock at compile time.
/// I.e., if users does not follow the lock ordering, the code will not compile.
///
/// Note: this is a rather experimental thing that I'm quite excited about.
/// Think of this as specification in verification context.
///
/// Also read: <https://lwn.net/Articles/995814/>
/// # Safety
/// Not safe.
pub(crate) unsafe trait LockBefore<Other> {}

/// Specify that `A` happens before `B`, and any `X` that happens before `A` must also happens before `B`.
macro_rules! impl_lock_before_all {
    ($A:ty => $B:ty) => {
        unsafe impl LockBefore<$B> for $A {}
        unsafe impl<X: LockBefore<$A>> LockBefore<$B> for X {}
    };
}

/// Specify that `A` happens before `B`, and only `A` can happen before `B`.
macro_rules! impl_lock_before_one {
    ($A:ty => $B:ty) => {
        unsafe impl LockBefore<$B> for $A {}
    };
}

#[derive(Debug)]
pub(crate) struct OrderedMutex<Id, T> {
    mtx: std::sync::Mutex<T>,
    _marker: std::marker::PhantomData<Id>,
}

impl<Id, T> OrderedMutex<Id, T> {
    pub(crate) fn new(mtx: T) -> Self {
        OrderedMutex {
            mtx: std::sync::Mutex::new(mtx),
            _marker: std::marker::PhantomData,
        }
    }

    pub fn lock<L>(&self, _ctx: &mut LockCtx<L>) -> (std::sync::MutexGuard<T>, LockCtx<Id>)
    where
        L: LockBefore<Id>,
    {
        (self.mtx.lock().unwrap(), LockCtx(std::marker::PhantomData))
    }
}

#[derive(Debug)]
pub(crate) struct OrderedRwLock<Id, T> {
    mtx: std::sync::RwLock<T>,
    _marker: std::marker::PhantomData<Id>,
}

impl<Id, T> OrderedRwLock<Id, T> {
    pub fn new(mtx: T) -> Self {
        OrderedRwLock {
            mtx: std::sync::RwLock::new(mtx),
            _marker: std::marker::PhantomData,
        }
    }

    pub fn read<L>(&self, _ctx: &mut LockCtx<L>) -> (std::sync::RwLockReadGuard<T>, LockCtx<Id>)
    where
        L: LockBefore<Id>,
    {
        (self.mtx.read().unwrap(), LockCtx(std::marker::PhantomData))
    }

    pub fn write<L>(&self, _ctx: &mut LockCtx<L>) -> (std::sync::RwLockWriteGuard<T>, LockCtx<Id>)
    where
        L: LockBefore<Id>,
    {
        (self.mtx.write().unwrap(), LockCtx(std::marker::PhantomData))
    }
}

#[derive(Debug)]
pub(crate) struct UnLocked;

#[derive(Debug)]
pub(crate) struct LockEtcCompressorMetadata;

#[derive(Debug)]
pub(crate) struct LockEtcFsstCompressor;

#[derive(Debug)]
pub(crate) struct LockedEntry;

#[derive(Debug)]
pub(crate) struct LockColumnMapping;

#[derive(Debug)]
pub(crate) struct LockDiskFile;

#[derive(Debug)]
pub(crate) struct LockVortexCompression;

impl_lock_before_one!(UnLocked => LockColumnMapping);
impl_lock_before_one!(LockColumnMapping => LockDiskFile);
impl_lock_before_one!(LockDiskFile => LockedEntry);

impl_lock_before_one!(LockColumnMapping => LockedEntry);

impl_lock_before_one!(LockColumnMapping => LockEtcCompressorMetadata);
impl_lock_before_all!(LockEtcCompressorMetadata => LockEtcFsstCompressor);

impl_lock_before_one!(LockColumnMapping=> LockVortexCompression);

pub(crate) struct LockCtx<ID>(std::marker::PhantomData<ID>);

impl LockCtx<UnLocked> {
    pub(crate) const UNLOCKED: LockCtx<UnLocked> = LockCtx(std::marker::PhantomData);
}
