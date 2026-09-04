//! Utility functions for the storage module.

use datafusion_common::ScalarValue;
pub(crate) mod byte_cache;
mod variant_schema;
mod variant_utils;

pub use variant_schema::VariantSchema;
pub use variant_utils::typed_struct_contains_path;

pub(crate) fn get_bytes_needle(value: &ScalarValue) -> Option<Vec<u8>> {
    match value {
        ScalarValue::Utf8(Some(v)) => Some(v.as_bytes().to_vec()),
        ScalarValue::Utf8View(Some(v)) => Some(v.as_bytes().to_vec()),
        ScalarValue::LargeUtf8(Some(v)) => Some(v.as_bytes().to_vec()),
        ScalarValue::Binary(Some(v)) => Some(v.clone()),
        ScalarValue::BinaryView(Some(v)) => Some(v.clone()),
        ScalarValue::FixedSizeBinary(_, Some(v)) => Some(v.clone()),
        ScalarValue::LargeBinary(Some(v)) => Some(v.clone()),
        ScalarValue::Dictionary(_, value) => get_bytes_needle(value.as_ref()),
        _ => None,
    }
}

pub(crate) fn yield_now_if_shuttle() {
    #[cfg(all(feature = "shuttle", test))]
    shuttle::thread::yield_now();
}

#[cfg(all(feature = "shuttle", test))]
pub(crate) fn shuttle_test(test: impl Fn() + Send + Sync + 'static) {
    _ = tracing_subscriber::fmt()
        .with_ansi(true)
        .with_thread_names(false)
        .with_target(false)
        .try_init();

    let mut runner = shuttle::PortfolioRunner::new(true, Default::default());

    let available_cores = std::thread::available_parallelism().unwrap().get().min(4);

    for _i in 0..available_cores {
        runner.add(shuttle::scheduler::PctScheduler::new(10, 1_000));
    }
    runner.run(test);
}

#[allow(unused)]
#[cfg(all(feature = "shuttle", test))]
pub(crate) fn shuttle_replay(test: impl Fn() + Send + Sync + 'static, schedule: &str) {
    _ = tracing_subscriber::fmt()
        .with_ansi(true)
        .with_thread_names(false)
        .with_target(false)
        .try_init();
    shuttle::replay(test, schedule);
}
