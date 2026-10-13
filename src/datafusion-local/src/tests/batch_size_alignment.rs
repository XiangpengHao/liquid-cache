//! Regression tests for batch-size alignment: the reader indexes the cache by batch
//! id, and the parquet fallback turns that id back into rows using the *cache* batch
//! size. Reading at the session batch size (`datafusion.execution.batch_size`)
//! instead addresses the wrong rows whenever the two differ — the scan either runs
//! off the end of the row group and fails, or silently returns rows from the wrong
//! offsets.
//!
//! `LiquidCacheLocalBuilder::build` pins `execution.batch_size` to the cache batch
//! size, which is why the default configuration never hit this. A host that sets a
//! per-query batch size afterwards — as `SET datafusion.execution.batch_size` does
//! here — used to break the alignment.

use std::path::Path;
use std::sync::Arc;

use arrow::array::{Array, Int64Array};
use arrow::record_batch::RecordBatch;
use arrow_schema::{DataType, Field, Schema};
use datafusion::error::Result;
use datafusion::prelude::{ParquetReadOptions, SessionConfig, SessionContext};
use parquet::arrow::ArrowWriter;
use parquet::file::properties::WriterProperties;
use tempfile::TempDir;

use crate::LiquidCacheLocalBuilder;

/// One row group of `rows` rows, `id` ascending from 0.
fn write_single_row_group(path: &Path, rows: i64) {
    write_parquet(path, rows, None);
}

/// `rows` rows of ascending `id`, optionally split into row groups of
/// `row_group_rows`.
fn write_parquet(path: &Path, rows: i64, row_group_rows: Option<usize>) {
    let schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Int64, false)]));
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![Arc::new(Int64Array::from((0..rows).collect::<Vec<_>>()))],
    )
    .unwrap();
    let mut props = WriterProperties::builder();
    if let Some(n) = row_group_rows {
        props = props.set_max_row_group_row_count(Some(n));
    }
    let file = std::fs::File::create(path).unwrap();
    let mut writer = ArrowWriter::try_new(file, schema, Some(props.build())).unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();
}

/// Cache batch size stays at the 8192 default; the session batch size is then set
/// to `session_batch_size`, as a host sizing batches by row width would do.
async fn ctx_with_session_batch_size(
    cache_dir: &Path,
    parquet_path: &Path,
    session_batch_size: usize,
) -> Result<SessionContext> {
    ctx_with_session_batch_size_and_partitions(cache_dir, parquet_path, session_batch_size, 1).await
}

async fn ctx_with_session_batch_size_and_partitions(
    cache_dir: &Path,
    parquet_path: &Path,
    session_batch_size: usize,
    target_partitions: usize,
) -> Result<SessionContext> {
    std::fs::create_dir_all(cache_dir)?;
    let mut config = SessionConfig::new();
    config.options_mut().execution.target_partitions = target_partitions;
    let (ctx, cache) = LiquidCacheLocalBuilder::new()
        .with_cache_dir(cache_dir.to_path_buf())
        .build(config)
        .await?;
    assert_eq!(cache.batch_size(), 8192, "cache batch size should be 8192");

    ctx.sql(&format!(
        "SET datafusion.execution.batch_size = {session_batch_size}"
    ))
    .await?
    .collect()
    .await?;

    ctx.register_parquet(
        "t",
        parquet_path.to_str().unwrap(),
        ParquetReadOptions::default(),
    )
    .await?;
    Ok(ctx)
}

async fn collect_batches(ctx: &SessionContext, sql: &str) -> Vec<RecordBatch> {
    ctx.sql(sql).await.unwrap().collect().await.unwrap()
}

fn ids_of(batches: &[RecordBatch]) -> Vec<i64> {
    let mut ids = Vec::new();
    for batch in batches {
        let col = batch
            .column(0)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap();
        ids.extend((0..col.len()).map(|i| col.value(i)));
    }
    ids
}

async fn collect_ids(ctx: &SessionContext, sql: &str) -> Vec<i64> {
    ids_of(&collect_batches(ctx, sql).await)
}

/// The reported failure: a 12000-row row group (fewer than `2 * 8192` rows) read
/// with a session batch size of 2145 used to drive the fallback past the end of
/// the row group on query batch 2, panicking with "parquet fallback ended before
/// batch 2".
#[tokio::test]
async fn narrowed_session_batch_size_reads_the_whole_row_group() {
    let tmp = TempDir::new().unwrap();
    let parquet_path = tmp.path().join("t.parquet");
    write_single_row_group(&parquet_path, 12000);
    let ctx = ctx_with_session_batch_size(&tmp.path().join("cache"), &parquet_path, 2145)
        .await
        .unwrap();

    for _ in 0..2 {
        let ids = collect_ids(&ctx, "SELECT id FROM t").await;
        assert_eq!(ids, (0..12000).collect::<Vec<_>>());
    }
}

/// Silent corruption, not an error: a LIMIT stopped the scan before the fallback
/// ran out of rows, so the query succeeded and returned rows 0-2144 followed by
/// rows 8192-10336 in place of rows 2145-4289.
#[tokio::test]
async fn narrowed_session_batch_size_with_limit_returns_aligned_rows() {
    let tmp = TempDir::new().unwrap();
    let parquet_path = tmp.path().join("t.parquet");
    write_single_row_group(&parquet_path, 12000);
    let ctx = ctx_with_session_batch_size(&tmp.path().join("cache"), &parquet_path, 2145)
        .await
        .unwrap();

    let ids = collect_ids(&ctx, "SELECT id FROM t LIMIT 4290").await;
    assert_eq!(ids, (0..4290).collect::<Vec<_>>());
}

/// The fully-cached path misaligns as well. The cold run
/// populates cache batches 0 and 1; the warm run reads only from the cache and
/// used to emit rows 8192.. for query batch 1, because `read_arrow_array` applied
/// the 2145-bit selection to the 8192-row stored chunk and `arrow::compute::filter`
/// only rejects a mask *longer* than its target.
#[tokio::test]
async fn narrowed_session_batch_size_stays_aligned_on_the_fully_cached_path() {
    let tmp = TempDir::new().unwrap();
    let parquet_path = tmp.path().join("t.parquet");
    write_single_row_group(&parquet_path, 12000);
    let ctx = ctx_with_session_batch_size(&tmp.path().join("cache"), &parquet_path, 2145)
        .await
        .unwrap();

    let sql = "SELECT id FROM t LIMIT 4290";
    let _cold = collect_ids(&ctx, sql).await;
    let warm = collect_ids(&ctx, sql).await;
    assert_eq!(warm, (0..4290).collect::<Vec<_>>());
}

/// A session batch size *larger* than the cache batch size misaligned in the other
/// direction: the selection mask outran the stored chunk, and
/// `arrow::compute::filter` rejected it with "Filter predicate of length .. is
/// larger than target array of length ..".
#[tokio::test]
async fn widened_session_batch_size_stays_aligned() {
    let tmp = TempDir::new().unwrap();
    let parquet_path = tmp.path().join("t.parquet");
    write_single_row_group(&parquet_path, 12000);
    let ctx = ctx_with_session_batch_size(&tmp.path().join("cache"), &parquet_path, 20000)
        .await
        .unwrap();

    for _ in 0..2 {
        let ids = collect_ids(&ctx, "SELECT id FROM t").await;
        assert_eq!(ids, (0..12000).collect::<Vec<_>>());
    }
}

/// The scan now reads in cache-sized batches, but the caller must still receive
/// session-sized ones: `DataSourceExec::execute` wraps every source stream in a
/// `BatchSplitStream` sized by the session config. This is what makes ignoring the
/// session batch size inside the reader safe.
#[tokio::test]
async fn session_batch_size_still_bounds_the_batches_the_caller_receives() {
    let tmp = TempDir::new().unwrap();
    let parquet_path = tmp.path().join("t.parquet");
    write_single_row_group(&parquet_path, 12000);
    let ctx = ctx_with_session_batch_size(&tmp.path().join("cache"), &parquet_path, 2145)
        .await
        .unwrap();

    for _ in 0..2 {
        let batches = collect_batches(&ctx, "SELECT id FROM t").await;
        let oversized: Vec<_> = batches
            .iter()
            .map(|b| b.num_rows())
            .filter(|rows| *rows > 2145)
            .collect();
        assert!(
            oversized.is_empty(),
            "batches exceeded the session batch size: {oversized:?}"
        );
        assert_eq!(ids_of(&batches), (0..12000).collect::<Vec<_>>());
    }
}

/// `current_batch_id` restarts at 0 for each row group, so a file with several of
/// them exercises the mapping repeatedly rather than once.
#[tokio::test]
async fn multi_row_group_scan_stays_aligned() {
    let tmp = TempDir::new().unwrap();
    let parquet_path = tmp.path().join("t.parquet");
    write_parquet(&parquet_path, 30000, Some(5000));
    let ctx = ctx_with_session_batch_size(&tmp.path().join("cache"), &parquet_path, 2145)
        .await
        .unwrap();

    for _ in 0..2 {
        let ids = collect_ids(&ctx, "SELECT id FROM t").await;
        assert_eq!(ids, (0..30000).collect::<Vec<_>>());
    }
}

/// Several partitions read the same file concurrently, each with its own reader
/// and its own `current_batch_id` sequence.
#[tokio::test]
async fn multi_partition_scan_stays_aligned() {
    let tmp = TempDir::new().unwrap();
    let parquet_path = tmp.path().join("t.parquet");
    write_parquet(&parquet_path, 30000, Some(5000));
    let ctx = ctx_with_session_batch_size_and_partitions(
        &tmp.path().join("cache"),
        &parquet_path,
        2145,
        4,
    )
    .await
    .unwrap();

    for _ in 0..2 {
        let mut ids = collect_ids(&ctx, "SELECT id FROM t").await;
        ids.sort_unstable();
        assert_eq!(ids, (0..30000).collect::<Vec<_>>());
    }
}
