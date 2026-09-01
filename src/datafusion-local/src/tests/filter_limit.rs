//! Regression tests for `WHERE <predicate> LIMIT n` (no ORDER BY).
//!
//! The scan-level limit used to truncate the row selection *before* the
//! pushed-down row filter ran, so any match past the first `limit + offset`
//! physical rows was silently dropped — `SELECT .. WHERE tag='MATCH' LIMIT 10`
//! returned 0 rows even with 5 real matches. These tables place every match at
//! the physical tail of the data, past any LIMIT-sized prefix, so a
//! prefix-then-filter scan returns strictly fewer rows than expected.

use std::path::Path;
use std::sync::Arc;

use arrow::array::{Array, Int64Array, StringArray};
use arrow::record_batch::RecordBatch;
use arrow_schema::{DataType, Field, Schema};
use datafusion::error::Result;
use datafusion::prelude::{ParquetReadOptions, SessionConfig, SessionContext};
use parquet::arrow::ArrowWriter;
use parquet::file::properties::WriterProperties;
use tempfile::TempDir;

use crate::LiquidCacheLocalBuilder;

/// 20 rows: `tag='other'` for id 0-14, `tag='MATCH'` for id 15-19 — all 5
/// matches sit at the physical end of the file.
fn write_tail_match_file(path: &Path, max_row_group_size: Option<usize>) {
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("tag", DataType::Utf8, false),
    ]));
    let ids = Int64Array::from((0..20i64).collect::<Vec<_>>());
    let tags = StringArray::from(
        (0..20)
            .map(|i| if i >= 15 { "MATCH" } else { "other" })
            .collect::<Vec<_>>(),
    );
    let batch = RecordBatch::try_new(schema.clone(), vec![Arc::new(ids), Arc::new(tags)]).unwrap();
    let props = max_row_group_size.map(|n| {
        WriterProperties::builder()
            .set_max_row_group_row_count(Some(n))
            .build()
    });
    let file = std::fs::File::create(path).unwrap();
    let mut writer = ArrowWriter::try_new(file, schema, props).unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();
}

async fn liquid_ctx(cache_dir: &Path) -> Result<SessionContext> {
    std::fs::create_dir_all(cache_dir)?;
    let mut config = SessionConfig::new();
    config.options_mut().execution.target_partitions = 4;
    let (ctx, _cache) = LiquidCacheLocalBuilder::new()
        .with_cache_dir(cache_dir.to_path_buf())
        .build(config)
        .await?;
    Ok(ctx)
}

/// Runs `sql` twice and asserts the row count both times, so the second
/// execution exercises the cached path where one exists. (Only the first
/// `assert_rows` against a table starts from a cold cache; later calls in the
/// same test run warm/warm.)
async fn assert_rows(ctx: &SessionContext, sql: &str, expected: usize) {
    for run in ["cold", "warm"] {
        let batches = ctx.sql(sql).await.unwrap().collect().await.unwrap();
        let rows: usize = batches.iter().map(|b| b.num_rows()).sum();
        assert_eq!(rows, expected, "{run} run returned wrong row count: {sql}");
    }
}

/// All returned ids must be actual matches (>= 15), not prefix rows.
async fn assert_ids_are_matches(ctx: &SessionContext, sql: &str, expected_len: usize) {
    let batches = ctx.sql(sql).await.unwrap().collect().await.unwrap();
    let mut ids: Vec<i64> = batches
        .iter()
        .flat_map(|b| {
            let col = b
                .column(0)
                .as_any()
                .downcast_ref::<Int64Array>()
                .unwrap()
                .clone();
            (0..col.len()).map(move |i| col.value(i))
        })
        .collect();
    ids.sort_unstable();
    assert_eq!(ids.len(), expected_len, "wrong number of rows: {sql}");
    assert!(
        ids.iter().all(|id| *id >= 15),
        "returned non-matching rows {ids:?}: {sql}"
    );
}

#[tokio::test]
async fn filter_with_limit_single_row_group() {
    let dir = TempDir::new().unwrap();
    let file = dir.path().join("tail20.parquet");
    write_tail_match_file(&file, None);
    let ctx = liquid_ctx(&dir.path().join("cache")).await.unwrap();
    ctx.register_parquet(
        "tail20",
        file.to_str().unwrap(),
        ParquetReadOptions::default(),
    )
    .await
    .unwrap();

    // Baselines.
    assert_rows(&ctx, "SELECT id FROM tail20 WHERE tag='MATCH'", 5).await;
    // The bug: LIMIT larger than the match count must still return every
    // match, even though all matches sit past the first LIMIT physical rows.
    assert_rows(&ctx, "SELECT id FROM tail20 WHERE tag='MATCH' LIMIT 10", 5).await;
    // LIMIT smaller than the match count caps *matches*, not scanned rows.
    assert_rows(&ctx, "SELECT id FROM tail20 WHERE tag='MATCH' LIMIT 3", 3).await;
    // ORDER BY variant was never broken (fetch stays in the TopK sort).
    assert_rows(
        &ctx,
        "SELECT id FROM tail20 WHERE tag='MATCH' ORDER BY id LIMIT 10",
        5,
    )
    .await;
    // OFFSET is applied to filtered rows: skip 2 of the 5 matches.
    assert_rows(
        &ctx,
        "SELECT id FROM tail20 WHERE tag='MATCH' LIMIT 18 OFFSET 2",
        3,
    )
    .await;
    // The rows themselves must be matches, not prefix rows.
    assert_ids_are_matches(&ctx, "SELECT id FROM tail20 WHERE tag='MATCH' LIMIT 10", 5).await;
    assert_ids_are_matches(&ctx, "SELECT id FROM tail20 WHERE tag='MATCH' LIMIT 3", 3).await;

    // Numeric predicate, same shape.
    assert_rows(&ctx, "SELECT id FROM tail20 WHERE id >= 15 LIMIT 3", 3).await;
}

#[tokio::test]
async fn filter_with_limit_across_row_groups() {
    let dir = TempDir::new().unwrap();
    let file = dir.path().join("tail20_rg8.parquet");
    // Row groups of 8 rows: matches (rows 15-19) span the 2nd and 3rd groups.
    write_tail_match_file(&file, Some(8));
    let ctx = liquid_ctx(&dir.path().join("cache")).await.unwrap();
    ctx.register_parquet(
        "tail20_rg8",
        file.to_str().unwrap(),
        ParquetReadOptions::default(),
    )
    .await
    .unwrap();

    assert_rows(&ctx, "SELECT id FROM tail20_rg8 WHERE tag='MATCH'", 5).await;
    assert_rows(
        &ctx,
        "SELECT id FROM tail20_rg8 WHERE tag='MATCH' LIMIT 10",
        5,
    )
    .await;
    assert_rows(
        &ctx,
        "SELECT id FROM tail20_rg8 WHERE tag='MATCH' LIMIT 3",
        3,
    )
    .await;
    assert_ids_are_matches(
        &ctx,
        "SELECT id FROM tail20_rg8 WHERE tag='MATCH' LIMIT 10",
        5,
    )
    .await;
}

#[tokio::test]
async fn filter_with_limit_across_files() {
    let dir = TempDir::new().unwrap();
    let table_dir = dir.path().join("two_files");
    std::fs::create_dir_all(&table_dir).unwrap();
    // Two identical files, 5 tail matches each: 10 matches total. The broken
    // behavior applied the limit per file pre-filter, so LIMIT 17 returned 4
    // rows (2 per file) instead of 10.
    write_tail_match_file(&table_dir.join("a.parquet"), None);
    write_tail_match_file(&table_dir.join("b.parquet"), None);
    let ctx = liquid_ctx(&dir.path().join("cache")).await.unwrap();
    ctx.register_parquet(
        "two_files",
        &format!("{}/", table_dir.to_str().unwrap()),
        ParquetReadOptions::default(),
    )
    .await
    .unwrap();

    assert_rows(&ctx, "SELECT id FROM two_files WHERE tag='MATCH'", 10).await;
    assert_rows(
        &ctx,
        "SELECT id FROM two_files WHERE tag='MATCH' LIMIT 10",
        10,
    )
    .await;
    assert_rows(
        &ctx,
        "SELECT id FROM two_files WHERE tag='MATCH' LIMIT 17",
        10,
    )
    .await;
    assert_rows(
        &ctx,
        "SELECT id FROM two_files WHERE tag='MATCH' LIMIT 40",
        10,
    )
    .await;
    // Global cap still enforced on filtered rows.
    assert_rows(
        &ctx,
        "SELECT id FROM two_files WHERE tag='MATCH' LIMIT 7",
        7,
    )
    .await;
    assert_ids_are_matches(
        &ctx,
        "SELECT id FROM two_files WHERE tag='MATCH' LIMIT 7",
        7,
    )
    .await;
}
