use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use arrow::array::ArrayRef;
use arrow::buffer::BooleanBuffer;
use clap::Parser;
use datafusion::logical_expr::Operator;
use datafusion::prelude::*;
use datafusion::scalar::ScalarValue;
use futures::StreamExt;
use liquid_cache_storage::cache::EntryID;
use liquid_cache_storage::cache::LiquidCache;
use liquid_cache_storage::cache::LiquidCacheBuilder;
use liquid_cache_storage::cache::squeeze_policies::TranscodeSqueezeEvict;
use liquid_cache_storage::cache_policies::FiloPolicy;

#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

#[derive(Parser, Debug, Default)]
#[command(name = "CacheStorage Benchmark")]
#[command(about = "Measure CacheStorage insert + scan with predicate pushdown")]
struct CliArgs {
    /// Parquet file to read (projected Referer column)
    #[arg(long, default_value = "../../benchmark/clickbench/data/hits.parquet")]
    parquet: String,

    /// Cache directory root (choose device/mount to test). If not set, uses a temp dir.
    #[arg(long)]
    cache_dir: Option<PathBuf>,

    /// Cargo passes --bench for harness=false binaries; accept it to avoid parse errors
    #[arg(long, default_value = "false")]
    bench: bool,
}

fn main() {
    let args = CliArgs::parse();

    // 1) Build cache storage with FILO and a small budget (100 MB)
    let mut builder = LiquidCacheBuilder::new()
        .with_max_cache_bytes(500 * 1024 * 1024)
        .with_squeeze_policy(Box::new(TranscodeSqueezeEvict))
        .with_cache_policy(Box::new(FiloPolicy::new()));
    if let Some(dir) = args.cache_dir.clone() {
        builder = builder.with_cache_dir(dir);
    }
    let storage = builder.build();

    // 2) Load Referer column from parquet using DataFusion
    let (ids, lens, total_size) = load_and_insert_referer(&storage, &args.parquet);
    let total_rows: usize = lens.iter().sum();
    eprintln!(
        "Inserted {} batches, total {} rows, total size {} bytes into cache",
        ids.len(),
        total_rows,
        total_size
    );

    // 3) Build predicate: column == "" (empty string)
    use datafusion::physical_plan::expressions::{BinaryExpr, Column, Literal};
    let pred_expr: Arc<dyn datafusion::physical_plan::PhysicalExpr> = Arc::new(BinaryExpr::new(
        Arc::new(Column::new("col", 0)),
        Operator::Eq,
        Arc::new(Literal::new(ScalarValue::Utf8View(Some(String::new())))),
    ));

    // 4) Scan all entries with selection=all-true and time get_with_predicate
    let rt = tokio::runtime::Runtime::new().expect("tokio runtime");
    let scan_elapsed = rt.block_on(async {
        let t0 = Instant::now();
        let mut evaluated = 0usize;
        for (i, id) in ids.iter().enumerate() {
            let len = lens[i];
            let selection = BooleanBuffer::new_set(len);
            if let Some(result) = storage
                .eval_predicate(id, &pred_expr)
                .with_selection(&selection)
                .await
                && result.is_ok()
            {
                evaluated += 1;
            }
        }
        let elapsed = t0.elapsed();
        eprintln!("Evaluated: {}", evaluated);
        elapsed
    });
    let stats = storage.stats();
    println!("Cache stats: {stats:#?}");
    println!(
        "Cache scan (get_with_predicate) completed:\n  batches: {}\n  rows: {}\n  time: {:.3}s",
        ids.len(),
        total_rows,
        scan_elapsed.as_secs_f64(),
    );
}

// Read Referer column and insert into cache. Returns (entry_ids, lengths)
fn load_and_insert_referer(
    storage: &Arc<LiquidCache>,
    parquet_path: &str,
) -> (Vec<EntryID>, Vec<usize>, usize) {
    let rt = tokio::runtime::Runtime::new().expect("tokio runtime");
    rt.block_on(async move {
        let mut config = SessionConfig::default().with_batch_size(8192 * 2);
        let options = config.options_mut();
        options.execution.parquet.schema_force_view_types = false;

        let ctx = SessionContext::new_with_config(config);
        ctx.register_parquet("hits", parquet_path, Default::default())
            .await
            .expect("register parquet");

        let sql = "SELECT \"Referer\" FROM \"hits\"".to_string();
        let df = ctx.sql(&sql).await.expect("create df");
        let mut stream = df.execute_stream().await.expect("execute stream");

        let mut ids = Vec::new();
        let mut lens = Vec::new();
        let mut total_size = 0;
        let mut idx: usize = 0;
        while let Some(batch_res) = stream.next().await {
            let batch = batch_res.expect("stream batch");
            let array: ArrayRef = batch.column(0).clone();
            lens.push(array.len());
            let id = EntryID::from(idx);
            ids.push(id);
            total_size += array.get_array_memory_size();
            storage.insert(id, array).await;
            idx += 1;
        }

        (ids, lens, total_size)
    })
}
