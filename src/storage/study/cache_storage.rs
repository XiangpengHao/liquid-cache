use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use arrow::array::ArrayRef;
use arrow::buffer::BooleanBuffer;
use bytes::Bytes;
use clap::Parser;
use datafusion::logical_expr::Operator;
use datafusion::prelude::*;
use datafusion::scalar::ScalarValue;
use futures::StreamExt;
use liquid_cache_common::LiquidCacheMode;
use liquid_cache_storage::cache::CacheStorage;
use liquid_cache_storage::cache::CacheStorageBuilder;
use liquid_cache_storage::cache::EntryID;
use liquid_cache_storage::cache::cached_data::GetWithPredicateResult;
use liquid_cache_storage::cache::io_state::IoRequest;
use liquid_cache_storage::cache::io_state::{IoStateMachine, SansIo, TryGet};
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

fn do_io(io_req: &IoRequest) -> Bytes {
    match io_req.range() {
        Some(range) => {
            let r = range.range();
            let len = (r.end - r.start) as usize;
            let mut file = File::open(io_req.path()).expect("open cache file");
            file.seek(SeekFrom::Start(r.start)).expect("seek start");
            let mut buf = vec![0u8; len];
            file.read_exact(&mut buf).expect("read exact range");
            Bytes::from(buf)
        }
        None => {
            let bytes = std::fs::read(io_req.path()).expect("read cache file");
            Bytes::from(bytes)
        }
    }
}

fn main() {
    let args = CliArgs::parse();

    // 1) Build cache storage with FILO and a small budget (100 MB)
    let mut builder = CacheStorageBuilder::new()
        .with_max_cache_bytes(500 * 1024 * 1024)
        .with_cache_mode(LiquidCacheMode::Liquid)
        .with_policy(Box::new(FiloPolicy::new()));
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
    let t0 = Instant::now();
    let mut evaluated = 0usize;
    let mut num_io = 0usize;
    let mut num_io_bytes = 0usize;
    for (i, id) in ids.iter().enumerate() {
        let cached = storage.get(id).expect("get cached data");
        let len = lens[i];
        let selection = BooleanBuffer::new_set(len);
        match cached.get_with_predicate(&selection, &pred_expr) {
            SansIo::Ready(res) => match res {
                GetWithPredicateResult::Evaluated(_) => evaluated += 1,
                GetWithPredicateResult::Filtered(_filtered) => {}
            },
            SansIo::Pending((mut state, io_req)) => {
                let bytes = do_io(&io_req);
                num_io_bytes += bytes.len();
                num_io += 1;
                state.feed(bytes);
                match state.try_get() {
                    TryGet::Ready(GetWithPredicateResult::Evaluated(_)) => evaluated += 1,
                    TryGet::Ready(GetWithPredicateResult::Filtered(_filtered)) => {}
                    e => panic!("unexpected result: {e:?}"),
                }
            }
        }
    }
    let scan_elapsed = t0.elapsed();
    let stats = storage.stats();
    println!("Cache stats: {stats:#?}");
    println!(
        "Cache scan (get_with_predicate) completed:\n  batches: {}\n  rows: {}\n  time: {:.3}s\n  evaluated: {}\n  num_io: {}\n  num_io_bytes: {}",
        ids.len(),
        total_rows,
        scan_elapsed.as_secs_f64(),
        evaluated,
        num_io,
        num_io_bytes,
    );
}

// Read Referer column and insert into cache. Returns (entry_ids, lengths)
fn load_and_insert_referer(
    storage: &Arc<CacheStorage>,
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
            storage.insert(id, array);
            idx += 1;
        }

        (ids, lens, total_size)
    })
}
