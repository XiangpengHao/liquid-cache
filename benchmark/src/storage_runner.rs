#![cfg(target_os = "linux")]

/**
 * Benchmark to test the performance of io_uring runtime for clickbench queries. The queries are executed directly
 * on a LiquidCache instance to bypass datafusion, which is strongly coupled with tokio. The benchmark is based on
 * the arrow benchmark (https://github.com/apache/arrow-rs/blob/main/parquet/benches/arrow_reader_clickbench.rs#L729)
 */

use arrow::array::BooleanArray;
use arrow::buffer::BooleanBuffer;
use clap::Parser;
use datafusion::logical_expr::Operator;
use datafusion::physical_plan::expressions::{BinaryExpr, Column};
use datafusion::physical_plan::PhysicalExpr;
use datafusion::prelude::{SessionConfig, SessionContext};
use datafusion::scalar::ScalarValue;
use futures::StreamExt;
use liquid_cache_common::IoMode;
use liquid_cache_storage::cache::{EntryID, LiquidCacheBuilder, LiquidCache};
use liquid_cache_parquet::{ParquetIoContext, UringExecutor};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

#[derive(Parser)]
#[command(name = "storage_runner")]
struct Args {
    /// ClickBench query index (0-based). Only queries with filters are supported (e.g. 1, 10, 19, 20).
    #[arg(long)]
    query_index: usize,

    /// Number of partitions (tasks to spawn on UringExecutor).
    #[arg(long)]
    partitions: usize,

    #[arg(long)]
    worker_threads: usize,

    #[arg(long)]
    iterations: usize,

    /// Path to hits.parquet. Default: benchmark/clickbench/data/hits.parquet
    #[arg(long, default_value = "benchmark/clickbench/data/hits.parquet")]
    parquet: PathBuf,

    /// Directory for the liquid-cache storage. Default: $TMPDIR/liquid_cache_storage_runner
    #[arg(long)]
    cache_dir: Option<PathBuf>,
}

/// ClickBench query descriptor: filter column(s) and predicate expression(s).
/// Each expression is evaluated on a single column (column index 0 in the cached array).
/// TODO(): Add support for columns that are projected.
struct FilterQuery {
    /// Column names to load and cache (in schema order).
    filter_columns: Vec<&'static str>,
    /// One predicate per filter column; each expects Column(0) op Literal.
    predicates: Vec<Arc<dyn PhysicalExpr>>,
    /// Number of expected rows in result
    expected_row_count: usize,
}

fn all_filter_queries() -> Vec<Option<FilterQuery>> {
    use datafusion::physical_plan::expressions::Literal as Lit;
    let col = || Arc::new(Column::new("col", 0)) as Arc<dyn PhysicalExpr>;

    let mut q: Vec<Option<FilterQuery>> = (0..43).map(|_| None).collect();

    // Q1: AdvEngineID <> 0
    q[1] = Some(FilterQuery {
        filter_columns: vec!["AdvEngineID"],
        predicates: vec![Arc::new(BinaryExpr::new(
            col(),
            Operator::NotEq,
            Arc::new(Lit::new(ScalarValue::UInt64(Some(0)))),
        ))],
        expected_row_count: 3312,
    });

    // Q10: MobilePhoneModel <> ''
    q[10] = Some(FilterQuery {
        filter_columns: vec!["MobilePhoneModel"],
        predicates: vec![Arc::new(BinaryExpr::new(
            col(),
            Operator::NotEq,
            Arc::new(Lit::new(ScalarValue::Utf8(Some(String::new())))),
        ))],
        expected_row_count: 34276,
    });

    // Q12: SearchPhrase <> ''
    q[12] = Some(FilterQuery {
        filter_columns: vec!["SearchPhrase"],
        predicates: vec![Arc::new(BinaryExpr::new(
            col(),
            Operator::NotEq,
            Arc::new(Lit::new(ScalarValue::Utf8(Some(String::new())))),
        ))],
        expected_row_count: 131559,
    });

    // Q19: UserID = 3233473875476175636 (value that exists in hits_1)
    q[19] = Some(FilterQuery {
        filter_columns: vec!["UserID"],
        predicates: vec![Arc::new(BinaryExpr::new(
            col(),
            Operator::Eq,
            Arc::new(Lit::new(ScalarValue::UInt64(Some(3233473875476175636)))),
        ))],
        expected_row_count: 4,
    });

    q[20] = Some(FilterQuery { 
        filter_columns: vec!["URL"], 
        predicates: vec![Arc::new(BinaryExpr::new(
            col(),
            Operator::LikeMatch,
            Arc::new(Lit::new(ScalarValue::Utf8(Some("%google%".to_string())))),
        ))],
        expected_row_count: 137,
    });

    // Q27: URL <> ''
    q[27] = Some(FilterQuery {
        filter_columns: vec!["URL"],
        predicates: vec![Arc::new(BinaryExpr::new(
            col(),
            Operator::NotEq,
            Arc::new(Lit::new(ScalarValue::Utf8(Some(String::new())))),
        ))],
        expected_row_count: 999978,
    });

    // Q28: Referer <> ''
    q[28] = Some(FilterQuery {
        filter_columns: vec!["Referer"],
        predicates: vec![Arc::new(BinaryExpr::new(
            col(),
            Operator::NotEq,
            Arc::new(Lit::new(ScalarValue::Utf8(Some(String::new())))),
        ))],
        expected_row_count: 925813,
    });

    // Q30: SearchPhrase <> ''
    q[30] = Some(FilterQuery {
        filter_columns: vec!["SearchPhrase"],
        predicates: vec![Arc::new(BinaryExpr::new(
            col(),
            Operator::NotEq,
            Arc::new(Lit::new(ScalarValue::Utf8(Some(String::new())))),
        ))],
        expected_row_count: 131559,
    });

    // Q36: CounterID = 62, DontCountHits = 0, IsRefresh = 0, URL <> ''
    q[36] = Some(FilterQuery {
        filter_columns: vec!["CounterID", "EventDate", "DontCountHits", "IsRefresh", "URL"],
        predicates: vec![
            Arc::new(BinaryExpr::new(
                col(),
                Operator::Eq,
                Arc::new(Lit::new(ScalarValue::UInt32(Some(62)))),
            )),
            Arc::new(BinaryExpr::new(
                col(),
                Operator::Eq,
                Arc::new(Lit::new(ScalarValue::Int16(Some(0)))),
            )),
            Arc::new(BinaryExpr::new(
                col(),
                Operator::Eq,
                Arc::new(Lit::new(ScalarValue::UInt8(Some(0)))),
            )),
            Arc::new(BinaryExpr::new(
                col(),
                Operator::Eq,
                Arc::new(Lit::new(ScalarValue::UInt8(Some(0)))),
            )),
            Arc::new(BinaryExpr::new(
                col(),
                Operator::NotEq,
                Arc::new(Lit::new(ScalarValue::Utf8(Some(String::new())))),
            )),
        ],
        expected_row_count: 181198,
    });

    q
}

fn run_single_iter(
    num_batches: usize,
    num_partitions: usize,
    query: &FilterQuery, 
    storage: Arc<LiquidCache>,
    entry_ids: &Vec<EntryID>,
    batch_lengths: &Vec<usize>,
    executor: &mut UringExecutor
) {
    // 2) Partition batch indices evenly across workers.
    let batches_per_partition = (num_batches + num_partitions - 1) / num_partitions;
    let num_cols = query.filter_columns.len();

    // 3) Create futures for every partition
    let mut futures = Vec::new();
    for p in 0..num_partitions {
        let start = p * batches_per_partition;
        let end = (start + batches_per_partition).min(num_batches);
        if start >= end {
            continue;
        }
        let storage_clone = Arc::clone(&storage);
        let batch_range = start..end;
        let predicates = query.predicates.iter().map(Arc::clone).collect::<Vec<_>>();
        let entry_ids_clone = entry_ids.clone();
        let batch_lengths_clone = batch_lengths.clone();
        futures.push(run_partition(
                storage_clone,
                batch_range,
                num_cols,
                predicates,
                entry_ids_clone,
                batch_lengths_clone,
        ));
    }
        
    let start = Instant::now();
    let receiver = executor.spawn_many(&mut futures);

    let mut tasks_completed = 0;
    let mut total_rows = 0;
    while tasks_completed < num_partitions {
        total_rows += receiver.recv().expect("Failed to receive result");
        tasks_completed += 1;
    }
    let elapsed = start.elapsed();
    if total_rows != query.expected_row_count {
        log::warn!("Expected row count doesn't match. Actual: {}, expected: {}", total_rows, query.expected_row_count);
    }
    log::info!("Partitions: {}, Time: {:.3}s, Total rows: {}", num_partitions, elapsed.as_secs_f64(), total_rows);
}

fn run_bench(
    cache_dir: PathBuf,
    parquet_path: PathBuf,
    query: &FilterQuery,
    num_partitions: usize,
    num_iter: usize,
    num_workers: usize,
) {
    let _ = std::fs::create_dir_all(&cache_dir);
    let io_context = Arc::new(ParquetIoContext::new(
        cache_dir.clone(),
        IoMode::UringNonBlocking,
        4096,
    ));
    let storage = LiquidCacheBuilder::new()
        .with_io_context(io_context)
        .with_cache_dir(cache_dir)
        .build();

    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .expect("tokio runtime");
    let storage_clone = storage.clone();
    let (num_batches, entry_ids, batch_lengths) = rt.block_on(
    async move {
        // 1) Load parquet into record batches (filter columns only) and insert into cache.
        let (entry_ids, batch_lengths) = load_and_insert(storage_clone.clone(), parquet_path, query).await;
        let num_batches = entry_ids.len() / query.filter_columns.len();
        log::info!(
            "Populated cache: {} batches, {} filter columns, {} entries",
            num_batches,
            query.filter_columns.len(),
            entry_ids.len()
        );

        storage_clone.flush_all_to_disk().await;
        (num_batches, entry_ids, batch_lengths)
    });
    let mut executor = UringExecutor::new(num_workers);

    for _i in 0..num_iter {
        run_single_iter(num_batches, 
            num_partitions, 
            &query, 
            storage.clone(), 
            &entry_ids, 
            &batch_lengths, 
            &mut executor
        );
    }
}

async fn run_partition(
    storage: Arc<LiquidCache>,
    batch_range: std::ops::Range<usize>,
    num_cols: usize,
    predicates: Vec::<Arc<dyn PhysicalExpr>>,
    entry_ids: Vec::<EntryID>,
    batch_lengths: Vec::<usize>,
) -> usize {
    let mut total_matched = 0usize;
    for batch_idx in batch_range {
        let mut combined_mask: Option<BooleanArray> = None;
        for (col_idx, pred) in predicates.iter().enumerate() {
            let entry_idx = batch_idx * num_cols + col_idx;
            let entry_id = &entry_ids[entry_idx];
            let len = batch_lengths[entry_idx];
            let selection = BooleanBuffer::new_set(len);
            let result = storage
                .eval_predicate(entry_id, pred)
                .with_selection(&selection)
                .await;
            match result {
                Some(Ok(mask)) => {
                    combined_mask = Some(match combined_mask.take() {
                        Some(prev) => arrow::compute::and(&prev, &mask).unwrap(),
                        None => mask,
                    });
                }
                Some(Err(_)) | None => {
                    // Predicate could not be evaluated in cache; treat as no match for this batch.
                    combined_mask = Some(BooleanArray::from(vec![false; len]));
                }
            }
        }
        if let Some(m) = combined_mask {
            total_matched += m.true_count();
        }
    }
    total_matched
}

/// Load parquet with projection = query.filter_columns, insert each (batch, column) into cache.
/// Returns (entry_ids in order batch0_col0, batch0_col1, ..., batch1_col0, ...), (length per entry).
async fn load_and_insert(
    storage: Arc<liquid_cache_storage::cache::LiquidCache>,
    parquet_path: PathBuf,
    query: &FilterQuery,
) -> (Vec<EntryID>, Vec<usize>) {
    let config = SessionConfig::default().with_batch_size(8192);
    let ctx = SessionContext::new_with_config(config);
    ctx.register_parquet("hits", parquet_path.to_string_lossy().as_ref(), Default::default())
        .await
        .expect("register parquet");

    let cols: String = query
        .filter_columns
        .iter()
        .map(|c| format!("\"{}\"", c))
        .collect::<Vec<_>>()
        .join(", ");
    let sql = format!("SELECT {} FROM \"hits\"", cols);
    let df = ctx.sql(&sql).await.expect("sql");
    let mut stream = df.execute_stream().await.expect("execute");

    let num_cols = query.filter_columns.len();
    let mut entry_ids = Vec::new();
    let mut batch_lengths = Vec::new();
    let mut batch_idx = 0usize;

    while let Some(batch_res) = stream.next().await {
        let batch = batch_res.expect("batch");
        let nrows = batch.num_rows();
        for col_idx in 0..num_cols {
            let entry_id = EntryID::from(batch_idx * num_cols + col_idx);
            let array = batch.column(col_idx).clone();
            storage.insert(entry_id, array).await;
            entry_ids.push(entry_id);
            batch_lengths.push(nrows);
        }
        batch_idx += 1;
    }

    (entry_ids, batch_lengths)
}

fn main() {
    let args = Args::parse();

    let queries = all_filter_queries();
    let query = match args.query_index {
        i if i < queries.len() => match &queries[i] {
            Some(q) => q,
            None => {
                eprintln!(
                    "Query index {} has no filters. Only filter queries are supported. \
                     Try e.g. 1, 10, 12, 19, 27, 28, 30, 36.",
                    args.query_index
                );
                std::process::exit(1);
            }
        },
        _ => {
            eprintln!("Query index {} out of range (0..{}).", args.query_index, queries.len());
            std::process::exit(1);
        }
    };

    if args.partitions == 0 {
        eprintln!("partitions must be >= 1.");
        std::process::exit(1);
    }

    if !args.parquet.exists() {
        eprintln!(
            "Parquet file not found: {}. Download e.g. wget https://datasets.clickhouse.com/hits_compatible/athena/hits.parquet -O {}",
            args.parquet.display(),
            args.parquet.display()
        );
        std::process::exit(1);
    }
    let cache_dir = args.cache_dir.unwrap_or_else(|| {
        std::env::temp_dir().join("lc_cache_dir")
    });
    run_bench(cache_dir, 
        args.parquet, 
        query, 
        args.partitions, 
        args.iterations,
        args.worker_threads
    );
}