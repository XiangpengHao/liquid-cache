use std::sync::Arc;
use std::time::Instant;

use arrow::array::{Array, ArrayRef, StringArray, cast::AsArray};
use arrow_schema::DataType;
use clap::Parser;
use datafusion::prelude::*;
use liquid_cache_storage::liquid_array::raw::FsstArray;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand::seq::{SliceRandom, index::sample};

#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

#[derive(Parser, Debug, Clone)]
#[command(name = "FSST Selected Decode Benchmark")]
#[command(about = "Benchmark FsstArray::to_uncompressed_selected at multiple selectivities")]
struct CliArgs {
    /// Parquet file to read.
    #[arg(long, default_value = "../../benchmark/clickbench/data/hits.parquet")]
    parquet: String,

    /// Columns to process (comma-separated).
    #[arg(long, value_delimiter = ',', default_value = "Title,URL")]
    columns: Vec<String>,

    /// Optional row limit for each column (useful for faster runs).
    #[arg(long)]
    limit: Option<usize>,

    /// Parquet batch size (rows per RecordBatch).
    #[arg(long, default_value_t = 8192 * 2)]
    batch_size: usize,

    /// Selectivities to benchmark (percent, comma-separated).
    #[arg(long, value_delimiter = ',', default_value = "1,10,50,99")]
    selectivities: Vec<u8>,

    /// Iterations per selectivity.
    #[arg(long, default_value_t = 5)]
    iterations: usize,

    /// Cargo passes --bench for harness=false binaries; accept it to avoid parse errors.
    #[arg(long, default_value = "false")]
    bench: bool,
}

struct Selection {
    pct: u8,
    indices: Vec<usize>,
    approx_bytes: usize,
}

#[tokio::main]
async fn main() {
    let args = CliArgs::parse();

    let mut config = SessionConfig::default().with_batch_size(args.batch_size);
    let options = config.options_mut();
    options.execution.parquet.schema_force_view_types = false;

    let ctx = SessionContext::new_with_config(config);
    ctx.register_parquet("hits", &args.parquet, Default::default())
        .await
        .expect("register parquet");

    let mut score = 0;
    for (col_idx, column) in args.columns.iter().enumerate() {
        let array = load_column_array(&ctx, column, args.limit).await;
        let row_count = array.len();
        if row_count == 0 {
            println!("Column {column}: no rows");
            continue;
        }

        let compressor = Arc::new(FsstArray::train_compressor(
            array.iter().flatten().map(|value| value.as_bytes()),
        ));
        let fsst = FsstArray::from_byte_array_with_compressor(&array, compressor);
        let total_uncompressed = fsst.uncompressed_bytes();
        let avg_len = total_uncompressed as f64 / row_count as f64;
        drop(array);

        println!(
            "Column {column}\n  rows: {row_count}\n  uncompressed: {}\n  avg_len: {:.2} bytes",
            format_bytes(total_uncompressed),
            avg_len
        );

        let selections = build_selections(row_count, avg_len, &args.selectivities, col_idx as u64);

        for selection in selections {
            if selection.indices.is_empty() {
                println!(
                    "  selectivity {:>3}% -> selected: 0 (skipped)",
                    selection.pct
                );
                continue;
            }

            // Warm up once to reduce cold-start noise.
            std::hint::black_box(fsst.to_uncompressed_selected(&selection.indices));

            let mut total = 0.0;
            let mut min = f64::MAX;
            let mut max = 0.0_f64;
            for _ in 0..args.iterations {
                let start = Instant::now();
                let output = fsst.to_uncompressed_selected(&selection.indices);
                std::hint::black_box(output);
                let elapsed = start.elapsed().as_secs_f64();
                total += elapsed;
                min = min.min(elapsed);
                max = max.max(elapsed);
            }

            let avg = total / args.iterations as f64;
            let values_per_sec = selection.indices.len() as f64 / avg;
            let mb_per_sec = selection.approx_bytes as f64 / avg / (1024.0 * 1024.0);

            println!(
                "  selectivity {:>3}% -> selected: {:>8} | avg: {:>8.6}s | min: {:>8.6}s | max: {:>8.6}s | {:>10.1} values/s | {:>8.1} MiB/s",
                selection.pct,
                selection.indices.len(),
                avg,
                min,
                max,
                values_per_sec,
                mb_per_sec
            );
            score += mb_per_sec as usize;
        }
    }
    println!("Final score: {score}");
}

async fn load_column_array(
    ctx: &SessionContext,
    column: &str,
    limit: Option<usize>,
) -> StringArray {
    let sql = if let Some(limit) = limit {
        format!("SELECT \"{column}\" FROM \"hits\" LIMIT {limit}")
    } else {
        format!("SELECT \"{column}\" FROM \"hits\"")
    };
    let df = ctx.sql(&sql).await.expect("create df");
    let batches = df.collect().await.expect("collect");

    if batches.is_empty() {
        return StringArray::from(Vec::<Option<&str>>::new());
    }

    let mut arrays = Vec::with_capacity(batches.len());
    for batch in batches {
        let array = batch.column(0).clone();
        let array = if array.data_type() == &DataType::Utf8 {
            array
        } else {
            arrow::compute::cast(&array, &DataType::Utf8).expect("cast to Utf8")
        };
        arrays.push(array);
    }

    concat_utf8_arrays(arrays)
}

fn concat_utf8_arrays(arrays: Vec<ArrayRef>) -> StringArray {
    if arrays.is_empty() {
        return StringArray::from(Vec::<Option<&str>>::new());
    }

    let refs: Vec<&dyn Array> = arrays.iter().map(|array| array.as_ref()).collect();
    let concatenated = arrow::compute::concat(&refs).expect("concat arrays");
    concatenated.as_string::<i32>().clone()
}

fn build_selections(len: usize, avg_len: f64, selectivities: &[u8], seed: u64) -> Vec<Selection> {
    selectivities
        .iter()
        .copied()
        .map(|pct| {
            let mut rng = StdRng::seed_from_u64(seed ^ (pct as u64).wrapping_mul(0x9e37_79b9));
            let indices = build_selection(len, pct, &mut rng);
            let approx_bytes = (avg_len * indices.len() as f64).round() as usize;
            Selection {
                pct,
                indices,
                approx_bytes,
            }
        })
        .collect()
}

fn build_selection(len: usize, pct: u8, rng: &mut StdRng) -> Vec<usize> {
    if len == 0 || pct == 0 {
        return Vec::new();
    }

    let mut target = len.saturating_mul(pct as usize) / 100;
    if target == 0 {
        target = 1;
    }
    if target >= len {
        return (0..len).collect();
    }

    let mut selected = if target > len / 2 {
        let remove_count = len - target;
        let mut remove = vec![false; len];
        for idx in sample(rng, len, remove_count) {
            remove[idx] = true;
        }
        let mut indices = Vec::with_capacity(target);
        for idx in 0..len {
            if !remove[idx] {
                indices.push(idx);
            }
        }
        indices
    } else {
        sample(rng, len, target).into_vec()
    };

    selected.shuffle(rng);
    selected
}

fn format_bytes(bytes: usize) -> String {
    const KB: f64 = 1024.0;
    const MB: f64 = 1024.0 * KB;
    const GB: f64 = 1024.0 * MB;
    let b = bytes as f64;
    if b >= GB {
        format!("{:.2} GiB", b / GB)
    } else if b >= MB {
        format!("{:.2} MiB", b / MB)
    } else if b >= KB {
        format!("{:.2} KiB", b / KB)
    } else {
        format!("{bytes} B")
    }
}
