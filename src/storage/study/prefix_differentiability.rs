use ahash::AHashSet;
use arrow::array::{Array, StringArray, cast::AsArray};
use arrow::record_batch::RecordBatch;
use arrow_schema::DataType;
use clap::Parser;
use datafusion::prelude::*;
use futures::StreamExt;

#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

#[derive(Parser, Debug, Clone)]
#[command(name = "Prefix Differentiability Study")]
#[command(about = "Compute average prefix differentiability per batch for ClickBench columns")]
struct CliArgs {
    /// Parquet file to read.
    #[arg(long, default_value = "../../benchmark/clickbench/data/hits.parquet")]
    parquet: String,

    /// Columns to process (comma-separated).
    #[arg(long, value_delimiter = ',', default_value = "SearchPhrase,URL,Title")]
    columns: Vec<String>,

    /// Parquet batch size (rows per RecordBatch).
    #[arg(long, default_value_t = 8192)]
    batch_size: usize,

    /// Optional row limit (useful for faster runs).
    #[arg(long)]
    limit: Option<usize>,

    /// Cargo passes --bench for harness=false binaries; accept it to avoid parse errors.
    #[arg(long, default_value = "false")]
    bench: bool,
}

struct ScanConfig<'a> {
    column: &'a str,
    limit: Option<usize>,
    prefix_lengths: &'a [usize],
}

struct PrefixStats {
    prefix_lengths: Vec<usize>,
    sum_ratios: Vec<f64>,
    batches: usize,
}

impl PrefixStats {
    fn new(prefix_lengths: &[usize]) -> Self {
        Self {
            prefix_lengths: prefix_lengths.to_vec(),
            sum_ratios: vec![0.0; prefix_lengths.len()],
            batches: 0,
        }
    }

    fn add_batch(&mut self, ratios: &[f64]) {
        for (idx, ratio) in ratios.iter().enumerate() {
            self.sum_ratios[idx] += ratio;
        }
        self.batches += 1;
    }
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

    let prefix_lengths: Vec<usize> = (1..=16).collect();
    for column in &args.columns {
        let scan_config = ScanConfig {
            column,
            limit: args.limit,
            prefix_lengths: &prefix_lengths,
        };
        let stats = scan_column(&ctx, &scan_config).await;

        if stats.batches == 0 {
            println!("Column {column}: no rows");
            continue;
        }

        println!("Column {column} (batches: {})", stats.batches);
        for (prefix_len, sum_ratio) in stats.prefix_lengths.iter().zip(stats.sum_ratios.iter()) {
            let avg = sum_ratio / stats.batches as f64;
            println!("  prefix {:>2} -> avg differentiability {:.6}", prefix_len, avg);
        }
    }
}

async fn scan_column(ctx: &SessionContext, config: &ScanConfig<'_>) -> PrefixStats {
    let sql = if let Some(limit) = config.limit {
        format!("SELECT \"{}\" FROM \"hits\" LIMIT {}", config.column, limit)
    } else {
        format!("SELECT \"{}\" FROM \"hits\"", config.column)
    };
    let df = ctx.sql(&sql).await.expect("create df");
    let mut stream = df.execute_stream().await.expect("execute stream");

    let mut stats = PrefixStats::new(config.prefix_lengths);
    while let Some(batch) = stream.next().await {
        let batch = batch.expect("fetch batch");
        if batch.num_rows() == 0 {
            continue;
        }
        let array = column_as_string_array(&batch, 0);
        if let Some(ratios) = batch_prefix_differentiability(&array, config.prefix_lengths) {
            stats.add_batch(&ratios);
        }
    }

    stats
}

fn batch_prefix_differentiability(
    array: &StringArray,
    prefix_lengths: &[usize],
) -> Option<Vec<f64>> {
    let mut values = Vec::with_capacity(array.len());
    for row in 0..array.len() {
        if array.is_null(row) {
            continue;
        }
        let value = array.value(row);
        if value.is_empty() {
            continue;
        }
        values.push(value.as_bytes().to_vec());
    }

    if values.is_empty() {
        return None;
    }

    let common_prefix_len = common_prefix_len(&values);
    let mut sets: Vec<AHashSet<Vec<u8>>> = prefix_lengths
        .iter()
        .map(|_| AHashSet::with_capacity(values.len()))
        .collect();
    let mut unique_values: AHashSet<Vec<u8>> = AHashSet::with_capacity(values.len());

    for value in values {
        let suffix = &value[common_prefix_len..];
        for (idx, prefix_len) in prefix_lengths.iter().enumerate() {
            let end = suffix.len().min(*prefix_len);
            sets[idx].insert(suffix[..end].to_vec());
        }
        unique_values.insert(suffix.to_vec());
    }

    let total_rows = unique_values.len() as f64;
    if total_rows == 0.0 {
        return None;
    }
    Some(
        sets.iter()
            .map(|set| set.len() as f64 / total_rows)
            .collect(),
    )
}

fn common_prefix_len(values: &[Vec<u8>]) -> usize {
    if values.is_empty() {
        return 0;
    }
    let mut prefix_len = values[0].len();
    for value in values.iter().skip(1) {
        let mut idx = 0;
        let max_len = prefix_len.min(value.len());
        while idx < max_len && values[0][idx] == value[idx] {
            idx += 1;
        }
        prefix_len = idx;
        if prefix_len == 0 {
            break;
        }
    }
    prefix_len
}

fn column_as_string_array(batch: &RecordBatch, index: usize) -> StringArray {
    let array = batch.column(index).clone();
    let array = if array.data_type() == &DataType::Utf8 {
        array
    } else {
        arrow::compute::cast(&array, &DataType::Utf8).expect("cast to Utf8")
    };
    array.as_string::<i32>().clone()
}
