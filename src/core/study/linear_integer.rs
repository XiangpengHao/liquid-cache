use std::time::Instant;

use arrow::array::{Array, ArrayRef, cast::AsArray};
use arrow::datatypes::DataType;
use clap::Parser;
use datafusion::prelude::*;
use futures::StreamExt;
use liquid_cache::liquid_array::{
    LiquidArray, LiquidLinearArray, LiquidPrimitiveArray, LiquidPrimitiveType,
};

#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

#[derive(Parser, Debug, Default, Clone)]
#[command(name = "Linear Integer Study")]
#[command(about = "Compare primitive bitpacking vs linear-model encoding for integer columns")]
struct CliArgs {
    /// Parquet file to read
    #[arg(long, default_value = "../../benchmark/clickbench/data/hits.parquet")]
    parquet: String,

    /// Comma-separated list of columns. If not set, auto-detect integer-like columns.
    #[arg(long)]
    columns: Option<String>,

    /// Optional row limit for each column (useful for faster runs)
    #[arg(long)]
    limit: Option<usize>,

    /// Cargo passes --bench for harness=false binaries; accept it to avoid parse errors
    #[arg(long, default_value = "false")]
    bench: bool,
}

#[derive(Default, Debug, Clone)]
struct Stats {
    rows: usize,
    arrow_bytes: usize,
    prim_bytes: usize,
    linear_bytes: usize,
    prim_encode_sec: f64,
    prim_decode_sec: f64,
    linear_encode_sec: f64,
    linear_decode_sec: f64,
}

impl Stats {
    fn add(&mut self, other: &Stats) {
        self.rows += other.rows;
        self.arrow_bytes += other.arrow_bytes;
        self.prim_bytes += other.prim_bytes;
        self.linear_bytes += other.linear_bytes;
        self.prim_encode_sec += other.prim_encode_sec;
        self.prim_decode_sec += other.prim_decode_sec;
        self.linear_encode_sec += other.linear_encode_sec;
        self.linear_decode_sec += other.linear_decode_sec;
    }
}

#[tokio::main]
async fn main() {
    let args = CliArgs::parse();

    let mut config = SessionConfig::default().with_batch_size(8192 * 2);
    let options = config.options_mut();
    options.execution.parquet.schema_force_view_types = false;

    let ctx = SessionContext::new_with_config(config);
    ctx.register_parquet("hits", &args.parquet, Default::default())
        .await
        .expect("register parquet");

    // Identify columns
    let columns = if let Some(cols) = args.columns.clone() {
        cols.split(',')
            .map(|s| s.trim().to_string())
            .collect::<Vec<_>>()
    } else {
        autodetect_integer_columns(&ctx).await
    };

    println!("Linear Integer Study on {} column(s)", columns.len());

    let mut grand = Stats::default();
    for col in columns {
        let stats = run_for_column(&ctx, &col, args.limit).await;
        println!(
            "Column: {col}\n  rows: {}\n  sizes (bytes) -> arrow: {}, prim: {}, linear: {}\n  encode (s) -> prim: {:.6}, linear: {:.6}\n  decode (s) -> prim: {:.6}, linear: {:.6}",
            stats.rows,
            stats.arrow_bytes,
            stats.prim_bytes,
            stats.linear_bytes,
            stats.prim_encode_sec,
            stats.linear_encode_sec,
            stats.prim_decode_sec,
            stats.linear_decode_sec
        );
        grand.add(&stats);
    }

    println!(
        "TOTAL\n  rows: {}\n  sizes (bytes) -> arrow: {}, prim: {}, linear: {}\n  encode (s) -> prim: {:.6}, linear: {:.6}\n  decode (s) -> prim: {:.6}, linear: {:.6}",
        grand.rows,
        grand.arrow_bytes,
        grand.prim_bytes,
        grand.linear_bytes,
        grand.prim_encode_sec,
        grand.linear_encode_sec,
        grand.prim_decode_sec,
        grand.linear_decode_sec
    );
}

async fn autodetect_integer_columns(ctx: &SessionContext) -> Vec<String> {
    // Run a small query to fetch schema
    let df = ctx.sql("SELECT * FROM \"hits\" LIMIT 1").await.unwrap();
    let batches = df.collect().await.unwrap();
    let schema = batches[0].schema();
    let mut cols = Vec::new();
    for f in schema.fields() {
        if is_integer_like(f.data_type()) {
            cols.push(f.name().to_string());
        }
    }
    cols
}

fn is_integer_like(dt: &DataType) -> bool {
    matches!(
        dt,
        DataType::Int8
            | DataType::Int16
            | DataType::Int32
            | DataType::Int64
            | DataType::UInt8
            | DataType::UInt16
            | DataType::UInt32
            | DataType::UInt64
            | DataType::Date32
            | DataType::Date64
    )
}

async fn run_for_column(ctx: &SessionContext, column: &str, limit: Option<usize>) -> Stats {
    let sql = if let Some(n) = limit {
        format!("SELECT \"{column}\" FROM \"hits\" LIMIT {n}")
    } else {
        format!("SELECT \"{column}\" FROM \"hits\"")
    };
    let df = ctx.sql(&sql).await.expect("create df");
    let mut stream = df.execute_stream().await.expect("execute stream");

    let mut stats = Stats::default();
    while let Some(batch_res) = stream.next().await {
        let batch = batch_res.expect("stream batch");
        let array: ArrayRef = batch.column(0).clone();
        stats.rows += array.len();
        stats.arrow_bytes += array.get_array_memory_size();

        let dt = array.data_type().clone();
        match dt {
            DataType::Int8 => accumulate::<arrow::datatypes::Int8Type>(&array, &mut stats),
            DataType::Int16 => accumulate::<arrow::datatypes::Int16Type>(&array, &mut stats),
            DataType::Int32 => accumulate::<arrow::datatypes::Int32Type>(&array, &mut stats),
            DataType::Int64 => accumulate::<arrow::datatypes::Int64Type>(&array, &mut stats),
            DataType::UInt8 => accumulate::<arrow::datatypes::UInt8Type>(&array, &mut stats),
            DataType::UInt16 => accumulate::<arrow::datatypes::UInt16Type>(&array, &mut stats),
            DataType::UInt32 => accumulate::<arrow::datatypes::UInt32Type>(&array, &mut stats),
            DataType::UInt64 => accumulate::<arrow::datatypes::UInt64Type>(&array, &mut stats),
            DataType::Date32 => accumulate::<arrow::datatypes::Date32Type>(&array, &mut stats),
            DataType::Date64 => accumulate::<arrow::datatypes::Date64Type>(&array, &mut stats),
            _ => {}
        }
    }

    stats
}

fn accumulate<T: LiquidPrimitiveType>(array: &ArrayRef, stats: &mut Stats)
where
    <T as arrow::array::ArrowPrimitiveType>::Native: num_traits::cast::AsPrimitive<f64>
        + num_traits::FromPrimitive
        + num_traits::bounds::Bounded,
{
    let prim = array.as_primitive::<T>().clone();

    // Primitive bitpacking encode
    let t0 = Instant::now();
    let lp = LiquidPrimitiveArray::<T>::from_arrow_array(prim.clone());
    let enc_prim = t0.elapsed().as_secs_f64();
    let prim_bytes = lp.get_array_memory_size();

    // Primitive decode
    let t0 = Instant::now();
    let _ = lp.to_arrow_array();
    let dec_prim = t0.elapsed().as_secs_f64();

    // Linear encode
    let t0 = Instant::now();
    let ll = LiquidLinearArray::<T>::from_arrow_array(prim);
    let enc_linear = t0.elapsed().as_secs_f64();
    let linear_bytes = ll.get_array_memory_size();

    // Linear decode
    let t0 = Instant::now();
    let _ = ll.to_arrow_array();
    let dec_linear = t0.elapsed().as_secs_f64();

    stats.prim_bytes += prim_bytes;
    stats.linear_bytes += linear_bytes;
    stats.prim_encode_sec += enc_prim;
    stats.prim_decode_sec += dec_prim;
    stats.linear_encode_sec += enc_linear;
    stats.linear_decode_sec += dec_linear;
}
