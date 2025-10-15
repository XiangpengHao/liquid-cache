use arrow::array::{Array, ArrayRef, cast::AsArray};
use arrow::datatypes::Date32Type;
use clap::Parser;
use datafusion::prelude::*;
use futures::StreamExt;
use liquid_cache_storage::liquid_array::{Date32Field, LiquidPrimitiveArray, SqueezedDate32Array};

#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

#[derive(Parser, Debug, Clone)]
#[command(name = "Squeeze Date32 Study")]
#[command(about = "Compare size of full Date32 vs squeezed YEAR/MONTH/DAY on TPCH lineitem")]
struct CliArgs {
    /// Parquet file to read
    #[arg(
        long,
        default_value = "../../benchmark/tpch/data/sf1.0/lineitem.parquet"
    )]
    parquet: String,

    /// Optional row limit for faster runs
    #[arg(long)]
    limit: Option<usize>,

    /// Cargo passes --bench for harness=false binaries; accept it to avoid parse errors
    #[arg(long, default_value = "false")]
    bench: bool,
}

#[tokio::main]
async fn main() {
    let args = CliArgs::parse();

    let mut config = SessionConfig::default().with_batch_size(8192 * 2);
    let options = config.options_mut();
    options.execution.parquet.schema_force_view_types = false;

    let ctx = SessionContext::new_with_config(config);
    ctx.register_parquet("lineitem", &args.parquet, Default::default())
        .await
        .expect("register parquet");

    let cols = ["l_commitdate", "l_receiptdate", "l_shipdate"];

    for col in cols {
        run_for_column(&ctx, col, args.limit).await;
    }
}

async fn run_for_column(ctx: &SessionContext, col: &str, limit: Option<usize>) {
    let sql = if let Some(n) = limit {
        format!("SELECT {} FROM lineitem LIMIT {n}", col)
    } else {
        format!("SELECT {} FROM lineitem", col)
    };
    let df = ctx.sql(&sql).await.expect("create df");
    let mut stream = df.execute_stream().await.expect("execute stream");

    let mut total_rows = 0usize;
    let mut total_arrow_bytes = 0usize;
    let mut total_liquid_bytes = 0usize;
    let mut total_year_bytes = 0usize;
    let mut total_month_bytes = 0usize;
    let mut total_day_bytes = 0usize;

    while let Some(batch_res) = stream.next().await {
        let batch = batch_res.expect("stream batch");
        let arr: ArrayRef = batch.column(0).clone();
        assert_eq!(arr.data_type(), &arrow_schema::DataType::Date32);

        total_rows += arr.len();
        total_arrow_bytes += arr.get_array_memory_size();

        let prim = arr.as_primitive::<Date32Type>().clone();
        let liquid = LiquidPrimitiveArray::<Date32Type>::from_arrow_array(prim.clone());
        total_liquid_bytes += liquid.get_array_memory_size();

        let squeezed_year = SqueezedDate32Array::from_liquid_date32(&liquid, Date32Field::Year);
        let squeezed_month = SqueezedDate32Array::from_liquid_date32(&liquid, Date32Field::Month);
        let squeezed_day = SqueezedDate32Array::from_liquid_date32(&liquid, Date32Field::Day);

        total_year_bytes += squeezed_year.get_array_memory_size();
        total_month_bytes += squeezed_month.get_array_memory_size();
        total_day_bytes += squeezed_day.get_array_memory_size();
    }

    println!(
        "Column {col} on {total_rows} rows:\n  Arrow(Date32): {total_arrow_bytes} bytes\n  Liquid(Date32): {total_liquid_bytes} bytes\n  Squeezed YEAR: {total_year_bytes} bytes\n  Squeezed MONTH: {total_month_bytes} bytes\n  Squeezed DAY: {total_day_bytes} bytes"
    );
}
