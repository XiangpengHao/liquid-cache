use arrow::array::{Array, ArrayRef, BooleanArray, PrimitiveArray, cast::AsArray};
use arrow::buffer::BooleanBuffer;
use arrow::datatypes::DataType;
use bytes::Bytes;
use clap::Parser;
use datafusion::prelude::*;
use datafusion::scalar::ScalarValue;
use futures::StreamExt;
use liquid_cache_storage::liquid_array::{
    IntegerSqueezePolicy, LiquidArray, LiquidHybridArray, LiquidPrimitiveArray, LiquidPrimitiveType,
};

#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

#[derive(Parser, Debug, Default, Clone)]
#[command(name = "Squeeze Integer Study")]
#[command(about = "Clamp vs Quantize squeeze on representative integer filters from ClickBench")]
struct CliArgs {
    /// Parquet file to read
    #[arg(long, default_value = "../../benchmark/clickbench/data/hits.parquet")]
    parquet: String,

    /// Optional row limit for each column (useful for faster runs)
    #[arg(long)]
    limit: Option<usize>,

    /// Cargo passes --bench for harness=false binaries; accept it to avoid parse errors
    #[arg(long, default_value = "false")]
    bench: bool,
}

#[derive(Debug, Clone)]
struct FilterCase {
    column: String,
    op: datafusion::logical_expr::Operator,
    scalar: ScalarValue,
}

#[derive(Default, Debug, Clone)]
struct Stats {
    rows: usize,
    arrow_bytes: usize,
    clamp_mem_bytes: usize,
    clamp_disk_bytes: usize,
    quant_mem_bytes: usize,
    quant_disk_bytes: usize,
    // IO incurred (bytes) for try_eval_predicate
    clamp_pred_io_bytes: usize,
    quant_pred_io_bytes: usize,
    // IO incurred (bytes) for get_with_selection (filter_to_arrow)
    clamp_select_io_bytes: usize,
    quant_select_io_bytes: usize,
    // number of predicate cases executed on this column
    pred_cases: usize,
}

impl Stats {
    fn add(&mut self, other: &Stats) {
        self.rows += other.rows;
        self.arrow_bytes += other.arrow_bytes;
        self.clamp_mem_bytes += other.clamp_mem_bytes;
        self.clamp_disk_bytes += other.clamp_disk_bytes;
        self.quant_mem_bytes += other.quant_mem_bytes;
        self.quant_disk_bytes += other.quant_disk_bytes;
        self.clamp_pred_io_bytes += other.clamp_pred_io_bytes;
        self.quant_pred_io_bytes += other.quant_pred_io_bytes;
        self.clamp_select_io_bytes += other.clamp_select_io_bytes;
        self.quant_select_io_bytes += other.quant_select_io_bytes;
        self.pred_cases += other.pred_cases;
    }
}

// Hardcoded representative integer filters based on ClickBench queries
fn representative_integer_filters() -> Vec<FilterCase> {
    use datafusion::logical_expr::Operator as Op;
    vec![
        // SELECT COUNT(*) FROM hits WHERE "AdvEngineID" <> 0;
        FilterCase {
            column: "AdvEngineID".to_string(),
            op: Op::NotEq,
            scalar: ScalarValue::Int64(Some(0)),
        },
        // SELECT "UserID" FROM hits WHERE "UserID" = 435090932899640449;
        FilterCase {
            column: "UserID".to_string(),
            op: Op::Eq,
            scalar: ScalarValue::Int64(Some(435_090_932_899_640_449)),
        },
        // WHERE "CounterID" = 62
        FilterCase {
            column: "CounterID".to_string(),
            op: Op::Eq,
            scalar: ScalarValue::Int64(Some(62)),
        },
        // WHERE "IsRefresh" = 0
        FilterCase {
            column: "IsRefresh".to_string(),
            op: Op::Eq,
            scalar: ScalarValue::Int64(Some(0)),
        },
        // WHERE "DontCountHits" = 0
        FilterCase {
            column: "DontCountHits".to_string(),
            op: Op::Eq,
            scalar: ScalarValue::Int64(Some(0)),
        },
        // WHERE "IsLink" <> 0
        FilterCase {
            column: "IsLink".to_string(),
            op: Op::NotEq,
            scalar: ScalarValue::Int64(Some(0)),
        },
        // WHERE "IsDownload" = 0
        FilterCase {
            column: "IsDownload".to_string(),
            op: Op::Eq,
            scalar: ScalarValue::Int64(Some(0)),
        },
        // WHERE "TraficSourceID" IN (-1, 6)
        FilterCase {
            column: "TraficSourceID".to_string(),
            op: Op::Eq,
            scalar: ScalarValue::Int64(Some(-1)),
        },
        FilterCase {
            column: "TraficSourceID".to_string(),
            op: Op::Eq,
            scalar: ScalarValue::Int64(Some(6)),
        },
        // WHERE "RefererHash" = 3594120000172545465
        FilterCase {
            column: "RefererHash".to_string(),
            op: Op::Eq,
            scalar: ScalarValue::Int64(Some(3_594_120_000_172_545_465)),
        },
        // WHERE "URLHash" = 2868770270353813622
        FilterCase {
            column: "URLHash".to_string(),
            op: Op::Eq,
            scalar: ScalarValue::Int64(Some(2_868_770_270_353_813_622)),
        },
    ]
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

    // Hardcoded representative integer filters inspired by ClickBench queries
    let cases = representative_integer_filters();

    println!("Squeeze Integer Study over {} case(s)", cases.len());

    // Stream the requested columns; for simplicity scan per-case so we only pull needed column
    // and accumulate stats per column then sum totals.
    let mut grand = Stats::default();
    for case in &cases {
        let stats = run_case(&ctx, case, args.limit).await;
        println!(
            "Case on column '{}', op '{:?}', scalar {:?}:\n  rows: {}\n  sizes (bytes) -> arrow: {}, clamp: {} (mem: {}, disk: {}), quant: {} (mem: {}, disk: {})\n  io (bytes)   -> pred: clamp {}, quant {}; select: clamp {}, quant {}",
            case.column,
            case.op,
            case.scalar,
            stats.rows,
            stats.arrow_bytes,
            stats.clamp_mem_bytes + stats.clamp_disk_bytes,
            stats.clamp_mem_bytes,
            stats.clamp_disk_bytes,
            stats.quant_mem_bytes + stats.quant_disk_bytes,
            stats.quant_mem_bytes,
            stats.quant_disk_bytes,
            stats.clamp_pred_io_bytes,
            stats.quant_pred_io_bytes,
            stats.clamp_select_io_bytes,
            stats.quant_select_io_bytes
        );
        grand.add(&stats);
    }

    println!(
        "TOTAL\n  rows: {}\n  sizes (bytes) -> arrow: {}, clamp: {} (mem: {}, disk: {}), quant: {} (mem: {}, disk: {})\n  io (bytes)   -> pred: clamp {}, quant {}; select: clamp {}, quant {}",
        grand.rows,
        grand.arrow_bytes,
        grand.clamp_mem_bytes + grand.clamp_disk_bytes,
        grand.clamp_mem_bytes,
        grand.clamp_disk_bytes,
        grand.quant_mem_bytes + grand.quant_disk_bytes,
        grand.quant_mem_bytes,
        grand.quant_disk_bytes,
        grand.clamp_pred_io_bytes,
        grand.quant_pred_io_bytes,
        grand.clamp_select_io_bytes,
        grand.quant_select_io_bytes
    );
}

// --- end of helpers removed after switching to fixed cases ---

async fn run_case(ctx: &SessionContext, case: &FilterCase, limit: Option<usize>) -> Stats {
    let sql = if let Some(n) = limit {
        format!("SELECT \"{}\" FROM \"hits\" LIMIT {n}", case.column)
    } else {
        format!("SELECT \"{}\" FROM \"hits\"", case.column)
    };
    let df = ctx.sql(&sql).await.expect("create df");
    let mut stream = df.execute_stream().await.expect("execute stream");

    let mut stats = Stats::default();
    while let Some(batch_res) = stream.next().await {
        let batch = batch_res.expect("stream batch");
        let array: ArrayRef = batch.column(0).clone();
        stats.rows += array.len();
        stats.arrow_bytes += array.get_array_memory_size();

        // Dispatch by datatype
        match array.data_type() {
            DataType::Int8 => run_for_array::<arrow::datatypes::Int8Type>(&array, case, &mut stats),
            DataType::Int16 => {
                run_for_array::<arrow::datatypes::Int16Type>(&array, case, &mut stats)
            }
            DataType::Int32 => {
                run_for_array::<arrow::datatypes::Int32Type>(&array, case, &mut stats)
            }
            DataType::Int64 => {
                run_for_array::<arrow::datatypes::Int64Type>(&array, case, &mut stats)
            }
            DataType::UInt8 => {
                run_for_array::<arrow::datatypes::UInt8Type>(&array, case, &mut stats)
            }
            DataType::UInt16 => {
                run_for_array::<arrow::datatypes::UInt16Type>(&array, case, &mut stats)
            }
            DataType::UInt32 => {
                run_for_array::<arrow::datatypes::UInt32Type>(&array, case, &mut stats)
            }
            DataType::UInt64 => {
                run_for_array::<arrow::datatypes::UInt64Type>(&array, case, &mut stats)
            }
            DataType::Date32 => {
                run_for_array::<arrow::datatypes::Date32Type>(&array, case, &mut stats)
            }
            DataType::Date64 => {
                run_for_array::<arrow::datatypes::Date64Type>(&array, case, &mut stats)
            }
            _ => {}
        }
    }

    stats
}

fn run_for_array<T: LiquidPrimitiveType>(array: &ArrayRef, case: &FilterCase, stats: &mut Stats)
where
    <T as arrow::array::ArrowPrimitiveType>::Native: num_traits::cast::AsPrimitive<f64>
        + num_traits::FromPrimitive
        + num_traits::bounds::Bounded,
{
    let prim = array.as_primitive::<T>().clone();

    // Build Liquid primitive array and squeeze with Clamp
    let mut lp = LiquidPrimitiveArray::<T>::from_arrow_array(prim.clone());
    let clamp_hybrid_and_bytes = {
        lp.set_squeeze_policy(IntegerSqueezePolicy::Clamp);
        lp.squeeze()
    };

    // Build Quantize
    let mut lq = LiquidPrimitiveArray::<T>::from_arrow_array(prim.clone());
    let quant_hybrid_and_bytes = {
        lq.set_squeeze_policy(IntegerSqueezePolicy::Quantize);
        lq.squeeze()
    };

    // Size accounting (for squeezable ones)
    if let Some((h, bytes)) = clamp_hybrid_and_bytes.as_ref() {
        stats.clamp_mem_bytes += h.get_array_memory_size();
        stats.clamp_disk_bytes += bytes.len();
    }
    if let Some((h, bytes)) = quant_hybrid_and_bytes.as_ref() {
        stats.quant_mem_bytes += h.get_array_memory_size();
        stats.quant_disk_bytes += bytes.len();
    }

    // Build predicate expr: Column op Literal(scalar)
    use datafusion::physical_plan::expressions::{BinaryExpr, Column, Literal};
    let expr: std::sync::Arc<dyn datafusion::physical_plan::PhysicalExpr> =
        std::sync::Arc::new(BinaryExpr::new(
            std::sync::Arc::new(Column::new("col", 0)),
            case.op,
            std::sync::Arc::new(Literal::new(case.scalar.clone())),
        ));

    let all_true = BooleanBuffer::new_set(prim.len());

    // Evaluate predicate on clamp
    if let Some((hy, full_bytes)) = clamp_hybrid_and_bytes.clone() {
        let (mask, pred_io_bytes) = try_eval_or_fetch(&*hy, &full_bytes, &expr, &all_true);
        stats.clamp_pred_io_bytes += pred_io_bytes;
        let sel = bool_array_to_selection(&mask);
        // Expected selection result from Arrow
        let expected_filtered = filter_expected::<T>(&prim, &case.op, &case.scalar);
        // Try get with selection from hybrid
        let sel_io =
            get_with_selection_or_fetch(&*hy, &full_bytes, &sel, expected_filtered.as_ref());
        stats.clamp_select_io_bytes += sel_io;
    }

    // Evaluate predicate on quantized
    if let Some((hy, full_bytes)) = quant_hybrid_and_bytes.clone() {
        let (mask, pred_io_bytes) = try_eval_or_fetch(&*hy, &full_bytes, &expr, &all_true);
        stats.quant_pred_io_bytes += pred_io_bytes;
        let sel = bool_array_to_selection(&mask);
        // Expected selection result from Arrow
        let expected_filtered = filter_expected::<T>(&prim, &case.op, &case.scalar);
        let sel_io =
            get_with_selection_or_fetch(&*hy, &full_bytes, &sel, expected_filtered.as_ref());
        stats.quant_select_io_bytes += sel_io;
    }

    stats.pred_cases += 1;
}

fn try_eval_or_fetch(
    hybrid: &dyn LiquidHybridArray,
    full_bytes: &Bytes,
    expr: &std::sync::Arc<dyn datafusion::physical_plan::PhysicalExpr>,
    filter: &BooleanBuffer,
) -> (BooleanArray, usize) {
    match hybrid.try_eval_predicate(expr, filter) {
        Ok(Some(mask)) => (mask, 0),
        Ok(None) => {
            // Not supported: materialize and compute expected via Arrow
            let io = hybrid.to_liquid();
            let slice = full_bytes.slice(io.range().start as usize..io.range().end as usize);
            let liq = hybrid.soak(slice);
            let arr = liq.to_arrow_array();
            let mask = eval_on_arrow(&arr, expr);
            (mask, (io.range().end - io.range().start) as usize)
        }
        Err(io) => {
            let slice = full_bytes.slice(io.range().start as usize..io.range().end as usize);
            let liq = hybrid.soak(slice);
            let arr = liq.to_arrow_array();
            let mask = eval_on_arrow(&arr, expr);
            (mask, (io.range().end - io.range().start) as usize)
        }
    }
}

fn get_with_selection_or_fetch(
    hybrid: &dyn LiquidHybridArray,
    full_bytes: &Bytes,
    selection: &BooleanBuffer,
    expected: &dyn Array,
) -> usize {
    match hybrid.filter(selection) {
        Ok(arr) => {
            assert_eq!(arr.as_ref(), expected);
            0
        }
        Err(io) => {
            let slice = full_bytes.slice(io.range().start as usize..io.range().end as usize);
            let liq = hybrid.soak(slice);
            let arr = liq.filter(selection);
            assert_eq!(arr.as_ref(), expected);
            (io.range().end - io.range().start) as usize
        }
    }
}

fn eval_on_arrow(
    array: &ArrayRef,
    expr: &std::sync::Arc<dyn datafusion::physical_plan::PhysicalExpr>,
) -> BooleanArray {
    use arrow::compute::{cast, kernels};
    use datafusion::logical_expr::ColumnarValue;
    use datafusion::physical_expr_common::datum::apply_cmp;
    use datafusion::physical_plan::expressions::{BinaryExpr, Literal};

    if let Some(be) = expr.as_any().downcast_ref::<BinaryExpr>()
        && let Some(lit) = be.right().as_any().downcast_ref::<Literal>()
    {
        let target_dt = scalar_data_type(lit.value()).unwrap_or_else(|| array.data_type().clone());
        let lhs_arr = if &target_dt == array.data_type() {
            array.clone()
        } else {
            cast(array, &target_dt).expect("cast lhs for comparison")
        };
        let lhs = ColumnarValue::Array(lhs_arr);
        let rhs = ColumnarValue::Scalar(lit.value().clone());
        let res = match be.op() {
            datafusion::logical_expr::Operator::Eq => apply_cmp(&lhs, &rhs, kernels::cmp::eq),
            datafusion::logical_expr::Operator::NotEq => apply_cmp(&lhs, &rhs, kernels::cmp::neq),
            datafusion::logical_expr::Operator::Lt => apply_cmp(&lhs, &rhs, kernels::cmp::lt),
            datafusion::logical_expr::Operator::LtEq => apply_cmp(&lhs, &rhs, kernels::cmp::lt_eq),
            datafusion::logical_expr::Operator::Gt => apply_cmp(&lhs, &rhs, kernels::cmp::gt),
            datafusion::logical_expr::Operator::GtEq => apply_cmp(&lhs, &rhs, kernels::cmp::gt_eq),
            _ => panic!("unsupported operator"),
        }
        .expect("cmp ok");
        let arr = res.into_array(array.len()).unwrap();
        arr.as_boolean().clone()
    } else {
        panic!("unexpected expression kind for numeric predicate")
    }
}

fn bool_array_to_selection(mask: &BooleanArray) -> BooleanBuffer {
    // selection must be non-nullable; treat nulls as false
    let iter = (0..mask.len()).map(|i| mask.is_valid(i) && mask.value(i));
    BooleanBuffer::from_iter(iter)
}

fn filter_expected<T: arrow::array::ArrowPrimitiveType>(
    prim: &PrimitiveArray<T>,
    op: &datafusion::logical_expr::Operator,
    scalar: &ScalarValue,
) -> ArrayRef {
    use datafusion::physical_plan::expressions::{BinaryExpr, Column, Literal};
    // Build the same predicate expr and evaluate mask on Arrow (eval_on_arrow may cast internally).
    let arr: ArrayRef = std::sync::Arc::new(prim.clone());
    let expr: std::sync::Arc<dyn datafusion::physical_plan::PhysicalExpr> =
        std::sync::Arc::new(BinaryExpr::new(
            std::sync::Arc::new(Column::new("col", 0)),
            *op,
            std::sync::Arc::new(Literal::new(scalar.clone())),
        ));
    let mask = eval_on_arrow(&arr, &expr);
    // Apply mask to the original-typed array to keep dtype identical to hybrid’s result
    arrow::compute::kernels::filter::filter(&arr, &mask).unwrap()
}

fn scalar_data_type(sv: &ScalarValue) -> Option<DataType> {
    Some(match sv {
        ScalarValue::Int8(_) => DataType::Int8,
        ScalarValue::Int16(_) => DataType::Int16,
        ScalarValue::Int32(_) => DataType::Int32,
        ScalarValue::Int64(_) => DataType::Int64,
        ScalarValue::UInt8(_) => DataType::UInt8,
        ScalarValue::UInt16(_) => DataType::UInt16,
        ScalarValue::UInt32(_) => DataType::UInt32,
        ScalarValue::UInt64(_) => DataType::UInt64,
        ScalarValue::Date32(_) => DataType::Date32,
        ScalarValue::Date64(_) => DataType::Date64,
        _ => return None,
    })
}
