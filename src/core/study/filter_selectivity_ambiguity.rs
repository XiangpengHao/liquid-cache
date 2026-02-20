use arrow::array::{Array, StringArray, cast::AsArray};
use arrow::record_batch::RecordBatch;
use arrow_schema::DataType;
use clap::Parser;
use datafusion::prelude::*;
use futures::StreamExt;
use liquid_cache::liquid_array::LiquidByteViewArray;
use liquid_cache::liquid_array::byte_view_array::Comparison;
use liquid_cache::liquid_array::raw::FsstArray;

#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

#[derive(Parser, Debug, Clone)]
#[command(name = "SearchPhrase Filter Selectivity Study")]
#[command(about = "Compute prefix selectivity and ambiguity for ClickBench SearchPhrase filters")]
struct CliArgs {
    /// Parquet file to read.
    #[arg(long, default_value = "../../benchmark/clickbench/data/hits.parquet")]
    parquet: String,

    /// Column to evaluate.
    #[arg(long, default_value = "SearchPhrase")]
    column: String,

    /// Parquet batch size (rows per RecordBatch).
    #[arg(long, default_value_t = 8192 * 2)]
    batch_size: usize,

    /// Optional row limit (useful for faster runs).
    #[arg(long)]
    limit: Option<usize>,

    /// Cargo passes --bench for harness=false binaries; accept it to avoid parse errors.
    #[arg(long, default_value = "false")]
    bench: bool,
}

#[derive(Debug, Clone, Copy)]
struct FilterSpec {
    generation: u64,
    count: usize,
    literal: &'static str,
}

#[derive(Debug)]
struct FilterQuery {
    spec: FilterSpec,
    op: Comparison,
    needle: Vec<u8>,
    selected_rows: usize,
    ambiguous_rows: usize,
    distinct_rows: usize,
}

struct ScanConfig<'a> {
    column: &'a str,
    limit: Option<usize>,
}

const FILTERS: &[FilterSpec] = &[
    FilterSpec {
        generation: 2,
        count: 2,
        literal: "in-grid madonnasekret@yandex спб",
    },
    FilterSpec {
        generation: 3,
        count: 3,
        literal: "erection пермь курском звучка штильники скривода моряков адлера",
    },
    FilterSpec {
        generation: 4,
        count: 36,
        literal: "0б1 купить билето.одноклавович и сотряд",
    },
    FilterSpec {
        generation: 5,
        count: 3,
        literal: "0б1 купить бамбарды",
    },
    FilterSpec {
        generation: 6,
        count: 1,
        literal: "0986 года на скрыть",
    },
    FilterSpec {
        generation: 7,
        count: 1,
        literal: "03-85 серпухоль по краснодар",
    },
    FilterSpec {
        generation: 8,
        count: 1,
        literal: "0б1 купить клин-себационные моя мультики на карта",
    },
    FilterSpec {
        generation: 9,
        count: 17,
        literal: "03-85 серпухоль по краснодар",
    },
    FilterSpec {
        generation: 10,
        count: 3,
        literal: "(http://kommer aspire",
    },
    FilterSpec {
        generation: 11,
        count: 19,
        literal: "(http://kommedium=cpc&utm_source=main происход",
    },
    FilterSpec {
        generation: 12,
        count: 1,
        literal: "(http://kommedium=cpc&utm_source=main произвестивозачать на автомобиле",
    },
    FilterSpec {
        generation: 13,
        count: 9,
        literal: "(http://kommed acce maximum 5*",
    },
    FilterSpec {
        generation: 14,
        count: 2,
        literal: "'kbnysq rbyjgjbcr ghjbpdjlcndf dbltj ujhs",
    },
    FilterSpec {
        generation: 15,
        count: 25,
        literal: "'kbnysq gbhj;rb gjkmpjdfz",
    },
    FilterSpec {
        generation: 16,
        count: 155,
        literal: "'kbnysq ctrcfylhtq d vfrcbvev",
    },
    FilterSpec {
        generation: 17,
        count: 96,
        literal: "'kbnysq ctrcdfqafq yjxm. usb-накомсомосква",
    },
    FilterSpec {
        generation: 18,
        count: 131,
        literal: "'kbnysq ctrcdfqafq yf cello (feat nival telligar",
    },
    FilterSpec {
        generation: 19,
        count: 84,
        literal: "'kbnysq ctrcdfqafq lj;lz",
    },
    FilterSpec {
        generation: 20,
        count: 155,
        literal: "'kbnysq cgtrnjh xperia mazda trash",
    },
    FilterSpec {
        generation: 21,
        count: 294,
        literal: "'kbnysq cgtrnjh xperia mazda temperia ua авторіа.ua автория",
    },
    FilterSpec {
        generation: 22,
        count: 254,
        literal: "'kbnysq cgtrnjh xperia mazda te editional",
    },
    FilterSpec {
        generation: 23,
        count: 543,
        literal: "'kbnyjuj ljvfiyb[ pfcnhjqrb bp vfccfl 634 картак",
    },
    FilterSpec {
        generation: 24,
        count: 509,
        literal: "'kbnyjuj ljvfiyb[ pfcnhjqcndj vfr 16523-28 днепродажа",
    },
    FilterSpec {
        generation: 25,
        count: 2963,
        literal: "'kbnyjuj ljvfiyb[ pfcjkbndf",
    },
    FilterSpec {
        generation: 26,
        count: 284,
        literal: "'exist.androit dogs/tags tuning dogg",
    },
    FilterSpec {
        generation: 27,
        count: 250,
        literal: "'exist.androit dogs купить памятников шарарок сатист онлайн",
    },
    FilterSpec {
        generation: 28,
        count: 1,
        literal: "'exist.androit dogs ever",
    },
    FilterSpec {
        generation: 29,
        count: 226,
        literal: "'cgjhnhtnm jnpsds madded",
    },
    FilterSpec {
        generation: 30,
        count: 177,
        literal: "'atrn dynami инстроги на добразовая",
    },
    FilterSpec {
        generation: 31,
        count: 5193,
        literal: "$dmini+7",
    },
    FilterSpec {
        generation: 32,
        count: 144,
        literal: "$_posten of greenjera mi 300 мегафонов (1944-105 отзывы",
    },
    FilterSpec {
        generation: 36,
        count: 62,
        literal: "$dmini+7",
    },
    FilterSpec {
        generation: 37,
        count: 994,
        literal: "$_posten of greenjera mi 300 мегафонов (1944-105 отзывы",
    },
];

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

    let mut filters = build_filters();
    let scan_config = ScanConfig {
        column: &args.column,
        limit: args.limit,
    };
    scan_column(&ctx, &scan_config, &mut filters).await;

    for filter in &filters {
        let selectivity = filter.selected_rows as f64 / filter.distinct_rows as f64;
        let ambiguous_rate = filter.ambiguous_rows as f64 / filter.distinct_rows as f64;

        println!(
            r#""gen": {}, "filter": "{}", "count": {}, "selectivity": {}, "ambiguous_rate": {}"#,
            filter.spec.generation,
            filter.spec.literal,
            filter.spec.count,
            selectivity,
            ambiguous_rate
        );
    }
}

fn build_filters() -> Vec<FilterQuery> {
    FILTERS
        .iter()
        .map(|spec| FilterQuery {
            spec: *spec,
            op: Comparison::Lt,
            needle: spec.literal.as_bytes().to_vec(),
            selected_rows: 0,
            ambiguous_rows: 0,
            distinct_rows: 0,
        })
        .collect()
}

async fn scan_column(
    ctx: &SessionContext,
    config: &ScanConfig<'_>,
    filters: &mut [FilterQuery],
) -> usize {
    let sql = if let Some(limit) = config.limit {
        format!("SELECT \"{}\" FROM \"hits\" LIMIT {}", config.column, limit)
    } else {
        format!("SELECT \"{}\" FROM \"hits\"", config.column)
    };
    let df = ctx.sql(&sql).await.expect("create df");
    let mut stream = df.execute_stream().await.expect("execute stream");

    let mut total_rows = 0usize;
    while let Some(batch) = stream.next().await {
        let batch = batch.expect("fetch batch");
        if batch.num_rows() == 0 {
            continue;
        }
        let array = column_as_string_array(&batch, 0);
        total_rows += array.len();

        let (_compressor, byte_view) = LiquidByteViewArray::<FsstArray>::train_from_arrow(&array);
        for filter in filters.iter_mut() {
            let (selected_rows, ambiguous_rows, distinct_rows) =
                byte_view.prefix_compare_counts(&filter.needle, &filter.op);
            filter.selected_rows += selected_rows;
            filter.ambiguous_rows += ambiguous_rows;
            filter.distinct_rows += distinct_rows;
        }
    }

    total_rows
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
