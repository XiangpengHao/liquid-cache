use clap::Parser;
use datafusion::dataframe::DataFrameWriteOptions;
use datafusion::prelude::*;
use std::path::Path;

#[derive(Parser)]
#[command(name = "convert_parquet")]
#[command(about = "Converts CSV/TSV files to Parquet format")]
struct Args {
    /// Input path to the CSV or TSV file
    input_path: String,
}

#[tokio::main]
async fn main() -> datafusion::error::Result<()> {
    let args = Args::parse();
    convert_to_parquet(&args.input_path).await
}

async fn convert_to_parquet(input_path: &str) -> datafusion::error::Result<()> {
    let ctx = SessionContext::new();
    let path = Path::new(input_path);

    let output_file = path
        .file_name()
        .unwrap()
        .to_str()
        .unwrap()
        .replace(".csv", ".parquet")
        .replace(".tsv", ".parquet");

    let df = ctx
        .read_csv(
            input_path,
            CsvReadOptions::new()
                .has_header(false) // later set to right headers
                .delimiter(b'|'),
        )
        .await?;

    let output_path = Path::new("data/").join(output_file);
    df.write_parquet(
        &output_path.to_string_lossy(),
        DataFrameWriteOptions::new(),
        None,
    )
    .await?;

    let parquet_df = ctx
        .read_parquet(
            output_path.to_string_lossy().as_ref(),
            ParquetReadOptions::default(),
        )
        .await?;
    println!("Parquet Schema: {:?}", parquet_df.schema());

    Ok(())
}
