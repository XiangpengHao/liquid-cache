use datafusion::prelude::*;
use datafusion::dataframe::DataFrameWriteOptions;
use std::path::Path;
use tokio;

#[tokio::main]
async fn main() -> datafusion::error::Result<()> {
    let input_path = "";   # <-- change this to your input .csv or .tsv path

    convert_to_parquet(input_path).await
}

async fn convert_to_parquet(input_path: &str) -> datafusion::error::Result<()> {
    let ctx = SessionContext::new();
    let path = Path::new(input_path);

    
    let output_file = path.file_name().unwrap().to_str().unwrap()
    .replace(".csv", ".parquet").replace(".tsv", ".parquet");

    let df = ctx
        .read_csv(
            input_path,
            CsvReadOptions::new()
                .has_header(false) // later set to right headers
                .delimiter(b'|' as u8),
        )
        .await?;

    let output_path = Path::new("data/").join(output_file);
    df.write_parquet(
        &output_path,
        DataFrameWriteOptions::new(),
        None,
    )
    .await?;

    let parquet_df = ctx.read_parquet(&output_path, ParquetReadOptions::default()).await?;
    println!("Parquet Schema: {:?}", parquet_df.schema());

    Ok(())
}
