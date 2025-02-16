use parquet::arrow::ArrowWriter;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use parquet::file::properties::WriterProperties;
use std::fs::File;

fn main() {
    let file = File::open("benchmark/data/hits.parquet").unwrap();
    let builder = ParquetRecordBatchReaderBuilder::try_new(file).unwrap();

    let compression_alg = builder.metadata().row_groups()[0].columns()[0].compression();

    let mut arrow_reader = builder.with_batch_size(8192 * 3).build().unwrap();

    let batch_one = arrow_reader.next().unwrap().unwrap();
    let batch_two = arrow_reader.next().unwrap().unwrap();

    let props = WriterProperties::builder()
        .set_compression(compression_alg)
        .set_max_row_group_size(8192 * 3)
        .build();

    let file = File::create("benchmark/data/nano_hits.parquet").unwrap();

    let mut writer = ArrowWriter::try_new(file, batch_one.schema(), Some(props)).unwrap();

    writer
        .write(&batch_one)
        .expect("Writing batch 1 (Full Batch)");
    writer
        .write(&batch_two.slice(0, 10))
        .expect("Writing batch 2 (Small Batch)");

    writer.close().unwrap();
}
