use std::fs::File;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use liquid_cache_common::CacheMode;

use crate::tests::run_sql;

fn gen_parquet(dir: impl AsRef<Path>) -> PathBuf {
    use arrow::array::UInt32Array;
    use arrow::datatypes::{DataType, Field, Schema};
    use arrow::record_batch::RecordBatch;
    use parquet::arrow::ArrowWriter;
    use parquet::file::properties::WriterProperties;
    let temp_path = dir.as_ref().join("parquet_page_index.parquet");
    let file = File::create(&temp_path).unwrap();
    let schema = Arc::new(Schema::new(vec![Field::new("id", DataType::UInt32, false)]));
    let id_array = UInt32Array::from_iter_values(0..200_000);
    let id_batch = RecordBatch::try_new(Arc::clone(&schema), vec![Arc::new(id_array)]).unwrap();
    let props = WriterProperties::builder()
        .set_offset_index_disabled(false)
        .build();
    let mut writer = ArrowWriter::try_new(file, Arc::clone(&schema), Some(props)).unwrap();
    writer.write(&id_batch).unwrap();
    writer.into_inner().unwrap();
    temp_path
}

#[tokio::test]
async fn test_parquet_with_page_index() {
    let temp_dir = tempfile::tempdir().unwrap();
    let file = gen_parquet(&temp_dir);
    let file_path = file.to_str().unwrap();

    let result = run_sql(
        "SELECT * FROM hits WHERE id = 0",
        CacheMode::LiquidEagerTranscode,
        1000,
        file_path,
    )
    .await;
    insta::assert_snapshot!(result);
}
