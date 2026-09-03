use std::{fs::File, sync::Arc};

use arrow::{
    array::{AsArray, Int32Array, StructArray},
    datatypes::{DataType, Field, Fields, Schema},
    record_batch::RecordBatch,
};
use datafusion::prelude::{ParquetReadOptions, SessionConfig};
use parquet::arrow::ArrowWriter;
use tempfile::TempDir;

use crate::LiquidCacheLocalBuilder;

fn write_people(path: &std::path::Path) {
    let person_fields = Fields::from(vec![Field::new("age", DataType::Int32, false)]);
    let schema = Arc::new(Schema::new(vec![
        Field::new("person", DataType::Struct(person_fields.clone()), false),
        Field::new("cohort", DataType::Int32, false),
    ]));
    let person = StructArray::new(
        person_fields,
        vec![Arc::new(Int32Array::from(vec![1, 1, 2, 2]))],
        None,
    );
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(person),
            Arc::new(Int32Array::from(vec![10, 20, 20, 30])),
        ],
    )
    .unwrap();

    let mut writer = ArrowWriter::try_new(File::create(path).unwrap(), schema, None).unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();
}

async fn query_rows(sql: &str) -> Vec<(i32, i32)> {
    let temp_dir = TempDir::new().unwrap();
    let parquet_path = temp_dir.path().join("people.parquet");
    write_people(&parquet_path);

    let (ctx, _) = LiquidCacheLocalBuilder::new()
        .with_cache_dir(temp_dir.path().to_path_buf())
        .build(SessionConfig::new())
        .await
        .unwrap();
    ctx.register_parquet(
        "people",
        parquet_path.to_str().unwrap(),
        ParquetReadOptions::default(),
    )
    .await
    .unwrap();

    ctx.sql(sql)
        .await
        .unwrap()
        .collect()
        .await
        .unwrap()
        .into_iter()
        .flat_map(|batch| {
            let ages = batch
                .column(0)
                .as_primitive::<arrow::datatypes::Int32Type>();
            let cohorts = batch
                .column(1)
                .as_primitive::<arrow::datatypes::Int32Type>();
            (0..batch.num_rows())
                .map(|row| (ages.value(row), cohorts.value(row)))
                .collect::<Vec<_>>()
        })
        .collect()
}

#[tokio::test]
async fn nested_struct_field_filter_keeps_only_matching_rows() {
    let rows = query_rows(
        "SELECT person['age'], cohort \
         FROM people WHERE person['age'] = 2 ORDER BY cohort",
    )
    .await;

    assert_eq!(rows, vec![(2, 20), (2, 30)]);
}

#[tokio::test]
async fn nested_and_primitive_filters_are_both_applied() {
    let rows = query_rows(
        "SELECT person['age'], cohort \
         FROM people WHERE person['age'] = 2 AND cohort = 20",
    )
    .await;

    assert_eq!(rows, vec![(2, 20)]);
}
