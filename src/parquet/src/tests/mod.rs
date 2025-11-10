//! This file is mostly used for Xiangpeng and llm to understand how variant works.
//! todo: DELETE ME when variant is implemented.

use arrow::array::{ArrayRef, RecordBatch, StringArray, StructArray};
use arrow::array::{AsArray, RecordBatchReader};
use arrow::datatypes::{DataType, Field, Fields, Schema};
use bytes::Bytes;
use parquet::arrow::ArrowWriter;
use parquet::arrow::arrow_reader::ArrowReaderBuilder;
use parquet::variant::VariantType;
use parquet::variant::{
    GetOptions, ShortString, Variant, VariantArray, VariantPath, VariantPathElement,
    json_to_variant, variant_get,
};
use std::sync::Arc;

fn write_variant_data() -> Vec<u8> {
    let input = StringArray::from(vec![
        Some(r#"{"name": "Alice", "age": 30}"#),
        Some(r#"{"name": "Bob", "age": 25, "address": {"city": "New York"}}"#),
        Some(r#"{"name": "Charlie", "age": 30, "address": {"zipcode": 90001}}"#),
        None,
        Some("{}"),
    ]);
    let input_array: ArrayRef = Arc::new(input);
    let array = json_to_variant(&input_array).unwrap();
    let schema = Arc::new(Schema::new(vec![array.field("data")]));
    let batch = RecordBatch::try_new(schema, vec![ArrayRef::from(array)]).unwrap();
    let mut bytes = Vec::new();
    let mut writer = ArrowWriter::try_new(&mut bytes, batch.schema(), None).unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();
    bytes
}

fn read_variant_file(bytes: Bytes) -> (VariantArray, ArrayRef) {
    let mut reader = ArrowReaderBuilder::try_new(bytes).unwrap().build().unwrap();
    let schema = reader.schema();
    let field = schema.field_with_name("data").unwrap();
    assert!(field.try_extension_type::<VariantType>().is_ok());

    let batch = reader.next().unwrap().unwrap();
    let col = batch.column_by_name("data").unwrap();
    let variant_array = VariantArray::try_new(col).unwrap();
    (variant_array, col.clone())
}

fn setup_variant() -> (VariantArray, ArrayRef) {
    let bytes = write_variant_data();
    read_variant_file(Bytes::from(bytes))
}

#[test]
fn read_json_variant() {
    let (variant_array, col) = setup_variant();
    assert_eq!(variant_array.len(), 5);
    {
        let var_value = variant_array.value(0);
        println!("First value: {:?}", var_value);

        let name = var_value.get_object_field("name").unwrap();
        assert_eq!(
            name,
            Variant::ShortString(ShortString::try_new("Alice").unwrap())
        );

        let age = var_value.get_object_field("age").unwrap();
        assert_eq!(age, Variant::Int8(30));

        let address = var_value.get_object_field("address");
        assert_eq!(address, None);
    }
    {
        let var_value = variant_array.value(1);
        println!("Second value: {:?}", var_value);

        let address = var_value.get_object_field("address").unwrap();
        let city = address.get_object_field("city").unwrap();
        assert_eq!(
            city,
            Variant::ShortString(ShortString::try_new("New York").unwrap())
        );
    }
    {
        let result = variant_get(
            &col,
            GetOptions::new_with_path(VariantPath::new(vec![
                VariantPathElement::field("address"),
                VariantPathElement::field("city"),
            ])),
        )
        .unwrap();
        let variant_array = VariantArray::try_new(&result).unwrap();
        for i in 0..variant_array.len() {
            println!("Value {}: {:?}", i, variant_array.value(i));
        }
    }
    {
        let field = Field::new("address", DataType::Utf8, true);
        let result = variant_get(
            &col,
            GetOptions::new_with_path(VariantPath::from("address").join("city"))
                .with_as_type(Some(Arc::new(field))),
        )
        .unwrap();
        let address = result.as_string::<i32>();
        for v in address {
            println!("Address: {:?}", v);
        }
    }
}

#[test]
fn shred_typed_variant() {
    let (_variant_array, col) = setup_variant();

    let field = Arc::new(Field::new("name", DataType::Utf8, true));
    let result = variant_get(
        &col,
        GetOptions::new_with_path(VariantPath::from("name")).with_as_type(Some(field.clone())),
    )
    .unwrap();
    let name = result.as_string::<i32>().clone();
    for v in &name {
        println!("Name: {:?}", v);
    }

    let old_struct = col.as_struct();
    let metadata = old_struct.column_by_name("metadata").unwrap().clone();
    let value = old_struct.column_by_name("value").unwrap().clone();
    assert!(old_struct.column_by_name("typed_value").is_none());

    let (typed_value, fields) = {
        let inner_field = Field::new("typed_value", DataType::Utf8, true);
        let struct_field = Fields::from(vec![Arc::new(inner_field)]);
        let shred_field =
            StructArray::new(struct_field.clone(), vec![Arc::new(name.clone())], None);

        let typed_value_field = Field::new("name", DataType::Struct(struct_field.clone()), true);
        let fields = Fields::from(vec![typed_value_field.clone()]);

        let v = StructArray::new(fields.clone(), vec![Arc::new(shred_field)], None);
        (Arc::new(v), fields)
    };

    let variant_struct = StructArray::new(
        Fields::from(vec![
            Field::new("metadata", DataType::BinaryView, true),
            Field::new("value", DataType::BinaryView, true),
            Field::new("typed_value", DataType::Struct(fields), true),
        ]),
        vec![metadata, value, typed_value],
        None,
    );

    let variant = VariantArray::try_new(&variant_struct).unwrap();
    assert!(variant.typed_value_field().is_some());
    let variant_array = ArrayRef::from(variant);
    let result = variant_get(
        &variant_array,
        GetOptions::new_with_path(VariantPath::from("name")).with_as_type(Some(field.clone())),
    )
    .unwrap();
    let result = result.as_string::<i32>();
    for v in result {
        println!("Result: {:?}", v);
    }
}

#[test]
fn shred_general_variant() {
    let (_variant_array, col) = setup_variant();

    let shred_result = variant_get(
        &col,
        GetOptions::new_with_path(VariantPath::from("address")),
    )
    .unwrap();
    let shred_result = VariantArray::try_new(&shred_result).unwrap();
    for i in 0..shred_result.len() {
        println!("Shred result {}: {:?}", i, shred_result.value(i));
    }

    let old_struct = col.as_struct();
    let metadata = old_struct.column_by_name("metadata").unwrap().clone();
    let value = old_struct.column_by_name("value").unwrap().clone();
    assert!(old_struct.column_by_name("typed_value").is_none());

    let (shredded_value, fields) = {
        let inner_meta_field = Field::new("metadata", DataType::BinaryView, false);
        let inner_value_field = Field::new("value", DataType::BinaryView, true);
        let outer_field = Field::new(
            "address",
            DataType::Struct(Fields::from(vec![
                Arc::new(inner_meta_field),
                Arc::new(inner_value_field),
            ])),
            true,
        );
        let outer_field = Arc::new(outer_field);
        let struct_field = Fields::from(vec![outer_field.clone()]);
        let shredded_value = StructArray::new(
            struct_field.clone(),
            vec![Arc::new(shred_result.inner().clone())],
            None,
        );
        (Arc::new(shredded_value), struct_field)
    };

    let variant_struct = StructArray::new(
        Fields::from(vec![
            Field::new("metadata", DataType::BinaryView, true),
            Field::new("value", DataType::BinaryView, true),
            Field::new("typed_value", DataType::Struct(fields), true),
        ]),
        vec![metadata.clone(), value.clone(), shredded_value.clone()],
        None,
    );

    let variant = VariantArray::try_new(&variant_struct).unwrap();
    assert!(variant.typed_value_field().is_some());
    let variant_array = ArrayRef::from(variant.clone());
    let result = variant_get(
        &variant_array,
        GetOptions::new_with_path(VariantPath::from("address")),
    )
    .unwrap();
    let result = VariantArray::try_new(&result).unwrap();
    for i in 0..result.len() {
        println!("Result {}: {:?}", i, result.value(i));
    }

    let address_array = variant
        .typed_value_field()
        .unwrap()
        .as_struct()
        .column_by_name("address")
        .unwrap();
    let as_type = Arc::new(Field::new("city", DataType::Utf8, true));
    let result = variant_get(
        address_array,
        GetOptions::new_with_path(VariantPath::from("city")).with_as_type(Some(as_type)),
    )
    .unwrap();
    let result = result.as_string::<i32>();
    for v in result {
        println!("Result: {:?}", v);
    }
}
