#![cfg_attr(not(doctest), doc = include_str!(concat!("../", std::env!("CARGO_PKG_README"))))]

use std::ops::Deref;
use std::str::FromStr;
use std::{fmt::Display, sync::Arc};

use arrow_schema::{DataType, Field, FieldRef, Schema, SchemaRef};
pub mod rpc;
pub mod utils;
/// Specify how LiquidCache should cache the data
#[derive(Clone, Debug, Default, Copy, PartialEq, Eq)]
pub enum CacheMode {
    /// Cache parquet files
    Parquet,
    /// Cache LiquidArray, transcode happens in background
    #[default]
    Liquid,
    /// Transcode blocks query execution
    LiquidEagerTranscode,
    /// Cache Arrow, transcode happens in background
    Arrow,
}

impl Display for CacheMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}",
            match self {
                CacheMode::Parquet => "parquet",
                CacheMode::Liquid => "liquid",
                CacheMode::LiquidEagerTranscode => "liquid_eager_transcode",
                CacheMode::Arrow => "arrow",
            }
        )
    }
}

impl FromStr for CacheMode {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Ok(match s {
            "parquet" => CacheMode::Parquet,
            "liquid" => CacheMode::Liquid,
            "liquid_eager_transcode" => CacheMode::LiquidEagerTranscode,
            "arrow" => CacheMode::Arrow,
            _ => {
                return Err(format!(
                    "Invalid cache mode: {}, must be one of: parquet, liquid, liquid_eager_transcode, arrow",
                    s
                ));
            }
        })
    }
}

/// Create a new field with the specified data type, copying the other
/// properties from the input field
fn field_with_new_type(field: &FieldRef, new_type: DataType) -> FieldRef {
    Arc::new(field.as_ref().clone().with_data_type(new_type))
}

pub fn coerce_to_liquid_cache_types(schema: &Schema) -> Schema {
    let transformed_fields: Vec<Arc<Field>> = schema
        .fields
        .iter()
        .map(|field| match field.data_type() {
            DataType::Utf8 | DataType::LargeUtf8 | DataType::Utf8View => field_with_new_type(
                field,
                DataType::Dictionary(Box::new(DataType::UInt16), Box::new(DataType::Utf8)),
            ),
            DataType::Binary | DataType::LargeBinary | DataType::BinaryView => field_with_new_type(
                field,
                DataType::Dictionary(Box::new(DataType::UInt16), Box::new(DataType::Utf8)),
            ),
            _ => field.clone(),
        })
        .collect();
    Schema::new_with_metadata(transformed_fields, schema.metadata.clone())
}

/// Coerce the schema from LiquidParquetReader to LiquidCache types.
pub fn coerce_from_parquet_reader_to_liquid_types(schema: &Schema) -> Schema {
    let transformed_fields: Vec<Arc<Field>> = schema
        .fields
        .iter()
        .map(|field| {
            if field.data_type().equals_datatype(&DataType::Utf8View) {
                field_with_new_type(
                    field,
                    DataType::Dictionary(Box::new(DataType::UInt16), Box::new(DataType::Utf8)),
                )
            } else {
                field.clone()
            }
        })
        .collect();
    Schema::new_with_metadata(transformed_fields, schema.metadata.clone())
}

pub fn coerce_binary_to_string(schema: &Schema) -> Schema {
    let transformed_fields: Vec<Arc<Field>> = schema
        .fields
        .iter()
        .map(|field| match field.data_type() {
            DataType::Binary | DataType::LargeBinary | DataType::BinaryView => {
                field_with_new_type(field, DataType::Utf8)
            }
            _ => field.clone(),
        })
        .collect();
    Schema::new_with_metadata(transformed_fields, schema.metadata.clone())
}

pub fn coerce_string_to_view(schema: &Schema) -> Schema {
    let transformed_fields: Vec<Arc<Field>> = schema
        .fields
        .iter()
        .map(|field| match field.data_type() {
            DataType::Utf8 | DataType::LargeUtf8 => field_with_new_type(field, DataType::Utf8View),
            _ => field.clone(),
        })
        .collect();
    Schema::new_with_metadata(transformed_fields, schema.metadata.clone())
}

/// A schema that where strings are stored as `Utf8View`
pub struct StringViewSchema {
    schema: SchemaRef,
}

impl Deref for StringViewSchema {
    type Target = SchemaRef;

    fn deref(&self) -> &Self::Target {
        &self.schema
    }
}

impl From<&DictStringSchema> for StringViewSchema {
    fn from(schema: &DictStringSchema) -> Self {
        let dict_string_type =
            DataType::Dictionary(Box::new(DataType::UInt16), Box::new(DataType::Utf8));
        let transformed_fields: Vec<Arc<Field>> = schema
            .schema
            .fields
            .iter()
            .map(|field| {
                if field.data_type().equals_datatype(&dict_string_type) {
                    field_with_new_type(field, DataType::Utf8View)
                } else {
                    field.clone()
                }
            })
            .collect();
        Self {
            schema: Arc::new(Schema::new_with_metadata(
                transformed_fields,
                schema.schema.metadata.clone(),
            )),
        }
    }
}

/// A schema that where strings are stored as `Dictionary<UInt16, Utf8>`
pub struct DictStringSchema {
    schema: SchemaRef,
}

impl DictStringSchema {
    /// Create a new `DictStringSchema` from a `SchemaRef`.
    ///
    /// # Panics
    ///
    /// This function will panic if the schema contains a `Utf8` or `Utf8View` field.
    pub fn new(schema: SchemaRef) -> Self {
        {
            for field in schema.fields() {
                assert!(
                    !field.data_type().equals_datatype(&DataType::Utf8),
                    "Field {} must not be a Utf8",
                    field.name()
                );
                assert!(
                    !field.data_type().equals_datatype(&DataType::Utf8View),
                    "Field {} must not be a Utf8View",
                    field.name()
                );
            }
        }
        Self { schema }
    }
}

impl Deref for DictStringSchema {
    type Target = SchemaRef;

    fn deref(&self) -> &Self::Target {
        &self.schema
    }
}

/// A schema that where strings are stored as `Utf8`
pub struct StringSchema {
    schema: SchemaRef,
}

impl Deref for StringSchema {
    type Target = SchemaRef;

    fn deref(&self) -> &Self::Target {
        &self.schema
    }
}

impl From<&DictStringSchema> for StringSchema {
    fn from(schema: &DictStringSchema) -> Self {
        let dict_string_type =
            DataType::Dictionary(Box::new(DataType::UInt16), Box::new(DataType::Utf8));
        let transformed_fields: Vec<Arc<Field>> = schema
            .schema
            .fields
            .iter()
            .map(|field| {
                if field.data_type().equals_datatype(&dict_string_type) {
                    field_with_new_type(field, DataType::Utf8)
                } else {
                    field.clone()
                }
            })
            .collect();
        Self {
            schema: Arc::new(Schema::new_with_metadata(
                transformed_fields,
                schema.schema.metadata.clone(),
            )),
        }
    }
}
