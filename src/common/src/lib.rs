#![cfg_attr(not(doctest), doc = include_str!(concat!("../", std::env!("CARGO_PKG_README"))))]

use std::str::FromStr;
use std::{fmt::Display, sync::Arc};

use arrow::array::ArrayRef;
use arrow_schema::{DataType, Field, FieldRef, Schema, SchemaRef};
pub mod mock_store;
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
    /// Static file server mode
    StaticFileServer,
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
                CacheMode::StaticFileServer => "static_file_server",
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
            "static_file_server" => CacheMode::StaticFileServer,
            _ => {
                return Err(format!(
                    "Invalid cache mode: {s}, must be one of: parquet, liquid, liquid_eager_transcode, arrow, static_file_server"
                ));
            }
        })
    }
}

/// The mode of the cache.
#[derive(Debug, Copy, Clone)]
pub enum LiquidCacheMode {
    /// The baseline that reads the arrays as is.
    Arrow,
    /// The baseline that reads the arrays as is, but transcode the data into liquid arrays in the background.
    Liquid {
        /// Whether to transcode the data into liquid arrays in the background.
        transcode_in_background: bool,
    },
}

impl Default for LiquidCacheMode {
    fn default() -> Self {
        Self::Liquid {
            transcode_in_background: true,
        }
    }
}

impl From<CacheMode> for LiquidCacheMode {
    fn from(value: CacheMode) -> Self {
        match value {
            CacheMode::Liquid => LiquidCacheMode::Liquid {
                transcode_in_background: true,
            },
            CacheMode::Arrow => LiquidCacheMode::Arrow,
            CacheMode::LiquidEagerTranscode => LiquidCacheMode::Liquid {
                transcode_in_background: false,
            },
            CacheMode::Parquet => unreachable!(),
            CacheMode::StaticFileServer => unreachable!(),
        }
    }
}

/// Create a new field with the specified data type, copying the other
/// properties from the input field
fn field_with_new_type(field: &FieldRef, new_type: DataType) -> FieldRef {
    Arc::new(field.as_ref().clone().with_data_type(new_type))
}

pub fn coerce_string_to_liquid_type(data_type: &DataType, mode: &LiquidCacheMode) -> DataType {
    match mode {
        LiquidCacheMode::Arrow => data_type.clone(),
        LiquidCacheMode::Liquid { .. } => match data_type {
            DataType::Utf8
            | DataType::LargeUtf8
            | DataType::Utf8View
            | DataType::Binary
            | DataType::LargeBinary
            | DataType::BinaryView => {
                DataType::Dictionary(Box::new(DataType::UInt16), Box::new(DataType::Utf8))
            }
            _ => data_type.clone(),
        },
    }
}

pub fn coerce_to_liquid_cache_types(schema: &Schema, mode: &LiquidCacheMode) -> Schema {
    match mode {
        // if in memory arrow, we cache as utf8 not dict or utf8view
        LiquidCacheMode::Arrow => schema.clone(),
        LiquidCacheMode::Liquid { .. } => {
            let transformed_fields: Vec<Arc<Field>> = schema
                .fields
                .iter()
                .map(|field| {
                    field_with_new_type(
                        field,
                        coerce_string_to_liquid_type(field.data_type(), mode),
                    )
                })
                .collect();
            Schema::new_with_metadata(transformed_fields, schema.metadata.clone())
        }
    }
}

pub fn coerce_from_parquet_to_liquid_type(
    data_type: &DataType,
    cache_mode: &LiquidCacheMode,
) -> DataType {
    match cache_mode {
        LiquidCacheMode::Arrow => {
            if data_type.equals_datatype(&DataType::Utf8View) {
                DataType::Utf8
            } else {
                data_type.clone()
            }
        }
        LiquidCacheMode::Liquid { .. } => {
            if data_type.equals_datatype(&DataType::Utf8View) {
                DataType::Dictionary(Box::new(DataType::UInt16), Box::new(DataType::Utf8))
            } else {
                data_type.clone()
            }
        }
    }
}

pub fn cast_from_parquet_to_liquid_type(array: ArrayRef, cache_mode: &LiquidCacheMode) -> ArrayRef {
    match cache_mode {
        LiquidCacheMode::Arrow => {
            if array.data_type() == &DataType::Utf8View {
                arrow::compute::kernels::cast(&array, &DataType::Utf8).unwrap()
            } else {
                array
            }
        }
        LiquidCacheMode::Liquid { .. } => {
            if array.data_type() == &DataType::Utf8View {
                arrow::compute::kernels::cast(
                    &array,
                    &DataType::Dictionary(Box::new(DataType::UInt16), Box::new(DataType::Utf8)),
                )
                .unwrap()
            } else {
                array
            }
        }
    }
}
/// Coerce the schema from LiquidParquetReader to LiquidCache types.
pub fn coerce_from_parquet_reader_to_liquid_types(
    schema: &Schema,
    cache_mode: &LiquidCacheMode,
) -> Schema {
    let transformed_fields: Vec<Arc<Field>> = schema
        .fields
        .iter()
        .map(|field| {
            field_with_new_type(
                field,
                coerce_from_parquet_to_liquid_type(field.data_type(), cache_mode),
            )
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
pub struct ParquetReaderSchema {}

impl ParquetReaderSchema {
    pub fn from(schema: &Schema) -> SchemaRef {
        let transformed_fields: Vec<Arc<Field>> = schema
            .fields
            .iter()
            .map(|field| {
                let data_type = field.data_type();
                match data_type {
                    DataType::Utf8 | DataType::LargeUtf8 | DataType::Utf8View => {
                        field_with_new_type(field, DataType::Utf8View)
                    }
                    DataType::Binary | DataType::LargeBinary | DataType::BinaryView => {
                        field_with_new_type(field, DataType::BinaryView)
                    }
                    _ => field.clone(),
                }
            })
            .collect();
        Arc::new(Schema::new_with_metadata(
            transformed_fields,
            schema.metadata.clone(),
        ))
    }
}
