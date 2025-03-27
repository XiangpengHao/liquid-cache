use std::{any::Any, sync::Arc};

use arrow_schema::{DataType, Field, FieldRef, Schema, SchemaRef};
use async_trait::async_trait;
use datafusion::{
    catalog::Session,
    common::{GetExt, Statistics, internal_err, parsers::CompressionTypeVariant},
    config::TableParquetOptions,
    datasource::{
        file_format::{
            FileFormat, FilePushdownSupport, file_compression_type::FileCompressionType,
        },
        physical_plan::{FileScanConfig, FileSource},
    },
    error::Result,
    physical_plan::{ExecutionPlan, PhysicalExpr},
    prelude::*,
};
use liquid_cache_common::coerce_to_liquid_cache_types;
use log::info;
use object_store::{ObjectMeta, ObjectStore};
#[cfg(test)]
pub(crate) use source::CachedMetaReaderFactory;
pub use source::LiquidParquetSource;
pub(crate) use source::ParquetMetadataCacheReader;

use crate::{LiquidCacheMode, LiquidCacheRef};

// This is entirely copied from DataFusion
// We should make DataFusion to public this
mod row_filter;

// This is entirely copied from DataFusion
// We should make DataFusion to public this
mod opener;
mod row_group_filter;
mod source;

#[derive(Debug)]
pub struct LiquidParquetFactory {}

impl GetExt for LiquidParquetFactory {
    fn get_ext(&self) -> String {
        "parquet".to_string()
    }
}

/// A file format for liquid parquet.
#[derive(Debug)]
pub struct LiquidParquetFileFormat {
    options: TableParquetOptions,
    inner: Arc<dyn FileFormat>,   // is actually ParquetFormat
    liquid_cache: LiquidCacheRef, // a file format deals with multiple files
    liquid_cache_mode: LiquidCacheMode,
}

impl LiquidParquetFileFormat {
    /// Creates a new liquid parquet file format.
    pub fn new(
        options: TableParquetOptions,
        inner: Arc<dyn FileFormat>,
        liquid_cache: LiquidCacheRef,
        liquid_cache_mode: LiquidCacheMode,
    ) -> Self {
        Self {
            options,
            liquid_cache,
            liquid_cache_mode,
            inner,
        }
    }

    /// Return the metadata size hint if set
    pub fn metadata_size_hint(&self) -> Option<usize> {
        self.options.global.metadata_size_hint
    }
}

#[async_trait]
impl FileFormat for LiquidParquetFileFormat {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn get_ext(&self) -> String {
        LiquidParquetFactory {}.get_ext()
    }

    fn get_ext_with_compression(
        &self,
        file_compression_type: &FileCompressionType,
    ) -> Result<String> {
        let ext = self.get_ext();
        match file_compression_type.get_variant() {
            CompressionTypeVariant::UNCOMPRESSED => Ok(ext),
            _ => internal_err!("Unsupported compression type: {:?}", file_compression_type),
        }
    }

    async fn infer_schema(
        &self,
        state: &dyn Session,
        store: &Arc<dyn ObjectStore>,
        objects: &[ObjectMeta],
    ) -> Result<SchemaRef> {
        let parquet_schema = self.inner.infer_schema(state, store, objects).await?;
        let mut transformed = coerce_binary_to_string(&parquet_schema);
        if matches!(
            self.liquid_cache_mode,
            LiquidCacheMode::InMemoryLiquid { .. }
        ) {
            transformed = coerce_to_liquid_cache_types(&transformed);
        }
        Ok(Arc::new(transformed))
    }

    async fn infer_stats(
        &self,
        state: &dyn Session,
        store: &Arc<dyn ObjectStore>,
        table_schema: SchemaRef,
        object: &ObjectMeta,
    ) -> Result<Statistics> {
        self.inner
            .infer_stats(state, store, table_schema, object)
            .await
    }

    async fn create_physical_plan(
        &self,
        _state: &dyn Session,
        conf: FileScanConfig,
        filters: Option<&Arc<dyn PhysicalExpr>>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        info!("create_physical_plan for liquid parquet");
        let mut predicate = None;
        let mut metadata_size_hint = None;

        // If enable pruning then combine the filters to build the predicate.
        // If disable pruning then set the predicate to None, thus readers
        // will not prune data based on the statistics.
        if let Some(pred) = filters.cloned() {
            predicate = Some(pred);
        }

        if let Some(metadata) = self.metadata_size_hint() {
            metadata_size_hint = Some(metadata);
        }

        let mut source = LiquidParquetSource::new(
            self.options.clone(),
            self.liquid_cache.clone(),
            self.liquid_cache_mode,
        );

        if let Some(predicate) = predicate {
            source = source.with_predicate(Arc::clone(&conf.file_schema), predicate);
        }
        if let Some(metadata_size_hint) = metadata_size_hint {
            source = source.with_metadata_size_hint(metadata_size_hint)
        }

        Ok(conf.with_source(Arc::new(source)).build())
    }

    fn supports_filters_pushdown(
        &self,
        file_schema: &Schema,
        table_schema: &Schema,
        filters: &[&Expr],
    ) -> Result<FilePushdownSupport> {
        self.inner
            .supports_filters_pushdown(file_schema, table_schema, filters)
    }

    fn file_source(&self) -> Arc<dyn FileSource> {
        // this can be anything, as we don't really use it.
        self.inner.file_source()
    }
}

/// Create a new field with the specified data type, copying the other
/// properties from the input field
fn field_with_new_type(field: &FieldRef, new_type: DataType) -> FieldRef {
    Arc::new(field.as_ref().clone().with_data_type(new_type))
}

pub(crate) fn coerce_binary_to_string(schema: &Schema) -> Schema {
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

pub(crate) fn coerce_string_to_view(schema: &Schema) -> Schema {
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

/// Liquid cache reads binary as strings.
pub(crate) fn coerce_from_reader_to_liquid_types(schema: &Schema) -> Schema {
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
