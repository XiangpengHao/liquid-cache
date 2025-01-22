use std::{any::Any, collections::HashMap, sync::Arc};

use arrow_schema::{DataType, Field, FieldRef, Schema, SchemaRef};
use async_trait::async_trait;
use datafusion::{
    common::{GetExt, Statistics, internal_err, parsers::CompressionTypeVariant},
    config::TableParquetOptions,
    datasource::{
        file_format::{
            FileFormat, FilePushdownSupport, file_compression_type::FileCompressionType,
        },
        physical_plan::FileScanConfig,
    },
    error::Result,
    execution::SessionState,
    physical_optimizer::pruning::PruningPredicate,
    physical_plan::{ExecutionPlan, PhysicalExpr, metrics::ExecutionPlanMetricsSet},
    prelude::*,
};
use exec::LiquidParquetExec;
use log::{debug, info};
use object_store::{ObjectMeta, ObjectStore};
use page_filter::PagePruningAccessPlanFilter;

use crate::cache::{LiquidCache, LiquidCacheMode};

// This is entirely copied from DataFusion
// We should make DataFusion to public this
mod page_filter;

// This is entirely copied from DataFusion
// We should make DataFusion to public this
mod row_filter;

// This is entirely copied from DataFusion
// We should make DataFusion to public this
mod row_group_filter;

mod opener;

mod exec;

#[derive(Debug)]
pub struct LiquidParquetFactory {}

impl GetExt for LiquidParquetFactory {
    fn get_ext(&self) -> String {
        "parquet".to_string()
    }
}

#[derive(Debug)]
pub struct LiquidParquetFileFormat {
    options: TableParquetOptions,
    liquid_cache_mode: LiquidCacheMode,
    inner: Arc<dyn FileFormat>, // is actually ParquetFormat
}

impl LiquidParquetFileFormat {
    pub fn new(
        options: TableParquetOptions,
        liquid_cache_mode: LiquidCacheMode,
        inner: Arc<dyn FileFormat>,
    ) -> Self {
        Self {
            options,
            liquid_cache_mode,
            inner,
        }
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
        state: &SessionState,
        store: &Arc<dyn ObjectStore>,
        objects: &[ObjectMeta],
    ) -> Result<SchemaRef> {
        self.inner.infer_schema(state, store, objects).await
    }

    async fn infer_stats(
        &self,
        state: &SessionState,
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
        state: &SessionState,
        conf: FileScanConfig,
        filters: Option<&Arc<dyn PhysicalExpr>>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        info!("create_physical_plan for liquid parquet");

        let metrics = ExecutionPlanMetricsSet::new();

        let file_schema = &conf.file_schema;
        let pruning_predicate = filters
            .and_then(|predicate_expr| {
                match PruningPredicate::try_new(Arc::clone(predicate_expr), Arc::clone(file_schema))
                {
                    Ok(pruning_predicate) => Some(Arc::new(pruning_predicate)),
                    Err(e) => {
                        debug!("Could not create pruning predicate for: {e}");
                        None
                    }
                }
            })
            .filter(|p| !p.always_true());

        let page_pruning_predicate = filters
            .as_ref()
            .map(|predicate_expr| {
                PagePruningAccessPlanFilter::new(predicate_expr, Arc::clone(file_schema))
            })
            .map(Arc::new);

        let (projected_schema, _, projected_statistics, projected_output_ordering) = conf.project();

        let cache = LiquidParquetExec::compute_properties(
            projected_schema,
            &projected_output_ordering,
            &conf,
        );

        let batch_size = state.config().batch_size();
        let liquid_cache = Arc::new(LiquidCache::new(self.liquid_cache_mode, batch_size));

        let exec = LiquidParquetExec {
            base_config: conf,
            table_parquet_options: self.options.clone(),
            predicate: filters.cloned(),
            metadata_size_hint: None,
            cache,
            projected_statistics,
            metrics,
            pruning_predicate,
            page_pruning_predicate,
            liquid_cache,
        };
        Ok(Arc::new(exec))
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
}

pub(crate) fn coerce_file_schema_to_string_type(
    table_schema: &Schema,
    file_schema: &Schema,
) -> Option<Schema> {
    let mut transform = false;
    let table_fields: HashMap<_, _> = table_schema
        .fields
        .iter()
        .map(|f| (f.name(), f.data_type()))
        .collect();
    let transformed_fields: Vec<Arc<Field>> = file_schema
        .fields
        .iter()
        .map(
            |field| match (table_fields.get(field.name()), field.data_type()) {
                // table schema uses string type, coerce the file schema to use string type
                (
                    Some(DataType::Utf8),
                    DataType::Binary | DataType::LargeBinary | DataType::BinaryView,
                ) => {
                    transform = true;
                    field_with_new_type(field, DataType::Utf8)
                }
                // table schema uses large string type, coerce the file schema to use large string type
                (
                    Some(DataType::LargeUtf8),
                    DataType::Binary | DataType::LargeBinary | DataType::BinaryView,
                ) => {
                    transform = true;
                    field_with_new_type(field, DataType::LargeUtf8)
                }
                // table schema uses string view type, coerce the file schema to use view type
                (
                    Some(DataType::Utf8View),
                    DataType::Binary | DataType::LargeBinary | DataType::BinaryView,
                ) => {
                    transform = true;
                    field_with_new_type(field, DataType::Utf8View)
                }
                _ => Arc::clone(field),
            },
        )
        .collect();

    if !transform {
        None
    } else {
        Some(Schema::new_with_metadata(
            transformed_fields,
            file_schema.metadata.clone(),
        ))
    }
}

/// Coerces the file schema if the table schema uses a view type.
pub(crate) fn coerce_file_schema_to_view_type(
    table_schema: &Schema,
    file_schema: &Schema,
) -> Option<Schema> {
    let mut transform = false;
    let table_fields: HashMap<_, _> = table_schema
        .fields
        .iter()
        .map(|f| {
            let dt = f.data_type();
            if dt.equals_datatype(&DataType::Utf8View) || dt.equals_datatype(&DataType::BinaryView)
            {
                transform = true;
            }
            (f.name(), dt)
        })
        .collect();

    if !transform {
        return None;
    }

    let transformed_fields: Vec<Arc<Field>> = file_schema
        .fields
        .iter()
        .map(
            |field| match (table_fields.get(field.name()), field.data_type()) {
                (Some(DataType::Utf8View), DataType::Utf8 | DataType::LargeUtf8) => {
                    field_with_new_type(field, DataType::Utf8View)
                }
                (Some(DataType::BinaryView), DataType::Binary | DataType::LargeBinary) => {
                    field_with_new_type(field, DataType::BinaryView)
                }
                _ => Arc::clone(field),
            },
        )
        .collect();

    Some(Schema::new_with_metadata(
        transformed_fields,
        file_schema.metadata.clone(),
    ))
}

/// Create a new field with the specified data type, copying the other
/// properties from the input field
fn field_with_new_type(field: &FieldRef, new_type: DataType) -> FieldRef {
    Arc::new(field.as_ref().clone().with_data_type(new_type))
}
