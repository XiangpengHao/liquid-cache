//! Optimizers for the Parquet module

mod lineage_opt;

use std::{str::FromStr, sync::Arc};

use arrow_schema::{DataType, Field, Fields, Schema, SchemaRef};
use datafusion::{
    catalog::memory::DataSourceExec,
    common::tree_node::{Transformed, TreeNode, TreeNodeRecursion},
    config::ConfigOptions,
    datasource::{
        physical_plan::{FileSource, ParquetSource},
        source::DataSource,
        table_schema::TableSchema,
    },
    physical_expr_adapter::PhysicalExprAdapterFactory,
    physical_optimizer::PhysicalOptimizerRule,
    physical_plan::ExecutionPlan,
};
pub use lineage_opt::LineageOptimizer;
pub(crate) use lineage_opt::VariantField;

use crate::{
    LiquidCacheParquetRef, LiquidParquetSource,
    optimizers::lineage_opt::{ColumnAnnotation, metadata_from_factory, serialize_date_part},
};
use liquid_cache_storage::utils::VariantSchema;
use serde::{Deserialize, Serialize};

pub(crate) const DATE_MAPPING_METADATA_KEY: &str = "liquid.cache.date_mapping";
pub(crate) const VARIANT_MAPPING_METADATA_KEY: &str = "liquid.cache.variant_path";
pub(crate) const VARIANT_MAPPING_TYPE_METADATA_KEY: &str = "liquid.cache.variant_type";
pub(crate) const STRING_FINGERPRINT_METADATA_KEY: &str = "liquid.cache.string_fingerprint";

#[derive(Debug, Clone, Serialize, Deserialize)]
struct VariantMappingSerdeEntry {
    path: String,
    #[serde(rename = "type", skip_serializing_if = "Option::is_none")]
    data_type: Option<String>,
}

pub(crate) fn serialize_variant_mappings(fields: &[VariantField]) -> Option<String> {
    if fields.is_empty() {
        return None;
    }

    let entries: Vec<VariantMappingSerdeEntry> = fields
        .iter()
        .map(|field| VariantMappingSerdeEntry {
            path: field.path.clone(),
            data_type: field
                .data_type
                .as_ref()
                .map(|data_type| data_type.to_string()),
        })
        .collect();

    serde_json::to_string(&entries).ok()
}

fn deserialize_variant_mappings(raw: &str) -> Option<Vec<VariantField>> {
    let entries: Vec<VariantMappingSerdeEntry> = serde_json::from_str(raw).ok()?;
    let mut fields = Vec::with_capacity(entries.len());
    for entry in entries {
        let data_type = match entry.data_type {
            Some(spec) => Some(DataType::from_str(&spec).ok()?),
            None => None,
        };
        fields.push(VariantField {
            path: entry.path,
            data_type,
        });
    }
    Some(fields)
}

pub(crate) fn variant_mappings_from_field(field: &Field) -> Option<Vec<VariantField>> {
    let metadata = field.metadata();
    let raw = metadata.get(VARIANT_MAPPING_METADATA_KEY)?;
    if let Some(parsed) = deserialize_variant_mappings(raw) {
        return Some(parsed);
    }

    let data_type = metadata
        .get(VARIANT_MAPPING_TYPE_METADATA_KEY)
        .and_then(|spec| DataType::from_str(spec).ok());

    Some(vec![VariantField {
        path: raw.clone(),
        data_type,
    }])
}

/// Physical optimizer rule for local mode liquid cache
///
/// This optimizer rewrites DataSourceExec nodes that read Parquet files
/// to use LiquidParquetSource instead of the default ParquetSource
#[derive(Debug)]
pub struct LocalModeOptimizer {
    cache: LiquidCacheParquetRef,
    eager_shredding: bool,
}

impl LocalModeOptimizer {
    /// Create an optimizer with an existing cache instance
    pub fn new(cache: LiquidCacheParquetRef, eager_shredding: bool) -> Self {
        Self {
            cache,
            eager_shredding,
        }
    }

    /// Create an optimizer with an existing cache instance
    pub fn with_cache(cache: LiquidCacheParquetRef) -> Self {
        Self {
            cache,
            eager_shredding: true,
        }
    }
}

impl PhysicalOptimizerRule for LocalModeOptimizer {
    fn optimize(
        &self,
        plan: Arc<dyn ExecutionPlan>,
        _config: &ConfigOptions,
    ) -> Result<Arc<dyn ExecutionPlan>, datafusion::error::DataFusionError> {
        Ok(rewrite_data_source_plan(
            plan,
            &self.cache,
            self.eager_shredding,
        ))
    }

    fn name(&self) -> &str {
        "LocalModeLiquidCacheOptimizer"
    }

    fn schema_check(&self) -> bool {
        // We deliberately enrich scan schemas with metadata describing variant/date
        // extractions, so allow the optimizer to adjust schema metadata.
        false
    }
}

/// Rewrite the data source plan to use liquid cache.
pub fn rewrite_data_source_plan(
    plan: Arc<dyn ExecutionPlan>,
    cache: &LiquidCacheParquetRef,
    eager_shredding: bool,
) -> Arc<dyn ExecutionPlan> {
    let rewritten = plan
        .transform_up(|node| try_optimize_parquet_source(node, cache, eager_shredding))
        .unwrap();
    rewritten.data
}

fn try_optimize_parquet_source(
    plan: Arc<dyn ExecutionPlan>,
    cache: &LiquidCacheParquetRef,
    eager_shredding: bool,
) -> Result<Transformed<Arc<dyn ExecutionPlan>>, datafusion::error::DataFusionError> {
    let any_plan = plan.as_any();
    if let Some(data_source_exec) = any_plan.downcast_ref::<DataSourceExec>()
        && let Some((file_scan_config, parquet_source)) =
            data_source_exec.downcast_to_file_source::<ParquetSource>()
    {
        let mut new_config = file_scan_config.clone();

        let mut new_source =
            LiquidParquetSource::from_parquet_source(parquet_source.clone(), cache.clone());
        if let Some(expr_adapter_factory) = file_scan_config.expr_adapter_factory.as_ref() {
            let new_schema = enrich_source_schema(
                file_scan_config.file_schema(),
                expr_adapter_factory,
                eager_shredding,
            );
            let table_partition_cols = new_source.table_schema().table_partition_cols();
            let new_table_schema =
                TableSchema::new(Arc::new(new_schema), table_partition_cols.clone());
            new_source = new_source.with_table_schema(new_table_schema);
        }

        new_config.file_source = Arc::new(new_source);
        let new_file_source: Arc<dyn DataSource> = Arc::new(new_config);
        let new_plan = Arc::new(DataSourceExec::new(new_file_source));

        return Ok(Transformed::new(
            new_plan,
            true,
            TreeNodeRecursion::Continue,
        ));
    }
    Ok(Transformed::no(plan))
}

fn enrich_source_schema(
    file_schema: &SchemaRef,
    expr_adapter_factory: &Arc<dyn PhysicalExprAdapterFactory>,
    eager_shredding: bool,
) -> Schema {
    let mut new_fields = vec![];
    for field in file_schema.fields() {
        if let Some(annotation) = metadata_from_factory(expr_adapter_factory, field.name()) {
            new_fields.push(process_field_annotation(field, annotation, eager_shredding));
        } else {
            new_fields.push(field.clone());
        }
    }
    Schema::new(new_fields)
}

fn process_field_annotation(
    field: &Arc<Field>,
    annotation: ColumnAnnotation,
    eager_shredding: bool,
) -> Arc<Field> {
    let mut field_metadata = field.metadata().clone();
    let mut updated_field = Field::clone(field.as_ref());
    match annotation {
        ColumnAnnotation::DatePart(unit) => {
            field_metadata.insert(
                DATE_MAPPING_METADATA_KEY.to_string(),
                serialize_date_part(&unit),
            );
        }
        ColumnAnnotation::VariantPaths(paths) => {
            if eager_shredding {
                if let Some(serialized) = serialize_variant_mappings(&paths) {
                    field_metadata.insert(VARIANT_MAPPING_METADATA_KEY.to_string(), serialized);
                }
                updated_field = enrich_variant_field_type(&updated_field, &paths);
            }
        }
        ColumnAnnotation::SubstringSearch => {
            field_metadata.insert(
                STRING_FINGERPRINT_METADATA_KEY.to_string(),
                "substring".into(),
            );
        }
    }
    Arc::new(updated_field.with_metadata(field_metadata))
}

pub(crate) fn enrich_variant_field_type(field: &Field, fields: &[VariantField]) -> Field {
    let typed_specs: Vec<&VariantField> = fields
        .iter()
        .filter(|field| field.data_type.is_some())
        .collect();
    if typed_specs.is_empty() {
        return Field::clone(field);
    }

    let new_type = match field.data_type() {
        DataType::Struct(children) => {
            let mut rewritten = Vec::with_capacity(children.len() + 1);
            let mut replaced = false;
            for child in children.iter() {
                if child.name() == "typed_value" {
                    rewritten.push(build_variant_typed_value_field(
                        Some(child.as_ref()),
                        &typed_specs,
                    ));
                    replaced = true;
                } else {
                    let mut child_field = child.as_ref().clone();
                    if child_field.name() == "value" {
                        child_field =
                            Field::new(child_field.name(), child_field.data_type().clone(), true)
                                .with_metadata(child_field.metadata().clone());
                    }
                    rewritten.push(Arc::new(child_field));
                }
            }
            if !replaced {
                rewritten.push(build_variant_typed_value_field(None, &typed_specs));
            }
            DataType::Struct(Fields::from(rewritten))
        }
        other => other.clone(),
    };
    Field::clone(field).with_data_type(new_type)
}

pub(crate) fn enrich_schema_for_cache(schema: &SchemaRef) -> SchemaRef {
    let mut fields = vec![];
    for field in schema.fields() {
        let new_field = if let Some(mappings) = variant_mappings_from_field(field.as_ref()) {
            Arc::new(enrich_variant_field_type(field.as_ref(), &mappings))
        } else {
            field.clone()
        };
        fields.push(new_field);
    }
    Arc::new(Schema::new(fields))
}

fn build_variant_typed_value_field(
    existing: Option<&Field>,
    specs: &[&VariantField],
) -> Arc<Field> {
    let mut schema = VariantSchema::new(existing);
    for spec in specs {
        if let Some(data_type) = spec.data_type.as_ref() {
            schema.insert_path(&spec.path, data_type);
        }
    }

    Arc::new(Field::new(
        "typed_value",
        DataType::Struct(Fields::from(schema.typed_fields())),
        true,
    ))
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use datafusion::{datasource::physical_plan::FileScanConfig, prelude::SessionContext};
    use liquid_cache_storage::{
        cache::{AlwaysHydrate, squeeze_policies::TranscodeSqueezeEvict},
        cache_policies::LiquidPolicy,
    };

    use crate::LiquidCacheParquet;
    use liquid_cache_common::IoMode;

    use super::*;

    fn rewrite_plan_inner(plan: Arc<dyn ExecutionPlan>) {
        let expected_schema = plan.schema();
        let liquid_cache = Arc::new(LiquidCacheParquet::new(
            8192,
            1000000,
            PathBuf::from("test"),
            Box::new(LiquidPolicy::new()),
            Box::new(TranscodeSqueezeEvict),
            Box::new(AlwaysHydrate::new()),
            IoMode::Uring,
        ));
        let rewritten = rewrite_data_source_plan(plan, &liquid_cache, true);

        rewritten
            .apply(|node| {
                if let Some(plan) = node.as_any().downcast_ref::<DataSourceExec>() {
                    let data_source = plan.data_source();
                    let any_source = data_source.as_any();
                    let source = any_source.downcast_ref::<FileScanConfig>().unwrap();
                    let file_source = source.file_source();
                    let any_file_source = file_source.as_any();
                    let _parquet_source = any_file_source
                        .downcast_ref::<LiquidParquetSource>()
                        .unwrap();
                    let schema = source.file_schema().as_ref();
                    assert_eq!(schema, expected_schema.as_ref());
                }
                Ok(TreeNodeRecursion::Continue)
            })
            .unwrap();
    }

    #[tokio::test]
    async fn test_plan_rewrite() {
        let ctx = SessionContext::new();
        ctx.register_parquet(
            "nano_hits",
            "../../examples/nano_hits.parquet",
            Default::default(),
        )
        .await
        .unwrap();
        let df = ctx
            .sql("SELECT * FROM nano_hits WHERE \"URL\" like 'https://%' limit 10")
            .await
            .unwrap();
        let plan = df.create_physical_plan().await.unwrap();
        rewrite_plan_inner(plan.clone());
    }
}
