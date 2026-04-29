//! Optimizers for the Parquet module

mod lineage_opt;

use std::sync::Arc;

use arrow_schema::{Field, Schema, SchemaRef};
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

use crate::{
    LiquidCacheParquetRef, LiquidParquetSource,
    optimizers::lineage_opt::{ColumnAnnotation, metadata_from_factory, serialize_date_part},
};

pub(crate) const DATE_MAPPING_METADATA_KEY: &str = "liquid.cache.date_mapping";
pub(crate) const STRING_FINGERPRINT_METADATA_KEY: &str = "liquid.cache.string_fingerprint";

/// Physical optimizer rule for local mode liquid cache
///
/// This optimizer rewrites DataSourceExec nodes that read Parquet files
/// to use LiquidParquetSource instead of the default ParquetSource
#[derive(Debug)]
pub struct LocalModeOptimizer {
    cache: LiquidCacheParquetRef,
}

impl LocalModeOptimizer {
    /// Create an optimizer with an existing cache instance
    pub fn new(cache: LiquidCacheParquetRef) -> Self {
        Self { cache }
    }

    /// Create an optimizer with an existing cache instance
    pub fn with_cache(cache: LiquidCacheParquetRef) -> Self {
        Self { cache }
    }
}

impl PhysicalOptimizerRule for LocalModeOptimizer {
    fn optimize(
        &self,
        plan: Arc<dyn ExecutionPlan>,
        _config: &ConfigOptions,
    ) -> Result<Arc<dyn ExecutionPlan>, datafusion::error::DataFusionError> {
        Ok(rewrite_data_source_plan(plan, &self.cache))
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
) -> Arc<dyn ExecutionPlan> {
    let rewritten = plan
        .transform_up(|node| try_optimize_parquet_source(node, cache))
        .unwrap();
    rewritten.data
}

fn try_optimize_parquet_source(
    plan: Arc<dyn ExecutionPlan>,
    cache: &LiquidCacheParquetRef,
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
) -> Schema {
    let mut new_fields = vec![];
    for field in file_schema.fields() {
        if let Some(annotation) = metadata_from_factory(expr_adapter_factory, field.name()) {
            new_fields.push(process_field_annotation(field, annotation));
        } else {
            new_fields.push(field.clone());
        }
    }
    Schema::new(new_fields)
}

fn process_field_annotation(
    field: &Arc<Field>,
    annotation: ColumnAnnotation,
) -> Arc<Field> {
    let mut field_metadata = field.metadata().clone();
    match annotation {
        ColumnAnnotation::DatePart(unit) => {
            field_metadata.insert(
                DATE_MAPPING_METADATA_KEY.to_string(),
                serialize_date_part(&unit),
            );
        }
        ColumnAnnotation::VariantPaths(_) => {}
        ColumnAnnotation::SubstringSearch => {
            field_metadata.insert(
                STRING_FINGERPRINT_METADATA_KEY.to_string(),
                "substring".into(),
            );
        }
    }
    Arc::new(Field::clone(field.as_ref()).with_metadata(field_metadata))
}

#[cfg(test)]
mod tests {
    use datafusion::{datasource::physical_plan::FileScanConfig, prelude::SessionContext};
    use liquid_cache::{
        cache::{AlwaysHydrate, squeeze_policies::TranscodeSqueezeEvict},
        cache_policies::LiquidPolicy,
    };

    use crate::LiquidCacheParquet;

    use super::*;

    async fn rewrite_plan_inner(plan: Arc<dyn ExecutionPlan>) {
        let expected_schema = plan.schema();
        let tmp_dir = tempfile::tempdir().unwrap();
        let store = t4::mount(tmp_dir.path().join("liquid_cache.t4"))
            .await
            .unwrap();
        let liquid_cache = Arc::new(
            LiquidCacheParquet::new(
                8192,
                1000000,
                store,
                Box::new(LiquidPolicy::new()),
                Box::new(TranscodeSqueezeEvict),
                Box::new(AlwaysHydrate::new()),
            )
            .await,
        );
        let rewritten = rewrite_data_source_plan(plan, &liquid_cache);

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
        rewrite_plan_inner(plan.clone()).await;
    }
}
