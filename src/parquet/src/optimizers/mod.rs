//! Optimizers for the Parquet module

mod lineage_opt;

use std::sync::Arc;

use arrow_schema::{Field, Schema};
use datafusion::{
    catalog::memory::DataSourceExec,
    common::tree_node::{Transformed, TreeNode, TreeNodeRecursion},
    config::ConfigOptions,
    datasource::{
        physical_plan::{FileSource, ParquetSource},
        source::DataSource,
        table_schema::TableSchema,
    },
    physical_optimizer::PhysicalOptimizerRule,
    physical_plan::ExecutionPlan,
};
pub use lineage_opt::LineageOptimizer;

use crate::{
    LiquidCacheRef, LiquidParquetSource,
    optimizers::lineage_opt::{ColumnAnnotation, metadata_from_factory},
};

pub(crate) const DATE_MAPPING_METADATA_KEY: &str = "liquid.cache.date_mapping";
pub(crate) const VARIANT_MAPPING_METADATA_KEY: &str = "liquid.cache.variant_path";

/// Physical optimizer rule for local mode liquid cache
///
/// This optimizer rewrites DataSourceExec nodes that read Parquet files
/// to use LiquidParquetSource instead of the default ParquetSource
#[derive(Debug)]
pub struct LocalModeOptimizer {
    cache: LiquidCacheRef,
}

impl LocalModeOptimizer {
    /// Create an optimizer with an existing cache instance
    pub fn with_cache(cache: LiquidCacheRef) -> Self {
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
    cache: &LiquidCacheRef,
) -> Arc<dyn ExecutionPlan> {
    let rewritten = plan
        .transform_up(|node| {
            let any_plan = node.as_any();
            if let Some(data_source_exec) = any_plan.downcast_ref::<DataSourceExec>() {
                if let Some((file_scan_config, parquet_source)) =
                    data_source_exec.downcast_to_file_source::<ParquetSource>()
                {
                    let mut new_config = file_scan_config.clone();

                    let mut new_source = LiquidParquetSource::from_parquet_source(
                        parquet_source.clone(),
                        cache.clone(),
                    );
                    if let Some(schema_factory) =
                        file_scan_config.file_source().schema_adapter_factory()
                    {
                        let file_schema = file_scan_config.file_schema().clone();
                        let mut new_fields = vec![];
                        for field in file_schema.fields() {
                            if let Some(annotation) =
                                metadata_from_factory(&schema_factory, field.name())
                            {
                                let (metadata_key, metadata_value): (&str, String) =
                                    match annotation {
                                        ColumnAnnotation::DatePart(unit) => (
                                            DATE_MAPPING_METADATA_KEY,
                                            unit.metadata_value().to_string(),
                                        ),
                                        ColumnAnnotation::VariantPath(path) => {
                                            (VARIANT_MAPPING_METADATA_KEY, path)
                                        }
                                    };
                                let mut field_metadata = field.metadata().clone();
                                field_metadata.insert(metadata_key.to_string(), metadata_value);
                                let new_field =
                                    Field::clone(field.as_ref()).with_metadata(field_metadata);
                                new_fields.push(Arc::new(new_field));
                            } else {
                                new_fields.push(field.clone());
                            }
                        }
                        let new_schema = Schema::new(new_fields);
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

                return Ok(Transformed::no(node));
            }
            Ok(Transformed::no(node))
        })
        .unwrap();
    rewritten.data
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use datafusion::{datasource::physical_plan::FileScanConfig, prelude::SessionContext};
    use liquid_cache_storage::{
        cache::squeeze_policies::TranscodeSqueezeEvict, cache_policies::LiquidPolicy,
    };

    use crate::LiquidCache;
    use liquid_cache_common::IoMode;

    use super::*;

    fn rewrite_plan_inner(plan: Arc<dyn ExecutionPlan>) {
        let expected_schema = plan.schema();
        let liquid_cache = Arc::new(LiquidCache::new(
            8192,
            1000000,
            PathBuf::from("test"),
            Box::new(LiquidPolicy::new()),
            Box::new(TranscodeSqueezeEvict),
            IoMode::Uring,
        ));
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
        rewrite_plan_inner(plan.clone());
    }
}
