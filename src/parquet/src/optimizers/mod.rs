//! Optimizers for the Parquet module

mod date_extract_opt;

use std::sync::Arc;

use datafusion::{
    catalog::memory::DataSourceExec,
    common::tree_node::{Transformed, TreeNode, TreeNodeRecursion},
    datasource::{physical_plan::ParquetSource, source::DataSource},
    physical_plan::ExecutionPlan,
};
pub use date_extract_opt::DateExtractOptimizer;

use crate::{LiquidCacheRef, LiquidParquetSource};

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
                    let new_source = LiquidParquetSource::from_parquet_source(
                        parquet_source.clone(),
                        file_scan_config.file_schema.clone(),
                        cache.clone(),
                    );
                    let mut new_config = file_scan_config.clone();
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

    use super::*;

    fn rewrite_plan_inner(plan: Arc<dyn ExecutionPlan>) {
        let expected_schema = plan.schema();
        let liquid_cache = Arc::new(LiquidCache::new(
            8192,
            1000000,
            PathBuf::from("test"),
            Box::new(LiquidPolicy::new()),
            Box::new(TranscodeSqueezeEvict),
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
                    let schema = source.file_schema.as_ref();
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
