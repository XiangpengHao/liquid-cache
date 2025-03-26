use std::{collections::HashMap, sync::Arc};

use datafusion::{
    common::tree_node::{Transformed, TransformedResult, TreeNode},
    config::ConfigOptions,
    datasource::source::DataSourceExec,
    error::Result,
    execution::object_store::ObjectStoreUrl,
    physical_optimizer::PhysicalOptimizerRule,
    physical_plan::ExecutionPlan,
};
use liquid_cache_common::CacheMode;

use crate::client_exec::LiquidCacheClientExec;

/// PushdownOptimizer is a physical optimizer rule that pushes down filters to the liquid cache server.
#[derive(Debug)]
pub struct PushdownOptimizer {
    cache_server: String,
    cache_mode: CacheMode,
    object_stores: Vec<(ObjectStoreUrl, HashMap<String, String>)>,
}

impl PushdownOptimizer {
    /// Create a new PushdownOptimizer
    ///
    /// # Arguments
    ///
    /// * `cache_server` - The address of the liquid cache server
    /// * `cache_mode` - The cache mode to use
    ///
    /// # Returns
    ///
    pub fn new(
        cache_server: String,
        cache_mode: CacheMode,
        object_stores: Vec<(ObjectStoreUrl, HashMap<String, String>)>,
    ) -> Self {
        Self {
            cache_server,
            cache_mode,
            object_stores,
        }
    }
}

impl PhysicalOptimizerRule for PushdownOptimizer {
    fn optimize(
        &self,
        plan: Arc<dyn ExecutionPlan>,
        _config: &ConfigOptions,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        plan.transform_up(|plan| {
            let plan_any = plan.as_any();
            if let Some(_data_source) = plan_any.downcast_ref::<DataSourceExec>() {
                Ok(Transformed::yes(Arc::new(LiquidCacheClientExec::new(
                    plan,
                    self.cache_server.clone(),
                    self.cache_mode,
                    self.object_stores.clone(),
                ))))
            } else {
                Ok(Transformed::no(plan))
            }
        })
        .data()
    }

    fn name(&self) -> &str {
        "PushdownOptimizer"
    }

    fn schema_check(&self) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use datafusion::{
        execution::SessionStateBuilder,
        physical_plan::display::DisplayableExecutionPlan,
        prelude::{SessionConfig, SessionContext},
    };

    use super::*;

    #[tokio::test]
    async fn test_plan_rewrite() {
        let mut config = SessionConfig::from_env().unwrap();
        config.options_mut().execution.parquet.pushdown_filters = true;
        let builder = SessionStateBuilder::new()
            .with_config(config)
            .with_physical_optimizer_rule(Arc::new(PushdownOptimizer::new(
                "localhost:50051".to_string(),
                CacheMode::Liquid,
                vec![],
            )));
        let state = builder.build();
        let ctx = SessionContext::new_with_state(state);
        ctx.register_parquet(
            "nano_hits",
            "../../examples/nano_hits.parquet",
            Default::default(),
        )
        .await
        .unwrap();
        let df = ctx
            .sql("SELECT \"URL\" FROM nano_hits WHERE \"URL\" like 'https://%' limit 10")
            .await
            .unwrap();
        let plan = df.create_physical_plan().await.unwrap();
        let display_plan = DisplayableExecutionPlan::new(plan.as_ref());
        let plan_str = display_plan.indent(false).to_string();

        assert!(plan_str.starts_with("LiquidCacheClientExec\n  DataSourceExec:"));
    }
}
