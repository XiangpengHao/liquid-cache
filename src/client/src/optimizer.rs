use std::{collections::HashMap, sync::Arc};

use datafusion::{
    config::ConfigOptions, datasource::source::DataSourceExec, error::Result,
    execution::object_store::ObjectStoreUrl, physical_optimizer::PhysicalOptimizerRule,
    physical_plan::ExecutionPlan, physical_plan::aggregates::AggregateExec,
    physical_plan::aggregates::AggregateMode, physical_plan::repartition::RepartitionExec,
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

    /// Apply the optimization by finding nodes to push down and wrapping them
    fn optimize_plan(&self, plan: Arc<dyn ExecutionPlan>) -> Result<Arc<dyn ExecutionPlan>> {
        // If this node is already a LiquidCacheClientExec, return it as is
        if plan
            .as_any()
            .downcast_ref::<LiquidCacheClientExec>()
            .is_some()
        {
            return Ok(plan);
        }

        // Find the candidate to push down in this branch of the tree
        if let Some(candidate) = find_pushdown_candidate(&plan) {
            // If the current node is the one to be pushed down, wrap it
            if Arc::ptr_eq(&plan, &candidate) {
                return Ok(Arc::new(LiquidCacheClientExec::new(
                    plan,
                    self.cache_server.clone(),
                    self.cache_mode,
                    self.object_stores.clone(),
                )));
            }
        }

        // Otherwise, recurse into children
        let mut new_children = Vec::with_capacity(plan.children().len());
        let mut children_changed = false;

        for child in plan.children() {
            let new_child = self.optimize_plan(child.clone())?;
            if !Arc::ptr_eq(child, &new_child) {
                children_changed = true;
            }
            new_children.push(new_child);
        }

        // If any children were changed, create a new plan with the updated children
        if children_changed {
            plan.with_new_children(new_children)
        } else {
            Ok(plan)
        }
    }
}

/// Find the highest pushable node
fn find_pushdown_candidate(plan: &Arc<dyn ExecutionPlan>) -> Option<Arc<dyn ExecutionPlan>> {
    // Check if this node is already a LiquidCacheClientExec to avoid redundant wrapping
    if plan
        .as_any()
        .downcast_ref::<LiquidCacheClientExec>()
        .is_some()
    {
        return None;
    }

    let plan_any = plan.as_any();

    // If we have an AggregateExec (partial, no group by) with a pushable child (direct or through RepartitionExec), push it down
    if let Some(agg_exec) = plan_any.downcast_ref::<AggregateExec>()
        && matches!(agg_exec.mode(), AggregateMode::Partial)
        && agg_exec.group_expr().is_empty()
    {
        let child = agg_exec.input();

        // Check if child is DataSourceExec or RepartitionExec->DataSourceExec
        if child.as_any().downcast_ref::<DataSourceExec>().is_some() {
            return Some(plan.clone());
        } else if let Some(repart) = child.as_any().downcast_ref::<RepartitionExec>()
            && let Some(repart_child) = repart.children().first()
            && repart_child
                .as_any()
                .downcast_ref::<DataSourceExec>()
                .is_some()
        {
            return Some(plan.clone());
        }
    }

    // If we have a RepartitionExec with a DataSourceExec child, push it down
    if let Some(repart_exec) = plan_any.downcast_ref::<RepartitionExec>()
        && let Some(child) = repart_exec.children().first()
        && child.as_any().downcast_ref::<DataSourceExec>().is_some()
    {
        return Some(plan.clone());
    }

    // If this is a DataSourceExec, push it down
    if plan_any.downcast_ref::<DataSourceExec>().is_some() {
        return Some(plan.clone());
    }

    // Otherwise, recurse into children looking for pushdown candidates
    for child in plan.children() {
        if let Some(candidate) = find_pushdown_candidate(child) {
            return Some(candidate);
        }
    }

    None
}

impl PhysicalOptimizerRule for PushdownOptimizer {
    fn optimize(
        &self,
        plan: Arc<dyn ExecutionPlan>,
        _config: &ConfigOptions,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        self.optimize_plan(plan)
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
    use std::sync::Arc;

    use arrow::datatypes::{DataType, Field, Schema, SchemaRef};
    use datafusion::{
        config::ConfigOptions,
        datasource::memory::MemorySourceConfig,
        error::Result,
        execution::SessionStateBuilder,
        physical_plan::{
            ExecutionPlan,
            aggregates::{AggregateExec, AggregateMode, PhysicalGroupBy},
            display::DisplayableExecutionPlan,
            repartition::RepartitionExec,
        },
        prelude::{SessionConfig, SessionContext},
    };
    use liquid_cache_common::CacheMode;

    use super::*;

    async fn create_session_context() -> SessionContext {
        let mut config = SessionConfig::from_env().unwrap();
        config.options_mut().execution.parquet.pushdown_filters = true;
        let builder = SessionStateBuilder::new()
            .with_config(config)
            .with_default_features()
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
        ctx
    }

    #[tokio::test]
    async fn test_plan_rewrite() {
        let ctx = create_session_context().await;
        let df = ctx
            .sql("SELECT \"URL\" FROM nano_hits WHERE \"URL\" like 'https://%' limit 10")
            .await
            .unwrap();
        let plan = df.create_physical_plan().await.unwrap();
        let display_plan = DisplayableExecutionPlan::new(plan.as_ref());
        let plan_str = display_plan.indent(false).to_string();

        assert!(plan_str.contains("LiquidCacheClientExec"));
        assert!(plan_str.contains("DataSourceExec"));
    }

    #[tokio::test]
    async fn test_aggregate_pushdown() {
        let ctx = create_session_context().await;

        let df = ctx
            .sql("SELECT MAX(\"URL\") FROM nano_hits WHERE \"URL\" like 'https://%'")
            .await
            .unwrap();
        let plan = df.create_physical_plan().await.unwrap();
        let display_plan = DisplayableExecutionPlan::new(plan.as_ref());
        let plan_str = display_plan.indent(false).to_string();

        println!("Plan: {}", plan_str);

        // With the top-down approach, the LiquidCacheClientExec should contain:
        // 1. The AggregateExec with mode=Partial
        // 2. Any RepartitionExec below that
        // 3. The DataSourceExec at the bottom

        // Verify that AggregateExec: mode=Partial is inside the LiquidCacheClientExec
        assert!(plan_str.contains("LiquidCacheClientExec"));

        let parts: Vec<&str> = plan_str.split("LiquidCacheClientExec").collect();
        assert!(parts.len() > 1);

        let higher_layers = parts[0];
        let pushed_down = parts[1];

        assert!(higher_layers.contains("AggregateExec: mode=Final"));
        assert!(pushed_down.contains("AggregateExec: mode=Partial"));
        assert!(pushed_down.contains("DataSourceExec"));
    }

    // Create a test schema for our mock plans
    fn create_test_schema() -> SchemaRef {
        Arc::new(Schema::new(vec![
            Field::new("c1", DataType::Int32, true),
            Field::new("c2", DataType::Utf8, true),
            Field::new("c3", DataType::Float64, true),
        ]))
    }

    // Mock DataSourceExec that we can use in our tests
    fn create_datasource_exec(schema: SchemaRef) -> Arc<dyn ExecutionPlan> {
        Arc::new(DataSourceExec::new(Arc::new(
            MemorySourceConfig::try_new(&[vec![]], schema, None).unwrap(),
        )))
    }

    // Apply the PushdownOptimizer to a plan and get the result as a string for comparison
    fn apply_optimizer(plan: Arc<dyn ExecutionPlan>) -> String {
        let optimizer =
            PushdownOptimizer::new("localhost:50051".to_string(), CacheMode::Liquid, vec![]);

        let optimized = optimizer.optimize(plan, &ConfigOptions::default()).unwrap();
        let display_plan = DisplayableExecutionPlan::new(optimized.as_ref());
        display_plan.indent(false).to_string()
    }

    #[test]
    fn test_simple_datasource_pushdown() -> Result<()> {
        let schema = create_test_schema();
        let datasource = create_datasource_exec(schema);
        let result = apply_optimizer(datasource);
        assert!(result.starts_with("LiquidCacheClientExec"));
        Ok(())
    }

    #[test]
    fn test_repartition_datasource_pushdown() -> Result<()> {
        let schema = create_test_schema();
        let datasource = create_datasource_exec(schema);
        let repartition = Arc::new(RepartitionExec::try_new(
            datasource,
            datafusion::physical_plan::Partitioning::RoundRobinBatch(4),
        )?);

        let result = apply_optimizer(repartition);

        assert!(result.starts_with("LiquidCacheClientExec"));
        assert!(result.contains("RepartitionExec"));

        Ok(())
    }

    #[test]
    fn test_partial_aggregate_pushdown() -> Result<()> {
        // Create an AggregateExec (Partial, no group by) -> DataSourceExec plan
        let schema = create_test_schema();
        let datasource = create_datasource_exec(schema.clone());

        let group_by = PhysicalGroupBy::new_single(vec![]);

        let aggregate = Arc::new(AggregateExec::try_new(
            AggregateMode::Partial,
            group_by,
            vec![],
            vec![],
            datasource,
            schema.clone(),
        )?);

        let result = apply_optimizer(aggregate);

        assert!(result.starts_with("LiquidCacheClientExec"));
        assert!(result.contains("AggregateExec: mode=Partial"));

        Ok(())
    }

    #[test]
    fn test_aggregate_with_repartition_pushdown() -> Result<()> {
        // Create an AggregateExec (Partial, no group by) -> RepartitionExec -> DataSourceExec plan
        let schema = create_test_schema();
        let datasource = create_datasource_exec(schema.clone());

        let repartition = Arc::new(RepartitionExec::try_new(
            datasource,
            datafusion::physical_plan::Partitioning::RoundRobinBatch(4),
        )?);

        let group_by = PhysicalGroupBy::new_single(vec![]);
        let aggregate = Arc::new(AggregateExec::try_new(
            AggregateMode::Partial,
            group_by,
            vec![],
            vec![],
            repartition,
            schema.clone(),
        )?);

        let result = apply_optimizer(aggregate);

        assert!(result.starts_with("LiquidCacheClientExec"));
        assert!(result.contains("AggregateExec: mode=Partial"));
        assert!(result.contains("RepartitionExec"));

        Ok(())
    }

    #[test]
    fn test_non_pushable_aggregate() -> Result<()> {
        // Create an AggregateExec (Final, no group by) -> DataSourceExec plan
        // This should not push down the AggregateExec
        let schema = create_test_schema();
        let datasource = create_datasource_exec(schema.clone());

        let group_by = PhysicalGroupBy::new_single(vec![]);

        let aggregate = Arc::new(AggregateExec::try_new(
            AggregateMode::Final,
            group_by,
            vec![],
            vec![],
            datasource,
            schema.clone(),
        )?);

        let result = apply_optimizer(aggregate);

        let parts: Vec<&str> = result.split("LiquidCacheClientExec").collect();
        assert!(parts.len() > 1);

        let higher_layers = parts[0];
        assert!(higher_layers.contains("AggregateExec: mode=Final"));
        let lower_layers = parts[1];
        assert!(lower_layers.contains("DataSourceExec"));

        Ok(())
    }
}
