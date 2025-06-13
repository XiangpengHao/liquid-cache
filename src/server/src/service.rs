use crate::{ExecutionStats, local_cache::LocalCache};
use anyhow::Result;
use datafusion::{
    common::tree_node::{TreeNode, TreeNodeRecursion},
    datasource::source::DataSourceExec,
    execution::{SendableRecordBatchStream, object_store::ObjectStoreUrl},
    physical_plan::{ExecutionPlan, display::DisplayableExecutionPlan, metrics::MetricValue},
    prelude::SessionContext,
};
use liquid_cache_common::{CacheEvictionStrategy, CacheMode, rpc::ExecutionMetricsResponse};
use liquid_cache_parquet::{LiquidCache, LiquidCacheRef};
use liquid_cache_parquet::{
    policies::{CachePolicy, DiscardPolicy, FiloPolicy, LruPolicy, ToDiskPolicy},
    rewrite_data_source_plan,
};
use log::{debug, info};
use object_store::ObjectStore;
use std::sync::RwLock;
use std::{collections::HashMap, path::PathBuf, sync::Arc, time::SystemTime};
use url::Url;
use uuid::Uuid;

#[derive(Clone)]
pub(crate) struct ExecutionPlanEntry {
    pub plan: Arc<dyn ExecutionPlan>,
    pub created_at: SystemTime,
}

impl ExecutionPlanEntry {
    pub fn new(plan: Arc<dyn ExecutionPlan>) -> Self {
        Self {
            plan,
            created_at: SystemTime::now(),
        }
    }
}

pub(crate) struct LiquidCacheServiceInner {
    execution_plans: RwLock<HashMap<Uuid, ExecutionPlanEntry>>,
    execution_stats: RwLock<Vec<ExecutionStats>>,
    default_ctx: Arc<SessionContext>,
    liquid_cache: Option<LiquidCacheRef>,
    parquet_cache_dir: PathBuf,
}

impl LiquidCacheServiceInner {
    pub fn new(
        default_ctx: Arc<SessionContext>,
        max_cache_bytes: Option<usize>,
        disk_cache_dir: PathBuf,
        cache_mode: CacheMode,
        case_eviction_policy: CacheEvictionStrategy,
    ) -> Self {
        let batch_size = default_ctx.state().config().batch_size();

        let parquet_cache_dir = disk_cache_dir.join("parquet");
        let liquid_cache_dir = disk_cache_dir.join("liquid");

        let cache_policy: Box<dyn CachePolicy> = match case_eviction_policy {
            CacheEvictionStrategy::Lru => Box::new(LruPolicy::new()),
            CacheEvictionStrategy::Discard => Box::new(DiscardPolicy),
            CacheEvictionStrategy::Filo => Box::new(FiloPolicy::new()),
            CacheEvictionStrategy::ToDisk => Box::new(ToDiskPolicy::new()),
        };

        let liquid_cache = match cache_mode {
            CacheMode::Parquet => None,
            _ => Some(Arc::new(LiquidCache::new(
                batch_size,
                max_cache_bytes.unwrap_or(usize::MAX),
                liquid_cache_dir,
                cache_mode.into(),
                cache_policy,
            ))),
        };

        Self {
            execution_plans: Default::default(),
            execution_stats: Default::default(),
            default_ctx,
            liquid_cache,
            parquet_cache_dir,
        }
    }

    pub(crate) fn cache(&self) -> &Option<LiquidCacheRef> {
        &self.liquid_cache
    }

    pub(crate) async fn register_object_store(
        &self,
        url: &Url,
        options: HashMap<String, String>,
    ) -> Result<()> {
        let (object_store, path) = object_store::parse_url_opts(url, options)?;
        if path.as_ref() != "" {
            return Err(anyhow::anyhow!(
                "object store url should not be a full path, got {}",
                path.as_ref()
            ));
        }
        let object_store_url = ObjectStoreUrl::parse(url.as_str())?;
        let existing = self
            .default_ctx
            .runtime_env()
            .object_store(&object_store_url);

        if existing.is_ok() {
            // if the object store is already registered, we don't need to register it again
            info!("object store already registered at {url}");
            return Ok(());
        }

        let object_store: Arc<dyn ObjectStore> = {
            let sanitized_url =
                liquid_cache_common::utils::sanitize_object_store_url_for_dirname(url);
            let store_cache_dir = self.parquet_cache_dir.join(sanitized_url);
            let local_cache = LocalCache::new(Arc::new(object_store), store_cache_dir);
            Arc::new(local_cache)
        };

        self.default_ctx
            .runtime_env()
            .register_object_store(object_store_url.as_ref(), object_store);
        Ok(())
    }

    /// Get a registered object store by its URL
    pub(crate) fn get_object_store(&self, url: &Url) -> Result<Arc<dyn ObjectStore>> {
        let (_, path) = object_store::parse_url(url)?;
        if path.as_ref() != "" {
            return Err(anyhow::anyhow!(
                "object store url should not be a full path, got {}",
                path.as_ref()
            ));
        }

        let object_store_url = ObjectStoreUrl::parse(url.as_str())?;
        let existing = self
            .default_ctx
            .runtime_env()
            .object_store(&object_store_url);

        match existing {
            Ok(store) => Ok(store),
            Err(_) => Err(anyhow::anyhow!("Object store {url} not registered")),
        }
    }

    pub(crate) fn register_plan(&self, handle: Uuid, plan: Arc<dyn ExecutionPlan>) {
        match self.cache() {
            Some(cache) => {
                self.execution_plans.write().unwrap().insert(
                    handle,
                    ExecutionPlanEntry::new(rewrite_data_source_plan(plan, cache)),
                );
            }
            None => {
                self.execution_plans
                    .write()
                    .unwrap()
                    .insert(handle, ExecutionPlanEntry::new(plan));
            }
        }
    }

    pub(crate) async fn execute_plan(
        &self,
        handle: &Uuid,
        partition: usize,
    ) -> Result<SendableRecordBatchStream> {
        let plan = self
            .execution_plans
            .read()
            .unwrap()
            .get(handle)
            .ok_or_else(|| anyhow::anyhow!("Plan not found"))?
            .plan
            .clone();
        let displayable = DisplayableExecutionPlan::new(plan.as_ref());
        debug!("physical plan:\n{}", displayable.indent(false));
        let ctx = self.default_ctx.clone();

        Ok(plan.execute(partition, ctx.task_ctx())?)
    }

    pub(crate) fn batch_size(&self) -> usize {
        self.default_ctx.state().config().batch_size()
    }

    pub(crate) fn get_metrics(&self, plan_id: &Uuid) -> Option<ExecutionMetricsResponse> {
        let plan = self
            .execution_plans
            .read()
            .unwrap()
            .get(plan_id)?
            .plan
            .clone();

        // Traverse the plan tree to find all DataSourceExec nodes and collect their metrics
        let mut time_elapsed_processing_millis = 0;
        let mut bytes_scanned = 0;

        plan.apply(|node| {
            let any_plan = node.as_any();
            if let Some(data_source_exec) = any_plan.downcast_ref::<DataSourceExec>()
                && let Some(metrics) = data_source_exec.metrics()
            {
                let aggregated_metrics = metrics
                    .aggregate_by_name()
                    .sorted_for_display()
                    .timestamps_removed();

                for metric in aggregated_metrics.iter() {
                    if let MetricValue::Time { name, time } = metric.value()
                        && name == "time_elapsed_processing"
                    {
                        time_elapsed_processing_millis += time.value() / 1_000_000;
                    } else if let MetricValue::Count { name, count } = metric.value()
                        && name == "bytes_scanned"
                    {
                        bytes_scanned += count.value();
                    }
                }
            }
            Ok(TreeNodeRecursion::Continue)
        })
        .ok()?;

        let liquid_cache_usage = self
            .cache()
            .as_ref()
            .map(|cache| cache.compute_memory_usage_bytes())
            .unwrap_or(0);
        let cache_memory_usage = liquid_cache_usage + bytes_scanned as u64;

        let response = ExecutionMetricsResponse {
            pushdown_eval_time: time_elapsed_processing_millis as u64,
            cache_memory_usage,
            liquid_cache_usage,
        };
        Some(response)
    }

    pub(crate) fn get_parquet_cache_dir(&self) -> &PathBuf {
        &self.parquet_cache_dir
    }

    pub(crate) fn get_ctx(&self) -> &Arc<SessionContext> {
        &self.default_ctx
    }

    pub(crate) fn get_plan(&self, id: &Uuid) -> Option<ExecutionPlanEntry> {
        self.execution_plans.read().unwrap().get(id).cloned()
    }

    pub(crate) fn get_execution_stats(&self) -> Vec<ExecutionStats> {
        self.execution_stats.read().unwrap().clone()
    }

    pub(crate) fn add_execution_stats(&self, execution_stats: ExecutionStats) {
        self.execution_stats.write().unwrap().push(execution_stats);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use liquid_cache_common::CacheEvictionStrategy::Discard;
    #[tokio::test]
    async fn test_register_object_store() {
        let server = LiquidCacheServiceInner::new(
            Arc::new(SessionContext::new()),
            None,
            PathBuf::from("test"),
            CacheMode::LiquidEagerTranscode,
            Discard,
        );
        let url = Url::parse("file:///").unwrap();
        server
            .register_object_store(&url, HashMap::new())
            .await
            .unwrap();

        let url = Url::parse("s3://test_data").unwrap();
        server
            .register_object_store(&url, HashMap::new())
            .await
            .unwrap();

        // reregister should be ok
        let url = Url::parse("s3://test_data").unwrap();
        server
            .register_object_store(&url, HashMap::new())
            .await
            .unwrap();

        let url = Url::parse("http://test_data").unwrap();
        server
            .register_object_store(&url, HashMap::new())
            .await
            .unwrap();

        let url = Url::parse("http://test_data/asdf").unwrap();
        let v = server.register_object_store(&url, HashMap::new()).await;
        assert!(v.is_err());

        let url = Url::parse("s3://test_data2").unwrap();
        let config = HashMap::from([
            ("access_key_id".to_string(), "test".to_string()),
            ("secret_access_key".to_string(), "test".to_string()),
        ]);
        server.register_object_store(&url, config).await.unwrap();

        let url = Url::parse("s3://test_data3/a.parquet").unwrap();
        let v = server.register_object_store(&url, HashMap::new()).await;
        assert!(v.is_err());

        let url = Url::parse("s3://test_data3/").unwrap();
        server
            .register_object_store(&url, HashMap::new())
            .await
            .unwrap();
    }
}
