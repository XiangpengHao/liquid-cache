use crate::ExecutionStats;
use anyhow::Result;
use datafusion::{
    execution::{SendableRecordBatchStream, object_store::ObjectStoreUrl},
    physical_plan::{ExecutionPlan, display::DisplayableExecutionPlan},
    prelude::SessionContext,
};
use liquid_cache::{ByteCache, cache::squeeze_policies::SqueezePolicy};
use liquid_cache::{cache::HydrationPolicy, cache_policies::CachePolicy};
use liquid_cache_common::{IoMode, rpc::ExecutionMetricsResponse};
use liquid_cache_datafusion::{
    cache::{LiquidCacheParquet, LiquidCacheParquetRef},
    extract_execution_metrics,
    optimizers::rewrite_data_source_plan,
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
    liquid_cache: LiquidCacheParquetRef,
    parquet_cache_dir: PathBuf,
}

impl LiquidCacheServiceInner {
    pub fn new(
        default_ctx: Arc<SessionContext>,
        max_cache_bytes: Option<usize>,
        disk_cache_dir: PathBuf,
        cache_policy: Box<dyn CachePolicy>,
        squeeze_policy: Box<dyn SqueezePolicy>,
        hydration_policy: Box<dyn HydrationPolicy>,
        io_mode: IoMode,
    ) -> Self {
        let batch_size = default_ctx.state().config().batch_size();

        let parquet_cache_dir = disk_cache_dir.join("parquet");
        let liquid_cache_dir = disk_cache_dir.join("liquid");

        let liquid_cache = Arc::new(LiquidCacheParquet::new(
            batch_size,
            max_cache_bytes.unwrap_or(usize::MAX),
            liquid_cache_dir,
            cache_policy,
            squeeze_policy,
            hydration_policy,
            io_mode,
        ));

        Self {
            execution_plans: Default::default(),
            execution_stats: Default::default(),
            default_ctx,
            liquid_cache,
            parquet_cache_dir,
        }
    }

    pub(crate) fn cache(&self) -> &LiquidCacheParquetRef {
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
            let local_cache = ByteCache::new(Arc::new(object_store), store_cache_dir);
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
        let cache = self.cache();
        self.execution_plans.write().unwrap().insert(
            handle,
            ExecutionPlanEntry::new(rewrite_data_source_plan(plan, cache, true)),
        );
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

        let response = extract_execution_metrics(&plan, Some(self.cache()));
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
    use liquid_cache::{
        cache::{AlwaysHydrate, squeeze_policies::TranscodeSqueezeEvict},
        cache_policies::LiquidPolicy,
    };
    #[tokio::test]
    async fn test_register_object_store() {
        let server = LiquidCacheServiceInner::new(
            Arc::new(SessionContext::new()),
            None,
            PathBuf::from("test"),
            Box::new(LiquidPolicy::new()),
            Box::new(TranscodeSqueezeEvict),
            Box::new(AlwaysHydrate::new()),
            IoMode::Uring,
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
