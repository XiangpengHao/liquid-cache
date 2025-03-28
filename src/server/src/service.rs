use arrow_schema::SchemaRef;
use dashmap::DashMap;
use datafusion::{
    common::tree_node::{Transformed, TreeNode},
    datasource::{
        physical_plan::{FileScanConfig, ParquetSource},
        source::{DataSource, DataSourceExec},
    },
    error::{DataFusionError, Result},
    execution::{SendableRecordBatchStream, object_store::ObjectStoreUrl},
    physical_plan::{ExecutionPlan, display::DisplayableExecutionPlan, metrics::MetricValue},
    prelude::SessionContext,
};
use liquid_cache_common::{CacheMode, rpc::ExecutionMetricsResponse};
use liquid_cache_parquet::{LiquidCache, LiquidCacheRef, LiquidParquetSource};
use log::{debug, info};
use object_store::ObjectStore;
use std::{collections::HashMap, path::PathBuf, sync::Arc};
use tokio::sync::Mutex;
use tonic::Status;
use url::Url;
use uuid::Uuid;

use crate::local_cache::LocalCache;

pub(crate) struct LiquidCacheServiceInner {
    execution_plans: Arc<DashMap<Uuid, Arc<dyn ExecutionPlan>>>,
    registered_tables: Mutex<HashMap<String, (String, CacheMode)>>, // table name -> (path, cached file)
    default_ctx: Arc<SessionContext>,
    cache: LiquidCacheRef,
    parquet_cache_dir: PathBuf,
}

impl LiquidCacheServiceInner {
    pub fn new(
        default_ctx: Arc<SessionContext>,
        max_cache_bytes: Option<usize>,
        disk_cache_dir: Option<PathBuf>,
    ) -> Self {
        let batch_size = default_ctx.state().config().batch_size();

        let disk_cache_dir =
            disk_cache_dir.unwrap_or_else(|| tempfile::tempdir().unwrap().into_path());

        let parquet_cache_dir = disk_cache_dir.join("parquet");
        let liquid_cache_dir = disk_cache_dir.join("liquid");
        let liquid_cache = Arc::new(LiquidCache::new(
            batch_size,
            max_cache_bytes.unwrap_or(usize::MAX),
            liquid_cache_dir,
        ));

        Self {
            execution_plans: Default::default(),
            registered_tables: Default::default(),
            default_ctx,
            cache: liquid_cache,
            parquet_cache_dir,
        }
    }

    pub(crate) fn cache(&self) -> &LiquidCacheRef {
        &self.cache
    }

    pub(crate) async fn register_object_store(
        &self,
        url: &Url,
        options: HashMap<String, String>,
    ) -> Result<()> {
        let (object_store, path) = object_store::parse_url_opts(url, options)?;
        if path.as_ref() != "" {
            return Err(DataFusionError::Configuration(format!(
                "object store url should not be a full path, got {}",
                path.as_ref()
            )));
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

    pub(crate) async fn get_table_schema(&self, table_name: &str) -> Result<SchemaRef, Status> {
        let schema = self
            .default_ctx
            .table_provider(table_name)
            .await
            .unwrap()
            .schema();
        Ok(schema)
    }

    pub(crate) async fn prepare_and_register_plan(
        &self,
        query: &str,
        handle: Uuid,
    ) -> Result<Arc<dyn ExecutionPlan>, Status> {
        info!("Planning query: {query}");
        let ctx = self.default_ctx.clone();
        let plan = ctx.sql(query).await.expect("Error generating plan");
        let (state, plan) = plan.into_parts();
        let plan = state.optimize(&plan).expect("Error optimizing plan");
        let physical_plan = state
            .create_physical_plan(&plan)
            .await
            .expect("Error creating physical plan");

        self.execution_plans.insert(handle, physical_plan.clone());

        Ok(physical_plan)
    }

    pub(crate) fn register_plan(
        &self,
        handle: Uuid,
        plan: Arc<dyn ExecutionPlan>,
        cache_mode: CacheMode,
    ) {
        match cache_mode {
            CacheMode::Parquet => {
                self.execution_plans.insert(handle, plan);
            }
            _ => {
                self.execution_plans.insert(
                    handle,
                    rewrite_data_source_plan(plan, &self.cache, cache_mode),
                );
            }
        }
    }

    pub(crate) fn get_plan(&self, handle: &Uuid) -> Option<Arc<dyn ExecutionPlan>> {
        let plan = self.execution_plans.get(handle)?;
        Some(plan.clone())
    }

    pub(crate) async fn execute_plan(
        &self,
        handle: &Uuid,
        partition: usize,
    ) -> SendableRecordBatchStream {
        let plan = self.execution_plans.get(handle).unwrap();
        let displayable = DisplayableExecutionPlan::new(plan.as_ref());
        debug!("physical plan:\n{}", displayable.indent(false));
        let ctx = self.default_ctx.clone();

        plan.execute(partition, ctx.task_ctx()).unwrap()
    }

    pub(crate) fn batch_size(&self) -> usize {
        self.default_ctx.state().config().batch_size()
    }

    pub(crate) fn get_metrics(&self, plan_id: &Uuid) -> Option<ExecutionMetricsResponse> {
        let maybe_leaf = self.get_plan(plan_id)?;

        let displayable = DisplayableExecutionPlan::with_metrics(maybe_leaf.as_ref());
        debug!("physical plan:\n{}", displayable.indent(true));

        let plan = if let Some(plan) = maybe_leaf.children().first() {
            *plan
        } else {
            &maybe_leaf
        };
        let metrics = plan
            .metrics()
            .unwrap()
            .aggregate_by_name()
            .sorted_for_display()
            .timestamps_removed();

        let mut time_elapsed_processing_millis = 0;
        let mut bytes_scanned = 0;
        for metric in metrics.iter() {
            if let MetricValue::Time { name, time } = metric.value() {
                if name == "time_elapsed_processing" {
                    time_elapsed_processing_millis = time.value() / 1_000_000;
                }
            } else if let MetricValue::Count { name, count } = metric.value() {
                if name == "bytes_scanned" {
                    bytes_scanned = count.value();
                }
            }
        }
        let liquid_cache_usage = self.cache().compute_memory_usage_bytes();
        let cache_memory_usage = liquid_cache_usage + bytes_scanned as u64;

        let response = ExecutionMetricsResponse {
            pushdown_eval_time: time_elapsed_processing_millis as u64,
            cache_memory_usage,
            liquid_cache_usage,
        };
        Some(response)
    }

    pub(crate) async fn get_registered_tables(&self) -> HashMap<String, (String, CacheMode)> {
        let tables = self.registered_tables.lock().await;
        tables.clone()
    }

    pub(crate) fn get_parquet_cache_dir(&self) -> &PathBuf {
        &self.parquet_cache_dir
    }

    pub(crate) fn get_ctx(&self) -> &Arc<SessionContext> {
        &self.default_ctx
    }
}

fn rewrite_data_source_plan(
    plan: Arc<dyn ExecutionPlan>,
    cache: &LiquidCacheRef,
    cache_mode: CacheMode,
) -> Arc<dyn ExecutionPlan> {
    let rewritten = plan
        .transform_up(|node| {
            let any_plan = node.as_any();
            if let Some(plan) = any_plan.downcast_ref::<DataSourceExec>() {
                let data_source = plan.data_source();
                let any_source = data_source.as_any();
                if let Some(source) = any_source.downcast_ref::<FileScanConfig>() {
                    let file_source = source.file_source();
                    let any_file_source = file_source.as_any();
                    if let Some(file_source) = any_file_source.downcast_ref::<ParquetSource>() {
                        let new_source = LiquidParquetSource::from_parquet_source(
                            file_source.clone(),
                            cache.clone(),
                            cache_mode.into(),
                        );
                        let mut new_file_source = source.clone();
                        new_file_source.file_source = Arc::new(new_source);
                        // let coerced_schema =
                        //     coerce_to_liquid_cache_types(new_file_source.file_schema.as_ref());
                        new_file_source.projection = new_file_source.projection.map(|mut v| {
                            v.sort();
                            v
                        });
                        // new_file_source.file_schema = Arc::new(coerced_schema);
                        let new_file_source: Arc<dyn DataSource> = Arc::new(new_file_source);
                        let new_plan = Arc::new(DataSourceExec::new(new_file_source));

                        // data source is at the bottom of the plan tree, so we can stop the recursion
                        return Ok(Transformed::new(
                            new_plan,
                            true,
                            datafusion::common::tree_node::TreeNodeRecursion::Stop,
                        ));
                    }
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
    use datafusion::common::tree_node::TreeNodeRecursion;

    use super::*;

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
            .sql("SELECT \"URL\" FROM nano_hits WHERE \"URL\" like 'https://%' limit 10")
            .await
            .unwrap();
        let plan = df.create_physical_plan().await.unwrap();
        let liquid_cache = Arc::new(LiquidCache::new(8192, 1000000, PathBuf::from("test")));
        let rewritten = rewrite_data_source_plan(plan, &liquid_cache, CacheMode::Liquid);

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
                }
                Ok(TreeNodeRecursion::Continue)
            })
            .unwrap();
    }

    #[tokio::test]
    async fn test_register_object_store() {
        let server = LiquidCacheServiceInner::new(Arc::new(SessionContext::new()), None, None);
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
