use arrow_schema::SchemaRef;
use dashmap::DashMap;
use datafusion::{
    error::{DataFusionError, Result},
    execution::{SendableRecordBatchStream, object_store::ObjectStoreUrl, options::ReadOptions},
    physical_plan::{ExecutionPlan, display::DisplayableExecutionPlan, metrics::MetricValue},
    prelude::{ParquetReadOptions, SessionContext},
};
use liquid_cache_common::{CacheMode, rpc::ExecutionMetricsResponse};
use liquid_cache_parquet::{LiquidCache, LiquidCacheMode, LiquidCacheRef, LiquidParquetFileFormat};
use log::{debug, info};
use object_store::ObjectStore;
use std::{collections::HashMap, path::PathBuf, sync::Arc};
use tokio::sync::Mutex;
use tonic::Status;
use url::Url;

use crate::local_cache::LocalCache;

pub(crate) struct LiquidCacheServiceInner {
    execution_plans: Arc<DashMap<u64, Arc<dyn ExecutionPlan>>>,
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

    pub(crate) async fn register_table(
        &self,
        url_str: &str,
        table_name: &str,
        parquet_mode: CacheMode,
    ) -> Result<()> {
        let url =
            Url::parse(url_str).map_err(|e| DataFusionError::Configuration(format!("{e:?}")))?;

        let mut registered_tables = self.registered_tables.lock().await;
        if let Some((path, mode)) = registered_tables.get(table_name) {
            if path.as_str() == url_str && mode == &parquet_mode {
                info!("table {table_name} already registered at {path}");
                return Ok(());
            } else {
                panic!(
                    "table {table_name} already registered at {path} but not at {url} or not {parquet_mode}"
                );
            }
        }
        match parquet_mode {
            CacheMode::Parquet => {
                self.default_ctx
                    .register_parquet(table_name, url.as_str(), Default::default())
                    .await?;
            }
            CacheMode::Liquid => {
                let mode = LiquidCacheMode::InMemoryLiquid {
                    transcode_in_background: true,
                };
                self.register_liquid_parquet(table_name, url.as_str(), mode)
                    .await?;
            }
            CacheMode::LiquidEagerTranscode => {
                let mode = LiquidCacheMode::InMemoryLiquid {
                    transcode_in_background: false,
                };
                self.register_liquid_parquet(table_name, url.as_str(), mode)
                    .await?;
            }
            CacheMode::Arrow => {
                let mode = LiquidCacheMode::InMemoryArrow;
                self.register_liquid_parquet(table_name, url.as_str(), mode)
                    .await?;
            }
        }
        info!("registered table {table_name} from {url} as {parquet_mode}");
        registered_tables.insert(table_name.to_string(), (url_str.to_string(), parquet_mode));
        Ok(())
    }

    async fn register_liquid_parquet(
        &self,
        table_name: &str,
        url: &str,
        liquid_cache_mode: LiquidCacheMode,
    ) -> Result<()> {
        let parquet_options = ParquetReadOptions::default();
        let mut listing_options = parquet_options.to_listing_options(
            &self.default_ctx.copied_config(),
            self.default_ctx.copied_table_options(),
        );
        let format = listing_options.format;
        let table_parquet_options = self.default_ctx.state().table_options().parquet.clone();
        let liquid_parquet = LiquidParquetFileFormat::new(
            table_parquet_options,
            format,
            self.cache.clone(),
            liquid_cache_mode,
        );
        listing_options.format = Arc::new(liquid_parquet);

        self.default_ctx
            .register_listing_table(
                table_name,
                url,
                listing_options,
                parquet_options.schema.map(|s| Arc::new(s.to_owned())),
                None,
            )
            .await?;

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
        handle: u64,
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

    pub(crate) fn get_plan(&self, handle: u64) -> Option<Arc<dyn ExecutionPlan>> {
        let plan = self.execution_plans.get(&handle)?;
        Some(plan.clone())
    }

    pub(crate) async fn execute_plan(
        &self,
        handle: u64,
        partition: usize,
    ) -> SendableRecordBatchStream {
        let plan = self.execution_plans.get(&handle).unwrap();
        let displayable = DisplayableExecutionPlan::new(plan.as_ref());
        debug!("physical plan:\n{}", displayable.indent(false));
        let schema = plan.schema();
        debug!("execution plan schema: {:?}", schema);

        let ctx = self.default_ctx.clone();

        plan.execute(partition, ctx.task_ctx()).unwrap()
    }

    pub(crate) fn batch_size(&self) -> usize {
        self.default_ctx.state().config().batch_size()
    }

    pub(crate) fn get_metrics(&self, plan_id: u64) -> Option<ExecutionMetricsResponse> {
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
}

#[cfg(test)]
mod tests {
    use super::*;

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
