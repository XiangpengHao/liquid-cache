use std::{collections::HashMap, sync::Arc};

use arrow_schema::SchemaRef;
use dashmap::DashMap;
use datafusion::{
    error::{DataFusionError, Result},
    execution::{SendableRecordBatchStream, options::ReadOptions},
    physical_plan::{ExecutionPlan, display::DisplayableExecutionPlan},
    prelude::{ParquetReadOptions, SessionContext},
};
use log::{debug, info};
use tokio::sync::Mutex;
use tonic::Status;
use url::Url;

use liquid_parquet::{LiquidCache, LiquidCacheRef, LiquidParquetFileFormat};

use super::LiquidCacheConfig;

pub(crate) struct LiquidCacheServiceInner {
    execution_plans: Arc<DashMap<String, Arc<dyn ExecutionPlan>>>,
    registered_tables: Mutex<HashMap<String, String>>, // table name -> path
    default_ctx: Arc<SessionContext>,
    liquid_cache: LiquidCacheRef,
}

impl LiquidCacheServiceInner {
    pub fn new(default_ctx: Arc<SessionContext>, config: LiquidCacheConfig) -> Self {
        let batch_size = default_ctx.state().config().batch_size();
        let liquid_cache = Arc::new(LiquidCache::new(config.liquid_cache_mode, batch_size));
        Self {
            execution_plans: Default::default(),
            registered_tables: Default::default(),
            default_ctx,
            liquid_cache,
        }
    }

    pub fn new_ctx_and_cache(ctx: Arc<SessionContext>, cache: LiquidCacheRef) -> Self {
        Self {
            execution_plans: Default::default(),
            registered_tables: Default::default(),
            default_ctx: ctx,
            liquid_cache: cache,
        }
    }

    pub(crate) async fn register_table(&self, url_str: &str, table_name: &str) -> Result<()> {
        let url =
            Url::parse(url_str).map_err(|e| DataFusionError::Configuration(format!("{e:?}")))?;

        let mut registered_tables = self.registered_tables.lock().await;
        if let Some(path) = registered_tables.get(table_name) {
            if path.as_str() == url_str {
                info!("table {table_name} already registered at {path}");
                return Ok(());
            } else {
                panic!("table {table_name} already registered at {path} but not at {url}");
            }
        }

        // here we can't use register_parquet because it will use the default parquet format.
        // we want to override with liquid parquet format.
        self.register_liquid_parquet(table_name, url.as_str())
            .await?;
        info!("registered table {table_name} from {url}");
        registered_tables.insert(table_name.to_string(), url_str.to_string());
        Ok(())
    }

    async fn register_liquid_parquet(&self, table_name: &str, url: &str) -> Result<()> {
        let parquet_options = ParquetReadOptions::default();
        let mut listing_options = parquet_options.to_listing_options(
            &self.default_ctx.copied_config(),
            self.default_ctx.copied_table_options(),
        );
        let format = listing_options.format;
        let table_parquet_options = self.default_ctx.state().table_options().parquet.clone();
        let liquid_parquet =
            LiquidParquetFileFormat::new(table_parquet_options, format, self.liquid_cache.clone());
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
        handle: &str,
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

        self.execution_plans
            .insert(handle.to_string(), physical_plan.clone());

        Ok(physical_plan)
    }

    pub(crate) async fn execute_plan(
        &self,
        handle: &str,
        partition: usize,
    ) -> SendableRecordBatchStream {
        let plan = self.execution_plans.get(handle).unwrap();
        let displayable = DisplayableExecutionPlan::new(plan.as_ref());
        debug!("physical plan:\n{}", displayable.indent(false));
        let schema = plan.schema();
        debug!("execution plan schema: {:?}", schema);

        let ctx = self.default_ctx.clone();

        plan.execute(partition, ctx.task_ctx()).unwrap()
    }

    pub(crate) fn remove_plan(&self, handle: &str) {
        self.execution_plans.remove(&handle.to_string());
    }

    pub(crate) fn batch_size(&self) -> usize {
        self.default_ctx.state().config().batch_size()
    }
}
