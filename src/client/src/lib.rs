#![warn(missing_docs)]
#![cfg_attr(not(doctest), doc = include_str!(concat!("../", std::env!("CARGO_PKG_README"))))]
use std::any::Any;
use std::collections::HashMap;
use std::error::Error;
use std::fmt::Debug;
use std::sync::Arc;
use std::time::Duration;
mod client_exec;
mod exec;
mod metrics;
mod optimizer;
mod sql;
use arrow_flight::FlightInfo;
use arrow_flight::error::FlightError;
use arrow_schema::{DataType, Field, Schema, SchemaRef};
use async_trait::async_trait;
pub use client_exec::LiquidCacheClientExec;
use datafusion::{
    catalog::{Session, TableProvider},
    common::{Statistics, ToDFSchema, stats::Precision},
    datasource::{DefaultTableSource, TableType, empty::EmptyTable},
    error::{DataFusionError, Result},
    execution::{SessionStateBuilder, object_store::ObjectStoreUrl, runtime_env::RuntimeEnv},
    logical_expr::{LogicalPlan, TableProviderFilterPushDown, TableScan},
    physical_plan::ExecutionPlan,
    prelude::*,
    sql::{
        TableReference,
        unparser::{Unparser, dialect::PostgreSqlDialect},
    },
};
pub use exec::FlightExec;
use liquid_cache_common::CacheMode;
use log::info;
pub use optimizer::PushdownOptimizer;
use owo_colors::OwoColorize;
use serde::{Deserialize, Serialize};
use sql::FlightSqlDriver;
use tonic::transport::Channel;

fn transform_flight_schema_to_output_schema(schema: &SchemaRef) -> Schema {
    let transformed_fields: Vec<Arc<Field>> = schema
        .fields
        .iter()
        .map(|field| match field.data_type() {
            DataType::Dictionary(key, value) => {
                if key.as_ref() == &DataType::UInt16 && value.as_ref() == &DataType::Utf8 {
                    Arc::new(field.as_ref().clone().with_data_type(DataType::Utf8))
                } else if key.as_ref() == &DataType::UInt16 && value.as_ref() == &DataType::Binary {
                    Arc::new(field.as_ref().clone().with_data_type(DataType::Binary))
                } else {
                    field.clone()
                }
            }
            _ => field.clone(),
        })
        .collect();
    Schema::new_with_metadata(transformed_fields, schema.metadata.clone())
}

/// The builder for LiquidCache client state.
pub struct LiquidCacheBuilder {
    object_stores: Vec<(ObjectStoreUrl, HashMap<String, String>)>,
    cache_mode: CacheMode,
    cache_server: String,
}

impl LiquidCacheBuilder {
    /// Create a new builder for LiquidCache client state.
    pub fn new(cache_server: impl AsRef<str>) -> Self {
        Self {
            object_stores: vec![],
            cache_mode: CacheMode::Liquid,
            cache_server: cache_server.as_ref().to_string(),
        }
    }

    /// Add an object store to the builder.
    pub fn with_object_store(
        mut self,
        url: ObjectStoreUrl,
        object_store_options: Option<HashMap<String, String>>,
    ) -> Self {
        self.object_stores
            .push((url, object_store_options.unwrap_or_default()));
        self
    }

    /// Set the cache mode for the builder.
    pub fn with_cache_mode(mut self, cache_mode: CacheMode) -> Self {
        self.cache_mode = cache_mode;
        self
    }

    /// Build the [SessionContext].
    pub fn build(self, config: SessionConfig) -> Result<SessionContext> {
        let mut session_config = config;
        session_config
            .options_mut()
            .execution
            .parquet
            .pushdown_filters = true;
        session_config
            .options_mut()
            .execution
            .parquet
            .schema_force_view_types = false;
        let session_state = SessionStateBuilder::new()
            .with_config(session_config)
            .with_runtime_env(Arc::new(RuntimeEnv::default()))
            .with_default_features()
            .with_physical_optimizer_rule(Arc::new(PushdownOptimizer::new(
                self.cache_server.clone(),
                self.cache_mode,
                self.object_stores.clone(),
            )))
            .build();
        Ok(SessionContext::new_with_state(session_state))
    }
}

/// The builder for a [LiquidCacheTable].
///
/// # Example
///
/// ```ignore
/// let mut session_config = SessionConfig::from_env()?;
/// session_config
///     .options_mut()
///     .execution
///     .parquet
///     .pushdown_filters = true;
/// let ctx = Arc::new(SessionContext::new_with_config(session_config));
/// let table = LiquidCacheTableBuilder::new(cache_server, table_name, url.as_ref())
///     .with_object_store(
///         format!("{}://{}", url.scheme(), url.host_str().unwrap_or_default()),
///         None,
///     )
///     .build()
///     .await?;
/// ctx.register_table(table_name, Arc::new(table))?;
/// ctx.sql(&sql).await?.show().await?;
/// ```
#[derive(Clone, Debug)]
pub struct LiquidCacheTableBuilder {
    driver: Arc<FlightSqlDriver>,
    object_stores: Vec<(String, HashMap<String, String>)>,
    cache_mode: CacheMode,
    cache_server: String,
    table_name: String,
    table_url: String,
}

impl LiquidCacheTableBuilder {
    /// Create a new builder for a [LiquidCacheTable].
    ///
    /// # Arguments
    ///
    /// * `cache_server` - The address of the cache server
    /// * `table_name` - The name of the table
    /// * `table_url` - The url of the table
    pub fn new(
        cache_server: impl AsRef<str>,
        table_name: impl AsRef<str>,
        table_url: impl AsRef<str>,
    ) -> Self {
        Self {
            driver: Arc::new(FlightSqlDriver::default()),
            object_stores: vec![],
            cache_mode: CacheMode::Liquid,
            cache_server: cache_server.as_ref().to_string(),
            table_name: table_name.as_ref().to_string(),
            table_url: table_url.as_ref().to_string(),
        }
    }

    /// Add an object store to the builder.
    ///
    /// # Arguments
    ///
    /// * `url` - The url of the object store
    /// * `object_store_options` - The options for the object store
    ///
    /// # Example
    ///
    /// ```rust
    /// use liquid_cache_client::LiquidCacheTableBuilder;
    /// use std::collections::HashMap;
    ///
    /// let mut builder = LiquidCacheTableBuilder::new("localhost:50051", "my_table", "my_table_url");
    /// let object_store_options = HashMap::new();
    /// builder.with_object_store("s3://my_bucket", Some(object_store_options)).build();
    /// ```
    pub fn with_object_store(
        mut self,
        url: impl AsRef<str>,
        object_store_options: Option<HashMap<String, String>>,
    ) -> Self {
        self.object_stores.push((
            url.as_ref().to_string(),
            object_store_options.unwrap_or_default(),
        ));
        self
    }

    /// Set the cache mode for the builder.
    ///
    /// # Arguments
    ///
    /// * `cache_mode` - The cache mode to use
    pub fn with_cache_mode(mut self, cache_mode: CacheMode) -> Self {
        self.cache_mode = cache_mode;
        self
    }

    /// Build the [LiquidCacheTable].
    /// It will communicate with the cache server to get the metadata of the table.
    pub async fn build(self) -> Result<LiquidCacheTable> {
        let channel = flight_channel(&self.cache_server).await?;
        for (url, object_store_options) in self.object_stores {
            self.driver
                .register_object_store(channel.clone(), &url, object_store_options)
                .await
                .map_err(to_df_err)?;
        }

        let metadata = self
            .driver
            .metadata(
                channel.clone(),
                &self.table_name,
                &self.table_url,
                self.cache_mode,
            )
            .await
            .map_err(to_df_err)?;
        let num_rows = precision(metadata.info.total_records);
        let total_byte_size = precision(metadata.info.total_bytes);
        let output_schema = Arc::new(transform_flight_schema_to_output_schema(&metadata.schema));
        let flight_schema = metadata.schema;
        let stats = Statistics {
            num_rows,
            total_byte_size,
            column_statistics: vec![],
        };
        Ok(LiquidCacheTable {
            driver: self.driver.clone(),
            channel,
            origin: self.cache_server,
            table_name: self.table_name.into(),
            flight_schema,
            output_schema,
            stats,
        })
    }
}

/// The information that a [FlightSqlDriver] must produce
/// in order to register flights as DataFusion tables.
#[derive(Clone, Debug)]
pub(crate) struct FlightMetadata {
    /// FlightInfo object produced by the driver
    pub(crate) info: FlightInfo,
    /// Various knobs that control execution
    pub(crate) props: FlightProperties,
    /// Arrow schema. Can be enforced by the driver or inferred from the FlightInfo
    pub(crate) schema: SchemaRef,
}

impl FlightMetadata {
    /// Customize everything that is in the driver's control
    pub fn new(info: FlightInfo, props: FlightProperties, schema: SchemaRef) -> Self {
        Self {
            info,
            props,
            schema,
        }
    }

    /// Customize flight properties and try to use the FlightInfo schema
    // TODO: fix from upstream
    #[allow(clippy::result_large_err)]
    pub fn try_new(info: FlightInfo, props: FlightProperties) -> arrow_flight::error::Result<Self> {
        let schema = Arc::new(info.clone().try_decode_schema()?);
        Ok(Self::new(info, props, schema))
    }
}

impl TryFrom<FlightInfo> for FlightMetadata {
    type Error = FlightError;

    fn try_from(info: FlightInfo) -> Result<Self, Self::Error> {
        Self::try_new(info, FlightProperties::default())
    }
}

/// Meant to gradually encapsulate all sorts of knobs required
/// for controlling the protocol and query execution details.
#[derive(Clone, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
pub(crate) struct FlightProperties {
    pub(crate) unbounded_stream: bool,
    pub(crate) grpc_headers: HashMap<String, String>,
}

impl FlightProperties {
    pub fn grpc_headers(mut self, grpc_headers: HashMap<String, String>) -> Self {
        self.grpc_headers = grpc_headers;
        self
    }
}

/// Table provider that wraps a specific flight from an Arrow Flight service
#[derive(Debug)]
pub struct LiquidCacheTable {
    driver: Arc<FlightSqlDriver>,
    channel: Channel,
    origin: String,
    table_name: TableReference,
    /// Flight schema is the schema sent between network.
    flight_schema: SchemaRef,
    /// Output schema is the schema we emit to users.
    /// The flight schema is only optimized for transmission, so we need a schema adapter to change it to the output schema.
    output_schema: SchemaRef,
    stats: Statistics,
}

#[async_trait]
impl TableProvider for LiquidCacheTable {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn schema(&self) -> SchemaRef {
        self.output_schema.clone()
    }

    fn table_type(&self) -> TableType {
        TableType::View
    }

    async fn scan(
        &self,
        _state: &dyn Session,
        projection: Option<&Vec<usize>>,
        filters: &[Expr],
        limit: Option<usize>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        let unparsed_sql = {
            // we don't care about actual source for the purpose of unparsing the sql.
            let empty_table_provider = EmptyTable::new(self.schema().clone());
            let table_source = Arc::new(DefaultTableSource::new(Arc::new(empty_table_provider)));

            let logical_plan = TableScan {
                table_name: self.table_name.clone(),
                source: table_source,
                projection: projection.map(|p| p.to_vec()),
                filters: filters.to_vec(),
                fetch: limit,
                projected_schema: Arc::new(self.schema().as_ref().clone().to_dfschema().unwrap()),
            };
            let unparser = Unparser::new(&PostgreSqlDialect {});
            let unparsed_sql = unparser
                .plan_to_sql(&LogicalPlan::TableScan(logical_plan))
                .unwrap();
            unparsed_sql.to_string()
        };

        info!("SQL send to cache: \n{}", unparsed_sql.cyan());

        let metadata = self
            .driver
            .run_sql(self.channel.clone(), &unparsed_sql)
            .await
            .map_err(to_df_err)?;

        Ok(Arc::new(FlightExec::try_new(
            self.flight_schema.clone(),
            self.output_schema.clone(),
            metadata,
            projection,
            &self.origin,
            limit,
        )?))
    }

    fn statistics(&self) -> Option<Statistics> {
        Some(self.stats.clone())
    }

    fn supports_filters_pushdown(
        &self,
        filters: &[&Expr],
    ) -> Result<Vec<TableProviderFilterPushDown>> {
        let filter_push_down: Vec<TableProviderFilterPushDown> = filters
            .iter()
            .map(
                |f| match Unparser::new(&PostgreSqlDialect {}).expr_to_sql(f) {
                    Ok(_) => TableProviderFilterPushDown::Exact,
                    Err(_) => TableProviderFilterPushDown::Unsupported,
                },
            )
            .collect();

        Ok(filter_push_down)
    }
}

pub(crate) fn to_df_err<E: Error + Send + Sync + 'static>(err: E) -> DataFusionError {
    DataFusionError::External(Box::new(err))
}

pub(crate) async fn flight_channel(source: impl Into<String>) -> Result<Channel> {
    // No tls here, to avoid the overhead of TLS
    // we assume both server and client are running on the trusted network.
    let endpoint = Channel::from_shared(source.into())
        .map_err(to_df_err)?
        .tcp_keepalive(Some(Duration::from_secs(10)));
    endpoint.connect().await.map_err(to_df_err)
}

fn precision(total: i64) -> Precision<usize> {
    if total < 0 {
        Precision::Absent
    } else {
        Precision::Exact(total as usize)
    }
}
