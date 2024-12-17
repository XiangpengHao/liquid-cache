use std::any::Any;
use std::collections::HashMap;
use std::error::Error;
use std::fmt::Debug;
use std::sync::Arc;

use crate::exec::FlightExec;
use crate::sql::FlightSqlDriver;
use arrow_flight::error::FlightError;
use arrow_flight::FlightInfo;
use arrow_schema::{DataType, Field, Schema, SchemaRef};
use async_trait::async_trait;
use datafusion::datasource::empty::EmptyTable;
use datafusion::datasource::DefaultTableSource;
use datafusion_catalog::{Session, TableProvider, TableProviderFactory};
use datafusion_common::error::Result;
use datafusion_common::stats::Precision;
use datafusion_common::{DataFusionError, Statistics, ToDFSchema};
use datafusion_expr::{
    CreateExternalTable, Expr, LogicalPlan, TableProviderFilterPushDown, TableScan, TableType,
};
use datafusion_physical_plan::ExecutionPlan;
use datafusion_sql::unparser::dialect::PostgreSqlDialect;
use datafusion_sql::unparser::Unparser;
use datafusion_sql::TableReference;
use log::info;
use serde::{Deserialize, Serialize};
use tonic::transport::Channel;

fn transform_flight_schema_to_original_type(schema: &SchemaRef) -> Schema {
    let transformed_fields: Vec<Arc<Field>> = schema
        .fields
        .iter()
        .map(|field| match field.data_type() {
            DataType::Dictionary(key, value) => {
                if key.as_ref() == &DataType::UInt16 && value.as_ref() == &DataType::Utf8 {
                    Arc::new(field.as_ref().clone().with_data_type(DataType::Utf8))
                } else {
                    field.clone()
                }
            }
            _ => field.clone(),
        })
        .collect();
    Schema::new_with_metadata(transformed_fields, schema.metadata.clone())
}

/// Generic Arrow Flight data source. Requires a [FlightDriver] that allows implementors
/// to integrate any custom Flight RPC service by producing a [FlightMetadata] for some DDL.
#[derive(Clone, Debug)]
pub struct FlightTableFactory {
    driver: Arc<FlightSqlDriver>,
}

impl FlightTableFactory {
    /// Create a data source using the provided driver
    pub fn new(driver: Arc<FlightSqlDriver>) -> Self {
        Self { driver }
    }

    /// Convenient way to create a [FlightTable] programatically, as an alternative to DDL.
    pub async fn open_table(
        &self,
        entry_point: impl Into<String>,
        options: HashMap<String, String>,
        table_name: impl Into<TableReference>,
    ) -> Result<FlightTable> {
        let origin = entry_point.into();
        let channel = flight_channel(&origin).await?;

        let metadata = self
            .driver
            .metadata(channel.clone(), &options)
            .await
            .map_err(to_df_err)?;
        let num_rows = precision(metadata.info.total_records);
        let total_byte_size = precision(metadata.info.total_bytes);
        let output_schema = Arc::new(transform_flight_schema_to_original_type(&metadata.schema));
        let flight_schema = metadata.schema;
        let stats = Statistics {
            num_rows,
            total_byte_size,
            column_statistics: vec![],
        };
        Ok(FlightTable {
            driver: self.driver.clone(),
            channel,
            options,
            origin,
            table_name: table_name.into(),
            flight_schema,
            output_schema,
            stats,
        })
    }
}

#[async_trait]
impl TableProviderFactory for FlightTableFactory {
    async fn create(
        &self,
        _state: &dyn Session,
        cmd: &CreateExternalTable,
    ) -> Result<Arc<dyn TableProvider>> {
        let table = self
            .open_table(&cmd.location, cmd.options.clone(), cmd.name.clone())
            .await?;
        Ok(Arc::new(table))
    }
}

/// The information that a [FlightDriver] must produce
/// in order to register flights as DataFusion tables.
#[derive(Clone, Debug)]
pub struct FlightMetadata {
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
pub struct FlightProperties {
    pub(crate) unbounded_stream: bool,
    pub(crate) grpc_headers: HashMap<String, String>,
}

impl FlightProperties {
    pub fn unbounded_stream(mut self, unbounded_stream: bool) -> Self {
        self.unbounded_stream = unbounded_stream;
        self
    }

    pub fn grpc_headers(mut self, grpc_headers: HashMap<String, String>) -> Self {
        self.grpc_headers = grpc_headers;
        self
    }
}

/// Table provider that wraps a specific flight from an Arrow Flight service
#[derive(Debug)]
pub struct FlightTable {
    driver: Arc<FlightSqlDriver>,
    channel: Channel,
    options: HashMap<String, String>,
    origin: String,
    table_name: TableReference,
    flight_schema: SchemaRef,
    output_schema: SchemaRef,
    stats: Statistics,
}

#[async_trait]
impl TableProvider for FlightTable {
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
            info!("unparsed sql: {}", unparsed_sql);
            unparsed_sql.to_string()
        };

        let metadata = self
            .driver
            .run_sql(self.channel.clone(), &self.options, &unparsed_sql)
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
    Channel::from_shared(source.into())
        .map_err(to_df_err)?
        .connect()
        .await
        .map_err(to_df_err)
}

fn precision(total: i64) -> Precision<usize> {
    if total < 0 {
        Precision::Absent
    } else {
        Precision::Exact(total as usize)
    }
}
