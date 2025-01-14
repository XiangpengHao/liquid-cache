use std::{any::Any, collections::HashMap, sync::Arc};

use arrow_schema::{Schema, SchemaRef};
use async_trait::async_trait;
use datafusion::{
    datasource::{
        file_format::{
            file_compression_type::FileCompressionType, parquet::ParquetFormatFactory, FileFormat,
            FileFormatFactory, FilePushdownSupport,
        },
        physical_plan::FileScanConfig,
    },
    error::Result,
    execution::SessionState,
};
use datafusion_common::{internal_err, parsers::CompressionTypeVariant, GetExt, Statistics};
use datafusion_expr::Expr;
use datafusion_physical_expr::PhysicalExpr;
use datafusion_physical_plan::ExecutionPlan;
use object_store::{ObjectMeta, ObjectStore};

#[derive(Debug, Default)]
pub struct LiquidParquetFactory {
    inner: ParquetFormatFactory,
}

impl LiquidParquetFactory {
    pub fn new() -> Self {
        Self {
            inner: ParquetFormatFactory::new(),
        }
    }
}

impl GetExt for LiquidParquetFactory {
    fn get_ext(&self) -> String {
        ".lpq".to_string()
    }
}

impl FileFormatFactory for LiquidParquetFactory {
    fn create(
        &self,
        state: &SessionState,
        options: &HashMap<String, String>,
    ) -> Result<Arc<dyn FileFormat>> {
        let inner = self.inner.create(state, options)?;
        Ok(Arc::new(LiquidParquetFileFormat { inner }))
    }

    fn default(&self) -> Arc<dyn FileFormat> {
        Arc::new(LiquidParquetFileFormat {
            inner: self.inner.default(),
        })
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

#[derive(Debug)]
pub struct LiquidParquetFileFormat {
    inner: Arc<dyn FileFormat>, // is actually ParquetFormat
}

#[async_trait]
impl FileFormat for LiquidParquetFileFormat {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn get_ext(&self) -> String {
        LiquidParquetFactory::new().get_ext()
    }

    fn get_ext_with_compression(
        &self,
        file_compression_type: &FileCompressionType,
    ) -> Result<String> {
        let ext = self.get_ext();
        match file_compression_type.get_variant() {
            CompressionTypeVariant::UNCOMPRESSED => Ok(ext),
            _ => internal_err!("Unsupported compression type: {:?}", file_compression_type),
        }
    }

    async fn infer_schema(
        &self,
        state: &SessionState,
        store: &Arc<dyn ObjectStore>,
        objects: &[ObjectMeta],
    ) -> Result<SchemaRef> {
        self.inner.infer_schema(state, store, objects).await
    }

    async fn infer_stats(
        &self,
        state: &SessionState,
        store: &Arc<dyn ObjectStore>,
        table_schema: SchemaRef,
        object: &ObjectMeta,
    ) -> Result<Statistics> {
        self.inner
            .infer_stats(state, store, table_schema, object)
            .await
    }

    async fn create_physical_plan(
        &self,
        state: &SessionState,
        conf: FileScanConfig,
        filters: Option<&Arc<dyn PhysicalExpr>>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        self.inner.create_physical_plan(state, conf, filters).await
    }

    fn supports_filters_pushdown(
        &self,
        file_schema: &Schema,
        table_schema: &Schema,
        filters: &[&Expr],
    ) -> Result<FilePushdownSupport> {
        self.inner
            .supports_filters_pushdown(file_schema, table_schema, filters)
    }
}
