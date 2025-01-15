use std::{any::Any, sync::Arc};

use datafusion::{
    common::Statistics,
    config::{ConfigOptions, TableParquetOptions},
    datasource::{
        listing::PartitionedFile,
        physical_plan::{
            parquet::DefaultParquetFileReaderFactory, FileGroupPartitioner, FileScanConfig,
            FileStream, ParquetFileReaderFactory,
        },
        schema_adapter::{DefaultSchemaAdapterFactory, SchemaAdapterFactory},
    },
    error::Result,
    execution::{SendableRecordBatchStream, TaskContext},
    physical_optimizer::pruning::PruningPredicate,
    physical_plan::{
        metrics::{ExecutionPlanMetricsSet, MetricsSet},
        DisplayAs, DisplayFormatType, ExecutionPlan, PhysicalExpr, PlanProperties,
    },
};
use itertools::Itertools;

use crate::liquid_parquet::opener::LiquidParquetOpener;

use super::page_filter::PagePruningAccessPlanFilter;

#[derive(Debug, Clone)]
pub(crate) struct LiquidParquetExec {
    base_config: FileScanConfig,
    projected_statistics: Statistics,
    metrics: ExecutionPlanMetricsSet,
    predicate: Option<Arc<dyn PhysicalExpr>>,
    pruning_predicate: Option<Arc<PruningPredicate>>,
    page_pruning_predicate: Option<Arc<PagePruningAccessPlanFilter>>,
    metadata_size_hint: Option<usize>,
    parquet_file_reader_factory: Option<Arc<dyn ParquetFileReaderFactory>>,
    cache: PlanProperties,
    table_parquet_options: TableParquetOptions,
    schema_adapter_factory: Option<Arc<dyn SchemaAdapterFactory>>,
}

impl LiquidParquetExec {
    fn pushdown_filters(&self) -> bool {
        self.table_parquet_options.global.pushdown_filters
    }

    fn reorder_filters(&self) -> bool {
        self.table_parquet_options.global.reorder_filters
    }

    fn enable_page_index(&self) -> bool {
        self.table_parquet_options.global.enable_page_index
    }

    fn bloom_filter_on_read(&self) -> bool {
        self.table_parquet_options.global.bloom_filter_on_read
    }

    fn with_file_groups_and_update_partitioning(
        mut self,
        file_groups: Vec<Vec<PartitionedFile>>,
    ) -> Self {
        let partition_cnt = file_groups.len();
        self.base_config.file_groups = file_groups;
        let output_partitioning =
            datafusion::physical_plan::Partitioning::UnknownPartitioning(partition_cnt);
        self.cache = self.cache.with_partitioning(output_partitioning);
        self
    }
}

impl ExecutionPlan for LiquidParquetExec {
    fn name(&self) -> &str {
        "LiquidParquetExec"
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn properties(&self) -> &PlanProperties {
        &self.cache
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        vec![]
    }

    fn with_new_children(
        self: Arc<Self>,
        _: Vec<Arc<dyn ExecutionPlan>>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        Ok(self)
    }

    fn fetch(&self) -> Option<usize> {
        self.base_config.limit
    }

    fn with_fetch(&self, limit: Option<usize>) -> Option<Arc<dyn ExecutionPlan>> {
        let new_config = self.base_config.clone().with_limit(limit);

        Some(Arc::new(Self {
            base_config: new_config,
            projected_statistics: self.projected_statistics.clone(),
            metrics: self.metrics.clone(),
            predicate: self.predicate.clone(),
            pruning_predicate: self.pruning_predicate.clone(),
            page_pruning_predicate: self.page_pruning_predicate.clone(),
            metadata_size_hint: self.metadata_size_hint,
            parquet_file_reader_factory: self.parquet_file_reader_factory.clone(),
            cache: self.cache.clone(),
            table_parquet_options: self.table_parquet_options.clone(),
            schema_adapter_factory: self.schema_adapter_factory.clone(),
        }))
    }

    fn repartitioned(
        &self,
        target_partitions: usize,
        config: &ConfigOptions,
    ) -> Result<Option<Arc<dyn ExecutionPlan>>> {
        let repartition_file_min_size = config.optimizer.repartition_file_min_size;
        let repartitioned_file_groups_option = FileGroupPartitioner::new()
            .with_target_partitions(target_partitions)
            .with_repartition_file_min_size(repartition_file_min_size)
            .with_preserve_order_within_groups(self.properties().output_ordering().is_some())
            .repartition_file_groups(&self.base_config.file_groups);

        let mut new_plan = self.clone();
        if let Some(repartitioned_file_groups) = repartitioned_file_groups_option {
            new_plan = new_plan.with_file_groups_and_update_partitioning(repartitioned_file_groups);
        }
        Ok(Some(Arc::new(new_plan)))
    }

    fn execute(
        &self,
        partition_index: usize,
        ctx: Arc<TaskContext>,
    ) -> Result<SendableRecordBatchStream> {
        let projection: Vec<usize> = self
            .base_config
            .projection
            .as_ref()
            .map(|p| {
                p.iter()
                    .filter(|col_idx| **col_idx < self.base_config.file_schema.fields().len())
                    .copied()
                    .collect()
            })
            .unwrap_or_else(|| (0..self.base_config.file_schema.fields().len()).collect());

        let reader_factory = ctx
            .runtime_env()
            .object_store(&self.base_config.object_store_url)
            .map(|store| Arc::new(DefaultParquetFileReaderFactory::new(store)))?;
        let schema_adapter = Arc::new(DefaultSchemaAdapterFactory);

        let opener = LiquidParquetOpener {
            partition_index,
            projection: Arc::from(projection),
            batch_size: ctx.session_config().batch_size(),
            limit: self.base_config.limit,
            predicate: self.predicate.clone(),
            pruning_predicate: self.pruning_predicate.clone(),
            page_pruning_predicate: self.page_pruning_predicate.clone(),
            table_schema: Arc::clone(&self.base_config.file_schema),
            metadata_size_hint: self.metadata_size_hint,
            metrics: self.metrics.clone(),
            parquet_file_reader_factory: reader_factory,
            pushdown_filters: self.pushdown_filters(),
            reorder_filters: self.reorder_filters(),
            enable_page_index: self.enable_page_index(),
            enable_bloom_filter: self.bloom_filter_on_read(),
            schema_adapter_factory: schema_adapter,
        };

        let stream = FileStream::new(&self.base_config, partition_index, opener, &self.metrics)?;

        Ok(Box::pin(stream))
    }

    fn metrics(&self) -> Option<MetricsSet> {
        Some(self.metrics.clone_inner())
    }

    fn statistics(&self) -> Result<Statistics> {
        let stats = if self.pruning_predicate.is_some()
            || (self.predicate.is_some() && self.pushdown_filters())
        {
            self.projected_statistics.clone().to_inexact()
        } else {
            self.projected_statistics.clone()
        };
        Ok(stats)
    }
}

impl DisplayAs for LiquidParquetExec {
    fn fmt_as(&self, t: DisplayFormatType, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match t {
            DisplayFormatType::Default | DisplayFormatType::Verbose => {
                let predicate_string = self
                    .predicate
                    .as_ref()
                    .map(|p| format!(", predicate={p}"))
                    .unwrap_or_default();

                let pruning_predicate_string = self
                    .pruning_predicate
                    .as_ref()
                    .map(|pre| {
                        let mut guarantees = pre
                            .literal_guarantees()
                            .iter()
                            .map(|item| format!("{}", item))
                            .collect_vec();
                        guarantees.sort();
                        format!(
                            ", pruning_predicate={}, required_guarantees=[{}]",
                            pre.predicate_expr(),
                            guarantees.join(", ")
                        )
                    })
                    .unwrap_or_default();

                write!(f, "LiquidParquetExec: ")?;
                self.base_config.fmt_as(t, f)?;
                write!(f, "{}{}", predicate_string, pruning_predicate_string,)
            }
        }
    }
}
