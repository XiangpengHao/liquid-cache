//! Optimizers for the Parquet module

mod squeeze_hint;

use std::sync::Arc;

use datafusion::{
    catalog::memory::DataSourceExec,
    common::tree_node::{Transformed, TreeNode, TreeNodeRecursion},
    config::ConfigOptions,
    datasource::{physical_plan::ParquetSource, source::DataSource},
    physical_optimizer::PhysicalOptimizerRule,
    physical_plan::ExecutionPlan,
};

pub(crate) use squeeze_hint::HintAnalyzer;
pub use squeeze_hint::SqueezeHintMap;

use crate::{LiquidCacheParquetRef, LiquidParquetSource, cache::ColumnSqueezeHints};

/// Physical optimizer rule for local mode liquid cache.
///
/// Rewrites `DataSourceExec` parquet scans to use [`LiquidParquetSource`], and
/// in the same pass derives typed squeeze hints from the full physical plan
/// (via the squeeze-hint analyzer) and attaches each scan's hints to its source.
#[derive(Debug)]
pub struct LocalModeOptimizer {
    cache: LiquidCacheParquetRef,
}

impl LocalModeOptimizer {
    /// Create an optimizer with an existing cache instance
    pub fn new(cache: LiquidCacheParquetRef) -> Self {
        Self { cache }
    }

    /// Create an optimizer with an existing cache instance
    pub fn with_cache(cache: LiquidCacheParquetRef) -> Self {
        Self { cache }
    }
}

impl PhysicalOptimizerRule for LocalModeOptimizer {
    fn optimize(
        &self,
        plan: Arc<dyn ExecutionPlan>,
        _config: &ConfigOptions,
    ) -> Result<Arc<dyn ExecutionPlan>, datafusion::error::DataFusionError> {
        let analysis = HintAnalyzer::analyze(&plan);
        let cache = self.cache.clone();
        let mut convert = |node: &Arc<dyn ExecutionPlan>, hints: ColumnSqueezeHints| {
            convert_parquet_scan(node, &cache, hints)
        };
        Ok(squeeze_hint::rewrite_with_hints(
            plan,
            &mut convert,
            &analysis,
        ))
    }

    fn name(&self) -> &str {
        "LocalModeLiquidCacheOptimizer"
    }

    fn schema_check(&self) -> bool {
        true
    }
}

/// Rewrite the data source plan to use liquid cache, attaching `hints` (keyed by
/// file-schema column name) to every parquet scan it rewrites.
///
/// This is the entry point used by the cache server, where hints are derived on
/// the client (which has the full plan) and shipped alongside the pushed
/// fragment, which is always single-scan.
pub fn rewrite_data_source_plan_with_hints(
    plan: Arc<dyn ExecutionPlan>,
    cache: &LiquidCacheParquetRef,
    hints: &ColumnSqueezeHints,
) -> Arc<dyn ExecutionPlan> {
    plan.transform_up(
        |node| match convert_parquet_scan(&node, cache, hints.clone()) {
            Some(new_node) => Ok(Transformed::new(
                new_node,
                true,
                TreeNodeRecursion::Continue,
            )),
            None => Ok(Transformed::no(node)),
        },
    )
    .unwrap()
    .data
}

/// Rewrite the data source plan to use liquid cache (no squeeze hints).
pub fn rewrite_data_source_plan(
    plan: Arc<dyn ExecutionPlan>,
    cache: &LiquidCacheParquetRef,
) -> Arc<dyn ExecutionPlan> {
    rewrite_data_source_plan_with_hints(plan, cache, &ColumnSqueezeHints::default())
}

/// If `node` is a `DataSourceExec` over a `ParquetSource`, return an equivalent
/// node backed by [`LiquidParquetSource`] carrying `hints`.
fn convert_parquet_scan(
    node: &Arc<dyn ExecutionPlan>,
    cache: &LiquidCacheParquetRef,
    hints: ColumnSqueezeHints,
) -> Option<Arc<dyn ExecutionPlan>> {
    let data_source_exec = node.downcast_ref::<DataSourceExec>()?;
    let (file_scan_config, parquet_source) =
        data_source_exec.downcast_to_file_source::<ParquetSource>()?;

    let new_source =
        LiquidParquetSource::from_parquet_source(parquet_source.clone(), cache.clone())
            .with_squeeze_hints(Arc::new(hints));

    let mut new_config = file_scan_config.clone();
    new_config.file_source = Arc::new(new_source);
    let new_file_source: Arc<dyn DataSource> = Arc::new(new_config);
    Some(Arc::new(DataSourceExec::new(new_file_source)))
}

#[cfg(test)]
mod tests {
    use std::{fs::File, path::Path};

    use arrow::{array::Int32Array, record_batch::RecordBatch};
    use arrow_schema::{DataType, Field, Schema};
    use datafusion::{
        common::{ScalarValue, stats::Precision},
        datasource::physical_plan::{FileScanConfig, FileSource},
        logical_expr::Operator,
        physical_expr::expressions::{BinaryExpr, Column, Literal},
        physical_plan::{
            PhysicalExpr, collect, display::DisplayableExecutionPlan, filter_pushdown::PushedDown,
        },
        prelude::SessionContext,
    };
    use liquid_cache::{
        cache::{AlwaysHydrate, squeeze_policies::TranscodeSqueezeEvict},
        cache_policies::LiquidPolicy,
    };
    use parquet::{arrow::ArrowWriter, file::properties::WriterProperties};

    use crate::LiquidCacheParquet;

    use super::*;

    async fn make_cache(path: &Path) -> LiquidCacheParquetRef {
        let store = t4::mount(path.join("liquid_cache.t4")).await.unwrap();
        Arc::new(
            LiquidCacheParquet::new(
                8192,
                1000000,
                usize::MAX,
                store,
                Box::new(LiquidPolicy::new()),
                Box::new(TranscodeSqueezeEvict),
                Box::new(AlwaysHydrate::new()),
            )
            .await,
        )
    }

    fn liquid_source(plan: &Arc<dyn ExecutionPlan>) -> LiquidParquetSource {
        let mut source = None;
        plan.apply(|node| {
            if let Some(plan) = node.downcast_ref::<DataSourceExec>() {
                let config = plan.data_source().downcast_ref::<FileScanConfig>().unwrap();
                source = Some(
                    config
                        .file_source()
                        .downcast_ref::<LiquidParquetSource>()
                        .unwrap()
                        .clone(),
                );
            }
            Ok(TreeNodeRecursion::Continue)
        })
        .unwrap();
        source.unwrap()
    }

    async fn rewrite_plan_inner(plan: Arc<dyn ExecutionPlan>) -> Arc<dyn ExecutionPlan> {
        let expected_schema = plan.schema();
        let tmp_dir = tempfile::tempdir().unwrap();
        let liquid_cache = make_cache(tmp_dir.path()).await;
        let rewritten = rewrite_data_source_plan(plan, &liquid_cache);

        rewritten
            .apply(|node| {
                if let Some(plan) = node.downcast_ref::<DataSourceExec>() {
                    let data_source = plan.data_source();
                    let source = data_source.downcast_ref::<FileScanConfig>().unwrap();
                    let file_source = source.file_source();
                    let _parquet_source =
                        file_source.downcast_ref::<LiquidParquetSource>().unwrap();
                    let schema = source.file_schema().as_ref();
                    assert_eq!(schema, expected_schema.as_ref());
                }
                Ok(TreeNodeRecursion::Continue)
            })
            .unwrap();

        rewritten
    }

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
            .sql("SELECT * FROM nano_hits WHERE \"URL\" like 'https://%' limit 10")
            .await
            .unwrap();
        let plan = df.create_physical_plan().await.unwrap();
        let rewritten = rewrite_plan_inner(plan).await;

        let displayed = DisplayableExecutionPlan::new(rewritten.as_ref())
            .indent(true)
            .to_string();
        assert!(displayed.contains("predicate="), "{displayed}");

        rewritten
            .apply(|node| {
                if let Some(plan) = node.downcast_ref::<DataSourceExec>() {
                    let statistics = plan.data_source().partition_statistics(None)?;
                    assert!(!matches!(statistics.num_rows, Precision::Exact(_)));
                }
                Ok(TreeNodeRecursion::Continue)
            })
            .unwrap();

        // Supported filters are conjoined onto the predicate; unsupported ones
        // are handed back to the parent.
        let source = liquid_source(&rewritten);
        let url_index = source.table_schema().file_schema().index_of("URL").unwrap();
        let supported: Arc<dyn PhysicalExpr> = Arc::new(BinaryExpr::new(
            Arc::new(Column::new("URL", url_index)),
            Operator::Eq,
            Arc::new(Literal::new(ScalarValue::Utf8(Some(
                "https://example.com".into(),
            )))),
        ));
        let unsupported: Arc<dyn PhysicalExpr> = Arc::new(BinaryExpr::new(
            Arc::new(Column::new("missing", 0)),
            Operator::Eq,
            Arc::new(Literal::new(ScalarValue::Utf8(Some("value".into())))),
        ));
        let result = source
            .try_pushdown_filters(
                vec![supported, unsupported],
                &datafusion::config::ConfigOptions::new(),
            )
            .unwrap();
        assert!(matches!(
            result.filters.as_slice(),
            [PushedDown::Yes, PushedDown::No]
        ));
        let predicate = result.updated_node.unwrap().filter().unwrap().to_string();
        assert!(predicate.contains(" AND "), "{predicate}");
        assert!(predicate.contains("https://example.com"), "{predicate}");
        assert!(!predicate.contains("missing"), "{predicate}");
    }

    fn write_bloom_file(path: &Path) {
        let schema = Arc::new(Schema::new(vec![Field::new("a", DataType::Int32, false)]));
        let properties = WriterProperties::builder()
            .set_bloom_filter_enabled(true)
            .build();
        let mut writer = ArrowWriter::try_new(
            File::create(path).unwrap(),
            schema.clone(),
            Some(properties),
        )
        .unwrap();
        for values in [[1, 2, 4], [1, 3, 4]] {
            writer
                .write(
                    &RecordBatch::try_new(
                        schema.clone(),
                        vec![Arc::new(Int32Array::from(values.to_vec()))],
                    )
                    .unwrap(),
                )
                .unwrap();
            writer.flush().unwrap();
        }
        writer.close().unwrap();
    }

    #[tokio::test]
    async fn prunes_row_group_with_bloom_filter() {
        let tmp_dir = tempfile::tempdir().unwrap();
        let parquet_path = tmp_dir.path().join("bloom.parquet");
        write_bloom_file(&parquet_path);

        let ctx = SessionContext::new();
        ctx.register_parquet("t", parquet_path.to_str().unwrap(), Default::default())
            .await
            .unwrap();
        let plan = ctx
            .sql("SELECT * FROM t WHERE a = 2")
            .await
            .unwrap()
            .create_physical_plan()
            .await
            .unwrap();
        let cache = make_cache(tmp_dir.path()).await;
        let rewritten = rewrite_data_source_plan(plan, &cache);
        let metrics = liquid_source(&rewritten).metrics().clone();

        let batches = collect(rewritten, ctx.task_ctx()).await.unwrap();
        assert_eq!(batches.iter().map(RecordBatch::num_rows).sum::<usize>(), 1);

        let metric = metrics
            .clone_inner()
            .sum_by_name("row_groups_pruned_bloom_filter")
            .unwrap();
        let datafusion::physical_plan::metrics::MetricValue::PruningMetrics {
            pruning_metrics, ..
        } = metric
        else {
            panic!("unexpected metric: {metric:?}");
        };
        assert_eq!(pruning_metrics.pruned(), 1);
    }
}
