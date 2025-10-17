use std::collections::HashMap;
use std::str::FromStr;
use std::sync::{Arc, Mutex};

use arrow::compute::kernels::cast_utils::IntervalUnit;
use arrow_schema::DataType;
use datafusion::common::tree_node::{Transformed, TreeNode, TreeNodeRecursion};
use datafusion::common::{
    Column, DFSchema, DataFusionError, ExprSchema, Result, ScalarValue, TableReference,
};
use datafusion::logical_expr::Expr;
use datafusion::logical_expr::logical_plan::{
    Aggregate, Distinct, DistinctOn, Filter, Join, Limit, LogicalPlan, Partitioning, Projection,
    Repartition, Sort, SubqueryAlias, TableScan, Union, Window,
};
use datafusion::optimizer::{OptimizerConfig, OptimizerRule};

/// Supported components for `EXTRACT` clauses.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum SupportedIntervalUnit {
    Year,
    Month,
    Day,
}

/// Metadata describing a Date32 column that participates in an `EXTRACT`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct DateExtraction {
    pub(crate) column: Column,
    pub(crate) component: SupportedIntervalUnit,
}

/// Logical optimizer that analyses the logical plan to detect columns that
/// are only used via compatible `EXTRACT` projections.
#[derive(Debug, Default)]
pub struct DateExtractOptimizer {
    extractions: Arc<Mutex<Vec<DateExtraction>>>,
}

impl DateExtractOptimizer {
    /// Create a new optimizer.
    pub fn new() -> Self {
        Self::default()
    }

    /// Retrieve the extractions discovered during the last optimization pass.
    #[cfg(test)]
    fn extractions(&self) -> Vec<DateExtraction> {
        self.extractions.lock().unwrap().clone()
    }
}

impl OptimizerRule for DateExtractOptimizer {
    fn name(&self) -> &str {
        "DateExtractOptimizer"
    }

    fn rewrite(
        &self,
        plan: LogicalPlan,
        _config: &dyn OptimizerConfig,
    ) -> Result<Transformed<LogicalPlan>, DataFusionError> {
        let mut analyzer = LineageAnalyzer::default();
        let _ = analyzer.analyze_plan(&plan)?;
        let mut findings = analyzer.finish();
        findings.sort_by(|a, b| a.column.flat_name().cmp(&b.column.flat_name()));
        *self.extractions.lock().unwrap() = findings;
        Ok(Transformed::no(plan))
    }
}

type LineageMap = HashMap<ColumnKey, Vec<ColumnUsage>>;

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct ColumnKey {
    relation: Option<TableReference>,
    name: String,
}

impl ColumnKey {
    fn from_column(column: &Column) -> Self {
        Self {
            relation: column.relation.clone(),
            name: column.name.clone(),
        }
    }

    fn to_column(&self) -> Column {
        Column::new(self.relation.clone(), self.name.clone())
    }
}

#[derive(Debug, Clone)]
struct ColumnUsage {
    base: ColumnKey,
    data_type: DataType,
    operations: Vec<Operation>,
}

impl ColumnUsage {
    fn new_base(column: &Column, data_type: DataType) -> Self {
        Self {
            base: ColumnKey::from_column(column),
            data_type,
            operations: Vec::new(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Operation {
    Extract(SupportedIntervalUnit),
    Other,
}

#[derive(Debug)]
struct UsageStats {
    data_type: DataType,
    component: Option<SupportedIntervalUnit>,
    only_extract: bool,
}

impl UsageStats {
    fn new(data_type: DataType) -> Self {
        Self {
            data_type,
            component: None,
            only_extract: true,
        }
    }

    fn apply(&mut self, usage: &ColumnUsage) {
        if !self.only_extract {
            return;
        }

        if usage.operations.is_empty() {
            self.only_extract = false;
            return;
        }

        let first = match usage.operations.first() {
            Some(Operation::Extract(unit)) => *unit,
            _ => {
                self.only_extract = false;
                return;
            }
        };

        if usage
            .operations
            .iter()
            .skip(1)
            .any(|op| !matches!(op, Operation::Extract(unit) if *unit == first))
        {
            self.only_extract = false;
            return;
        }

        match self.component {
            Some(existing) if existing != first => self.only_extract = false,
            None => self.component = Some(first),
            _ => {}
        }
    }

    fn candidate(&self) -> Option<(DataType, SupportedIntervalUnit)> {
        if self.only_extract {
            self.component
                .map(|component| (self.data_type.clone(), component))
        } else {
            None
        }
    }
}

#[derive(Default)]
struct LineageAnalyzer {
    stats: HashMap<ColumnKey, UsageStats>,
}

impl LineageAnalyzer {
    fn analyze_plan(&mut self, plan: &LogicalPlan) -> Result<LineageMap> {
        match plan {
            LogicalPlan::TableScan(scan) => self.analyze_table_scan(scan),
            LogicalPlan::Projection(projection) => self.analyze_projection(projection),
            LogicalPlan::Filter(filter) => self.analyze_filter(filter),
            LogicalPlan::Aggregate(aggregate) => self.analyze_aggregate(aggregate),
            LogicalPlan::Sort(sort) => self.analyze_sort(sort),
            LogicalPlan::Join(join) => self.analyze_join(join),
            LogicalPlan::SubqueryAlias(alias) => self.analyze_subquery_alias(alias),
            LogicalPlan::Window(window) => self.analyze_window(window),
            LogicalPlan::Limit(limit) => self.analyze_limit(limit),
            LogicalPlan::Repartition(repartition) => self.analyze_repartition(repartition),
            LogicalPlan::Union(union) => self.analyze_union(union),
            LogicalPlan::Distinct(distinct) => self.analyze_distinct(distinct),
            other => {
                let mut merged = LineageMap::new();
                for input in other.inputs() {
                    let child = self.analyze_plan(input)?;
                    merged = merge_lineage_maps(merged, child);
                }
                Ok(merged)
            }
        }
    }

    fn analyze_table_scan(&mut self, scan: &TableScan) -> Result<LineageMap> {
        let schema = scan.projected_schema.as_ref();
        let mut map = LineageMap::new();
        for (index, column) in schema.columns().iter().enumerate() {
            let field = schema.field(index);
            let usage = ColumnUsage::new_base(column, field.data_type().clone());
            map.insert(usage.base.clone(), vec![usage]);
        }

        for filter in &scan.filters {
            let usages = lineage_for_expr(filter, &map, schema)?;
            self.record(&usages);
        }

        Ok(map)
    }

    fn analyze_projection(&mut self, projection: &Projection) -> Result<LineageMap> {
        let input_map = self.analyze_plan(projection.input.as_ref())?;
        let input_schema = projection.input.schema();
        let mut output = LineageMap::new();
        for (expr, column) in projection.expr.iter().zip(projection.schema.columns()) {
            let usages = lineage_for_expr(expr, &input_map, input_schema.as_ref())?;
            self.record(&usages);
            output.insert(ColumnKey::from_column(&column), usages);
        }
        Ok(output)
    }

    fn analyze_filter(&mut self, filter: &Filter) -> Result<LineageMap> {
        let input_map = self.analyze_plan(filter.input.as_ref())?;
        let schema = filter.input.schema();
        let usages = lineage_for_expr(&filter.predicate, &input_map, schema.as_ref())?;
        self.record(&usages);
        Ok(input_map)
    }

    fn analyze_sort(&mut self, sort: &Sort) -> Result<LineageMap> {
        let input_map = self.analyze_plan(sort.input.as_ref())?;
        let schema = sort.input.schema();
        for sort_expr in &sort.expr {
            let usages = lineage_for_expr(&sort_expr.expr, &input_map, schema.as_ref())?;
            self.record(&usages);
        }
        Ok(input_map)
    }

    fn analyze_aggregate(&mut self, aggregate: &Aggregate) -> Result<LineageMap> {
        let input_map = self.analyze_plan(aggregate.input.as_ref())?;
        let schema = aggregate.input.schema();
        let mut output = LineageMap::new();
        let mut expr_iter = aggregate
            .group_expr
            .iter()
            .chain(aggregate.aggr_expr.iter());

        for column in aggregate.schema.columns() {
            if let Some(expr) = expr_iter.next() {
                let usages = lineage_for_expr(expr, &input_map, schema.as_ref())?;
                self.record(&usages);
                output.insert(ColumnKey::from_column(&column), usages);
            } else {
                output.insert(ColumnKey::from_column(&column), Vec::new());
            }
        }

        Ok(output)
    }

    fn analyze_join(&mut self, join: &Join) -> Result<LineageMap> {
        let left_map = self.analyze_plan(join.left.as_ref())?;
        let right_map = self.analyze_plan(join.right.as_ref())?;
        let left_schema = join.left.schema();
        let right_schema = join.right.schema();

        for (left_expr, right_expr) in &join.on {
            let left_usages = lineage_for_expr(left_expr, &left_map, left_schema.as_ref())?;
            self.record(&left_usages);
            let right_usages = lineage_for_expr(right_expr, &right_map, right_schema.as_ref())?;
            self.record(&right_usages);
        }

        if let Some(filter) = &join.filter {
            let mut combined = left_map.clone();
            merge_lineage_map_inplace(&mut combined, &right_map);
            let usages = lineage_for_expr(filter, &combined, join.schema.as_ref())?;
            self.record(&usages);
        }

        let left_columns = left_schema.columns();
        let right_columns = right_schema.columns();
        let mut output_columns = join.schema.columns().into_iter();
        let mut output = LineageMap::new();

        for column in left_columns {
            if let Some(output_column) = output_columns.next() {
                let key = ColumnKey::from_column(&output_column);
                let usages = left_map
                    .get(&ColumnKey::from_column(&column))
                    .cloned()
                    .unwrap_or_default();
                output.insert(key, usages);
            }
        }

        for column in right_columns {
            if let Some(output_column) = output_columns.next() {
                let key = ColumnKey::from_column(&output_column);
                let usages = right_map
                    .get(&ColumnKey::from_column(&column))
                    .cloned()
                    .unwrap_or_default();
                output.insert(key, usages);
            }
        }

        Ok(output)
    }

    fn analyze_subquery_alias(&mut self, alias: &SubqueryAlias) -> Result<LineageMap> {
        let input_map = self.analyze_plan(alias.input.as_ref())?;
        let input_columns = alias.input.schema().columns();
        let mut output = LineageMap::new();
        for (input_column, output_column) in
            input_columns.iter().zip(alias.schema.columns().into_iter())
        {
            let key = ColumnKey::from_column(&output_column);
            let usages = input_map
                .get(&ColumnKey::from_column(input_column))
                .cloned()
                .unwrap_or_default();
            output.insert(key, usages);
        }
        Ok(output)
    }

    fn analyze_window(&mut self, window: &Window) -> Result<LineageMap> {
        let input_map = self.analyze_plan(window.input.as_ref())?;
        let input_schema = window.input.schema();

        let input_cols = input_schema.columns();
        let output_cols = window.schema.columns();
        let mut output = LineageMap::new();

        for (input_column, output_column) in input_cols.iter().zip(output_cols.iter()) {
            let key = ColumnKey::from_column(output_column);
            let usages = input_map
                .get(&ColumnKey::from_column(input_column))
                .cloned()
                .unwrap_or_default();
            output.insert(key, usages);
        }

        for (expr, output_column) in window
            .window_expr
            .iter()
            .zip(output_cols.into_iter().skip(input_cols.len()))
        {
            let usages = lineage_for_expr(expr, &input_map, input_schema.as_ref())?;
            self.record(&usages);
            output.insert(ColumnKey::from_column(&output_column), usages);
        }

        Ok(output)
    }

    fn analyze_limit(&mut self, limit: &Limit) -> Result<LineageMap> {
        let map = self.analyze_plan(limit.input.as_ref())?;
        let schema = limit.input.schema();
        if let Some(skip) = &limit.skip {
            let usages = lineage_for_expr(skip, &map, schema.as_ref())?;
            self.record(&usages);
        }
        if let Some(fetch) = &limit.fetch {
            let usages = lineage_for_expr(fetch, &map, schema.as_ref())?;
            self.record(&usages);
        }
        Ok(map)
    }

    fn analyze_repartition(&mut self, repartition: &Repartition) -> Result<LineageMap> {
        let map = self.analyze_plan(repartition.input.as_ref())?;
        let schema = repartition.input.schema();
        if let Partitioning::Hash(exprs, _) | Partitioning::DistributeBy(exprs) =
            &repartition.partitioning_scheme
        {
            for expr in exprs {
                let usages = lineage_for_expr(expr, &map, schema.as_ref())?;
                self.record(&usages);
            }
        }
        Ok(map)
    }

    fn analyze_union(&mut self, union: &Union) -> Result<LineageMap> {
        let mut input_maps: Vec<LineageMap> = Vec::with_capacity(union.inputs.len());
        for input in &union.inputs {
            input_maps.push(self.analyze_plan(input.as_ref())?);
        }

        let mut output = LineageMap::new();
        for output_column in union.schema.columns() {
            let key = ColumnKey::from_column(&output_column);
            let mut combined: Vec<ColumnUsage> = Vec::new();
            for map in &input_maps {
                for (candidate_key, usages) in map {
                    if candidate_key.name == key.name {
                        combined.extend(usages.clone());
                    }
                }
            }
            output.insert(key, combined);
        }
        Ok(output)
    }

    fn analyze_distinct(&mut self, distinct: &Distinct) -> Result<LineageMap> {
        match distinct {
            Distinct::All(plan) => self.analyze_plan(plan.as_ref()),
            Distinct::On(distinct_on) => self.analyze_distinct_on(distinct_on),
        }
    }

    fn analyze_distinct_on(&mut self, distinct_on: &DistinctOn) -> Result<LineageMap> {
        let input_map = self.analyze_plan(distinct_on.input.as_ref())?;
        let schema = distinct_on.input.schema();

        for expr in &distinct_on.on_expr {
            let usages = lineage_for_expr(expr, &input_map, schema.as_ref())?;
            self.record(&usages);
        }
        for expr in &distinct_on.select_expr {
            let usages = lineage_for_expr(expr, &input_map, schema.as_ref())?;
            self.record(&usages);
        }
        if let Some(sort_exprs) = &distinct_on.sort_expr {
            for sort_expr in sort_exprs {
                let usages = lineage_for_expr(&sort_expr.expr, &input_map, schema.as_ref())?;
                self.record(&usages);
            }
        }

        let mut output = LineageMap::new();
        for (expr, column) in distinct_on
            .select_expr
            .iter()
            .zip(distinct_on.schema.columns().into_iter())
        {
            let usages = lineage_for_expr(expr, &input_map, schema.as_ref())?;
            output.insert(ColumnKey::from_column(&column), usages);
        }
        Ok(output)
    }

    fn record(&mut self, usages: &[ColumnUsage]) {
        for usage in usages {
            let entry = self
                .stats
                .entry(usage.base.clone())
                .or_insert_with(|| UsageStats::new(usage.data_type.clone()));
            entry.apply(usage);
        }
    }

    fn finish(self) -> Vec<DateExtraction> {
        self.stats
            .into_iter()
            .filter_map(|(key, stats)| {
                stats.candidate().and_then(|(data_type, component)| {
                    if matches!(data_type, DataType::Date32) {
                        Some(DateExtraction {
                            column: key.to_column(),
                            component,
                        })
                    } else {
                        None
                    }
                })
            })
            .collect()
    }
}

fn merge_lineage_maps(mut base: LineageMap, other: LineageMap) -> LineageMap {
    for (key, usages) in other {
        base.entry(key).or_default().extend(usages);
    }
    base
}

fn merge_lineage_map_inplace(base: &mut LineageMap, other: &LineageMap) {
    for (key, usages) in other {
        base.entry(key.clone()).or_default().extend(usages.clone());
    }
}

fn lineage_for_expr(
    expr: &Expr,
    input_lineage: &LineageMap,
    schema: &DFSchema,
) -> Result<Vec<ColumnUsage>> {
    match expr {
        Expr::Column(column) => {
            let key = ColumnKey::from_column(column);
            if let Some(usages) = input_lineage.get(&key) {
                Ok(usages.clone())
            } else {
                let field = schema.field_from_column(column)?;
                Ok(vec![ColumnUsage::new_base(
                    column,
                    field.data_type().clone(),
                )])
            }
        }
        Expr::Alias(alias) => lineage_for_expr(&alias.expr, input_lineage, schema),
        Expr::ScalarFunction(func) => {
            let func_name = func.func.name();
            if func_name.eq_ignore_ascii_case("date_part")
                && func.args.len() == 2
                && let Some(component) = part_to_unit(&func.args[0])
            {
                let mut usages = lineage_for_expr(&func.args[1], input_lineage, schema)?;
                for usage in &mut usages {
                    usage.operations.push(Operation::Extract(component));
                }
                return Ok(usages);
            }
            propagate_other(expr, input_lineage, schema)
        }
        Expr::Cast(cast) => {
            let mut usages = lineage_for_expr(&cast.expr, input_lineage, schema)?;
            for usage in &mut usages {
                usage.operations.push(Operation::Other);
            }
            Ok(usages)
        }
        Expr::TryCast(cast) => {
            let mut usages = lineage_for_expr(&cast.expr, input_lineage, schema)?;
            for usage in &mut usages {
                usage.operations.push(Operation::Other);
            }
            Ok(usages)
        }
        Expr::Literal(_, _) => Ok(Vec::new()),
        Expr::ScalarSubquery(_) | Expr::Exists { .. } => Ok(Vec::new()),
        Expr::Placeholder(_) => Ok(Vec::new()),
        #[allow(deprecated)]
        Expr::Wildcard { .. } => {
            let mut usages = Vec::new();
            for column_usages in input_lineage.values() {
                usages.extend(column_usages.clone());
            }
            Ok(usages)
        }
        _ => propagate_other(expr, input_lineage, schema),
    }
}

fn propagate_other(
    expr: &Expr,
    input_lineage: &LineageMap,
    schema: &DFSchema,
) -> Result<Vec<ColumnUsage>> {
    let mut combined: Vec<ColumnUsage> = Vec::new();
    expr.apply_children(|child| {
        let mut usages = lineage_for_expr(child, input_lineage, schema)?;
        for usage in &mut usages {
            usage.operations.push(Operation::Other);
        }
        combined.extend(usages);
        Ok(TreeNodeRecursion::Continue)
    })?;
    Ok(combined)
}

fn part_to_unit(expr: &Expr) -> Option<SupportedIntervalUnit> {
    let value = match expr {
        Expr::Literal(literal, _) => literal,
        _ => return None,
    };
    let text = match value {
        ScalarValue::Utf8(Some(v))
        | ScalarValue::LargeUtf8(Some(v))
        | ScalarValue::Utf8View(Some(v)) => v.as_str(),
        _ => return None,
    };
    let unit = IntervalUnit::from_str(text).ok()?;
    match unit {
        IntervalUnit::Year => Some(SupportedIntervalUnit::Year),
        IntervalUnit::Month => Some(SupportedIntervalUnit::Month),
        IntervalUnit::Day => Some(SupportedIntervalUnit::Day),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::{ArrayRef, Date32Array, TimestampMicrosecondArray};
    use arrow_schema::{Field, Schema, TimeUnit};
    use datafusion::execution::SessionStateBuilder;
    use datafusion::prelude::{ParquetReadOptions, SessionConfig, SessionContext};
    use parquet::arrow::ArrowWriter;
    use tempfile::TempDir;

    async fn create_test_ctx(
        table_a: &str,
        table_b: &str,
        optimizer: Arc<DateExtractOptimizer>,
    ) -> SessionContext {
        let schema = Arc::new(Schema::new(vec![
            Field::new(
                "event_ts",
                DataType::Timestamp(TimeUnit::Microsecond, None),
                false,
            ),
            Field::new("date", DataType::Date32, false),
        ]));

        let timestamps: ArrayRef = Arc::new(TimestampMicrosecondArray::from(vec![
            Some(1_609_459_200_000_000),
            Some(1_640_995_200_000_000),
            Some(1_672_358_400_000_000),
        ]));
        let dates: ArrayRef = Arc::new(Date32Array::from(vec![
            Some(20210101),
            Some(20220202),
            Some(20230303),
        ]));
        let batch =
            arrow::record_batch::RecordBatch::try_new(Arc::clone(&schema), vec![timestamps, dates])
                .unwrap();

        let file = std::fs::File::create(table_a).unwrap();
        let mut writer = ArrowWriter::try_new(file, Arc::clone(&schema), None).unwrap();
        writer.write(&batch).unwrap();
        writer.close().unwrap();

        let file = std::fs::File::create(table_b).unwrap();
        let mut writer = ArrowWriter::try_new(file, Arc::clone(&schema), None).unwrap();
        writer.write(&batch).unwrap();
        writer.close().unwrap();

        let state = SessionStateBuilder::new()
            .with_config(SessionConfig::new())
            .with_default_features()
            .with_optimizer_rule(optimizer as Arc<dyn OptimizerRule + Send + Sync>)
            .build();
        let ctx = SessionContext::new_with_state(state);

        ctx.register_parquet("table_a", table_a, ParquetReadOptions::default())
            .await
            .unwrap();
        ctx.register_parquet("table_b", table_b, ParquetReadOptions::default())
            .await
            .unwrap();
        ctx
    }

    #[tokio::test]
    async fn detects_consistent_extracts_for_date32() {
        let temp_dir = TempDir::new().unwrap();
        let table_a = temp_dir.path().join("table_a.parquet");
        let table_b = temp_dir.path().join("table_b.parquet");

        let optimizer = Arc::new(DateExtractOptimizer::new());
        let ctx = create_test_ctx(
            table_a.to_str().unwrap(),
            table_b.to_str().unwrap(),
            optimizer.clone(),
        )
        .await;

        let statements = vec![
            "SELECT EXTRACT(DAY FROM date) AS day FROM table_a",
            "SELECT EXTRACT(day FROM date) AS day FROM table_a",
            "SELECT EXTRACT(DAY FROM table_a.date) FROM table_a",
            // "SELECT EXTRACT(DAY FROM sub.order_date) FROM (SELECT date AS order_date FROM table_a) sub",
            "SELECT EXTRACT(YEAR FROM event_ts) AS year, EXTRACT(DAY FROM date) AS day FROM table_a",
        ];

        for sql in statements {
            let df = ctx.sql(sql).await.unwrap();
            let (state, plan) = df.into_parts();
            let optimized = state.optimize(&plan).unwrap();
            let _ = state.create_physical_plan(&optimized).await.unwrap();

            let extractions = optimizer.extractions();
            assert_eq!(
                extractions.len(),
                1,
                "expected one extraction for query `{sql}`"
            );
            let extraction = &extractions[0];
            assert_eq!(extraction.component, SupportedIntervalUnit::Day);
            assert_eq!(extraction.column.name(), "date");
            assert_eq!(
                extraction
                    .column
                    .relation
                    .as_ref()
                    .map(|r| r.table().to_string()),
                Some("table_a".to_string()),
                "expected base column relation for query `{sql}`"
            );
        }
    }

    #[tokio::test]
    async fn inconsistent_extracts_are_ignored() {
        let temp_dir = TempDir::new().unwrap();
        let table_a = temp_dir.path().join("table_a.parquet");
        let table_b = temp_dir.path().join("table_b.parquet");

        let optimizer = Arc::new(DateExtractOptimizer::new());
        let ctx = create_test_ctx(
            table_a.to_str().unwrap(),
            table_b.to_str().unwrap(),
            optimizer.clone(),
        )
        .await;

        let statements = vec![
            "SELECT EXTRACT(DAY FROM date) AS day, EXTRACT(MONTH FROM date) AS month FROM table_a",
            "SELECT EXTRACT(DAY FROM date + INTERVAL '1 day') AS day FROM table_a",
            "SELECT date FROM table_a",
            "SELECT EXTRACT(DAY FROM table_a.date) AS day FROM table_a INNER JOIN table_b ON table_a.date = table_b.date",
        ];

        for sql in statements {
            let df = ctx.sql(sql).await.unwrap();
            let (state, plan) = df.into_parts();
            let optimized = state.optimize(&plan).unwrap();
            let _ = state.create_physical_plan(&optimized).await.unwrap();

            let extractions = optimizer.extractions();
            assert!(
                extractions.is_empty(),
                "expected no extraction for query `{sql}`, found {extractions:?}"
            );
        }
    }
}
