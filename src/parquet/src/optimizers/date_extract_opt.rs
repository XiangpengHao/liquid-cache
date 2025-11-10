//! This module has a logical optimizer that detects columns that are only used via compatible `EXTRACT` projections.
//! It then attaches the metadata to schema adapter, which is then passed to the physical plan.
//! The physical optimizer will move the metadata to the fields of the schema.

use std::collections::HashMap;
use std::str::FromStr;
use std::sync::{Arc, Mutex, OnceLock};

use arrow::compute::kernels::cast_utils::IntervalUnit;
use arrow_schema::{DataType, Schema, SchemaRef};
use datafusion::common::tree_node::{Transformed, TreeNode, TreeNodeRecursion};
use datafusion::common::{
    Column, DFSchema, DataFusionError, ExprSchema, Result, ScalarValue, TableReference,
};
use datafusion::datasource::listing::ListingTable;
use datafusion::datasource::schema_adapter::{
    DefaultSchemaAdapterFactory, SchemaAdapter, SchemaAdapterFactory, SchemaMapper,
};
use datafusion::datasource::{TableProvider, provider_as_source, source_as_provider};
use datafusion::logical_expr::logical_plan::{
    Aggregate, Distinct, DistinctOn, Filter, Join, Limit, LogicalPlan, Partitioning, Projection,
    Repartition, Sort, SubqueryAlias, TableScan, Union, Window,
};
use datafusion::logical_expr::{Expr, TableSource};
use datafusion::optimizer::{ApplyOrder, OptimizerConfig, OptimizerRule};

/// Supported components for `EXTRACT` clauses.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum SupportedIntervalUnit {
    Year,
    Month,
    Day,
}

impl SupportedIntervalUnit {
    fn metadata_value(self) -> &'static str {
        match self {
            SupportedIntervalUnit::Year => "YEAR",
            SupportedIntervalUnit::Month => "MONTH",
            SupportedIntervalUnit::Day => "DAY",
        }
    }
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

    fn apply_order(&self) -> Option<ApplyOrder> {
        // so that it won't recursively apply the rule to every node.
        None
    }

    fn rewrite(
        &self,
        plan: LogicalPlan,
        _config: &dyn OptimizerConfig,
    ) -> Result<Transformed<LogicalPlan>, DataFusionError> {
        let mut analyzer = LineageAnalyzer::default();
        let _ = analyzer.analyze_plan(&plan)?;
        let table_usage = analyzer.finish();
        let mut findings = table_usage.find_date32_extractions();
        findings.sort_by(|a, b| a.column.flat_name().cmp(&b.column.flat_name()));
        let annotations = build_annotation_map(&findings);
        let transformed_plan = annotate_plan_with_extractions(plan, &annotations)?;
        *self.extractions.lock().unwrap() = findings;
        Ok(transformed_plan)
    }
}

type LineageMap = HashMap<ColumnKey, Vec<ColumnUsage>>;

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct ColumnKey {
    relation: Option<TableReference>,
    name: String,
}

impl ColumnKey {
    fn new(relation: Option<TableReference>, name: impl Into<String>) -> Self {
        Self {
            relation,
            name: name.into(),
        }
    }

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
    usages: Vec<Vec<Operation>>,
}

impl UsageStats {
    fn new(data_type: DataType) -> Self {
        Self {
            data_type,
            usages: Vec::new(),
        }
    }

    fn apply(&mut self, usage: &ColumnUsage) {
        self.usages.push(usage.operations.clone());
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

    fn finish(self) -> TableColumnUsage {
        TableColumnUsage { usage: self.stats }
    }
}

struct TableColumnUsage {
    usage: HashMap<ColumnKey, UsageStats>,
}

impl TableColumnUsage {
    fn find_date32_extractions(&self) -> Vec<DateExtraction> {
        let mut extractions = Vec::new();
        for (key, stats) in self.usage.iter() {
            if matches!(stats.data_type, DataType::Date32) {
                // Check if every usage's first operation is Extract with the same unit
                let first_unit = stats.usages.first().and_then(|usage| {
                    if let Some(Operation::Extract(unit)) = usage.first() {
                        Some(unit)
                    } else {
                        None
                    }
                });
                if let Some(first_unit) = first_unit {
                    let all_matches = stats.usages.iter().all(|usage| {
                        matches!(usage.first(), Some(Operation::Extract(unit)) if unit == first_unit)
                    });
                    if all_matches {
                        extractions.push(DateExtraction {
                            column: key.to_column(),
                            component: *first_unit,
                        });
                    }
                }
            }
        }
        extractions
    }
}

fn build_annotation_map(findings: &[DateExtraction]) -> HashMap<ColumnKey, SupportedIntervalUnit> {
    findings
        .iter()
        .map(|extraction| {
            (
                ColumnKey::from_column(&extraction.column),
                extraction.component,
            )
        })
        .collect()
}

fn annotate_plan_with_extractions(
    plan: LogicalPlan,
    annotations: &HashMap<ColumnKey, SupportedIntervalUnit>,
) -> Result<Transformed<LogicalPlan>, DataFusionError> {
    if annotations.is_empty() {
        return Ok(Transformed::no(plan));
    }

    plan.transform_up(|logical_plan| match logical_plan {
        LogicalPlan::TableScan(mut scan) => {
            let table_annotations = annotations_for_table_scan(&scan, annotations);
            let mut changed = false;

            if let Some(source) = annotate_listing_table_source(&scan.source, &table_annotations)? {
                scan.source = source;
                changed = true;
            }

            if changed {
                Ok(Transformed::yes(LogicalPlan::TableScan(scan)))
            } else {
                Ok(Transformed::no(LogicalPlan::TableScan(scan)))
            }
        }
        other => Ok(Transformed::no(other)),
    })
}

fn annotations_for_table_scan(
    scan: &TableScan,
    annotations: &HashMap<ColumnKey, SupportedIntervalUnit>,
) -> HashMap<String, SupportedIntervalUnit> {
    let mut table_annotations = HashMap::new();

    for (qualifier_opt, field_ref) in scan.projected_schema.iter() {
        let qualifier_owned = qualifier_opt.cloned();
        let name = field_ref.name().clone();
        if let Some(unit) = annotations
            .get(&ColumnKey::new(qualifier_owned.clone(), name.clone()))
            .copied()
            .or_else(|| {
                annotations
                    .get(&ColumnKey::new(None, name.clone()))
                    .copied()
            })
        {
            table_annotations.insert(name, unit);
        }
    }

    table_annotations
}

fn annotate_listing_table_source(
    source: &Arc<dyn TableSource>,
    annotations: &HashMap<String, SupportedIntervalUnit>,
) -> Result<Option<Arc<dyn TableSource>>, DataFusionError> {
    if annotations.is_empty() {
        return Ok(None);
    }

    let provider = match source_as_provider(source) {
        Ok(provider) => provider,
        Err(_) => return Ok(None),
    };

    let Some(listing) = provider.as_any().downcast_ref::<ListingTable>() else {
        return Ok(None);
    };

    let base_factory = listing.schema_adapter_factory().map(Arc::clone);

    let encoded_annotations: HashMap<String, String> = annotations
        .iter()
        .map(|(name, unit)| (name.clone(), unit.metadata_value().to_string()))
        .collect();

    let metadata_copy = encoded_annotations.clone();
    let new_factory: Arc<dyn SchemaAdapterFactory> = Arc::new(
        DateExtractSchemaAdapterFactory::new(base_factory, encoded_annotations),
    );
    register_factory_metadata(&new_factory, metadata_copy);
    let new_listing = listing.clone().with_schema_adapter_factory(new_factory);

    let new_provider: Arc<dyn TableProvider> = Arc::new(new_listing);
    Ok(Some(provider_as_source(new_provider)))
}

#[derive(Debug)]
struct DateExtractSchemaAdapterFactory {
    base: Option<Arc<dyn SchemaAdapterFactory>>,
    _annotations: HashMap<String, String>,
}

impl DateExtractSchemaAdapterFactory {
    fn new(
        base: Option<Arc<dyn SchemaAdapterFactory>>,
        annotations: HashMap<String, String>,
    ) -> Self {
        Self {
            base,
            _annotations: annotations,
        }
    }
}

impl SchemaAdapterFactory for DateExtractSchemaAdapterFactory {
    fn create(
        &self,
        projected_table_schema: SchemaRef,
        table_schema: SchemaRef,
    ) -> Box<dyn SchemaAdapter> {
        let inner = match &self.base {
            Some(base) => base.create(projected_table_schema, table_schema),
            None => DefaultSchemaAdapterFactory.create(projected_table_schema, table_schema),
        };
        Box::new(DateExtractSchemaAdapter { inner })
    }
}

struct DateExtractSchemaAdapter {
    inner: Box<dyn SchemaAdapter>,
}

impl SchemaAdapter for DateExtractSchemaAdapter {
    fn map_column_index(&self, index: usize, file_schema: &Schema) -> Option<usize> {
        self.inner.map_column_index(index, file_schema)
    }

    fn map_schema(
        &self,
        file_schema: &Schema,
    ) -> datafusion::common::Result<(Arc<dyn SchemaMapper>, Vec<usize>)> {
        self.inner.map_schema(file_schema)
    }
}

fn factory_registry() -> &'static Mutex<HashMap<usize, HashMap<String, String>>> {
    static REGISTRY: OnceLock<Mutex<HashMap<usize, HashMap<String, String>>>> = OnceLock::new();
    REGISTRY.get_or_init(|| Mutex::new(HashMap::new()))
}

fn register_factory_metadata(
    factory: &Arc<dyn SchemaAdapterFactory>,
    metadata: HashMap<String, String>,
) {
    let key = Arc::as_ptr(factory) as *const () as usize;
    factory_registry().lock().unwrap().insert(key, metadata);
}

pub(crate) fn metadata_from_factory(
    factory: &Arc<dyn SchemaAdapterFactory>,
    column: &str,
) -> Option<String> {
    let key = Arc::as_ptr(factory) as *const () as usize;
    factory_registry()
        .lock()
        .unwrap()
        .get(&key)
        .and_then(|map| map.get(column).cloned())
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
    use std::path::PathBuf;

    use crate::LiquidCache;
    use crate::optimizers::{DATE_MAPPING_METADATA_KEY, LocalModeOptimizer};
    use liquid_cache_common::IoMode;

    use super::*;
    use arrow::array::{ArrayRef, Date32Array, TimestampMicrosecondArray};
    use arrow_schema::{Field, Schema, TimeUnit};
    use datafusion::catalog::memory::DataSourceExec;
    use datafusion::datasource::physical_plan::FileScanConfig;
    use datafusion::execution::SessionStateBuilder;
    use datafusion::physical_plan::ExecutionPlan;
    use datafusion::prelude::{ParquetReadOptions, SessionConfig, SessionContext};
    use liquid_cache_storage::cache::squeeze_policies::TranscodeSqueezeEvict;
    use liquid_cache_storage::cache_policies::LiquidPolicy;
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
            Field::new("date_copy", DataType::Date32, false),
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
        let batch = arrow::record_batch::RecordBatch::try_new(
            Arc::clone(&schema),
            vec![timestamps, dates.clone(), dates],
        )
        .unwrap();

        let file = std::fs::File::create(table_a).unwrap();
        let mut writer = ArrowWriter::try_new(file, Arc::clone(&schema), None).unwrap();
        writer.write(&batch).unwrap();
        writer.close().unwrap();

        let file = std::fs::File::create(table_b).unwrap();
        let mut writer = ArrowWriter::try_new(file, Arc::clone(&schema), None).unwrap();
        writer.write(&batch).unwrap();
        writer.close().unwrap();

        let physical_optimizer = LocalModeOptimizer::with_cache(Arc::new(LiquidCache::new(
            1024,
            1024 * 1024 * 1024,
            PathBuf::from("test"),
            Box::new(LiquidPolicy::new()),
            Box::new(TranscodeSqueezeEvict),
            IoMode::Uring,
        )));

        let state = SessionStateBuilder::new()
            .with_config(SessionConfig::new())
            .with_default_features()
            .with_optimizer_rule(optimizer as Arc<dyn OptimizerRule + Send + Sync>)
            .with_physical_optimizer_rule(Arc::new(physical_optimizer))
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

    fn extract_field_metadata_from_physical_plan(
        plan: &Arc<dyn ExecutionPlan>,
    ) -> HashMap<String, String> {
        let mut field_metadata_map = HashMap::new();

        plan.apply(|node| {
            let Some(data_source) = node.as_any().downcast_ref::<DataSourceExec>() else {
                return Ok(TreeNodeRecursion::Continue);
            };
            let Some(file_scan_config) = data_source
                .data_source()
                .as_any()
                .downcast_ref::<FileScanConfig>()
            else {
                return Ok(TreeNodeRecursion::Continue);
            };

            // Extract metadata from all fields in file_schema
            let file_schema = &file_scan_config.file_schema();
            for field in file_schema.fields() {
                if let Some(metadata_value) = field.metadata().get(DATE_MAPPING_METADATA_KEY) {
                    field_metadata_map.insert(field.name().to_string(), metadata_value.clone());
                }
            }
            Ok(TreeNodeRecursion::Continue)
        })
        .unwrap();
        field_metadata_map
    }

    async fn general_test(sql: &str, expected: Vec<DateExtraction>) {
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

        let df = ctx.sql(sql).await.unwrap();
        let (state, plan) = df.into_parts();
        let optimized = state.optimize(&plan).unwrap();
        let extractions = optimizer.extractions();
        assert_eq!(extractions, expected);

        let physical_plan = state.create_physical_plan(&optimized).await.unwrap();
        let field_metadata_map = extract_field_metadata_from_physical_plan(&physical_plan);

        let expected_field_metadata = expected
            .iter()
            .map(|extraction| {
                (
                    extraction.column.name().to_string(),
                    extraction.component.metadata_value().to_string(),
                )
            })
            .collect::<HashMap<String, String>>();
        assert_eq!(field_metadata_map, expected_field_metadata);
    }

    #[tokio::test]
    async fn multi_table_extracts() {
        general_test(
            "SELECT EXTRACT(YEAR FROM table_a.date) AS year, EXTRACT(DAY FROM table_b.date) AS day FROM table_a INNER JOIN table_b ON table_a.event_ts = table_b.event_ts",
            vec![
                DateExtraction { column: Column::new(Some("table_a"), "date"), component: SupportedIntervalUnit::Year },
                DateExtraction { column: Column::new(Some("table_b"), "date"), component: SupportedIntervalUnit::Day },
            ],
        )
        .await;
    }

    #[tokio::test]
    async fn single_table_multiple_extracts() {
        general_test(
            "SELECT EXTRACT(YEAR FROM date_copy) AS year, EXTRACT(DAY FROM date) AS day FROM table_a",
            vec![
                DateExtraction { column: Column::new(Some("table_a"), "date"), component: SupportedIntervalUnit::Day },
                DateExtraction { column: Column::new(Some("table_a"), "date_copy"), component: SupportedIntervalUnit::Year },
            ],
        )
        .await;
    }

    #[tokio::test]
    async fn single_table_extracts() {
        let statements = vec![
            "SELECT EXTRACT(DAY FROM date) AS day FROM table_a",
            "SELECT EXTRACT(day FROM date) AS day FROM table_a",
            "SELECT EXTRACT(DAY FROM table_a.date) FROM table_a",
            "SELECT AVG(EXTRACT(DAY FROM date)) AS avg_day FROM table_a",
            "SELECT AVG(EXTRACT(DAY FROM date) + 1) AS avg_day FROM table_a",
            "SELECT (SELECT MAX(EXTRACT(DAY FROM date)) FROM table_a) AS max_day, (SELECT MIN(EXTRACT(DAY FROM date)) FROM table_a) AS min_day",
        ];
        let expected = vec![DateExtraction {
            column: Column::new(Some("table_a"), "date"),
            component: SupportedIntervalUnit::Day,
        }];
        for sql in statements {
            general_test(sql, expected.clone()).await;
        }
    }

    #[tokio::test]
    async fn test_no_metadata_on_unused_fields() {
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

        // Query that extracts from date but doesn't use event_ts for extraction
        let df = ctx
            .sql("SELECT EXTRACT(YEAR FROM date) AS year, event_ts FROM table_a")
            .await
            .unwrap();
        let (state, plan) = df.into_parts();
        let optimized = state.optimize(&plan).unwrap();
        let physical_plan = state.create_physical_plan(&optimized).await.unwrap();
        let field_metadata_map = extract_field_metadata_from_physical_plan(&physical_plan);

        assert_eq!(field_metadata_map.get("date"), Some(&"YEAR".to_string()),);

        assert!(!field_metadata_map.contains_key("event_ts"));

        assert_eq!(field_metadata_map.len(), 1);

        let expected_keys: Vec<&str> = vec!["date"];
        let actual_keys: Vec<&str> = field_metadata_map.keys().map(|s| s.as_str()).collect();
        assert_eq!(actual_keys, expected_keys);
    }

    #[tokio::test]
    async fn inconsistent_extracts_are_ignored() {
        let statements = vec![
            "SELECT EXTRACT(DAY FROM date) AS day, EXTRACT(MONTH FROM date) AS month FROM table_a",
            "SELECT EXTRACT(DAY FROM date + INTERVAL '1 day') AS day FROM table_a",
            "SELECT date FROM table_a",
            "SELECT EXTRACT(DAY FROM table_a.date) AS day FROM table_a INNER JOIN table_b ON table_a.date = table_b.date",
            "SELECT (SELECT MAX(EXTRACT(DAY FROM date)) FROM table_a) AS max_day, (SELECT MIN(EXTRACT(Month FROM date)) FROM table_a) AS min_day",
            "SELECT EXTRACT(YEAR FROM event_ts) AS year FROM table_a", // todo: time stamp is not supported yet.
        ];

        for sql in statements {
            general_test(sql, vec![]).await;
        }
    }
}
