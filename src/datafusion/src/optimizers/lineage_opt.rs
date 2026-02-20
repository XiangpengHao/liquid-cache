//! This module has a logical optimizer that detects columns that are only used via compatible `EXTRACT` projections.
//! It then attaches the metadata to expression adapter factory, which is then passed to the physical plan.
//! The physical optimizer will move the metadata to the fields of the schema.

use std::cmp::Ordering;
use std::collections::hash_map::Entry;
use std::collections::{HashMap, HashSet};
use std::str::FromStr;
use std::sync::{Arc, Mutex, OnceLock};

use arrow::compute::kernels::cast_utils::IntervalUnit;
use arrow_schema::{DataType, Schema, SchemaRef};
use datafusion::common::tree_node::{Transformed, TreeNode, TreeNodeRecursion};
use datafusion::common::{
    Column, Constraints, DFSchema, DataFusionError, ExprSchema, Result, ScalarValue, TableReference,
};
use datafusion::datasource::listing::{ListingTable, ListingTableConfig};
use datafusion::datasource::{TableProvider, provider_as_source, source_as_provider};
use datafusion::logical_expr::logical_plan::{
    Aggregate, Distinct, DistinctOn, Filter, Join, Limit, LogicalPlan, Partitioning, Projection,
    Repartition, Sort, SubqueryAlias, TableScan, Union, Window,
};
use datafusion::logical_expr::{Expr, TableSource};
use datafusion::optimizer::{ApplyOrder, OptimizerConfig, OptimizerRule};
use datafusion::physical_expr_adapter::{
    DefaultPhysicalExprAdapterFactory, PhysicalExprAdapter, PhysicalExprAdapterFactory,
};

/// Supported components for `EXTRACT` clauses.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum SupportedIntervalUnit {
    Year,
    Month,
    Day,
    DayOfWeek,
}

impl SupportedIntervalUnit {
    pub(crate) fn metadata_value(self) -> &'static str {
        match self {
            SupportedIntervalUnit::Year => "YEAR",
            SupportedIntervalUnit::Month => "MONTH",
            SupportedIntervalUnit::Day => "DAY",
            SupportedIntervalUnit::DayOfWeek => "DOW",
        }
    }
}

/// Metadata describing a Date32/Timestamp column that participates in an `EXTRACT`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct DateExtraction {
    pub(crate) column: Column,
    pub(crate) components: HashSet<SupportedIntervalUnit>,
}

/// Metadata describing a Variant column that participates in a `variant_get`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct VariantExtraction {
    pub(crate) column: Column,
    pub(crate) fields: Vec<VariantField>,
}

impl PartialOrd for VariantExtraction {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for VariantExtraction {
    fn cmp(&self, other: &Self) -> Ordering {
        self.column
            .flat_name()
            .cmp(&other.column.flat_name())
            .then_with(|| self.fields.cmp(&other.fields))
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct VariantField {
    pub(crate) path: String,
    pub(crate) data_type: Option<DataType>,
}

impl PartialOrd for VariantField {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for VariantField {
    fn cmp(&self, other: &Self) -> Ordering {
        self.path.cmp(&other.path).then_with(|| {
            let self_ty = self
                .data_type
                .as_ref()
                .map(|dt| dt.to_string())
                .unwrap_or_default();
            let other_ty = other
                .data_type
                .as_ref()
                .map(|dt| dt.to_string())
                .unwrap_or_default();
            self_ty.cmp(&other_ty)
        })
    }
}

/// Annotation that should be attached to a column in the file schema.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum ColumnAnnotation {
    DatePart(HashSet<SupportedIntervalUnit>),
    VariantPaths(Vec<VariantField>),
    SubstringSearch,
}

pub(crate) fn serialize_date_part(units: &HashSet<SupportedIntervalUnit>) -> String {
    let mut sorted_units: Vec<&SupportedIntervalUnit> = units.iter().collect();
    // Sort by a consistent order: Year, Month, Day, DayOfWeek
    sorted_units.sort_by_key(|unit| match unit {
        SupportedIntervalUnit::Year => 0,
        SupportedIntervalUnit::Month => 1,
        SupportedIntervalUnit::Day => 2,
        SupportedIntervalUnit::DayOfWeek => 3,
    });
    sorted_units
        .iter()
        .map(|unit| unit.metadata_value())
        .collect::<Vec<_>>()
        .join(",")
}

/// Logical optimizer that analyses the logical plan to detect columns that
/// are only used via compatible `EXTRACT` or `variant_get` projections.
#[derive(Debug, Default)]
pub struct LineageOptimizer;

impl LineageOptimizer {
    /// Create a new optimizer.
    pub fn new() -> Self {
        Self
    }
}

impl OptimizerRule for LineageOptimizer {
    fn name(&self) -> &str {
        "LineageOptimizer"
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
        let mut date_findings = table_usage.find_date32_extractions();
        date_findings.sort_by(|a, b| a.column.flat_name().cmp(&b.column.flat_name()));

        let mut variant_findings = table_usage.find_variant_gets();
        variant_findings.sort();

        let mut substring_findings = table_usage.find_substring_searches();
        substring_findings.sort_by_key(|a| a.flat_name());

        let annotations =
            build_annotation_map(&date_findings, &variant_findings, &substring_findings);
        annotate_plan_with_extractions(plan, &annotations)
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

#[derive(Debug, Clone, PartialEq, Eq)]
enum Operation {
    Extract(SupportedIntervalUnit),
    VariantGet {
        path: String,
        data_type: Option<DataType>,
    },
    SubstringSearch,
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
            if is_date_part_type(&stats.data_type) {
                // Collect all extract units from paths where the first n operations are all extracts
                let mut all_units = HashSet::new();
                let mut all_paths_valid = true;

                for usage in &stats.usages {
                    // Collect all Extract units from the leading sequence of extracts
                    let mut path_units = HashSet::new();
                    for op in usage {
                        match op {
                            Operation::Extract(unit) => {
                                path_units.insert(unit);
                            }
                            _ => {
                                // Stop at first non-extract operation
                                break;
                            }
                        }
                    }

                    if path_units.is_empty() {
                        // This path doesn't start with Extract, so skip this column
                        all_paths_valid = false;
                        break;
                    }

                    // Union the units from this path into the overall set
                    all_units.extend(path_units);
                }

                if all_paths_valid && !all_units.is_empty() {
                    extractions.push(DateExtraction {
                        column: key.to_column(),
                        components: all_units,
                    });
                }
            }
        }
        extractions
    }

    fn find_variant_gets(&self) -> Vec<VariantExtraction> {
        let mut gets = Vec::new();
        for (key, stats) in self.usage.iter() {
            if stats.usages.is_empty() {
                continue;
            }

            let mut field_map: HashMap<String, VariantField> = HashMap::new();
            let mut valid = true;
            let mut saw_variant_get = false;
            for usage in &stats.usages {
                match usage.first() {
                    Some(Operation::VariantGet { path, data_type }) => {
                        saw_variant_get = true;
                        match field_map.entry(path.clone()) {
                            Entry::Vacant(entry) => {
                                entry.insert(VariantField {
                                    path: path.clone(),
                                    data_type: data_type.clone(),
                                });
                            }
                            Entry::Occupied(entry) => {
                                let current = entry.into_mut();
                                let conflict = match (&current.data_type, data_type) {
                                    (Some(existing), Some(new_ty)) => existing != new_ty,
                                    (Some(_), None) | (None, Some(_)) => true,
                                    (None, None) => false,
                                };
                                if conflict {
                                    valid = false;
                                    break;
                                }
                            }
                        }
                    }
                    // A passthrough of the base column (no operations) should not invalidate
                    // the variant metadata, but also does not contribute a path.
                    None => continue,
                    _ => {
                        valid = false;
                        break;
                    }
                }
            }

            if valid && saw_variant_get && !field_map.is_empty() {
                let mut fields: Vec<VariantField> = field_map.into_values().collect();
                fields.sort();
                gets.push(VariantExtraction {
                    column: key.to_column(),
                    fields,
                });
            }
        }
        gets
    }

    fn find_substring_searches(&self) -> Vec<Column> {
        let mut columns = Vec::new();
        for (key, stats) in self.usage.iter() {
            if !is_string_type(&stats.data_type) {
                continue;
            }

            if stats.usages.is_empty() {
                continue;
            }

            let mut saw_substring = false;
            let mut valid = true;
            for usage in &stats.usages {
                let has_substring = usage
                    .iter()
                    .any(|op| matches!(op, Operation::SubstringSearch));
                if has_substring {
                    saw_substring = true;
                    continue;
                }
                if !usage.is_empty() {
                    valid = false;
                    break;
                }
            }

            if valid && saw_substring {
                columns.push(key.to_column());
            }
        }
        columns
    }
}

fn build_annotation_map(
    date_findings: &[DateExtraction],
    variant_findings: &[VariantExtraction],
    substring_findings: &[Column],
) -> HashMap<ColumnKey, ColumnAnnotation> {
    let mut annotations: HashMap<ColumnKey, ColumnAnnotation> = HashMap::new();
    for extraction in date_findings {
        annotations.insert(
            ColumnKey::from_column(&extraction.column),
            ColumnAnnotation::DatePart(extraction.components.clone()),
        );
    }
    for extraction in variant_findings {
        annotations.insert(
            ColumnKey::from_column(&extraction.column),
            ColumnAnnotation::VariantPaths(extraction.fields.clone()),
        );
    }
    for column in substring_findings {
        annotations.insert(
            ColumnKey::from_column(column),
            ColumnAnnotation::SubstringSearch,
        );
    }
    annotations
}

fn annotate_plan_with_extractions(
    plan: LogicalPlan,
    annotations: &HashMap<ColumnKey, ColumnAnnotation>,
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
    annotations: &HashMap<ColumnKey, ColumnAnnotation>,
) -> HashMap<String, ColumnAnnotation> {
    let mut table_annotations = HashMap::new();

    for (qualifier_opt, field_ref) in scan.projected_schema.iter() {
        let qualifier_owned = qualifier_opt.cloned();
        let name = field_ref.name().clone();
        if let Some(unit) = annotations
            .get(&ColumnKey::new(qualifier_owned.clone(), name.clone()))
            .cloned()
            .or_else(|| {
                annotations
                    .get(&ColumnKey::new(None, name.clone()))
                    .cloned()
            })
        {
            table_annotations.insert(name, unit);
        }
    }

    table_annotations
}

fn annotate_listing_table_source(
    source: &Arc<dyn TableSource>,
    annotations: &HashMap<String, ColumnAnnotation>,
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

    let metadata_copy = annotations.clone();
    let new_factory: Arc<dyn PhysicalExprAdapterFactory> = Arc::new(
        LineageExtractPhysicalExprAdapterFactory::new(annotations.clone()),
    );
    register_factory_metadata(&new_factory, metadata_copy);
    let mut new_listing = ListingTable::try_new(
        ListingTableConfig::new_with_multi_paths(listing.table_paths().clone())
            .with_listing_options(listing.options().clone())
            .with_schema(listing_file_schema(listing))
            .with_expr_adapter_factory(new_factory),
    )?;
    new_listing = new_listing.with_constraints(listing_constraints(listing));
    new_listing = new_listing.with_definition(
        listing
            .get_table_definition()
            .map(std::string::ToString::to_string),
    );

    let new_provider: Arc<dyn TableProvider> = Arc::new(new_listing);
    Ok(Some(provider_as_source(new_provider)))
}

#[derive(Debug)]
struct LineageExtractPhysicalExprAdapterFactory {
    base: Arc<dyn PhysicalExprAdapterFactory>,
    _annotations: HashMap<String, ColumnAnnotation>,
}

impl LineageExtractPhysicalExprAdapterFactory {
    fn new(annotations: HashMap<String, ColumnAnnotation>) -> Self {
        Self {
            base: Arc::new(DefaultPhysicalExprAdapterFactory),
            _annotations: annotations,
        }
    }
}

impl PhysicalExprAdapterFactory for LineageExtractPhysicalExprAdapterFactory {
    fn create(
        &self,
        logical_file_schema: SchemaRef,
        physical_file_schema: SchemaRef,
    ) -> Arc<dyn PhysicalExprAdapter> {
        self.base.create(logical_file_schema, physical_file_schema)
    }
}

fn listing_file_schema(listing: &ListingTable) -> SchemaRef {
    let table_schema = listing.schema();
    let file_field_count = table_schema
        .fields()
        .len()
        .saturating_sub(listing.options().table_partition_cols.len());
    let fields = table_schema
        .fields()
        .iter()
        .take(file_field_count)
        .cloned()
        .collect::<Vec<_>>();
    Arc::new(Schema::new(fields).with_metadata(table_schema.metadata().clone()))
}

fn listing_constraints(listing: &ListingTable) -> Constraints {
    listing.constraints().cloned().unwrap_or_default()
}

fn factory_registry() -> &'static Mutex<HashMap<usize, HashMap<String, ColumnAnnotation>>> {
    static REGISTRY: OnceLock<Mutex<HashMap<usize, HashMap<String, ColumnAnnotation>>>> =
        OnceLock::new();
    REGISTRY.get_or_init(|| Mutex::new(HashMap::new()))
}

fn register_factory_metadata(
    factory: &Arc<dyn PhysicalExprAdapterFactory>,
    metadata: HashMap<String, ColumnAnnotation>,
) {
    let key = Arc::as_ptr(factory) as *const () as usize;
    factory_registry().lock().unwrap().insert(key, metadata);
}

pub(crate) fn metadata_from_factory(
    factory: &Arc<dyn PhysicalExprAdapterFactory>,
    column: &str,
) -> Option<ColumnAnnotation> {
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
            } else if func_name.eq_ignore_ascii_case("variant_get")
                && (func.args.len() == 2 || func.args.len() == 3)
                && let Some(path) = literal_utf8(&func.args[1])
            {
                let type_hint = func.args.get(2).and_then(literal_data_type);
                let mut usages = lineage_for_expr(&func.args[0], input_lineage, schema)?;
                for usage in &mut usages {
                    usage.operations.push(Operation::VariantGet {
                        path: path.clone(),
                        data_type: type_hint.clone(),
                    });
                }
                return Ok(usages);
            }
            propagate_other(expr, input_lineage, schema)
        }
        Expr::Like(like) => {
            if !like.case_insensitive
                && like.escape_char.is_none()
                && let Some(pattern) = literal_utf8(&like.pattern)
                && is_substring_pattern(pattern.as_bytes())
            {
                let mut usages = lineage_for_expr(&like.expr, input_lineage, schema)?;
                for usage in &mut usages {
                    usage.operations.push(Operation::SubstringSearch);
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

fn literal_utf8(expr: &Expr) -> Option<String> {
    match expr {
        Expr::Literal(value, _) => match value {
            ScalarValue::Utf8(Some(v)) | ScalarValue::LargeUtf8(Some(v)) => Some(v.clone()),
            ScalarValue::Utf8View(Some(v)) => Some(v.clone()),
            _ => None,
        },
        _ => None,
    }
}

fn is_substring_pattern(pattern: &[u8]) -> bool {
    if pattern.len() < 2 {
        return false;
    }
    if pattern[0] != b'%' || pattern[pattern.len() - 1] != b'%' {
        return false;
    }
    let inner = &pattern[1..pattern.len() - 1];
    if inner.is_empty() {
        return false;
    }
    !inner.iter().any(|b| *b == b'%' || *b == b'_')
}

fn is_string_type(data_type: &DataType) -> bool {
    match data_type {
        DataType::Utf8 | DataType::Utf8View | DataType::LargeUtf8 => true,
        DataType::Dictionary(_, value_type) => is_string_type(value_type.as_ref()),
        _ => false,
    }
}

fn is_date_part_type(data_type: &DataType) -> bool {
    matches!(data_type, DataType::Date32 | DataType::Timestamp(_, _))
}

fn literal_data_type(expr: &Expr) -> Option<DataType> {
    literal_utf8(expr).and_then(|spec| DataType::from_str(&spec).ok())
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
    let lowered = text.to_ascii_lowercase();
    match lowered.as_str() {
        "dow" | "dayofweek" | "day_of_week" => {
            return Some(SupportedIntervalUnit::DayOfWeek);
        }
        _ => {}
    }
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

    use crate::optimizers::{
        DATE_MAPPING_METADATA_KEY, LocalModeOptimizer, VARIANT_MAPPING_METADATA_KEY,
    };
    use crate::{LiquidCacheParquet, VariantGetUdf, VariantToJsonUdf};
    use liquid_cache_common::IoMode;
    use liquid_cache::cache::AlwaysHydrate;

    use super::*;
    use arrow::array::{ArrayRef, Date32Array, StringArray, TimestampMicrosecondArray};
    use arrow_schema::{Field, Schema, TimeUnit};
    use datafusion::catalog::memory::DataSourceExec;
    use datafusion::datasource::physical_plan::FileScanConfig;
    use datafusion::execution::SessionStateBuilder;
    use datafusion::logical_expr::ScalarUDF;
    use datafusion::physical_plan::ExecutionPlan;
    use datafusion::prelude::{ParquetReadOptions, SessionConfig, SessionContext};
    use liquid_cache::cache::squeeze_policies::TranscodeSqueezeEvict;
    use liquid_cache::cache_policies::LiquidPolicy;
    use parquet::arrow::ArrowWriter;
    use parquet::variant::{VariantArray, json_to_variant};
    use serde::Deserialize;
    use tempfile::TempDir;

    // ─────────────────────────────────────────────────────────────────────────────
    // Setup helpers - lean versions for different test scenarios
    // ─────────────────────────────────────────────────────────────────────────────

    fn create_physical_optimizer() -> LocalModeOptimizer {
        LocalModeOptimizer::with_cache(Arc::new(LiquidCacheParquet::new(
            1024,
            1024 * 1024 * 1024,
            PathBuf::from("test"),
            Box::new(LiquidPolicy::new()),
            Box::new(TranscodeSqueezeEvict),
            Box::new(AlwaysHydrate::new()),
            IoMode::Uring,
        )))
    }

    fn create_session_context(optimizer: Arc<LineageOptimizer>) -> SessionContext {
        let state = SessionStateBuilder::new()
            .with_config(SessionConfig::new())
            .with_default_features()
            .with_optimizer_rule(optimizer as Arc<dyn OptimizerRule + Send + Sync>)
            .with_physical_optimizer_rule(Arc::new(create_physical_optimizer()))
            .build();
        SessionContext::new_with_state(state)
    }

    fn write_date_parquet(path: &std::path::Path) {
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

        let file = std::fs::File::create(path).unwrap();
        let mut writer = ArrowWriter::try_new(file, Arc::clone(&schema), None).unwrap();
        writer.write(&batch).unwrap();
        writer.close().unwrap();
    }

    fn write_variant_parquet(path: &std::path::Path) {
        let values = StringArray::from(vec![
            Some(r#"{"name": "Alice", "age": 30}"#),
            Some(r#"{"name": "Bob", "age": 25}"#),
            Some(r#"{"name": "Charlie"}"#),
        ]);
        let input_array: ArrayRef = Arc::new(values);
        let variant: VariantArray =
            json_to_variant(&input_array).expect("variant conversion for test data");

        let schema = Arc::new(Schema::new(vec![variant.field("data")]));
        let batch = arrow::record_batch::RecordBatch::try_new(
            Arc::clone(&schema),
            vec![ArrayRef::from(variant)],
        )
        .expect("variant batch");

        let file = std::fs::File::create(path).expect("create variant parquet file");
        let mut writer =
            ArrowWriter::try_new(file, batch.schema(), None).expect("create variant writer");
        writer.write(&batch).expect("write variant batch");
        writer.close().expect("close variant writer");
    }

    /// Setup for tests that only need a single date table (table_a)
    async fn setup_single_date_table() -> (TempDir, SessionContext, Arc<LineageOptimizer>) {
        let temp_dir = TempDir::new().unwrap();
        let table_path = temp_dir.path().join("table_a.parquet");
        write_date_parquet(&table_path);

        let optimizer = Arc::new(LineageOptimizer::new());
        let ctx = create_session_context(optimizer.clone());
        ctx.register_parquet(
            "table_a",
            table_path.to_str().unwrap(),
            ParquetReadOptions::default(),
        )
        .await
        .unwrap();

        (temp_dir, ctx, optimizer)
    }

    /// Setup for tests that need two date tables (table_a and table_b) for joins
    async fn setup_dual_date_tables() -> (TempDir, SessionContext, Arc<LineageOptimizer>) {
        let temp_dir = TempDir::new().unwrap();
        let table_a_path = temp_dir.path().join("table_a.parquet");
        let table_b_path = temp_dir.path().join("table_b.parquet");
        write_date_parquet(&table_a_path);
        write_date_parquet(&table_b_path);

        let optimizer = Arc::new(LineageOptimizer::new());
        let ctx = create_session_context(optimizer.clone());
        ctx.register_parquet(
            "table_a",
            table_a_path.to_str().unwrap(),
            ParquetReadOptions::default(),
        )
        .await
        .unwrap();
        ctx.register_parquet(
            "table_b",
            table_b_path.to_str().unwrap(),
            ParquetReadOptions::default(),
        )
        .await
        .unwrap();

        (temp_dir, ctx, optimizer)
    }

    /// Setup for tests that only need a variant table
    async fn setup_variant_table() -> (TempDir, SessionContext, Arc<LineageOptimizer>) {
        let temp_dir = TempDir::new().unwrap();
        let variant_path = temp_dir.path().join("variants_test.parquet");
        write_variant_parquet(&variant_path);

        let optimizer = Arc::new(LineageOptimizer::new());
        let ctx = create_session_context(optimizer.clone());
        ctx.register_udf(ScalarUDF::new_from_impl(VariantGetUdf::default()));
        ctx.register_udf(ScalarUDF::new_from_impl(VariantToJsonUdf::default()));
        ctx.register_parquet(
            "variants_test",
            variant_path.to_str().unwrap(),
            ParquetReadOptions::default().skip_metadata(false),
        )
        .await
        .unwrap();

        (temp_dir, ctx, optimizer)
    }

    /// Setup for tests that need both date table and variant table
    async fn setup_date_and_variant_tables() -> (TempDir, SessionContext, Arc<LineageOptimizer>) {
        let temp_dir = TempDir::new().unwrap();
        let table_a_path = temp_dir.path().join("table_a.parquet");
        let variant_path = temp_dir.path().join("variants_test.parquet");
        write_date_parquet(&table_a_path);
        write_variant_parquet(&variant_path);

        let optimizer = Arc::new(LineageOptimizer::new());
        let ctx = create_session_context(optimizer.clone());
        ctx.register_udf(ScalarUDF::new_from_impl(VariantGetUdf::default()));
        ctx.register_udf(ScalarUDF::new_from_impl(VariantToJsonUdf::default()));
        ctx.register_parquet(
            "table_a",
            table_a_path.to_str().unwrap(),
            ParquetReadOptions::default(),
        )
        .await
        .unwrap();
        ctx.register_parquet(
            "variants_test",
            variant_path.to_str().unwrap(),
            ParquetReadOptions::default().skip_metadata(false),
        )
        .await
        .unwrap();

        (temp_dir, ctx, optimizer)
    }

    // ─────────────────────────────────────────────────────────────────────────────
    // Test utilities
    // ─────────────────────────────────────────────────────────────────────────────

    fn extract_field_metadata(
        plan: &Arc<dyn ExecutionPlan>,
        metadata_key: &str,
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

            let file_schema = &file_scan_config.file_schema();
            for field in file_schema.fields() {
                if let Some(metadata_value) = field.metadata().get(metadata_key) {
                    field_metadata_map.insert(field.name().to_string(), metadata_value.clone());
                }
            }
            Ok(TreeNodeRecursion::Continue)
        })
        .unwrap();
        field_metadata_map
    }

    #[derive(Debug, Deserialize)]
    struct VariantMetadataEntry {
        path: String,
        #[serde(rename = "type")]
        data_type: Option<String>,
    }

    fn parse_variant_metadata(value: &str) -> Vec<VariantMetadataEntry> {
        serde_json::from_str(value).unwrap_or_else(|_| {
            vec![VariantMetadataEntry {
                path: value.to_string(),
                data_type: None,
            }]
        })
    }

    fn variant_paths_from_metadata(value: &str) -> Vec<String> {
        parse_variant_metadata(value)
            .into_iter()
            .map(|entry| entry.path)
            .collect()
    }

    /// Assert metadata on physical plan matches expected date and variant extractions
    async fn assert_metadata(
        ctx: &SessionContext,
        sql: &str,
        expected_date: Vec<(&str, &str)>,
        expected_variant: Vec<&str>,
    ) {
        let df = ctx.sql(sql).await.unwrap();
        let (state, plan) = df.into_parts();
        let optimized = state.optimize(&plan).unwrap();
        let physical_plan = state.create_physical_plan(&optimized).await.unwrap();

        let date_metadata = extract_field_metadata(&physical_plan, DATE_MAPPING_METADATA_KEY);
        let variant_metadata = extract_field_metadata(&physical_plan, VARIANT_MAPPING_METADATA_KEY);

        // Check date metadata
        let expected_date_map: HashMap<String, String> = expected_date
            .into_iter()
            .map(|(col, val)| (col.to_string(), val.to_string()))
            .collect();
        assert_eq!(
            date_metadata, expected_date_map,
            "date metadata mismatch for SQL: {}",
            sql
        );

        // Check variant metadata
        if expected_variant.is_empty() {
            assert!(
                !variant_metadata.contains_key("data"),
                "variant metadata should not be present for SQL: {}",
                sql
            );
        } else {
            let mut actual = variant_metadata
                .get("data")
                .map(|v| variant_paths_from_metadata(v))
                .unwrap_or_default();
            actual.sort();
            let mut expected: Vec<String> = expected_variant
                .into_iter()
                .map(|s| s.to_string())
                .collect();
            expected.sort();
            assert_eq!(
                actual, expected,
                "variant metadata mismatch for SQL: {}",
                sql
            );
        }
    }

    // ─────────────────────────────────────────────────────────────────────────────
    // Date extraction tests - single table
    // ─────────────────────────────────────────────────────────────────────────────

    #[tokio::test]
    async fn extract_day_basic() {
        let (_dir, ctx, _) = setup_single_date_table().await;
        assert_metadata(
            &ctx,
            "SELECT EXTRACT(DAY FROM date) AS day FROM table_a",
            vec![("date", "DAY")],
            vec![],
        )
        .await;
    }

    #[tokio::test]
    async fn extract_dow_basic() {
        let (_dir, ctx, _) = setup_single_date_table().await;
        assert_metadata(
            &ctx,
            "SELECT date_part('dow', date) AS dow FROM table_a",
            vec![("date", "DOW")],
            vec![],
        )
        .await;
    }

    #[tokio::test]
    async fn extract_day_lowercase() {
        let (_dir, ctx, _) = setup_single_date_table().await;
        assert_metadata(
            &ctx,
            "SELECT EXTRACT(day FROM date) AS day FROM table_a",
            vec![("date", "DAY")],
            vec![],
        )
        .await;
    }

    #[tokio::test]
    async fn extract_day_qualified_column() {
        let (_dir, ctx, _) = setup_single_date_table().await;
        assert_metadata(
            &ctx,
            "SELECT EXTRACT(DAY FROM table_a.date) FROM table_a",
            vec![("date", "DAY")],
            vec![],
        )
        .await;
    }

    #[tokio::test]
    async fn extract_day_in_avg() {
        let (_dir, ctx, _) = setup_single_date_table().await;
        assert_metadata(
            &ctx,
            "SELECT AVG(EXTRACT(DAY FROM date)) AS avg_day FROM table_a",
            vec![("date", "DAY")],
            vec![],
        )
        .await;
    }

    #[tokio::test]
    async fn extract_day_in_expression() {
        let (_dir, ctx, _) = setup_single_date_table().await;
        assert_metadata(
            &ctx,
            "SELECT AVG(EXTRACT(DAY FROM date) + 1) AS avg_day FROM table_a",
            vec![("date", "DAY")],
            vec![],
        )
        .await;
    }

    #[tokio::test]
    async fn extract_day_in_subqueries() {
        let (_dir, ctx, _) = setup_single_date_table().await;
        assert_metadata(
            &ctx,
            "SELECT (SELECT MAX(EXTRACT(DAY FROM date)) FROM table_a) AS max_day, (SELECT MIN(EXTRACT(DAY FROM date)) FROM table_a) AS min_day",
            vec![("date", "DAY")],
            vec![],
        )
        .await;
    }

    // ─────────────────────────────────────────────────────────────────────────────
    // Date extraction tests - multiple components
    // ─────────────────────────────────────────────────────────────────────────────

    #[tokio::test]
    async fn extract_day_and_month() {
        let (_dir, ctx, _) = setup_single_date_table().await;
        assert_metadata(
            &ctx,
            "SELECT EXTRACT(DAY FROM date) AS day, EXTRACT(MONTH FROM date) AS month FROM table_a",
            vec![("date", "MONTH,DAY")],
            vec![],
        )
        .await;
    }

    #[tokio::test]
    async fn extract_day_and_month_subqueries() {
        let (_dir, ctx, _) = setup_single_date_table().await;
        assert_metadata(
            &ctx,
            "SELECT (SELECT MAX(EXTRACT(DAY FROM date)) FROM table_a) AS max_day, (SELECT MIN(EXTRACT(Month FROM date)) FROM table_a) AS min_day",
            vec![("date", "MONTH,DAY")],
            vec![],
        )
        .await;
    }

    // ─────────────────────────────────────────────────────────────────────────────
    // Date extraction tests - multi table (joins)
    // ─────────────────────────────────────────────────────────────────────────────

    #[tokio::test]
    async fn extract_from_joined_tables() {
        let (_dir, ctx, _) = setup_dual_date_tables().await;
        // Both tables have "date" column - metadata HashMap stores by column name only,
        // so we verify at least one extraction is present (iteration order determines which)
        let df = ctx
            .sql("SELECT EXTRACT(YEAR FROM table_a.date) AS year, EXTRACT(DAY FROM table_b.date) AS day FROM table_a INNER JOIN table_b ON table_a.event_ts = table_b.event_ts")
            .await
            .unwrap();
        let (state, plan) = df.into_parts();
        let optimized = state.optimize(&plan).unwrap();
        let physical_plan = state.create_physical_plan(&optimized).await.unwrap();
        let metadata = extract_field_metadata(&physical_plan, DATE_MAPPING_METADATA_KEY);

        // Both tables' date columns should have extraction metadata
        assert!(
            metadata.contains_key("date"),
            "date column should have extraction metadata"
        );
        let value = metadata.get("date").unwrap();
        assert!(
            value == "YEAR" || value == "DAY",
            "expected YEAR or DAY, got {}",
            value
        );
    }

    // ─────────────────────────────────────────────────────────────────────────────
    // Date extraction tests - no metadata (inconsistent usage)
    // ─────────────────────────────────────────────────────────────────────────────

    #[tokio::test]
    async fn no_extraction_with_interval_arithmetic() {
        let (_dir, ctx, _) = setup_single_date_table().await;
        assert_metadata(
            &ctx,
            "SELECT EXTRACT(DAY FROM date + INTERVAL '1 day') AS day FROM table_a",
            vec![],
            vec![],
        )
        .await;
    }

    #[tokio::test]
    async fn no_extraction_when_column_used_directly() {
        let (_dir, ctx, _) = setup_single_date_table().await;
        assert_metadata(&ctx, "SELECT date FROM table_a", vec![], vec![]).await;
    }

    #[tokio::test]
    async fn no_extraction_when_used_in_join_condition() {
        let (_dir, ctx, _) = setup_dual_date_tables().await;
        assert_metadata(
            &ctx,
            "SELECT EXTRACT(DAY FROM table_a.date) AS day FROM table_a INNER JOIN table_b ON table_a.date = table_b.date",
            vec![],
            vec![],
        )
        .await;
    }

    #[tokio::test]
    async fn timestamp_extraction_supported() {
        let (_dir, ctx, _) = setup_single_date_table().await;
        assert_metadata(
            &ctx,
            "SELECT EXTRACT(YEAR FROM event_ts) AS year FROM table_a",
            vec![("event_ts", "YEAR")],
            vec![],
        )
        .await;
    }

    #[tokio::test]
    async fn timestamp_dow_extraction() {
        let (_dir, ctx, _) = setup_single_date_table().await;
        assert_metadata(
            &ctx,
            "SELECT date_part('dow', event_ts) AS dow FROM table_a",
            vec![("event_ts", "DOW")],
            vec![],
        )
        .await;
    }

    // ─────────────────────────────────────────────────────────────────────────────
    // Date extraction - metadata isolation test
    // ─────────────────────────────────────────────────────────────────────────────

    #[tokio::test]
    async fn metadata_only_on_extracted_fields() {
        let (_dir, ctx, _) = setup_single_date_table().await;
        // Only 'date' should have metadata, not 'event_ts'
        assert_metadata(
            &ctx,
            "SELECT EXTRACT(YEAR FROM date) AS year, event_ts FROM table_a",
            vec![("date", "YEAR")],
            vec![],
        )
        .await;
    }

    // ─────────────────────────────────────────────────────────────────────────────
    // Variant extraction tests - basic
    // ─────────────────────────────────────────────────────────────────────────────

    #[tokio::test]
    async fn variant_get_single_path() {
        let (_dir, ctx, _) = setup_variant_table().await;
        assert_metadata(
            &ctx,
            "SELECT variant_to_json(variant_get(data, 'name')) FROM variants_test",
            vec![],
            vec!["name"],
        )
        .await;
    }

    #[tokio::test]
    async fn variant_get_duplicate_paths() {
        let (_dir, ctx, _) = setup_variant_table().await;
        assert_metadata(
            &ctx,
            "SELECT variant_get(data, 'name'), variant_get(data, 'name') AS name2 FROM variants_test",
            vec![],
            vec!["name"],
        )
        .await;
    }

    #[tokio::test]
    async fn variant_get_with_to_json() {
        let (_dir, ctx, _) = setup_variant_table().await;
        assert_metadata(
            &ctx,
            "SELECT variant_to_json(variant_get(data, 'age')), variant_to_json(variant_get(data, 'age')) AS age2 FROM variants_test",
            vec![],
            vec!["age"],
        )
        .await;
    }

    #[tokio::test]
    async fn variant_get_qualified_column() {
        let (_dir, ctx, _) = setup_variant_table().await;
        assert_metadata(
            &ctx,
            "SELECT variant_get(variants_test.data, 'name') as name1, variant_get(variants_test.data, 'name') as name2 FROM variants_test",
            vec![],
            vec!["name"],
        )
        .await;
    }

    #[tokio::test]
    async fn variant_get_in_aggregates() {
        let (_dir, ctx, _) = setup_variant_table().await;
        assert_metadata(
            &ctx,
            "SELECT COUNT(variant_get(data, 'age')), MAX(variant_get(data, 'age')) FROM variants_test",
            vec![],
            vec!["age"],
        )
        .await;
    }

    #[tokio::test]
    async fn variant_get_with_where_clause() {
        let (_dir, ctx, _) = setup_variant_table().await;
        assert_metadata(
            &ctx,
            "SELECT variant_get(data, 'name') FROM variants_test WHERE variant_get(data, 'name') IS NOT NULL",
            vec![],
            vec!["name"],
        )
        .await;
    }

    #[tokio::test]
    async fn variant_get_in_subqueries() {
        let (_dir, ctx, _) = setup_variant_table().await;
        assert_metadata(
            &ctx,
            "SELECT (SELECT MAX(variant_get(data, 'name')) FROM variants_test) AS max_name, (SELECT MIN(variant_get(data, 'name')) FROM variants_test) AS min_name",
            vec![],
            vec!["name"],
        )
        .await;
    }

    #[tokio::test]
    async fn variant_get_multiple_paths() {
        let (_dir, ctx, _) = setup_variant_table().await;
        assert_metadata(
            &ctx,
            "SELECT variant_get(data, 'name'), variant_get(data, 'date') FROM variants_test",
            vec![],
            vec!["date", "name"],
        )
        .await;
    }

    #[tokio::test]
    async fn variant_get_nested() {
        let (_dir, ctx, _) = setup_variant_table().await;
        assert_metadata(
            &ctx,
            "SELECT variant_get(variant_get(data, 'name'), 'age') FROM variants_test",
            vec![],
            vec!["name"],
        )
        .await;
    }

    #[tokio::test]
    async fn variant_get_conflicting_types_no_metadata() {
        let (_dir, ctx, _) = setup_variant_table().await;
        // Same path with different type hints - should not produce metadata
        assert_metadata(
            &ctx,
            "SELECT variant_get(data, 'name', 'Utf8'), variant_get(data, 'name') FROM variants_test",
            vec![],
            vec![],
        )
        .await;
    }

    // ─────────────────────────────────────────────────────────────────────────────
    // Variant extraction tests - type metadata
    // ─────────────────────────────────────────────────────────────────────────────

    #[tokio::test]
    async fn variant_get_type_hint_propagated() {
        let (_dir, ctx, _) = setup_variant_table().await;

        let df = ctx
            .sql("SELECT variant_get(data, 'name', 'Utf8') FROM variants_test")
            .await
            .unwrap();
        let (state, plan) = df.into_parts();
        let optimized = state.optimize(&plan).unwrap();
        let physical_plan = state.create_physical_plan(&optimized).await.unwrap();

        let metadata = extract_field_metadata(&physical_plan, VARIANT_MAPPING_METADATA_KEY);

        let entries = metadata
            .get("data")
            .map(|value| parse_variant_metadata(value))
            .unwrap_or_default();
        let entry = entries
            .iter()
            .find(|entry| entry.path == "name")
            .expect("variant metadata entry for name");
        assert_eq!(entry.data_type.as_deref(), Some("Utf8"));
    }

    #[tokio::test]
    async fn variant_get_conflicting_types_in_filter() {
        let (_dir, ctx, _) = setup_variant_table().await;
        assert_metadata(
            &ctx,
            "SELECT variant_to_json(variant_get(data, 'name')) FROM variants_test WHERE variant_get(data, 'name', 'Utf8') = 'Bob'",
            vec![],
            vec![],
        )
        .await;
    }

    // ─────────────────────────────────────────────────────────────────────────────
    // Variant extraction tests - edge cases
    // ─────────────────────────────────────────────────────────────────────────────

    #[tokio::test]
    async fn variant_get_multiple_paths_with_types() {
        let (_dir, ctx, _) = setup_variant_table().await;
        assert_metadata(
            &ctx,
            "SELECT variant_get(data, 'did', 'Utf8') as user_id,
             MAX(TO_TIMESTAMP_MICROS(variant_get(data, 'time_us', 'Int64'))) - MIN(TO_TIMESTAMP_MICROS(variant_get(data, 'time_us', 'Int64')))
            FROM variants_test GROUP BY user_id",
            vec![],
            vec!["did", "time_us"],
        )
        .await;
    }

    // ─────────────────────────────────────────────────────────────────────────────
    // Mixed date extract and variant tests
    // ─────────────────────────────────────────────────────────────────────────────

    #[tokio::test]
    async fn mixed_date_and_variant_basic() {
        let (_dir, ctx, _) = setup_date_and_variant_tables().await;
        assert_metadata(
            &ctx,
            "SELECT EXTRACT(DAY FROM table_a.date) AS day, variant_get(variants_test.data, 'name') AS name FROM table_a CROSS JOIN variants_test",
            vec![("date", "DAY")],
            vec!["name"],
        )
        .await;
    }

    #[tokio::test]
    async fn mixed_multiple_date_and_variant() {
        let (_dir, ctx, _) = setup_date_and_variant_tables().await;
        assert_metadata(
            &ctx,
            "SELECT EXTRACT(YEAR FROM table_a.date) AS year, EXTRACT(MONTH FROM table_a.date_copy) AS month, variant_get(variants_test.data, 'name') AS name, variant_get(variants_test.data, 'age') AS age FROM table_a CROSS JOIN variants_test",
            vec![("date", "YEAR"), ("date_copy", "MONTH")],
            vec!["age", "name"],
        )
        .await;
    }

    #[tokio::test]
    async fn mixed_date_in_where_clause() {
        let (_dir, ctx, _) = setup_date_and_variant_tables().await;
        assert_metadata(
            &ctx,
            "SELECT variant_get(variants_test.data, 'name') AS name FROM variants_test CROSS JOIN table_a WHERE EXTRACT(DAY FROM table_a.date) > 1",
            vec![("date", "DAY")],
            vec!["name"],
        )
        .await;
    }

    #[tokio::test]
    async fn mixed_variant_in_where_clause() {
        let (_dir, ctx, _) = setup_date_and_variant_tables().await;
        assert_metadata(
            &ctx,
            "SELECT EXTRACT(DAY FROM table_a.date) AS day FROM table_a CROSS JOIN variants_test WHERE variant_get(variants_test.data, 'name') IS NOT NULL",
            vec![("date", "DAY")],
            vec!["name"],
        )
        .await;
    }

    #[tokio::test]
    async fn mixed_date_with_variant_subquery() {
        let (_dir, ctx, _) = setup_date_and_variant_tables().await;
        assert_metadata(
            &ctx,
            "SELECT EXTRACT(YEAR FROM table_a.date) AS year, (SELECT variant_get(variants_test.data, 'name') FROM variants_test LIMIT 1) AS name FROM table_a",
            vec![("date", "YEAR")],
            vec!["name"],
        )
        .await;
    }

    #[tokio::test]
    async fn mixed_in_aggregates() {
        let (_dir, ctx, _) = setup_date_and_variant_tables().await;
        assert_metadata(
            &ctx,
            "SELECT AVG(EXTRACT(DAY FROM table_a.date)) AS avg_day, COUNT(variant_get(variants_test.data, 'name')) AS name_count FROM table_a CROSS JOIN variants_test",
            vec![("date", "DAY")],
            vec!["name"],
        )
        .await;
    }
}
