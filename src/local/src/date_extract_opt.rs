use std::collections::{HashMap, HashSet};
use std::str::FromStr;
use std::sync::{Arc, Mutex};

use arrow::compute::kernels::cast_utils::IntervalUnit;
use arrow_schema::DataType;
use datafusion::common::ScalarValue;
use datafusion::common::tree_node::{TreeNode, TreeNodeRecursion};
use datafusion::config::ConfigOptions;
use datafusion::error::Result;
use datafusion::physical_expr::expressions::{CastExpr, Column, Literal};
use datafusion::physical_expr::{PhysicalExpr, ScalarFunctionExpr};
use datafusion::physical_optimizer::PhysicalOptimizerRule;
use datafusion::physical_plan::ExecutionPlan;
use datafusion::physical_plan::projection::ProjectionExec;

/// Supported components for `EXTRACT` clauses.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum SupportedIntervalUnit {
    Year,
    Month,
    Day,
    Week,
    Full,
}

/// Metadata describing a column that participates in an `EXTRACT` operation.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(crate) struct DateExtraction {
    /// The 0-based column index within the input schema.
    pub(crate) column_index: usize,
    /// The column name.
    pub(crate) column_name: String,
    /// The component being extracted.
    pub(crate) component: SupportedIntervalUnit,
}

/// Physical optimizer that scans projection expressions to locate `EXTRACT`
/// calls over temporal columns.
///
/// The rule does not mutate the plan. It analyses the plan and records
/// matching columns so callers can take action (e.g. request additional cache
/// columns) after optimization.
#[derive(Debug, Default)]
pub(crate) struct DateExtractOptimizer {
    extracted: Arc<Mutex<Vec<DateExtraction>>>,
}

impl DateExtractOptimizer {
    /// Create a new optimizer.
    pub(crate) fn new() -> Self {
        Self::default()
    }

    /// Retrieve the extractions discovered during the last optimization pass.
    pub(crate) fn extractions(&self) -> Vec<DateExtraction> {
        self.extracted.lock().unwrap().clone()
    }
}

impl PhysicalOptimizerRule for DateExtractOptimizer {
    fn optimize(
        &self,
        plan: Arc<dyn ExecutionPlan>,
        _config: &ConfigOptions,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        let mut candidates = Vec::new();

        plan.apply(|node| {
            if let Some(projection) = node.as_any().downcast_ref::<ProjectionExec>() {
                let input_schema = projection.input().schema();
                for projection_expr in projection.expr() {
                    collect_date_extracts(
                        &projection_expr.expr,
                        input_schema.as_ref(),
                        &mut candidates,
                    );
                }
            }
            Ok(TreeNodeRecursion::Continue)
        })?;

        let mut deduped = Vec::new();
        let mut seen = HashSet::new();
        for candidate in candidates {
            if seen.insert(candidate.clone()) {
                deduped.push(candidate);
            }
        }

        let mut counts: HashMap<(usize, String), usize> = HashMap::new();
        for entry in &deduped {
            *counts
                .entry((entry.column_index, entry.column_name.clone()))
                .or_insert(0) += 1;
        }

        let filtered = deduped
            .into_iter()
            .filter(|entry| {
                counts[&(entry.column_index, entry.column_name.clone())] == 1
                    && entry.component != SupportedIntervalUnit::Full
            })
            .collect::<Vec<_>>();

        *self.extracted.lock().unwrap() = filtered;

        Ok(plan)
    }

    fn name(&self) -> &str {
        "DateExtractOptimizer"
    }

    fn schema_check(&self) -> bool {
        true
    }
}

// we only extract date where the right children is a column expression,
// i.e., no nested expression.
fn collect_date_extracts(
    expr: &Arc<dyn PhysicalExpr>,
    input_schema: &arrow_schema::Schema,
    output: &mut Vec<DateExtraction>,
) {
    if let Some(extraction) = try_extract_from_expr(expr, input_schema) {
        output.push(extraction);
    }
}

fn try_extract_from_expr(
    expr: &Arc<dyn PhysicalExpr>,
    input_schema: &arrow_schema::Schema,
) -> Option<DateExtraction> {
    // if it's full projection, return entire field
    if let Some(col) = expr.as_any().downcast_ref::<Column>() {
        let field = input_schema.field(col.index());
        if !is_temporal_type(field.data_type()) {
            return None;
        }
        return Some(DateExtraction {
            column_index: col.index(),
            column_name: field.name().clone(),
            component: SupportedIntervalUnit::Full,
        });
    }

    let scalar = expr.as_any().downcast_ref::<ScalarFunctionExpr>()?;
    if !scalar.name().eq_ignore_ascii_case("date_part") {
        return None;
    }

    let args = scalar.args();
    if args.len() != 2 {
        return None;
    }

    let component = part_to_unit(&args[0])?;
    let column = resolve_column(&args[1])?;

    let field = input_schema.field(column.index());
    if !is_temporal_type(field.data_type()) {
        return None;
    }

    Some(DateExtraction {
        column_index: column.index(),
        column_name: field.name().clone(),
        component,
    })
}

fn part_to_unit(expr: &Arc<dyn PhysicalExpr>) -> Option<SupportedIntervalUnit> {
    let literal = expr.as_any().downcast_ref::<Literal>()?;
    let part = match literal.value() {
        ScalarValue::Utf8(Some(value))
        | ScalarValue::LargeUtf8(Some(value))
        | ScalarValue::Utf8View(Some(value)) => value,
        _ => return None,
    };
    let interval_unit = IntervalUnit::from_str(part).ok()?;
    match interval_unit {
        IntervalUnit::Year => Some(SupportedIntervalUnit::Year),
        IntervalUnit::Month => Some(SupportedIntervalUnit::Month),
        IntervalUnit::Day => Some(SupportedIntervalUnit::Day),
        IntervalUnit::Week => Some(SupportedIntervalUnit::Week),
        _ => return None,
    }
}

fn resolve_column(expr: &Arc<dyn PhysicalExpr>) -> Option<Column> {
    if let Some(column) = expr.as_any().downcast_ref::<Column>() {
        return Some(column.clone());
    }

    if let Some(cast) = expr.as_any().downcast_ref::<CastExpr>() {
        return resolve_column(cast.expr());
    }

    None
}

fn is_temporal_type(data_type: &DataType) -> bool {
    matches!(
        data_type,
        DataType::Date32 | DataType::Date64 | DataType::Timestamp(_, _)
    )
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use super::*;
    use arrow::array::{ArrayRef, Date32Array, RecordBatch, TimestampMicrosecondArray};
    use arrow_schema::{Field, Schema, TimeUnit};
    use datafusion::{
        execution::SessionStateBuilder,
        physical_expr::expressions::col,
        physical_plan::display::DisplayableExecutionPlan,
        prelude::{ParquetReadOptions, SessionConfig, SessionContext},
    };
    use parquet::arrow::ArrowWriter;
    use tempfile::TempDir;

    #[test]
    fn resolves_column_through_casts() {
        let schema = Schema::new(vec![Field::new(
            "ts",
            DataType::Timestamp(arrow_schema::TimeUnit::Microsecond, None),
            false,
        )]);

        let column = col("ts", &schema).unwrap();
        let cast_expr = Arc::new(CastExpr::new(
            Arc::clone(&column),
            DataType::Timestamp(arrow_schema::TimeUnit::Nanosecond, None),
            None,
        )) as Arc<dyn PhysicalExpr>;

        let resolved = resolve_column(&cast_expr).expect("column");
        assert_eq!(resolved.name(), "ts");
        assert_eq!(resolved.index(), 0);
    }

    async fn create_test_ctx(
        table_a: &Path,
        table_b: &Path,
        optimizer: &Arc<DateExtractOptimizer>,
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
        let batch = RecordBatch::try_new(Arc::clone(&schema), vec![timestamps, dates]).unwrap();

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
            .with_physical_optimizer_rule(optimizer.clone())
            .build();
        let ctx = SessionContext::new_with_state(state);

        ctx.register_parquet(
            "table_a",
            table_a.to_str().unwrap(),
            ParquetReadOptions::default(),
        )
        .await
        .unwrap();
        ctx.register_parquet(
            "table_b",
            table_b.to_str().unwrap(),
            ParquetReadOptions::default(),
        )
        .await
        .unwrap();
        ctx
    }

    #[tokio::test]
    async fn date_extract_optimizer_detects() {
        let temp_dir = TempDir::new().unwrap();
        let table_a = temp_dir.path().join("table_a.parquet");
        let table_b = temp_dir.path().join("table_b.parquet");

        let optimizer = Arc::new(DateExtractOptimizer::new());
        let ctx = create_test_ctx(&table_a, &table_b, &optimizer).await;

        let sql_to_test = vec![
            "SELECT EXTRACT(YEAR FROM event_ts) AS year, EXTRACT(DAY FROM date) AS date FROM table_a",
            "SELECT EXTRACT(year FROM event_ts) AS year, EXTRACT(day FROM date) AS date FROM table_a",
            "SELECT EXTRACT(Year FROM event_ts) AS year, EXTRACT(Day FROM date) AS date FROM table_a",
            "SELECT EXTRACT(years FROM event_ts) AS year, EXTRACT(days FROM date) AS date FROM table_a",
            "SELECT EXTRACT(y FROM event_ts) AS year, EXTRACT(d FROM date) AS date FROM table_a",
        ];

        for sql in sql_to_test {
            let df = ctx.sql(sql).await.unwrap();
            let (state, plan) = df.into_parts();
            let _physical_plan = state.create_physical_plan(&plan).await.unwrap();
            let extractions = optimizer.extractions();
            assert_eq!(
                extractions,
                vec![
                    DateExtraction {
                        column_index: 0,
                        column_name: "event_ts".to_string(),
                        component: SupportedIntervalUnit::Year,
                    },
                    DateExtraction {
                        column_index: 1,
                        column_name: "date".to_string(),
                        component: SupportedIntervalUnit::Day,
                    }
                ]
            );
        }
    }

    #[tokio::test]
    async fn date_extract_optimizer_detects_conflict() {
        let temp_dir = TempDir::new().unwrap();
        let table_a = temp_dir.path().join("table_a.parquet");
        let table_b = temp_dir.path().join("table_b.parquet");

        let optimizer = Arc::new(DateExtractOptimizer::new());
        let ctx = create_test_ctx(&table_a, &table_b, &optimizer).await;

        let sql_to_test = vec![
            "SELECT EXTRACT(YEAR FROM event_ts) AS year, event_ts FROM table_a",
            "SELECT EXTRACT(YEAR FROM event_ts) AS year, EXTRACT(d FROM event_ts) AS day FROM table_a",
            "SELECT EXTRACT(Day FROM date) AS day, date FROM table_a",
            "SELECT EXTRACT(Day FROM date) AS day, EXTRACT(Month FROM date) as month FROM table_a",
            "SELECT EXTRACT(Day FROM event_ts + INTERVAL '1 day') AS day FROM table_a",
            "SELECT EXTRACT(Day FROM table_a.date) AS day FROM table_a Inner Join table_b ON table_a.date = table_b.date",
        ];

        for sql in sql_to_test {
            let df = ctx.sql(sql).await.unwrap();
            let (state, plan) = df.into_parts();
            let _physical_plan = state.create_physical_plan(&plan).await.unwrap();
            let extractions = optimizer.extractions();
            assert!(extractions.is_empty());
        }
    }

    #[tokio::test]
    async fn date_extract_optimizer_multi_table_conflict() {
        let temp_dir = TempDir::new().unwrap();
        let table_a = temp_dir.path().join("table_a.parquet");
        let table_b = temp_dir.path().join("table_b.parquet");

        let optimizer = Arc::new(DateExtractOptimizer::new());
        let ctx = create_test_ctx(&table_a, &table_b, &optimizer).await;

        let sql_to_test = vec![
            "SELECT EXTRACT(Day FROM table_a.date) AS day FROM table_a Inner Join table_b ON table_a.date = table_b.date",
        ];

        for sql in sql_to_test {
            let df = ctx.sql(sql).await.unwrap();
            let (state, plan) = df.into_parts();
            let _physical_plan = state.create_physical_plan(&plan).await.unwrap();
            let displayable =
                DisplayableExecutionPlan::new(_physical_plan.as_ref()).set_show_schema(true);
            println!("physical_plan: \n{}", displayable.indent(false).to_string());
            let extractions = optimizer.extractions();
            assert!(extractions.is_empty());
        }
    }

    #[tokio::test]
    async fn date_extract_optimizer_multi_table() {
        let temp_dir = TempDir::new().unwrap();
        let table_a = temp_dir.path().join("table_a.parquet");
        let table_b = temp_dir.path().join("table_b.parquet");

        let optimizer = Arc::new(DateExtractOptimizer::new());
        let ctx = create_test_ctx(&table_a, &table_b, &optimizer).await;

        let sql = "SELECT EXTRACT(YEAR FROM table_a.event_ts) AS year_a, EXTRACT(YEAR FROM table_b.event_ts) AS year_b FROM table_a, table_b";
        let df = ctx.sql(sql).await.unwrap();
        let (state, plan) = df.into_parts();
        let _physical_plan = state.create_physical_plan(&plan).await.unwrap();
        let displayable =
            DisplayableExecutionPlan::new(_physical_plan.as_ref()).set_show_schema(true);
        println!("physical_plan: \n{}", displayable.indent(false).to_string());
        let extractions = optimizer.extractions();
        assert_eq!(
            extractions,
            vec![
                DateExtraction {
                    column_index: 0,
                    column_name: "event_ts".to_string(),
                    component: SupportedIntervalUnit::Year
                },
                DateExtraction {
                    column_index: 1,
                    column_name: "event_ts".to_string(),
                    component: SupportedIntervalUnit::Year
                }
            ]
        );
    }
}
