#![allow(unused)]
// Not our code, some of them might be unused.

// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

//! Utilities to push down of DataFusion filter predicates (any DataFusion
//! `PhysicalExpr` that evaluates to a [`BooleanArray`]) to the parquet decoder
//! level in `arrow-rs`.
//!
//! DataFusion will use a `ParquetRecordBatchStream` to read data from parquet
//! into [`RecordBatch`]es.
//!
//! The `ParquetRecordBatchStream` takes an optional `RowFilter` which is itself
//! a Vec of `Box<dyn ArrowPredicate>`. During decoding, the predicates are
//! evaluated in order, to generate a mask which is used to avoid decoding rows
//! in projected columns which do not pass the filter which can significantly
//! reduce the amount of compute required for decoding and thus improve query
//! performance.
//!
//! Since the predicates are applied serially in the order defined in the
//! `RowFilter`, the optimal ordering depends on the exact filters. The best
//! filters to execute first have two properties:
//!
//! 1. They are relatively inexpensive to evaluate (e.g. they read
//!    column chunks which are relatively small)
//!
//! 2. They filter many (contiguous) rows, reducing the amount of decoding
//!    required for subsequent filters and projected columns
//!
//! If requested, this code will reorder the filters based on heuristics try and
//! reduce the evaluation cost.
//!
//! The basic algorithm for constructing the `RowFilter` is as follows
//!
//! 1. Break conjunctions into separate predicates. An expression
//!    like `a = 1 AND (b = 2 AND c = 3)` would be
//!    separated into the expressions `a = 1`, `b = 2`, and `c = 3`.
//! 2. Determine whether each predicate can be evaluated as an `ArrowPredicate`.
//! 3. Determine, for each predicate, the total compressed size of all
//!    columns required to evaluate the predicate.
//! 4. Determine, for each predicate, whether all columns required to
//!    evaluate the expression are sorted.
//! 5. Re-order the predicate by total size (from step 3).
//! 6. Partition the predicates according to whether they are sorted (from step 4)
//! 7. "Compile" each predicate `Expr` to a `DatafusionArrowPredicate`.
//! 8. Build the `RowFilter` with the sorted predicates followed by
//!    the unsorted predicates. Within each partition, predicates are
//!    still be sorted by size.

use std::cmp::Ordering;
use std::collections::BTreeSet;
use std::sync::Arc;

use arrow::array::{ArrayRef, AsArray, BooleanArray};
use arrow::compute::kernels;
use arrow::datatypes::{DataType, Schema};
use arrow::error::{ArrowError, Result as ArrowResult};
use arrow::record_batch::RecordBatch;
use arrow_schema::Field;
use datafusion::datasource::physical_plan::ParquetFileMetrics;
use datafusion::logical_expr::{ColumnarValue, Operator};
use datafusion::physical_expr_common::datum::apply_cmp;
use datafusion::physical_plan::expressions::{BinaryExpr, LikeExpr};
use datafusion::physical_plan::metrics;
use parquet::arrow::ProjectionMask;
use parquet::arrow::arrow_reader::{ArrowPredicate, RowFilter};
use parquet::file::metadata::ParquetMetaData;

use datafusion::common::cast::as_boolean_array;
use datafusion::common::tree_node::{
    Transformed, TransformedResult, TreeNode, TreeNodeRecursion, TreeNodeRewriter,
};
use datafusion::common::{DataFusionError, Result, ScalarValue, arrow_datafusion_err};
use datafusion::datasource::schema_adapter::SchemaMapper;
use datafusion::physical_expr::expressions::{Column, Literal};
use datafusion::physical_expr::utils::reassign_predicate_columns;
use datafusion::physical_expr::{PhysicalExpr, split_conjunction};

use crate::LiquidPredicate;
use crate::liquid_array::{AsLiquidArray, LiquidArray, LiquidArrayRef};
use crate::reader::runtime::LiquidRowFilter;

/// A "compiled" predicate passed to `ParquetRecordBatchStream` to perform
/// row-level filtering during parquet decoding.
///
/// See the module level documentation for more information.
///
/// Implements the `ArrowPredicate` trait used by the parquet decoder
///
/// An expression can be evaluated as a `DatafusionArrowPredicate` if it:
/// * Does not reference any projected columns
/// * Does not reference columns with non-primitive types (e.g. structs / lists)
#[derive(Debug)]
pub(crate) struct DatafusionArrowPredicate {
    /// the filter expression
    physical_expr: Arc<dyn PhysicalExpr>,
    /// Path to the columns in the parquet schema required to evaluate the
    /// expression
    projection_mask: ProjectionMask,
    /// Columns required to evaluate the expression in the arrow schema
    projection: Vec<usize>,
    /// how many rows were filtered out by this predicate
    rows_pruned: metrics::Count,
    /// how many rows passed this predicate
    rows_matched: metrics::Count,
    /// how long was spent evaluating this predicate
    time: metrics::Time,
}

impl DatafusionArrowPredicate {
    /// Create a new `DatafusionArrowPredicate` from a `FilterCandidate`
    pub fn try_new(
        candidate: FilterCandidate,
        schema: &Schema,
        metadata: &ParquetMetaData,
        rows_pruned: metrics::Count,
        rows_matched: metrics::Count,
        time: metrics::Time,
    ) -> Result<Self> {
        let schema = Arc::new(schema.project(&candidate.projection)?);
        let physical_expr = reassign_predicate_columns(candidate.expr, &schema, true)?;

        // ArrowPredicate::evaluate is passed columns in the order they appear in the file
        // If the predicate has multiple columns, we therefore must project the columns based
        // on the order they appear in the file
        let projection = match candidate.projection.len() {
            0 | 1 => vec![],
            2.. => remap_projection(&candidate.projection),
        };

        Ok(Self {
            physical_expr,
            projection,
            projection_mask: ProjectionMask::roots(
                metadata.file_metadata().schema_descr(),
                candidate.projection,
            ),
            rows_pruned,
            rows_matched,
            time,
        })
    }
}

fn get_string_needle(value: &ScalarValue) -> Option<&str> {
    if let ScalarValue::Utf8(Some(v)) = value {
        Some(v)
    } else if let ScalarValue::Dictionary(_, value) = value {
        if let ScalarValue::Utf8(Some(v)) = value.as_ref() {
            Some(v)
        } else {
            None
        }
    } else {
        None
    }
}

impl LiquidPredicate for DatafusionArrowPredicate {
    fn evaluate_liquid(&mut self, array: &LiquidArrayRef) -> Result<BooleanArray, ArrowError> {
        if let Some(binary_expr) = self.physical_expr.as_any().downcast_ref::<BinaryExpr>() {
            if let Some(literal) = binary_expr.right().as_any().downcast_ref::<Literal>() {
                let op = binary_expr.op();
                if let Some(array) = array.as_string_array_opt() {
                    if let Some(needle) = get_string_needle(literal.value()) {
                        if op == &Operator::Eq {
                            let result = array.compare_equals(needle);
                            return Ok(result);
                        } else if op == &Operator::NotEq {
                            let result = array.compare_not_equals(needle);
                            return Ok(result);
                        }
                    }

                    let dict_array = array.to_dict_arrow();
                    let lhs = ColumnarValue::Array(Arc::new(dict_array));
                    let rhs = ColumnarValue::Scalar(literal.value().clone());

                    let result = match op {
                        Operator::NotEq => apply_cmp(&lhs, &rhs, kernels::cmp::neq),
                        Operator::Eq => apply_cmp(&lhs, &rhs, kernels::cmp::eq),
                        Operator::Lt => apply_cmp(&lhs, &rhs, kernels::cmp::lt),
                        Operator::LtEq => apply_cmp(&lhs, &rhs, kernels::cmp::lt_eq),
                        Operator::Gt => apply_cmp(&lhs, &rhs, kernels::cmp::gt),
                        Operator::GtEq => apply_cmp(&lhs, &rhs, kernels::cmp::gt_eq),
                        Operator::LikeMatch => apply_cmp(&lhs, &rhs, arrow::compute::like),
                        Operator::ILikeMatch => apply_cmp(&lhs, &rhs, arrow::compute::ilike),
                        Operator::NotLikeMatch => apply_cmp(&lhs, &rhs, arrow::compute::nlike),
                        Operator::NotILikeMatch => apply_cmp(&lhs, &rhs, arrow::compute::nilike),
                        _ => panic!("unsupported operator: {:?}", op),
                    };
                    if let Ok(result) = result {
                        let filtered = result.into_array(array.len())?.as_boolean().clone();
                        return Ok(filtered);
                    }
                }
            }
        } else if let Some(like_expr) = self.physical_expr.as_any().downcast_ref::<LikeExpr>() {
            if let Some(literal) = like_expr.pattern().as_any().downcast_ref::<Literal>() {
                let arrow_dict = array.as_string().to_dict_arrow();

                let lhs = ColumnarValue::Array(Arc::new(arrow_dict));
                let rhs = ColumnarValue::Scalar(literal.value().clone());

                let result = match (like_expr.negated(), like_expr.case_insensitive()) {
                    (false, false) => apply_cmp(&lhs, &rhs, arrow::compute::like),
                    (true, false) => apply_cmp(&lhs, &rhs, arrow::compute::nlike),
                    (false, true) => apply_cmp(&lhs, &rhs, arrow::compute::ilike),
                    (true, true) => apply_cmp(&lhs, &rhs, arrow::compute::nilike),
                };
                if let Ok(result) = result {
                    let filtered = result.into_array(array.len())?.as_boolean().clone();
                    return Ok(filtered);
                }
            }
        }
        let arrow_array = array.to_arrow_array();
        let schema = Schema::new(vec![Field::new(
            "",
            arrow_array.data_type().clone(),
            arrow_array.is_nullable(),
        )]);
        let record_batch =
            RecordBatch::try_new(Arc::new(schema), vec![Arc::new(arrow_array)]).unwrap();
        self.evaluate(record_batch)
    }
}

impl ArrowPredicate for DatafusionArrowPredicate {
    fn projection(&self) -> &ProjectionMask {
        &self.projection_mask
    }

    fn evaluate(&mut self, mut batch: RecordBatch) -> ArrowResult<BooleanArray> {
        if !self.projection.is_empty() {
            batch = batch.project(&self.projection).unwrap();
        };

        // we deliberately don't map schema here, because when a schema mismatch occurs,
        // something deeper is wrong, and we should fix it instead of hiding it.
        // let batch = self.schema_mapping.map_partial_batch(batch)?;

        // scoped timer updates on drop
        let mut timer = self.time.timer();

        self.physical_expr
            .evaluate(&batch)
            .and_then(|v| v.into_array(batch.num_rows()))
            .and_then(|array| {
                let bool_arr = as_boolean_array(&array)?.clone();
                let num_matched = bool_arr.true_count();
                let num_pruned = bool_arr.len() - num_matched;
                self.rows_pruned.add(num_pruned);
                self.rows_matched.add(num_matched);
                timer.stop();
                Ok(bool_arr)
            })
            .map_err(|e| {
                ArrowError::ComputeError(format!("Error evaluating filter predicate: {e:?}"))
            })
    }
}

/// A candidate expression for creating a `RowFilter`.
///
/// Each candidate contains the expression as well as data to estimate the cost
/// of evaluating the resulting expression.
///
/// See the module level documentation for more information.
pub(crate) struct FilterCandidate {
    expr: Arc<dyn PhysicalExpr>,
    required_bytes: usize,
    can_use_index: bool,
    projection: Vec<usize>,
}

/// Helper to build a `FilterCandidate`.
///
/// This will do several things
/// 1. Determine the columns required to evaluate the expression
/// 2. Calculate data required to estimate the cost of evaluating the filter
/// 3. Rewrite column expressions in the predicate which reference columns not
///    in the particular file schema.
///
/// # Schema Rewrite
///
/// When parquet files are read in the context of "schema evolution" there are
/// potentially wo schemas:
///
/// 1. The table schema (the columns of the table that the parquet file is part of)
/// 2. The file schema (the columns actually in the parquet file)
///
/// There are times when the table schema contains columns that are not in the
/// file schema, such as when new columns have been added in new parquet files
/// but old files do not have the columns.
///
/// When a file is missing a column from the table schema, the value of the
/// missing column is filled in with `NULL`  via a `SchemaAdapter`.
///
/// When a predicate is pushed down to the parquet reader, the predicate is
/// evaluated in the context of the file schema. If the predicate references a
/// column that is in the table schema but not in the file schema, the column
/// reference must be rewritten to a literal expression that represents the
/// `NULL` value that would be produced by the `SchemaAdapter`.
///
/// For example, if:
/// * The table schema is `id, name, address`
/// * The file schema is  `id, name` (missing the `address` column)
/// * predicate is `address = 'foo'`
///
/// When evaluating the predicate as a filter on the parquet file, the predicate
/// must be rewritten to `NULL = 'foo'` as the `address` column will be filled
/// in with `NULL` values during the rest of the evaluation.
struct FilterCandidateBuilder<'a> {
    expr: Arc<dyn PhysicalExpr>,
    /// The schema of this parquet file
    file_schema: &'a Schema,
    /// The schema of the table (merged schema) -- columns may be in different
    /// order than in the file and have columns that are not in the file schema
    table_schema: &'a Schema,
}

impl<'a> FilterCandidateBuilder<'a> {
    pub fn new(
        expr: Arc<dyn PhysicalExpr>,
        file_schema: &'a Schema,
        table_schema: &'a Schema,
    ) -> Self {
        Self {
            expr,
            file_schema,
            table_schema,
        }
    }

    /// Attempt to build a `FilterCandidate` from the expression
    ///
    /// # Return values
    ///
    /// * `Ok(Some(candidate))` if the expression can be used as an ArrowFilter
    /// * `Ok(None)` if the expression cannot be used as an ArrowFilter
    /// * `Err(e)` if an error occurs while building the candidate
    pub fn build(self, metadata: &ParquetMetaData) -> Result<Option<FilterCandidate>> {
        let Some((required_indices, rewritten_expr)) =
            pushdown_columns(self.expr, self.file_schema, self.table_schema)?
        else {
            return Ok(None);
        };

        let required_bytes = size_of_columns(&required_indices, metadata)?;
        let can_use_index = columns_sorted(&required_indices, metadata)?;

        Ok(Some(FilterCandidate {
            expr: rewritten_expr,
            required_bytes,
            can_use_index,
            projection: required_indices.into_iter().collect(),
        }))
    }
}

// a struct that implements TreeNodeRewriter to traverse a PhysicalExpr tree structure to determine
// if any column references in the expression would prevent it from being predicate-pushed-down.
// if non_primitive_columns || projected_columns, it can't be pushed down.
// can't be reused between calls to `rewrite`; each construction must be used only once.
struct PushdownChecker<'schema> {
    /// Does the expression require any non-primitive columns (like structs)?
    non_primitive_columns: bool,
    /// Does the expression reference any columns that are in the table
    /// schema but not in the file schema?
    projected_columns: bool,
    // the indices of all the columns found within the given expression which exist inside the given
    // [`file_schema`]
    required_column_indices: BTreeSet<usize>,
    file_schema: &'schema Schema,
    table_schema: &'schema Schema,
}

impl<'schema> PushdownChecker<'schema> {
    fn new(file_schema: &'schema Schema, table_schema: &'schema Schema) -> Self {
        Self {
            non_primitive_columns: false,
            projected_columns: false,
            required_column_indices: BTreeSet::default(),
            file_schema,
            table_schema,
        }
    }

    fn check_single_column(&mut self, column_name: &str) -> Option<TreeNodeRecursion> {
        if let Ok(idx) = self.file_schema.index_of(column_name) {
            self.required_column_indices.insert(idx);

            if DataType::is_nested(self.file_schema.field(idx).data_type()) {
                self.non_primitive_columns = true;
                return Some(TreeNodeRecursion::Jump);
            }
        } else if self.table_schema.index_of(column_name).is_err() {
            // If the column does not exist in the (un-projected) table schema then
            // it must be a projected column.
            self.projected_columns = true;
            return Some(TreeNodeRecursion::Jump);
        }

        None
    }

    #[inline]
    fn prevents_pushdown(&self) -> bool {
        self.non_primitive_columns || self.projected_columns
    }
}

impl TreeNodeRewriter for PushdownChecker<'_> {
    type Node = Arc<dyn PhysicalExpr>;

    fn f_down(
        &mut self,
        node: Arc<dyn PhysicalExpr>,
    ) -> Result<Transformed<Arc<dyn PhysicalExpr>>> {
        if let Some(column) = node.as_any().downcast_ref::<Column>() {
            if let Some(recursion) = self.check_single_column(column.name()) {
                return Ok(Transformed::new(node, false, recursion));
            }
        }

        Ok(Transformed::no(node))
    }

    /// After visiting all children, rewrite column references to nulls if
    /// they are not in the file schema.
    /// We do this because they won't be relevant if they're not in the file schema, since that's
    /// the only thing we're dealing with here as this is only used for the parquet pushdown during
    /// scanning
    fn f_up(&mut self, expr: Arc<dyn PhysicalExpr>) -> Result<Transformed<Arc<dyn PhysicalExpr>>> {
        if let Some(column) = expr.as_any().downcast_ref::<Column>() {
            // if the expression is a column, is it in the file schema?
            if self.file_schema.field_with_name(column.name()).is_err() {
                return self
                    .table_schema
                    .field_with_name(column.name())
                    .and_then(|field| {
                        // Replace the column reference with a NULL (using the type from the table schema)
                        // e.g. `column = 'foo'` is rewritten be transformed to `NULL = 'foo'`
                        //
                        // See comments on `FilterCandidateBuilder` for more information
                        let null_value = ScalarValue::try_from(field.data_type())?;
                        Ok(Transformed::yes(Arc::new(Literal::new(null_value)) as _))
                    })
                    // If the column is not in the table schema, should throw the error
                    .map_err(|e| arrow_datafusion_err!(e));
            }
        }

        Ok(Transformed::no(expr))
    }
}

type ProjectionAndExpr = (BTreeSet<usize>, Arc<dyn PhysicalExpr>);

// Checks if a given expression can be pushed down into `ParquetExec` as opposed to being evaluated
// post-parquet-scan in a `FilterExec`. If it can be pushed down, this returns returns all the
// columns in the given expression so that they can be used in the parquet scanning, along with the
// expression rewritten as defined in [`PushdownChecker::f_up`]
fn pushdown_columns(
    expr: Arc<dyn PhysicalExpr>,
    file_schema: &Schema,
    table_schema: &Schema,
) -> Result<Option<ProjectionAndExpr>> {
    let mut checker = PushdownChecker::new(file_schema, table_schema);

    let expr = expr.rewrite(&mut checker).data()?;

    Ok((!checker.prevents_pushdown()).then_some((checker.required_column_indices, expr)))
}

/// creates a PushdownChecker for a single use to check a given column with the given schemes. Used
/// to check preemptively if a column name would prevent pushdowning.
/// effectively does the inverse of [`pushdown_columns`] does, but with a single given column
/// (instead of traversing the entire tree to determine this)
fn would_column_prevent_pushdown(
    column_name: &str,
    file_schema: &Schema,
    table_schema: &Schema,
) -> bool {
    let mut checker = PushdownChecker::new(file_schema, table_schema);

    // the return of this is only used for [`PushdownChecker::f_down()`], so we can safely ignore
    // it here. I'm just verifying we know the return type of this so nobody accidentally changes
    // the return type of this fn and it gets implicitly ignored here.
    let _: Option<TreeNodeRecursion> = checker.check_single_column(column_name);

    // and then return a value based on the state of the checker
    checker.prevents_pushdown()
}

/// Recurses through expr as a tree, finds all `column`s, and checks if any of them would prevent
/// this expression from being predicate pushed down. If any of them would, this returns false.
/// Otherwise, true.
pub fn can_expr_be_pushed_down_with_schemas(
    expr: &datafusion::logical_expr::Expr,
    file_schema: &Schema,
    table_schema: &Schema,
) -> bool {
    let mut can_be_pushed = true;
    expr.apply(|expr| match expr {
        datafusion::logical_expr::Expr::Column(column) => {
            can_be_pushed &=
                !would_column_prevent_pushdown(column.name(), file_schema, table_schema);
            Ok(if can_be_pushed {
                TreeNodeRecursion::Jump
            } else {
                TreeNodeRecursion::Stop
            })
        }
        _ => Ok(TreeNodeRecursion::Continue),
    })
    .unwrap(); // we never return an Err, so we can safely unwrap this
    can_be_pushed
}

/// Computes the projection required to go from the file's schema order to the projected
/// order expected by this filter
///
/// Effectively this computes the rank of each element in `src`
fn remap_projection(src: &[usize]) -> Vec<usize> {
    let len = src.len();

    // Compute the column mapping from projected order to file order
    // i.e. the indices required to sort projected schema into the file schema
    //
    // e.g. projection: [5, 9, 0] -> [2, 0, 1]
    let mut sorted_indexes: Vec<_> = (0..len).collect();
    sorted_indexes.sort_unstable_by_key(|x| src[*x]);

    // Compute the mapping from schema order to projected order
    // i.e. the indices required to sort file schema into the projected schema
    //
    // Above we computed the order of the projected schema according to the file
    // schema, and so we can use this as the comparator
    //
    // e.g. sorted_indexes [2, 0, 1] -> [1, 2, 0]
    let mut projection: Vec<_> = (0..len).collect();
    projection.sort_unstable_by_key(|x| sorted_indexes[*x]);
    projection
}

/// Calculate the total compressed size of all `Column`'s required for
/// predicate `Expr`.
///
/// This value represents the total amount of IO required to evaluate the
/// predicate.
fn size_of_columns(columns: &BTreeSet<usize>, metadata: &ParquetMetaData) -> Result<usize> {
    let mut total_size = 0;
    let row_groups = metadata.row_groups();
    for idx in columns {
        for rg in row_groups.iter() {
            total_size += rg.column(*idx).compressed_size() as usize;
        }
    }

    Ok(total_size)
}

/// For a given set of `Column`s required for predicate `Expr` determine whether
/// all columns are sorted.
///
/// Sorted columns may be queried more efficiently in the presence of
/// a PageIndex.
fn columns_sorted(_columns: &BTreeSet<usize>, _metadata: &ParquetMetaData) -> Result<bool> {
    // TODO How do we know this?
    Ok(false)
}

/// Build a [`RowFilter`] from the given predicate `Expr` if possible
///
/// # returns
/// * `Ok(Some(row_filter))` if the expression can be used as RowFilter
/// * `Ok(None)` if the expression cannot be used as an RowFilter
/// * `Err(e)` if an error occurs while building the filter
///
/// Note that the returned `RowFilter` may not contains all conjuncts in the
/// original expression. This is because some conjuncts may not be able to be
/// evaluated as an `ArrowPredicate` and will be ignored.
///
/// For example, if the expression is `a = 1 AND b = 2 AND c = 3` and `b = 2`
/// can not be evaluated for some reason, the returned `RowFilter` will contain
/// `a = 1` and `c = 3`.
pub fn build_row_filter(
    expr: &Arc<dyn PhysicalExpr>,
    file_schema: &Schema,
    table_schema: &Schema,
    metadata: &ParquetMetaData,
    reorder_predicates: bool,
    file_metrics: &ParquetFileMetrics,
) -> Result<Option<LiquidRowFilter>> {
    let rows_pruned = &file_metrics.pushdown_rows_pruned;
    let rows_matched = &file_metrics.pushdown_rows_matched;
    let time = &file_metrics.row_pushdown_eval_time;

    // Split into conjuncts:
    // `a = 1 AND b = 2 AND c = 3` -> [`a = 1`, `b = 2`, `c = 3`]
    let predicates = split_conjunction(expr);

    // Determine which conjuncts can be evaluated as ArrowPredicates, if any
    let mut candidates: Vec<FilterCandidate> = predicates
        .into_iter()
        .map(|expr| {
            FilterCandidateBuilder::new(Arc::clone(expr), file_schema, table_schema).build(metadata)
        })
        .collect::<Result<Vec<_>, _>>()?
        .into_iter()
        .flatten()
        .collect();

    // no candidates
    if candidates.is_empty() {
        return Ok(None);
    }

    if reorder_predicates {
        candidates.sort_unstable_by(|c1, c2| match c1.can_use_index.cmp(&c2.can_use_index) {
            Ordering::Equal => c1.required_bytes.cmp(&c2.required_bytes),
            ord => ord,
        });
    }

    candidates.sort_unstable_by(|c1, c2| {
        let p1 = get_priority(&c1.expr);
        let p2 = get_priority(&c2.expr);
        p1.cmp(&p2)
    });

    candidates
        .into_iter()
        .map(|candidate| {
            DatafusionArrowPredicate::try_new(
                candidate,
                file_schema,
                metadata,
                rows_pruned.clone(),
                rows_matched.clone(),
                time.clone(),
            )
            .map(|pred| Box::new(pred) as _)
        })
        .collect::<Result<Vec<_>, _>>()
        .map(|filters| Some(LiquidRowFilter::new(filters)))
}

fn get_priority(expr: &Arc<dyn PhysicalExpr>) -> u8 {
    if let Some(binary) = expr.as_any().downcast_ref::<BinaryExpr>() {
        match binary.op() {
            Operator::Eq | Operator::NotEq => 0, // Highest priority
            Operator::LikeMatch | Operator::ILikeMatch => 1,
            Operator::NotLikeMatch | Operator::NotILikeMatch => 2,
            Operator::Lt | Operator::LtEq | Operator::Gt | Operator::GtEq => 3,
            _ => 4,
        }
    } else if expr.as_any().downcast_ref::<LikeExpr>().is_some() {
        1 // LIKE expressions
    } else {
        5 // All other expression types
    }
}
