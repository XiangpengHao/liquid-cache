//! Definitions for cache-aware expressions that can be applied when materializing arrays.

use std::cmp::Ordering;
use std::collections::VecDeque;
use std::sync::{Arc, RwLock};

use ahash::AHashMap;

use crate::cache::utils::EntryID;
use crate::liquid_array::Date32Field;

/// Experimental expression descriptor for cache lookups.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum CacheExpression {
    /// Extract a specific component (YEAR/MONTH/DAY) from a `Date32` column.
    ExtractDate32 {
        /// Component to extract (YEAR/MONTH/DAY).
        field: Date32Field,
    },
    /// Extract a field from a variant column via `variant_get`.
    VariantGet {
        /// The dotted path requested by the query.
        path: Arc<str>,
    },
}

impl CacheExpression {
    /// Build an extract expression for a `Date32` column.
    pub fn extract_date32(field: Date32Field) -> Self {
        Self::ExtractDate32 { field }
    }

    /// Build a variant-get expression for the provided dotted path.
    pub fn variant_get(path: impl Into<Arc<str>>) -> Self {
        Self::VariantGet { path: path.into() }
    }

    /// Attempt to parse a metadata value (e.g. `"YEAR"`) into an expression.
    ///
    /// The value is compared case-insensitively against supported components.
    pub fn try_from_date_part_str(value: &str) -> Option<Self> {
        let upper = value.to_ascii_uppercase();
        let field = match upper.as_str() {
            "YEAR" => Date32Field::Year,
            "MONTH" => Date32Field::Month,
            "DAY" => Date32Field::Day,
            _ => return None,
        };
        Some(Self::ExtractDate32 { field })
    }

    /// Return the requested `Date32` component when this is an extract expression.
    pub fn as_date32_field(&self) -> Option<Date32Field> {
        match self {
            Self::ExtractDate32 { field } => Some(*field),
            Self::VariantGet { .. } => None,
        }
    }

    /// Return the associated variant path when this is a variant-get expression.
    pub fn variant_path(&self) -> Option<&str> {
        match self {
            Self::VariantGet { path } => Some(path.as_ref()),
            Self::ExtractDate32 { .. } => None,
        }
    }
}

const COLUMN_EXPRESSION_HISTORY: usize = 16;
const BATCH_ID_MASK: usize = 0xFFFF;

#[derive(Debug, Default, Clone)]
struct ColumnExpressionTracker {
    history: VecDeque<Arc<CacheExpression>>,
}

impl ColumnExpressionTracker {
    fn record(&mut self, expression: Arc<CacheExpression>) {
        if self.history.len() == COLUMN_EXPRESSION_HISTORY {
            self.history.pop_front();
        }
        self.history.push_back(expression);
    }

    fn majority(&self) -> Option<Arc<CacheExpression>> {
        let mut counts: AHashMap<Arc<CacheExpression>, (usize, usize)> = AHashMap::new();
        for (idx, expr) in self.history.iter().enumerate() {
            let entry = counts.entry(expr.clone()).or_insert((0, idx));
            entry.0 += 1;
            entry.1 = idx;
        }

        counts
            .into_iter()
            .max_by(|a, b| match a.1.0.cmp(&b.1.0) {
                Ordering::Less => Ordering::Less,
                Ordering::Greater => Ordering::Greater,
                Ordering::Equal => a.1.1.cmp(&b.1.1),
            })
            .map(|(expr, _)| expr)
    }
}

/// Identifies a column (file, row group, column triple) without the batch component.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ColumnID(usize);

impl ColumnID {
    /// Create a ColumnID from file, row group, and column identifiers.
    pub fn new(file_id: u64, row_group_id: u64, column_id: u64) -> Self {
        Self((file_id as usize) << 48 | (row_group_id as usize) << 32 | (column_id as usize) << 16)
    }

    /// Create a ColumnID from an EntryID by masking out the batch component.
    pub fn from_entry_id(entry_id: EntryID) -> Self {
        Self(usize::from(entry_id) & !BATCH_ID_MASK)
    }
}

/// Registry that tracks expression usage per column.
#[derive(Debug)]
pub struct ExpressionRegistry {
    column_trackers: RwLock<AHashMap<ColumnID, ColumnExpressionTracker>>,
}

impl ExpressionRegistry {
    /// Create an empty registry.
    pub fn new() -> Self {
        Self {
            column_trackers: RwLock::new(AHashMap::new()),
        }
    }

    /// Create an Arc for the provided expression and return it.
    /// If `column_id` is provided, also records this expression for the column.
    pub fn register(
        &self,
        expression: CacheExpression,
        column_id: Option<ColumnID>,
    ) -> Arc<CacheExpression> {
        let arc = Arc::new(expression);

        if let Some(column_id) = column_id {
            let mut guard = self.column_trackers.write().unwrap();
            guard
                .entry(column_id)
                .or_insert_with(ColumnExpressionTracker::default)
                .record(arc.clone());
        }

        arc
    }

    /// Return the majority expression previously recorded for entries of the same column.
    pub fn column_majority_expression(&self, entry_id: EntryID) -> Option<Arc<CacheExpression>> {
        let column_id = ColumnID::from_entry_id(entry_id);
        let guard = self.column_trackers.read().unwrap();
        guard
            .get(&column_id)
            .and_then(ColumnExpressionTracker::majority)
    }
}

impl Default for ExpressionRegistry {
    fn default() -> Self {
        Self::new()
    }
}
