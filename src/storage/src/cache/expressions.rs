//! Definitions for cache-aware expressions that can be applied when materializing arrays.

use std::sync::Arc;

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
