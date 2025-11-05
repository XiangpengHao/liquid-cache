//! Definitions for cache-aware expressions that can be applied when materializing arrays.

use crate::liquid_array::Date32Field;

/// Compact identifier for cache expressions, encoded into a `u16`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CacheExpressionId(pub u16);

impl CacheExpressionId {
    const KIND_SHIFT: u16 = 8;
    const KIND_MASK: u16 = 0xFF;
    const VARIANT_MASK: u16 = 0xFF;

    const KIND_EXTRACT_DATE32: u16 = 1;

    /// Build an identifier for extracting a specific `Date32` component.
    pub const fn from_date32_field(field: Date32Field) -> Self {
        let variant = match field {
            Date32Field::Year => 0,
            Date32Field::Month => 1,
            Date32Field::Day => 2,
        };
        let raw = (Self::KIND_EXTRACT_DATE32 << Self::KIND_SHIFT) | variant;
        Self(raw)
    }

    /// Create an identifier directly from its raw `u16` value.
    pub const fn from_raw(raw: u16) -> Self {
        Self(raw)
    }

    /// Return the raw `u16` backing this identifier.
    pub const fn raw(self) -> u16 {
        self.0
    }

    const fn kind(self) -> u16 {
        (self.0 >> Self::KIND_SHIFT) & Self::KIND_MASK
    }

    const fn variant(self) -> u16 {
        self.0 & Self::VARIANT_MASK
    }

    /// Interpret this identifier as a `Date32` extract expression, if applicable.
    pub const fn as_date32_field(self) -> Option<Date32Field> {
        if self.kind() != Self::KIND_EXTRACT_DATE32 {
            return None;
        }
        match self.variant() {
            0 => Some(Date32Field::Year),
            1 => Some(Date32Field::Month),
            2 => Some(Date32Field::Day),
            _ => None,
        }
    }
}

/// Experimental expression descriptor for cache lookups.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CacheExpression {
    /// Extract a specific component (YEAR/MONTH/DAY) from a `Date32` column.
    ExtractDate32 {
        /// Component to extract (YEAR/MONTH/DAY).
        field: Date32Field,
    },
}

impl CacheExpression {
    /// Build an extract expression for a `Date32` column.
    pub fn extract_date32(field: Date32Field) -> Self {
        Self::ExtractDate32 { field }
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
        }
    }

    /// Encode this expression into a compact identifier.
    pub fn hint_id(&self) -> CacheExpressionId {
        match self {
            Self::ExtractDate32 { field } => CacheExpressionId::from_date32_field(*field),
        }
    }
}
