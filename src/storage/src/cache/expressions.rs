//! Definitions for cache-aware expressions that can be applied when materializing arrays.

use std::collections::HashMap;
use std::num::NonZeroU16;
use std::sync::{Arc, RwLock};

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

/// Identifier assigned to a registered cache expression.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ExpressionId(NonZeroU16);

impl ExpressionId {
    /// Try to build an [`ExpressionId`] from a raw 16-bit value.
    pub fn from_raw(raw: u16) -> Option<Self> {
        NonZeroU16::new(raw).map(Self)
    }

    /// Numerical representation of the identifier.
    pub fn as_raw(self) -> u16 {
        self.0.get()
    }

    /// Zero-based index into the registry backing-store.
    fn index(self) -> usize {
        (self.0.get() as usize) - 1
    }
}

const MAX_REGISTERED_EXPRESSIONS: usize = u16::MAX as usize;

#[derive(Debug)]
struct ExpressionRegistryInner {
    expressions: Vec<Arc<CacheExpression>>,
    lookup: HashMap<CacheExpression, ExpressionId>,
}

impl ExpressionRegistryInner {
    fn new() -> Self {
        Self {
            expressions: Vec::new(),
            lookup: HashMap::new(),
        }
    }
}

/// Registry that assigns stable identifiers to [`CacheExpression`] values.
#[derive(Debug)]
pub struct ExpressionRegistry {
    inner: RwLock<ExpressionRegistryInner>,
}

impl ExpressionRegistry {
    /// Create an empty registry.
    pub fn new() -> Self {
        Self {
            inner: RwLock::new(ExpressionRegistryInner::new()),
        }
    }

    /// Register the provided expression and return its identifier.
    ///
    /// Returns `None` if the registry has reached capacity.
    pub fn register(&self, expression: CacheExpression) -> Option<ExpressionId> {
        {
            let guard = self.inner.read().unwrap();
            if let Some(id) = guard.lookup.get(&expression) {
                return Some(*id);
            }
        }

        let mut guard = self.inner.write().unwrap();
        if let Some(id) = guard.lookup.get(&expression) {
            return Some(*id);
        }

        if guard.expressions.len() == MAX_REGISTERED_EXPRESSIONS {
            return None;
        }

        let next_index = guard.expressions.len() + 1;
        let raw = u16::try_from(next_index).ok()?;
        let id = ExpressionId::from_raw(raw)?;
        guard.lookup.insert(expression.clone(), id);
        guard.expressions.push(Arc::new(expression));
        Some(id)
    }

    /// Look up the identifier for a previously registered expression.
    pub fn id(&self, expression: &CacheExpression) -> Option<ExpressionId> {
        let guard = self.inner.read().unwrap();
        guard.lookup.get(expression).copied()
    }

    /// Resolve the identifier into the stored expression.
    pub fn get(&self, id: ExpressionId) -> Option<Arc<CacheExpression>> {
        let guard = self.inner.read().unwrap();
        guard.expressions.get(id.index()).cloned()
    }
}

impl Default for ExpressionRegistry {
    fn default() -> Self {
        Self::new()
    }
}
