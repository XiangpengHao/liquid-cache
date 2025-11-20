//! Definitions for cache-aware expressions that can be applied when materializing arrays.

use std::collections::HashMap;
use std::num::NonZeroU16;
use std::sync::{Arc, RwLock};

use arrow::compute::DatePart;
use std::hash::{Hash, Hasher};

/// A set of date parts that can be extracted from a `Date32` array.
/// Does not include YearMonthDay because this should never be extracted as components.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DatePartSet {
    /// The year part.
    Year,
    /// The month part.
    Month,
    /// The day part.
    Day,
    /// The year and month parts.
    YearMonth,
    /// The year and day parts.
    YearDay,
    /// The month and day parts.
    MonthDay,
}

impl DatePartSet {
    /// Iterate over the date parts in the set.
    pub fn iter(&self) -> DatePartSetIter {
        let mut parts = [DatePart::Year; 3];
        let mut len = 0;
        let push = |part: DatePart, parts: &mut [DatePart; 3], len: &mut usize| {
            parts[*len] = part;
            *len += 1;
        };

        match self {
            Self::Year => push(DatePart::Year, &mut parts, &mut len),
            Self::Month => push(DatePart::Month, &mut parts, &mut len),
            Self::Day => push(DatePart::Day, &mut parts, &mut len),
            Self::YearMonth => {
                push(DatePart::Year, &mut parts, &mut len);
                push(DatePart::Month, &mut parts, &mut len);
            }
            Self::YearDay => {
                push(DatePart::Year, &mut parts, &mut len);
                push(DatePart::Day, &mut parts, &mut len);
            }
            Self::MonthDay => {
                push(DatePart::Month, &mut parts, &mut len);
                push(DatePart::Day, &mut parts, &mut len);
            }
        }

        DatePartSetIter { parts, len, idx: 0 }
    }

    /// Check if the set contains the given date part.
    pub fn contains(&self, part: DatePart) -> bool {
        match self {
            Self::Year => part == DatePart::Year,
            Self::Month => part == DatePart::Month,
            Self::Day => part == DatePart::Day,
            Self::YearMonth => part == DatePart::Year || part == DatePart::Month,
            Self::YearDay => part == DatePart::Year || part == DatePart::Day,
            Self::MonthDay => part == DatePart::Month || part == DatePart::Day,
        }
    }

    /// Check if the set contains the year part.
    pub fn has_year(&self) -> bool {
        matches!(self, Self::Year | Self::YearMonth | Self::YearDay)
    }

    /// Check if the set contains the month part.
    pub fn has_month(&self) -> bool {
        matches!(self, Self::Month | Self::YearMonth | Self::MonthDay)
    }

    /// Check if the set contains the day part.
    pub fn has_day(&self) -> bool {
        matches!(self, Self::Day | Self::YearDay | Self::MonthDay)
    }

    /// Get the length of the set.
    pub fn len(&self) -> usize {
        match self {
            Self::Year | Self::Month | Self::Day => 1,
            _ => 2,
        }
    }

    /// Check if the set is empty.
    pub fn is_empty(&self) -> bool {
        false
    }

    /// Check if the set is a superset of the other set.
    pub fn is_superset_of(&self, other: DatePartSet) -> bool {
        other.iter().all(|part| self.contains(part))
    }
}

pub struct DatePartSetIter {
    parts: [DatePart; 3],
    len: usize,
    idx: usize,
}

impl Iterator for DatePartSetIter {
    type Item = DatePart;

    fn next(&mut self) -> Option<Self::Item> {
        if self.idx >= self.len {
            return None;
        }
        let part = self.parts[self.idx];
        self.idx += 1;
        Some(part)
    }
}

impl From<DatePart> for DatePartSet {
    fn from(value: DatePart) -> Self {
        match value {
            DatePart::Year => DatePartSet::Year,
            DatePart::Month => DatePartSet::Month,
            DatePart::Day => DatePartSet::Day,
            _ => panic!("Unsupported DatePart variant for caching: {:?}", value),
        }
    }
}

/// Experimental expression descriptor for cache lookups.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CacheExpression {
    /// Extract a specific component (YEAR/MONTH/DAY) from a `Date32` column.
    ExtractDate32 {
        /// Components to extract (YEAR/MONTH/DAY).
        field: DatePartSet,
    },
    /// Extract a field from a variant column via `variant_get`.
    VariantGet {
        /// The dotted path requested by the query.
        path: Arc<str>,
    },
}

impl Hash for CacheExpression {
    fn hash<H: Hasher>(&self, state: &mut H) {
        match self {
            Self::ExtractDate32 { field } => {
                0u8.hash(state);
                std::mem::discriminant(field).hash(state);
            }
            Self::VariantGet { path } => {
                1u8.hash(state);
                path.hash(state);
            }
        }
    }
}

impl CacheExpression {
    /// Build an extract expression for a `Date32` column.
    pub fn extract_date32(field: impl Into<DatePartSet>) -> Self {
        Self::ExtractDate32 {
            field: field.into(),
        }
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
            "YEAR" => DatePartSet::Year,
            "MONTH" => DatePartSet::Month,
            "DAY" => DatePartSet::Day,
            "YEAR,MONTH" => DatePartSet::YearMonth,
            "YEAR,DAY" => DatePartSet::YearDay,
            "MONTH,DAY" => DatePartSet::MonthDay,
            _ => return None,
        };
        Some(Self::ExtractDate32 { field })
    }

    /// Return the requested `Date32` component when this is an extract expression.
    pub fn as_date32_field(&self) -> Option<DatePartSet> {
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
