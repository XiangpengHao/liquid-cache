//! Definitions for cache-aware expressions that can be applied when materializing arrays.

use std::sync::Arc;

use arrow_schema::DataType;

use crate::liquid_array::Date32Field;

/// A typed variant path requested by a query.
#[derive(Debug, Clone, PartialEq, Eq, Hash, serde::Serialize)]
pub struct VariantRequest {
    path: Arc<str>,
    data_type: Arc<DataType>,
}

impl VariantRequest {
    /// Create a new typed path request.
    pub fn new(path: impl Into<Arc<str>>, data_type: DataType) -> Self {
        Self {
            path: path.into(),
            data_type: Arc::new(data_type),
        }
    }

    /// Path string for this request.
    pub fn path(&self) -> &str {
        self.path.as_ref()
    }

    /// Requested Arrow data type for this path.
    pub fn data_type(&self) -> &DataType {
        self.data_type.as_ref()
    }
}

/// Experimental expression descriptor for cache lookups.
#[derive(Debug, Clone, PartialEq, Eq, Hash, serde::Serialize)]
pub enum CacheExpression {
    /// Extract a specific component (YEAR/MONTH/DAY) from a `Date32` column.
    ExtractDate32 {
        /// Component to extract (YEAR/MONTH/DAY).
        field: Date32Field,
    },
    /// Extract a field from a variant column via `variant_get`.
    VariantGet {
        /// The set of dotted paths requested by the query.
        requests: Arc<[VariantRequest]>,
    },
    /// A column used for predicate evaluation.
    PredicateColumn,
}

impl std::fmt::Display for CacheExpression {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::VariantGet { requests } => {
                write!(f, "VariantGet[")?;
                let requests = requests
                    .iter()
                    .map(|request| format!("{}:{}", request.path(), request.data_type()))
                    .collect::<Vec<_>>();
                write!(f, "{}", requests.join(","))?;
                write!(f, "]")
            }
            Self::ExtractDate32 { field } => {
                write!(f, "ExtractDate32:{:?}", field)
            }
            Self::PredicateColumn => {
                write!(f, "PredicateColumn")
            }
        }
    }
}

impl CacheExpression {
    /// Build an extract expression for a `Date32` column.
    pub fn extract_date32(field: Date32Field) -> Self {
        Self::ExtractDate32 { field }
    }

    /// Build a variant-get expression for the provided dotted path.
    pub fn variant_get(path: impl Into<Arc<str>>, data_type: DataType) -> Self {
        Self::VariantGet {
            requests: Arc::from(vec![VariantRequest::new(path, data_type)].into_boxed_slice()),
        }
    }

    /// Build a variant-get expression covering multiple paths.
    pub fn variant_get_many<I, S>(requests: I) -> Self
    where
        I: IntoIterator<Item = (S, DataType)>,
        S: Into<Arc<str>>,
    {
        let requests: Vec<VariantRequest> = requests
            .into_iter()
            .map(|(path, data_type)| VariantRequest::new(path.into(), data_type))
            .collect();
        assert!(
            !requests.is_empty(),
            "variant_get_many requires at least one path"
        );
        Self::VariantGet {
            requests: Arc::from(requests.into_boxed_slice()),
        }
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
            Self::VariantGet { .. } | Self::PredicateColumn => None,
        }
    }

    /// Return the associated variant path when this is a variant-get expression.
    pub fn variant_path(&self) -> Option<&str> {
        match self {
            Self::VariantGet { requests } => requests.first().map(|request| request.path()),
            Self::ExtractDate32 { .. } | Self::PredicateColumn => None,
        }
    }

    /// Return the associated Arrow data type when this is a variant-get expression.
    pub fn variant_data_type(&self) -> Option<&DataType> {
        match self {
            Self::VariantGet { requests } => requests.first().map(|request| request.data_type()),
            Self::ExtractDate32 { .. } | Self::PredicateColumn => None,
        }
    }

    /// Return all typed variant paths carried by this expression.
    pub fn variant_requests(&self) -> Option<&[VariantRequest]> {
        match self {
            Self::VariantGet { requests } => Some(requests.as_ref()),
            Self::ExtractDate32 { .. } | Self::PredicateColumn => None,
        }
    }
}
