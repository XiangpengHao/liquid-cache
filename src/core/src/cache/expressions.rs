//! Definitions for cache-aware expressions that can be applied when materializing arrays.

use std::str::FromStr;
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
///
/// A `CacheExpression` is a *squeeze hint*: it tells the cache how a column is
/// consumed by a query so that, under memory pressure, the cache can keep only
/// the part of the column the query actually needs (e.g. a single date
/// component, or a handful of variant paths) instead of evicting it wholesale.
#[derive(Debug, Clone, PartialEq, Eq, Hash, serde::Serialize)]
pub enum CacheExpression {
    /// Extract one or more components (YEAR/MONTH/DAY/DOW) from a `Date32` or
    /// timestamp column.
    ///
    /// The set is non-empty and de-duplicated. A column used as
    /// `EXTRACT(MONTH FROM d)` *and* `EXTRACT(DAY FROM d)` carries both
    /// components so neither is silently lost.
    ExtractDate32 {
        /// Components requested by the query, sorted and de-duplicated.
        fields: Arc<[Date32Field]>,
    },
    /// Extract a field from a variant column via `variant_get`.
    VariantGet {
        /// The set of dotted paths requested by the query.
        requests: Arc<[VariantRequest]>,
    },
    /// A column used for predicate evaluation.
    PredicateColumn,
    /// A column used primarily for substring search (LIKE '%foo%').
    SubstringSearch,
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
            Self::ExtractDate32 { fields } => {
                write!(f, "ExtractDate32{:?}", fields)
            }
            Self::PredicateColumn => {
                write!(f, "PredicateColumn")
            }
            Self::SubstringSearch => {
                write!(f, "SubstringSearch")
            }
        }
    }
}

impl CacheExpression {
    /// Build an extract expression for a single `Date32`/timestamp component.
    pub fn extract_date32(field: Date32Field) -> Self {
        Self::ExtractDate32 {
            fields: Arc::from(vec![field].into_boxed_slice()),
        }
    }

    /// Build an extract expression covering multiple `Date32`/timestamp
    /// components. The components are sorted and de-duplicated; returns `None`
    /// if `fields` is empty.
    pub fn extract_date32_many<I>(fields: I) -> Option<Self>
    where
        I: IntoIterator<Item = Date32Field>,
    {
        let mut fields: Vec<Date32Field> = fields.into_iter().collect();
        sort_dedup_fields(&mut fields);
        if fields.is_empty() {
            return None;
        }
        Some(Self::ExtractDate32 {
            fields: Arc::from(fields.into_boxed_slice()),
        })
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

    /// Build a substring-search expression hint.
    pub fn substring_search() -> Self {
        Self::SubstringSearch
    }

    /// Return the requested `Date32` component when this is an extract
    /// expression for exactly one component.
    ///
    /// Multi-component extractions return `None`: there is no single-component
    /// squeezed representation that satisfies all of them, so the squeeze path
    /// keeps the column intact rather than dropping a needed component.
    pub fn as_date32_field(&self) -> Option<Date32Field> {
        match self {
            Self::ExtractDate32 { fields } if fields.len() == 1 => Some(fields[0]),
            _ => None,
        }
    }

    /// Return all requested `Date32` components when this is an extract
    /// expression.
    pub fn date32_fields(&self) -> Option<&[Date32Field]> {
        match self {
            Self::ExtractDate32 { fields } => Some(fields.as_ref()),
            _ => None,
        }
    }

    /// Return the associated variant path when this is a variant-get expression.
    pub fn variant_path(&self) -> Option<&str> {
        match self {
            Self::VariantGet { requests } => requests.first().map(|request| request.path()),
            _ => None,
        }
    }

    /// Return the associated Arrow data type when this is a variant-get expression.
    pub fn variant_data_type(&self) -> Option<&DataType> {
        match self {
            Self::VariantGet { requests } => requests.first().map(|request| request.data_type()),
            _ => None,
        }
    }

    /// Return all typed variant paths carried by this expression.
    pub fn variant_requests(&self) -> Option<&[VariantRequest]> {
        match self {
            Self::VariantGet { requests } => Some(requests.as_ref()),
            _ => None,
        }
    }

    /// Encode this expression into a compact, lossless string suitable for
    /// carrying through Arrow schema field metadata (and therefore across the
    /// `physical_plan_to_bytes` serialization boundary).
    ///
    /// The DataFusion layer derives a typed [`CacheExpression`] from the query
    /// plan, stamps it onto the scan's file schema via [`Self::to_metadata_value`],
    /// and decodes it back here via [`Self::from_metadata_value`] when the cache
    /// column is created. The round-trip is exact — including multi-component
    /// dates and multi-path variant requests, both of which the previous
    /// ad-hoc string format silently dropped.
    pub fn to_metadata_value(&self) -> String {
        let dto = CacheExprDto::from(self);
        serde_json::to_string(&dto).expect("CacheExpression DTO is always serializable")
    }

    /// Decode an expression previously produced by [`Self::to_metadata_value`].
    ///
    /// Returns `None` if the value is not a recognized encoding.
    pub fn from_metadata_value(value: &str) -> Option<Self> {
        let dto: CacheExprDto = serde_json::from_str(value).ok()?;
        Self::try_from(dto)
    }
}

fn sort_dedup_fields(fields: &mut Vec<Date32Field>) {
    fields.sort_by_key(field_order);
    fields.dedup();
}

fn field_order(field: &Date32Field) -> u8 {
    match field {
        Date32Field::Year => 0,
        Date32Field::Month => 1,
        Date32Field::Day => 2,
        Date32Field::DayOfWeek => 3,
    }
}

/// Owned, `Arc`-free mirror of [`CacheExpression`] used purely for the metadata
/// codec, so the encoding does not depend on serde's `rc` feature. Variant data
/// types are carried as their Arrow `Display` form, which round-trips through
/// `DataType::from_str`.
#[derive(serde::Serialize, serde::Deserialize)]
enum CacheExprDto {
    Date { fields: Vec<Date32Field> },
    Variant { requests: Vec<VariantReqDto> },
    Predicate,
    Substring,
}

#[derive(serde::Serialize, serde::Deserialize)]
struct VariantReqDto {
    path: String,
    data_type: String,
}

impl From<&CacheExpression> for CacheExprDto {
    fn from(expr: &CacheExpression) -> Self {
        match expr {
            CacheExpression::ExtractDate32 { fields } => CacheExprDto::Date {
                fields: fields.to_vec(),
            },
            CacheExpression::VariantGet { requests } => CacheExprDto::Variant {
                requests: requests
                    .iter()
                    .map(|request| VariantReqDto {
                        path: request.path().to_string(),
                        data_type: request.data_type().to_string(),
                    })
                    .collect(),
            },
            CacheExpression::PredicateColumn => CacheExprDto::Predicate,
            CacheExpression::SubstringSearch => CacheExprDto::Substring,
        }
    }
}

impl CacheExpression {
    fn try_from(dto: CacheExprDto) -> Option<Self> {
        match dto {
            CacheExprDto::Date { fields } => CacheExpression::extract_date32_many(fields),
            CacheExprDto::Variant { requests } => {
                let mut parsed = Vec::with_capacity(requests.len());
                for request in requests {
                    let data_type = DataType::from_str(&request.data_type).ok()?;
                    parsed.push(VariantRequest::new(request.path, data_type));
                }
                if parsed.is_empty() {
                    return None;
                }
                Some(CacheExpression::VariantGet {
                    requests: Arc::from(parsed.into_boxed_slice()),
                })
            }
            CacheExprDto::Predicate => Some(CacheExpression::PredicateColumn),
            CacheExprDto::Substring => Some(CacheExpression::SubstringSearch),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn date_metadata_roundtrip_multi_component() {
        let expr = CacheExpression::extract_date32_many([
            Date32Field::Day,
            Date32Field::Month,
            // duplicate is collapsed
            Date32Field::Day,
        ])
        .unwrap();
        let encoded = expr.to_metadata_value();
        let decoded = CacheExpression::from_metadata_value(&encoded).unwrap();
        assert_eq!(decoded, expr);
        // Multi-component extractions do not collapse to a single squeezable field.
        assert_eq!(decoded.as_date32_field(), None);
        assert_eq!(
            decoded.date32_fields().unwrap(),
            &[Date32Field::Month, Date32Field::Day]
        );
    }

    #[test]
    fn date_metadata_roundtrip_single_component() {
        let expr = CacheExpression::extract_date32(Date32Field::Year);
        let decoded = CacheExpression::from_metadata_value(&expr.to_metadata_value()).unwrap();
        assert_eq!(decoded, expr);
        assert_eq!(decoded.as_date32_field(), Some(Date32Field::Year));
    }

    #[test]
    fn variant_metadata_roundtrip_multi_path() {
        let expr = CacheExpression::variant_get_many([
            ("name", DataType::Utf8),
            ("age", DataType::Int64),
            // paths can contain delimiter-like characters; JSON encoding is safe
            ("address.zip,code", DataType::Utf8),
        ]);
        let decoded = CacheExpression::from_metadata_value(&expr.to_metadata_value()).unwrap();
        assert_eq!(decoded, expr);
    }

    #[test]
    fn predicate_and_substring_roundtrip() {
        for expr in [
            CacheExpression::PredicateColumn,
            CacheExpression::substring_search(),
        ] {
            let decoded = CacheExpression::from_metadata_value(&expr.to_metadata_value()).unwrap();
            assert_eq!(decoded, expr);
        }
    }

    #[test]
    fn rejects_unknown_metadata() {
        assert_eq!(CacheExpression::from_metadata_value("not json"), None);
        assert_eq!(CacheExpression::from_metadata_value("{}"), None);
    }
}
