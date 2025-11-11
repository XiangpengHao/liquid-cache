use std::collections::{HashMap, VecDeque};
use std::sync::Mutex;

use crate::cache::CacheExpression;

#[cfg(test)]
pub(crate) fn gen_test_decimal_array<T: arrow::datatypes::DecimalType>(
    data_type: arrow_schema::DataType,
) -> arrow::array::PrimitiveArray<T> {
    use arrow::{
        array::{AsArray, Int64Builder},
        compute::kernels::cast,
    };

    let mut builder = Int64Builder::new();
    for i in 0..4096i64 {
        if i % 97 == 0 {
            builder.append_null();
        } else {
            let value = if i % 5 == 0 {
                i * 1000 + 123
            } else if i % 3 == 0 {
                42
            } else if i % 7 == 0 {
                i * 1_000_000 + 456789
            } else {
                i * 100 + 42
            };
            builder.append_value(value);
        }
    }
    let array = builder.finish();
    cast(&array, &data_type)
        .unwrap()
        .as_primitive::<T>()
        .clone()
}

const MAX_HINT_HISTORY: usize = 8;

#[derive(Debug)]
pub(crate) struct ExpressionHintTracker {
    history: Mutex<VecDeque<CacheExpression>>,
}

impl ExpressionHintTracker {
    pub(crate) fn new() -> Self {
        Self {
            history: Mutex::new(VecDeque::with_capacity(MAX_HINT_HISTORY)),
        }
    }

    pub(crate) fn record_expression(&self, expression: &CacheExpression) {
        let mut guard = self.history.lock().unwrap();
        if guard.len() == MAX_HINT_HISTORY {
            guard.pop_front();
        }
        guard.push_back(expression.clone());
    }

    pub(crate) fn majority_expression(&self) -> Option<CacheExpression> {
        let guard = self.history.lock().unwrap();
        if guard.is_empty() {
            return None;
        }
        let mut counts: HashMap<&CacheExpression, (usize, usize)> = HashMap::new();
        for (idx, expr) in guard.iter().enumerate() {
            let entry = counts.entry(expr).or_insert((0, idx));
            entry.0 += 1;
            entry.1 = idx;
        }
        counts
            .into_iter()
            .max_by(|a, b| a.1.0.cmp(&b.1.0).then(a.1.1.cmp(&b.1.1)))
            .map(|(expr, _)| expr.clone())
    }
}

impl Default for ExpressionHintTracker {
    fn default() -> Self {
        Self::new()
    }
}

impl Clone for ExpressionHintTracker {
    fn clone(&self) -> Self {
        let history = self.history.lock().unwrap().clone();
        Self {
            history: Mutex::new(history),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::ExpressionHintTracker;
    use crate::{cache::CacheExpression, liquid_array::Date32Field};

    #[test]
    fn majority_date32_field_returns_most_frequent_field() {
        let tracker = ExpressionHintTracker::new();

        tracker.record_expression(&CacheExpression::extract_date32(Date32Field::Year));
        tracker.record_expression(&CacheExpression::extract_date32(Date32Field::Month));
        tracker.record_expression(&CacheExpression::extract_date32(Date32Field::Year));
        tracker.record_expression(&CacheExpression::extract_date32(Date32Field::Year));

        assert_eq!(
            tracker.majority_expression(),
            Some(CacheExpression::extract_date32(Date32Field::Year))
        );
    }

    #[test]
    fn majority_date32_field_prefers_most_recent_on_tie() {
        let tracker = ExpressionHintTracker::new();

        tracker.record_expression(&CacheExpression::extract_date32(Date32Field::Year));
        tracker.record_expression(&CacheExpression::extract_date32(Date32Field::Month));
        tracker.record_expression(&CacheExpression::extract_date32(Date32Field::Year));
        tracker.record_expression(&CacheExpression::extract_date32(Date32Field::Month));

        assert_eq!(
            tracker.majority_expression(),
            Some(CacheExpression::extract_date32(Date32Field::Month))
        );
    }
}
