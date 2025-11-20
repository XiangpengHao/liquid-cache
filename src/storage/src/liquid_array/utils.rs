use std::sync::atomic::{AtomicU64, Ordering};

use crate::cache::ExpressionId;
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

#[derive(Debug)]
pub(crate) struct ExpressionHintTracker {
    slots: AtomicU64,
}

impl ExpressionHintTracker {
    pub(crate) fn new() -> Self {
        Self {
            slots: AtomicU64::new(0),
        }
    }

    pub(crate) fn record_expression(&self, expression: ExpressionId) {
        let value = expression.as_raw() as u64;
        loop {
            let current = self.slots.load(Ordering::Relaxed);
            let next = (current << 16) | value;
            if self
                .slots
                .compare_exchange(current, next, Ordering::AcqRel, Ordering::Relaxed)
                .is_ok()
            {
                break;
            }
        }
    }

    pub(crate) fn majority_expression(&self) -> Option<ExpressionId> {
        let mut slots = [0u16; 4];
        let mut value = self.slots.load(Ordering::Relaxed);
        for raw in &mut slots {
            *raw = (value & 0xFFFF) as u16;
            value >>= 16;
        }

        let mut best: Option<(ExpressionId, usize, usize)> = None;
        for (idx, raw) in slots.iter().enumerate() {
            if *raw == 0 {
                continue;
            }
            let Some(candidate) = ExpressionId::from_raw(*raw) else {
                continue;
            };
            let count = slots.iter().filter(|value| **value == *raw).count();
            match best {
                None => best = Some((candidate, count, idx)),
                Some((_, best_count, best_idx))
                    if count > best_count || (count == best_count && idx < best_idx) =>
                {
                    best = Some((candidate, count, idx));
                }
                _ => {}
            }
        }

        best.map(|(id, _, _)| id)
    }
}

impl Default for ExpressionHintTracker {
    fn default() -> Self {
        Self::new()
    }
}

impl Clone for ExpressionHintTracker {
    fn clone(&self) -> Self {
        Self {
            slots: AtomicU64::new(self.slots.load(Ordering::Relaxed)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::ExpressionHintTracker;
    use crate::cache::{CacheExpression, ExpressionRegistry};
    use arrow::compute::DatePart;

    #[test]
    fn majority_date32_field_returns_most_frequent_field() {
        let tracker = ExpressionHintTracker::new();
        let registry = ExpressionRegistry::new();

        let year = registry
            .register(CacheExpression::extract_date32(DatePart::Year))
            .expect("register year");
        let month = registry
            .register(CacheExpression::extract_date32(DatePart::Month))
            .expect("register month");

        tracker.record_expression(year);
        tracker.record_expression(month);
        tracker.record_expression(year);
        tracker.record_expression(year);

        let majority = tracker.majority_expression().expect("majority id");
        let expr = registry.get(majority).expect("resolve expression");
        assert_eq!(
            expr.as_ref(),
            &CacheExpression::extract_date32(DatePart::Year)
        );
    }

    #[test]
    fn majority_date32_field_prefers_most_recent_on_tie() {
        let tracker = ExpressionHintTracker::new();
        let registry = ExpressionRegistry::new();

        let year = registry
            .register(CacheExpression::extract_date32(DatePart::Year))
            .expect("register year");
        let month = registry
            .register(CacheExpression::extract_date32(DatePart::Month))
            .expect("register month");

        tracker.record_expression(year);
        tracker.record_expression(month);
        tracker.record_expression(year);
        tracker.record_expression(month);

        let majority = tracker.majority_expression().expect("majority id");
        let expr = registry.get(majority).expect("resolve expression");
        assert_eq!(
            expr.as_ref(),
            &CacheExpression::extract_date32(DatePart::Month)
        );
    }
}
