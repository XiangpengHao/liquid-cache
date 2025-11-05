use std::sync::atomic::{AtomicUsize, Ordering};

use crate::{cache::CacheExpressionId, liquid_array::Date32Field};

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

const EXPRESSION_SLOT_BITS: usize = 16;
const EXPRESSION_SLOT_MASK: usize = (1usize << EXPRESSION_SLOT_BITS) - 1;
const EXPRESSION_SLOT_COUNT: usize = 4;
const EXPRESSION_EMPTY_VALUE: u16 = u16::MAX;

const fn empty_expression_packed() -> usize {
    let mut value = 0usize;
    let mut idx = 0;
    while idx < EXPRESSION_SLOT_COUNT {
        value |= (EXPRESSION_EMPTY_VALUE as usize) << (idx * EXPRESSION_SLOT_BITS);
        idx += 1;
    }
    value
}

const EXPRESSION_EMPTY_PACKED: usize = empty_expression_packed();

#[cfg(not(target_pointer_width = "64"))]
compile_error!("Expression hint tracking requires a 64-bit target pointer width");

fn decode_expression_slots(raw: usize) -> [u16; EXPRESSION_SLOT_COUNT] {
    let mut slots = [EXPRESSION_EMPTY_VALUE; EXPRESSION_SLOT_COUNT];
    let mut idx = 0;
    while idx < EXPRESSION_SLOT_COUNT {
        let shift = idx * EXPRESSION_SLOT_BITS;
        let value = (raw >> shift) & EXPRESSION_SLOT_MASK;
        slots[idx] = value as u16;
        idx += 1;
    }
    slots
}

fn encode_expression_slots(slots: &[u16; EXPRESSION_SLOT_COUNT]) -> usize {
    let mut raw = 0usize;
    let mut idx = 0;
    while idx < EXPRESSION_SLOT_COUNT {
        raw |= (slots[idx] as usize) << (idx * EXPRESSION_SLOT_BITS);
        idx += 1;
    }
    raw
}

#[derive(Debug)]
pub(crate) struct ExpressionHintTracker {
    packed: AtomicUsize,
}

impl ExpressionHintTracker {
    pub(crate) fn new() -> Self {
        Self {
            packed: AtomicUsize::new(EXPRESSION_EMPTY_PACKED),
        }
    }

    pub(crate) fn record_date32_field(&self, field: Date32Field) {
        let id = CacheExpressionId::from_date32_field(field);
        self.record_expression_id(id);
    }

    fn record_expression_id(&self, id: CacheExpressionId) {
        let code = id.raw();
        if code == EXPRESSION_EMPTY_VALUE {
            return;
        }
        let mut slots = decode_expression_slots(self.packed.load(Ordering::Relaxed));
        let mut idx = 0;
        while idx + 1 < EXPRESSION_SLOT_COUNT {
            slots[idx] = slots[idx + 1];
            idx += 1;
        }
        slots[EXPRESSION_SLOT_COUNT - 1] = code;
        let encoded = encode_expression_slots(&slots);
        // here we didn't use compare and swap because we don't care about contention here.
        self.packed.store(encoded, Ordering::Relaxed);
    }

    pub(crate) fn majority_date32_field(&self) -> Option<Date32Field> {
        let slots = decode_expression_slots(self.packed.load(Ordering::Relaxed));
        let mut best_field = None;
        let mut best_count = 0u8;
        let mut best_index = 0usize;

        for (idx, &code) in slots.iter().enumerate() {
            if code == EXPRESSION_EMPTY_VALUE {
                continue;
            }

            let id = CacheExpressionId::from_raw(code);
            let Some(field) = id.as_date32_field() else {
                continue;
            };

            let count = slots.iter().filter(|&&value| value == code).count() as u8;
            if count > best_count || (count == best_count && idx > best_index) {
                best_count = count;
                best_index = idx;
                best_field = Some(field);
            }
        }

        best_field
    }
}

impl Default for ExpressionHintTracker {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::ExpressionHintTracker;
    use crate::liquid_array::Date32Field;

    #[test]
    fn majority_date32_field_returns_most_frequent_field() {
        let tracker = ExpressionHintTracker::new();

        tracker.record_date32_field(Date32Field::Year);
        tracker.record_date32_field(Date32Field::Month);
        tracker.record_date32_field(Date32Field::Year);
        tracker.record_date32_field(Date32Field::Year);

        assert_eq!(tracker.majority_date32_field(), Some(Date32Field::Year));
    }

    #[test]
    fn majority_date32_field_prefers_most_recent_on_tie() {
        let tracker = ExpressionHintTracker::new();

        tracker.record_date32_field(Date32Field::Year);
        tracker.record_date32_field(Date32Field::Month);
        tracker.record_date32_field(Date32Field::Year);
        tracker.record_date32_field(Date32Field::Month);

        assert_eq!(tracker.majority_date32_field(), Some(Date32Field::Month));
    }
}
