use divan::Bencher;
use rand::SeedableRng;
use rand::seq::SliceRandom;
use std::fmt::Write;
use std::sync::Arc;

extern crate arrow;

use arrow::array::{DictionaryArray, StringArray, UInt16Array};
use liquid_cache::liquid_array::LiquidByteViewArray;
use liquid_cache::liquid_array::byte_view_array::{ByteViewOperator, Comparison, Equality};
use liquid_cache::liquid_array::raw::FsstArray;

const ROW_COUNT: usize = 10_000;
const STRING_LEN: usize = 16;
// Keep in sync with PrefixKey::prefix_len().
const PREFIX_LEN: usize = 7;
const SUFFIX_LEN: usize = STRING_LEN - PREFIX_LEN;

const DECIDABLE_PCTS: [u8; 6] = [0, 5, 20, 50, 80, 100];

const NEEDLE_PREFIX: &str = "aaaaaaa";
const LOWER_PREFIX: &str = "0000000";
const HIGHER_PREFIX: &str = "zzzzzzz";
const BREAKER_PREFIX: &str = "bbbbbbb";

fn make_value(prefix: &str, idx: usize) -> String {
    debug_assert_eq!(prefix.len(), PREFIX_LEN);
    let mut value = String::with_capacity(STRING_LEN);
    value.push_str(prefix);
    write!(&mut value, "{:0width$}", idx, width = SUFFIX_LEN).expect("format suffix");
    debug_assert_eq!(value.len(), STRING_LEN);
    value
}

fn build_case(decidable_pct: u8) -> (LiquidByteViewArray<FsstArray>, Vec<u8>) {
    assert!(decidable_pct <= 100);
    assert!(ROW_COUNT <= u16::MAX as usize);

    let decidable_rows = ROW_COUNT * decidable_pct as usize / 100;
    let ambiguous_rows = ROW_COUNT - decidable_rows;

    let mut values = Vec::with_capacity(ROW_COUNT + 1);

    for idx in 0..ambiguous_rows {
        values.push(make_value(NEEDLE_PREFIX, idx));
    }

    let lower_count = decidable_rows / 2;
    let higher_count = decidable_rows - lower_count;

    for idx in 0..lower_count {
        values.push(make_value(LOWER_PREFIX, idx));
    }
    for idx in 0..higher_count {
        values.push(make_value(HIGHER_PREFIX, idx));
    }

    // Keep shared_prefix empty even when all rows share the needle prefix.
    values.push(make_value(BREAKER_PREFIX, 0));

    let dict_values = Arc::new(StringArray::from(values));
    let mut keys: Vec<u16> = (0..ROW_COUNT).map(|idx| idx as u16).collect();
    let mut rng = rand::rngs::StdRng::seed_from_u64(0x9e37_79b9 ^ decidable_pct as u64);
    keys.shuffle(&mut rng);
    let keys = UInt16Array::from(keys);
    let dict = DictionaryArray::new(keys, dict_values.clone());

    let compressor = LiquidByteViewArray::<FsstArray>::train_compressor(dict_values.iter());
    // Safety: dictionary values are unique, and we intentionally include an unused breaker entry.
    let array =
        unsafe { LiquidByteViewArray::<FsstArray>::from_unique_dict_array(&dict, compressor) };

    let needle = make_value(NEEDLE_PREFIX, 0).into_bytes();
    (array, needle)
}

#[divan::bench(args = DECIDABLE_PCTS)]
fn byte_view_eq_prefix_decidable(bencher: Bencher, decidable_pct: u8) {
    let (array, needle) = build_case(decidable_pct);

    bencher
        .with_inputs(|| (array.clone(), needle.clone()))
        .input_counter(|_| divan::counter::BytesCount::new(ROW_COUNT * STRING_LEN))
        .bench_values(|(array, needle)| {
            std::hint::black_box(
                array.compare_with(&needle, &ByteViewOperator::Equality(Equality::Eq)),
            )
        });
}

#[divan::bench(args = DECIDABLE_PCTS)]
fn byte_view_lt_prefix_decidable(bencher: Bencher, decidable_pct: u8) {
    let (array, needle) = build_case(decidable_pct);

    bencher
        .with_inputs(|| (array.clone(), needle.clone()))
        .input_counter(|_| divan::counter::BytesCount::new(ROW_COUNT * STRING_LEN))
        .bench_values(|(array, needle)| {
            std::hint::black_box(
                array.compare_with(&needle, &ByteViewOperator::Comparison(Comparison::Lt)),
            )
        });
}

fn main() {
    divan::main();
}
