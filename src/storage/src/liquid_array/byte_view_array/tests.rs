use super::*;
use arrow::array::{
    Array, ArrayRef, BooleanArray, DictionaryArray, StringArray, UInt16Array, cast::AsArray,
    types::UInt16Type,
};
use arrow::buffer::BooleanBuffer;
use arrow_schema::DataType;
use datafusion::logical_expr::Operator;
use rand::{Rng, SeedableRng};
use std::sync::Arc;

use crate::cache::CacheExpression;
use crate::cache::TestingSqueezeIo;
use crate::liquid_array::raw::fsst_buffer::{DiskBuffer, FsstArray, PrefixKey};
use crate::liquid_array::{LiquidArray, LiquidDataType, LiquidSqueezedArray};

#[test]
fn test_dictionary_view_structure() {
    // Test PrefixKey structure
    let prefix_key = PrefixKey::from_parts([1, 2, 3, 4, 5, 6, 7], 7);
    assert_eq!(prefix_key.prefix7(), &[1, 2, 3, 4, 5, 6, 7]);
    assert_eq!(prefix_key.len_byte(), 7);

    // Test UInt16Array creation (dictionary keys are now stored directly in UInt16Array)
    let keys = UInt16Array::from(vec![42, 100, 255]);
    assert_eq!(keys.value(0), 42);
    assert_eq!(keys.value(1), 100);
    assert_eq!(keys.value(2), 255);
}

#[test]
fn test_original_arrow_data_type_returns_utf8() {
    let input = StringArray::from(vec!["foo", "bar"]);
    let compressor = LiquidByteViewArray::<FsstArray>::train_compressor(input.iter());
    let array = LiquidByteViewArray::<FsstArray>::from_string_array(&input, compressor);
    assert_eq!(array.original_arrow_data_type(), DataType::Utf8);
}

#[test]
fn test_hybrid_original_arrow_data_type_returns_utf8() {
    let input = StringArray::from(vec!["foo", "bar"]);
    let compressor = LiquidByteViewArray::<FsstArray>::train_compressor(input.iter());
    let in_memory = LiquidByteViewArray::<FsstArray>::from_string_array(&input, compressor);
    let (hybrid, _) = in_memory
        .squeeze(
            Arc::new(TestingSqueezeIo),
            Some(&CacheExpression::PredicateColumn),
        )
        .expect("squeeze should succeed");
    let disk_view = hybrid
        .as_any()
        .downcast_ref::<LiquidByteViewArray<DiskBuffer>>()
        .expect("should downcast to disk array");
    assert_eq!(disk_view.original_arrow_data_type(), DataType::Utf8);
}

#[test]
fn test_ipc_roundtrip_sliced_dictionary_nulls() {
    let values: ArrayRef = Arc::new(StringArray::from(vec!["a", "b", "c", "d"]));
    let keys = UInt16Array::from(vec![
        Some(0u16),
        None,
        Some(2),
        Some(1),
        None,
        Some(3),
        Some(0),
        Some(2),
        Some(1),
    ]);
    let dict = DictionaryArray::<UInt16Type>::new(keys, values);

    // Slice to create a non-zero offset (and therefore a non-zero null bitmap bit offset).
    let sliced = dict.slice(1, 7);

    let compressor = LiquidByteViewArray::<FsstArray>::train_compressor(
        sliced.values().as_string::<i32>().iter(),
    );
    let original = unsafe {
        LiquidByteViewArray::<FsstArray>::from_unique_dict_array(&sliced, compressor.clone())
    };

    let before = original.to_arrow_array();
    let bytes = original.to_bytes();
    let decoded = LiquidByteViewArray::<FsstArray>::from_bytes(bytes.into(), compressor);
    let after = decoded.to_arrow_array();

    assert_eq!(before.as_ref(), after.as_ref());
}

#[test]
fn test_prefix_extraction() {
    let input = StringArray::from(vec!["hello", "world", "test"]);
    let compressor = LiquidByteViewArray::<FsstArray>::train_compressor(input.iter());
    let liquid_array = LiquidByteViewArray::<FsstArray>::from_string_array(&input, compressor);

    // With no shared prefix, the offset view prefixes should be the original strings (truncated to 7 bytes)
    assert_eq!(liquid_array.shared_prefix, Vec::<u8>::new());
    assert_eq!(liquid_array.prefix_keys[0].prefix7(), b"hello\0\0");
    assert_eq!(liquid_array.prefix_keys[1].prefix7(), b"world\0\0");
    assert_eq!(liquid_array.prefix_keys[2].prefix7(), b"test\0\0\0");
}

#[test]
fn test_shared_prefix_functionality() {
    // Test case with shared prefix
    let input = StringArray::from(vec![
        "hello_world",
        "hello_rust",
        "hello_test",
        "hello_code",
    ]);
    let compressor = LiquidByteViewArray::<FsstArray>::train_compressor(input.iter());
    let liquid_array = LiquidByteViewArray::<FsstArray>::from_string_array(&input, compressor);

    // Should extract "hello_" as shared prefix
    assert_eq!(liquid_array.shared_prefix, b"hello_");

    // Offset view prefixes (7 bytes) and lengths
    assert_eq!(liquid_array.prefix_keys[0].prefix7(), b"world\0\0");
    assert_eq!(liquid_array.prefix_keys[1].prefix7(), b"rust\0\0\0");
    assert_eq!(liquid_array.prefix_keys[2].prefix7(), b"test\0\0\0");
    assert_eq!(liquid_array.prefix_keys[3].prefix7(), b"code\0\0\0");

    // Test roundtrip - should reconstruct original strings correctly
    let output = liquid_array.to_arrow_array();
    assert_eq!(&input, output.as_string::<i32>());

    // Test comparison with shared prefix optimization
    let result = liquid_array.compare_equals(b"hello_rust");
    let expected = BooleanArray::from(vec![false, true, false, false]);
    assert_eq!(result, expected);

    // Test comparison that doesn't match shared prefix
    let result = liquid_array.compare_equals(b"goodbye_world");
    let expected = BooleanArray::from(vec![false, false, false, false]);
    assert_eq!(result, expected);

    // Test partial shared prefix match
    let result = liquid_array.compare_equals(b"hello_");
    let expected = BooleanArray::from(vec![false, false, false, false]);
    assert_eq!(result, expected);
}

#[test]
fn test_shared_prefix_with_short_strings() {
    // Test case: short strings that fit entirely in the 6-byte prefix
    let input = StringArray::from(vec!["abc", "abcde", "abcdef", "abcdefg"]);
    let compressor = LiquidByteViewArray::<FsstArray>::train_compressor(input.iter());
    let liquid_array = LiquidByteViewArray::<FsstArray>::from_string_array(&input, compressor);

    // Should extract "abc" as shared prefix
    assert_eq!(liquid_array.shared_prefix, b"abc");

    // Offset view prefixes should be the remaining parts after shared prefix (7 bytes)
    assert_eq!(liquid_array.prefix_keys[0].prefix7(), &[0u8; 7]); // empty after "abc"
    assert_eq!(liquid_array.prefix_keys[1].prefix7(), b"de\0\0\0\0\0"); // "de" after "abc"
    assert_eq!(liquid_array.prefix_keys[2].prefix7(), b"def\0\0\0\0"); // "def" after "abc"
    assert_eq!(liquid_array.prefix_keys[3].prefix7(), b"defg\0\0\0"); // "defg" after "abc"

    // Test roundtrip
    let output = liquid_array.to_arrow_array();
    assert_eq!(&input, output.as_string::<i32>());

    // Test equality comparisons with short strings
    let result = liquid_array.compare_equals(b"abc");
    let expected = BooleanArray::from(vec![true, false, false, false]);
    assert_eq!(result, expected);

    let result = liquid_array.compare_equals(b"abcde");
    let expected = BooleanArray::from(vec![false, true, false, false]);
    assert_eq!(result, expected);

    // Test ordering comparisons that can be resolved by shared prefix
    let result = liquid_array.compare_with(b"ab", &Operator::Gt);
    let expected = BooleanArray::from(vec![true, true, true, true]); // All start with "abc" > "ab"
    assert_eq!(result, expected);

    let result = liquid_array.compare_with(b"abcd", &Operator::Lt);
    let expected = BooleanArray::from(vec![true, false, false, false]); // Only "abc" < "abcd"
    assert_eq!(result, expected);
}

#[test]
fn test_shared_prefix_contains_complete_strings() {
    // Test case: shared prefix completely contains some strings
    let input = StringArray::from(vec!["data", "database", "data_entry", "data_", "datatype"]);
    let compressor = LiquidByteViewArray::<FsstArray>::train_compressor(input.iter());
    let liquid_array = LiquidByteViewArray::<FsstArray>::from_string_array(&input, compressor);

    // Should extract "data" as shared prefix
    assert_eq!(liquid_array.shared_prefix, b"data");

    // Offset view prefixes should be the remaining parts (7 bytes)
    assert_eq!(liquid_array.prefix_keys[0].prefix7(), &[0u8; 7]); // "data" - empty remainder
    assert_eq!(liquid_array.prefix_keys[1].prefix7(), b"base\0\0\0"); // "database" - "base" remainder
    assert_eq!(liquid_array.prefix_keys[2].prefix7(), b"_entry\0"); // "data_entry" - "_entry" remainder
    assert_eq!(liquid_array.prefix_keys[3].prefix7(), b"_\0\0\0\0\0\0"); // "data_" - "_" remainder
    assert_eq!(liquid_array.prefix_keys[4].prefix7(), b"type\0\0\0"); // "datatype" - "type" remainder

    // Test roundtrip
    let output = liquid_array.to_arrow_array();
    assert_eq!(&input, output.as_string::<i32>());

    // Test equality with exact shared prefix
    let result = liquid_array.compare_equals(b"data");
    let expected = BooleanArray::from(vec![true, false, false, false, false]);
    assert_eq!(result, expected);

    // Test comparisons where shared prefix helps
    let result = liquid_array.compare_with(b"dat", &Operator::Gt);
    let expected = BooleanArray::from(vec![true, true, true, true, true]); // All > "dat"
    assert_eq!(result, expected);

    let result = liquid_array.compare_with(b"datab", &Operator::Lt);
    let expected = BooleanArray::from(vec![true, false, true, true, false]); // "data", "data_entry", and "data_" < "datab"
    assert_eq!(result, expected);

    // Test comparison with needle shorter than shared prefix
    let result = liquid_array.compare_with(b"da", &Operator::Gt);
    let expected = BooleanArray::from(vec![true, true, true, true, true]); // All > "da"
    assert_eq!(result, expected);

    // Test comparison with needle equal to shared prefix
    let result = liquid_array.compare_with(b"data", &Operator::GtEq);
    let expected = BooleanArray::from(vec![true, true, true, true, true]); // All >= "data"
    assert_eq!(result, expected);

    let result = liquid_array.compare_with(b"data", &Operator::Gt);
    let expected = BooleanArray::from(vec![false, true, true, true, true]); // All except exact "data" > "data"
    assert_eq!(result, expected);
}

#[test]
fn test_compare_with_large_value_no_panic() {
    // Exercise the ordering-compare slow path which must resize its scratch buffer
    // for large values instead of panicking inside `fsst::Decompressor::decompress_into`.
    let big = "aaaaaaa".to_string() + &"b".repeat(2 * 1024 * 1024 + 128);
    let input = StringArray::from(vec![big.as_str()]);

    let compressor = LiquidByteViewArray::<FsstArray>::train_compressor(input.iter());
    let liquid_array = LiquidByteViewArray::<FsstArray>::from_string_array(&input, compressor);

    let result = liquid_array.compare_with(big.as_bytes(), &Operator::LtEq);
    assert_eq!(result.len(), 1);
    assert!(result.value(0));
}

#[test]
fn test_shared_prefix_corner_case() {
    let input = StringArray::from(vec!["data", "database", "data_entry", "data_", "datatype"]);
    let compressor = LiquidByteViewArray::<FsstArray>::train_compressor(input.iter());
    let liquid_array = LiquidByteViewArray::<FsstArray>::from_string_array(&input, compressor);
    let result = liquid_array.compare_with(b"data", &Operator::GtEq);
    let expected = BooleanArray::from(vec![true, true, true, true, true]); // All >= "data"
    assert_eq!(result, expected);
}

#[test]
fn test_shared_prefix_edge_cases() {
    // Test case 1: All strings are the same (full shared prefix)
    let input = StringArray::from(vec!["identical", "identical", "identical"]);
    let compressor = LiquidByteViewArray::<FsstArray>::train_compressor(input.iter());
    let liquid_array = LiquidByteViewArray::<FsstArray>::from_string_array(&input, compressor);

    assert_eq!(liquid_array.shared_prefix, b"identical");
    // All offset view prefixes should be empty
    for i in 0..liquid_array.prefix_keys.len() {
        assert_eq!(liquid_array.prefix_keys[i].prefix7(), &[0u8; 7]);
    }

    // Test roundtrip
    let output = liquid_array.to_arrow_array();
    assert_eq!(&input, output.as_string::<i32>());

    // Test case 2: One string is a prefix of others
    let input = StringArray::from(vec!["hello", "hello_world", "hello_test"]);
    let compressor = LiquidByteViewArray::<FsstArray>::train_compressor(input.iter());
    let liquid_array = LiquidByteViewArray::<FsstArray>::from_string_array(&input, compressor);

    assert_eq!(liquid_array.shared_prefix, b"hello");
    assert_eq!(liquid_array.prefix_keys[0].prefix7(), &[0u8; 7]); // empty after "hello"
    assert_eq!(liquid_array.prefix_keys[1].prefix7(), b"_world\0");
    assert_eq!(liquid_array.prefix_keys[2].prefix7(), b"_test\0\0");

    // Test roundtrip
    let output = liquid_array.to_arrow_array();
    assert_eq!(&input, output.as_string::<i32>());

    // Test case 3: Empty string in array (should limit shared prefix)
    let input = StringArray::from(vec!["", "hello", "hello_world"]);
    let compressor = LiquidByteViewArray::<FsstArray>::train_compressor(input.iter());
    let liquid_array = LiquidByteViewArray::<FsstArray>::from_string_array(&input, compressor);

    assert_eq!(liquid_array.shared_prefix, Vec::<u8>::new()); // empty shared prefix
    assert_eq!(liquid_array.prefix_keys[0].prefix7(), &[0u8; 7]);
    assert_eq!(liquid_array.prefix_keys[1].prefix7(), b"hello\0\0");
    assert_eq!(liquid_array.prefix_keys[2].prefix7(), b"hello_w"); // "hello_world" truncated to 7 bytes

    // Test roundtrip
    let output = liquid_array.to_arrow_array();
    assert_eq!(&input, output.as_string::<i32>());
}

#[test]
fn test_memory_layout() {
    let input = StringArray::from(vec!["hello", "world", "test"]);
    let compressor = LiquidByteViewArray::<FsstArray>::train_compressor(input.iter());
    let liquid_array = LiquidByteViewArray::<FsstArray>::from_string_array(&input, compressor);

    // Verify memory layout components
    assert_eq!(liquid_array.dictionary_keys.len(), 3);
    assert_eq!(liquid_array.fsst_buffer.offsets_len(), 4);
    assert!(liquid_array.nulls().is_none());
    let _first = liquid_array.fsst_buffer.get_compressed_slice(0);
}

fn check_filter_result(input: &StringArray, filter: BooleanBuffer) {
    let compressor = LiquidByteViewArray::<FsstArray>::train_compressor(input.iter());
    let liquid_array = LiquidByteViewArray::<FsstArray>::from_string_array(input, compressor);
    let output = liquid_array.filter(&filter);
    let expected = {
        let selection = BooleanArray::new(filter.clone(), None);
        let arrow_filtered = arrow::compute::filter(&input, &selection).unwrap();
        arrow_filtered.as_string::<i32>().clone()
    };
    assert_eq!(output.as_ref(), &expected);
}

#[test]
fn test_filter_functionality() {
    let input = StringArray::from(vec![
        Some("hello"),
        Some("test"),
        None,
        Some("test"),
        None,
        Some("test"),
        Some("rust"),
    ]);
    let mut seeded_rng = rand::rngs::StdRng::seed_from_u64(42);
    for _i in 0..100 {
        let filter =
            BooleanBuffer::from_iter((0..input.len()).map(|_| seeded_rng.random::<bool>()));
        check_filter_result(&input, filter);
    }
}

#[test]
fn test_memory_efficiency() {
    let input = StringArray::from(vec!["hello", "world", "hello", "world", "hello"]);
    let compressor = LiquidByteViewArray::<FsstArray>::train_compressor(input.iter());
    let liquid_array = LiquidByteViewArray::<FsstArray>::from_string_array(&input, compressor);

    // Verify that dictionary views store unique values efficiently
    assert_eq!(liquid_array.dictionary_keys.len(), 5);

    // Verify that FSST buffer contains unique values
    let dict = liquid_array.to_dict_arrow();
    assert_eq!(dict.values().len(), 2); // Only "hello" and "world"
}

#[test]
fn test_to_best_arrow_array() {
    let input = StringArray::from(vec!["hello", "world", "test"]);
    let compressor = LiquidByteViewArray::<FsstArray>::train_compressor(input.iter());
    let liquid_array = LiquidByteViewArray::<FsstArray>::from_string_array(&input, compressor);

    let best_array = liquid_array.to_best_arrow_array();
    let dict_array = best_array.as_dictionary::<UInt16Type>();

    // Should return dictionary array as the best encoding
    assert_eq!(dict_array.len(), 3);
    assert_eq!(dict_array.values().len(), 3); // Three unique values
}

#[test]
fn test_data_type() {
    let input = StringArray::from(vec!["hello", "world"]);
    let compressor = LiquidByteViewArray::<FsstArray>::train_compressor(input.iter());
    let liquid_array = LiquidByteViewArray::<FsstArray>::from_string_array(&input, compressor);

    // Just verify we can get the data type without errors
    let data_type = liquid_array.data_type();
    assert!(matches!(data_type, LiquidDataType::ByteViewArray));
}

#[test]
fn test_compare_with_prefix_optimization_fast_path() {
    // Test case 1: Prefix comparison can decide most results without decompression
    // Uses strings with distinct prefixes to test the fast path
    let input = StringArray::from(vec![
        "apple123",  // prefix: "apple\0"
        "banana456", // prefix: "banana"
        "cherry789", // prefix: "cherry"
        "apple999",  // prefix: "apple\0" (same as first)
        "zebra000",  // prefix: "zebra\0"
    ]);

    let compressor = LiquidByteViewArray::<FsstArray>::train_compressor(input.iter());
    let liquid_array = LiquidByteViewArray::<FsstArray>::from_string_array(&input, compressor);

    // Test Lt with needle "car" (prefix: "car\0\0\0")
    // Expected: "apple123" < "car" => true, "banana456" < "car" => true, others false
    let result = liquid_array.compare_with_inner(b"car", &Operator::Lt);
    let expected = BooleanArray::from(vec![true, true, false, true, false]);
    assert_eq!(result, expected);

    // Test Gt with needle "dog" (prefix: "dog\0\0\0")
    // Expected: only "zebra000" > "dog" => true
    let result = liquid_array.compare_with_inner(b"dog", &Operator::Gt);
    let expected = BooleanArray::from(vec![false, false, false, false, true]);
    assert_eq!(result, expected);

    // Test GtEq with needle "apple" (prefix: "apple\0")
    // Expected: all except "apple123" and "apple999" need decompression, others by prefix
    let result = liquid_array.compare_with_inner(b"apple", &Operator::GtEq);
    let expected = BooleanArray::from(vec![true, true, true, true, true]);
    assert_eq!(result, expected);
}

#[test]
fn test_compare_with_prefix_optimization_decompression_path() {
    // Test case 2: Cases where prefix comparison is inconclusive and requires decompression
    // Uses strings with identical prefixes but different suffixes
    let input = StringArray::from(vec![
        "prefix_aaa", // prefix: "prefix"
        "prefix_bbb", // prefix: "prefix" (same prefix)
        "prefix_ccc", // prefix: "prefix" (same prefix)
        "prefix_abc", // prefix: "prefix" (same prefix)
        "different",  // prefix: "differ" (different prefix)
    ]);

    let compressor = LiquidByteViewArray::<FsstArray>::train_compressor(input.iter());
    let liquid_array = LiquidByteViewArray::<FsstArray>::from_string_array(&input, compressor);

    // Test Lt with needle "prefix_b" - this will require decompression for prefix matches
    // Expected: "prefix_aaa" < "prefix_b" => true, "prefix_bbb" < "prefix_b" => false, etc.
    let result = liquid_array.compare_with_inner(b"prefix_b", &Operator::Lt);
    let expected = BooleanArray::from(vec![true, false, false, true, true]);
    assert_eq!(result, expected);

    // Test LtEq with needle "prefix_bbb" - exact match case with decompression
    let result = liquid_array.compare_with_inner(b"prefix_bbb", &Operator::LtEq);
    let expected = BooleanArray::from(vec![true, true, false, true, true]);
    assert_eq!(result, expected);

    // Test Gt with needle "prefix_abc" - requires decompression for prefix matches
    let result = liquid_array.compare_with_inner(b"prefix_abc", &Operator::Gt);
    let expected = BooleanArray::from(vec![false, true, true, false, false]);
    assert_eq!(result, expected);
}

#[test]
fn test_compare_with_prefix_optimization_edge_cases_and_nulls() {
    // Test case 3: Edge cases including nulls, empty strings, and boundary conditions
    let input = StringArray::from(vec![
        Some(""),           // Empty string (prefix: "\0\0\0\0\0\0")
        None,               // Null value
        Some("a"),          // Single character (prefix: "a\0\0\0\0\0")
        Some("abcdef"),     // Exactly 6 chars (prefix: "abcdef")
        Some("abcdefghij"), // Longer than 6 chars (prefix: "abcdef")
        Some("abcdeg"),     // Differs at position 5 (prefix: "abcdeg")
    ]);

    let compressor = LiquidByteViewArray::<FsstArray>::train_compressor(input.iter());
    let liquid_array = LiquidByteViewArray::<FsstArray>::from_string_array(&input, compressor);

    // Test Lt with empty string needle - should test null handling
    let result = liquid_array.compare_with_inner(b"", &Operator::Lt);
    let expected = BooleanArray::from(vec![
        Some(false),
        None,
        Some(false),
        Some(false),
        Some(false),
        Some(false),
    ]);
    assert_eq!(result, expected);

    // Test Gt with needle "abcdef" - tests exact prefix match requiring decompression
    let result = liquid_array.compare_with_inner(b"abcdef", &Operator::Gt);
    let expected = BooleanArray::from(vec![
        Some(false),
        None,
        Some(false),
        Some(false),
        Some(true),
        Some(true),
    ]);
    assert_eq!(result, expected);

    // Test LtEq with needle "b" - tests single character comparisons
    let result = liquid_array.compare_with_inner(b"b", &Operator::LtEq);
    let expected = BooleanArray::from(vec![
        Some(true),
        None,
        Some(true),
        Some(true),
        Some(true),
        Some(true),
    ]);
    assert_eq!(result, expected);

    // Test GtEq with needle "abcdeg" - tests decompression when prefix exactly matches needle prefix
    let result = liquid_array.compare_with_inner(b"abcdeg", &Operator::GtEq);
    // b"" >= b"abcdeg" => false
    // null => null
    // b"a" >= b"abcdeg" => false
    // b"abcdef" >= b"abcdeg" => false (because b"abcdef" < b"abcdeg" since 'f' < 'g')
    // b"abcdefghij" >= b"abcdeg" => false (because b"abcdefghij" < b"abcdeg" since 'f' < 'g')
    // b"abcdeg" >= b"abcdeg" => true (exact match)
    let expected = BooleanArray::from(vec![
        Some(false),
        None,
        Some(false),
        Some(false),
        Some(false),
        Some(true),
    ]);
    assert_eq!(result, expected);
}

#[test]
fn test_compare_with_prefix_optimization_utf8_and_binary() {
    // Test case 4: UTF-8 encoded strings and binary data comparisons
    // This demonstrates the advantage of byte-level comparison
    let input = StringArray::from(vec![
        "café",   // UTF-8: [99, 97, 102, 195, 169]
        "naïve",  // UTF-8: [110, 97, 195, 175, 118, 101]
        "résumé", // UTF-8: [114, 195, 169, 115, 117, 109, 195, 169]
        "hello",  // ASCII: [104, 101, 108, 108, 111]
        "世界",   // UTF-8: [228, 184, 150, 231, 149, 140]
    ]);

    let compressor = LiquidByteViewArray::<FsstArray>::train_compressor(input.iter());
    let liquid_array = LiquidByteViewArray::<FsstArray>::from_string_array(&input, compressor);

    // Test Lt with UTF-8 needle "naïve" (UTF-8: [110, 97, 195, 175, 118, 101])
    // Expected: "café" < "naïve" => true (99 < 110), "hello" < "naïve" => true, others false
    let naive_bytes = "naïve".as_bytes(); // [110, 97, 195, 175, 118, 101]
    let result = liquid_array.compare_with_inner(naive_bytes, &Operator::Lt);
    let expected = BooleanArray::from(vec![true, false, false, true, false]);
    assert_eq!(result, expected);

    // Test Gt with UTF-8 needle "café" (UTF-8: [99, 97, 102, 195, 169])
    // Expected: strings with first byte > 99 should be true
    let cafe_bytes = "café".as_bytes(); // [99, 97, 102, 195, 169]
    let result = liquid_array.compare_with_inner(cafe_bytes, &Operator::Gt);
    let expected = BooleanArray::from(vec![false, true, true, true, true]);
    assert_eq!(result, expected);

    // Test LtEq with Chinese characters "世界" (UTF-8: [228, 184, 150, 231, 149, 140])
    // Expected: only strings with first byte <= 228 should be true, but since 228 is quite high,
    // most Latin characters will be true
    let world_bytes = "世界".as_bytes(); // [228, 184, 150, 231, 149, 140]
    let result = liquid_array.compare_with_inner(world_bytes, &Operator::LtEq);
    let expected = BooleanArray::from(vec![true, true, true, true, true]);
    assert_eq!(result, expected);

    // Test exact equality with "résumé" using GtEq and LtEq to verify byte-level precision
    let resume_bytes = "résumé".as_bytes(); // [114, 195, 169, 115, 117, 109, 195, 169]
    let gte_result = liquid_array.compare_with_inner(resume_bytes, &Operator::GtEq);
    let lte_result = liquid_array.compare_with_inner(resume_bytes, &Operator::LtEq);

    // Check GtEq and LtEq results separately
    // GtEq: "café"(99) >= "résumé"(114) => false, "naïve"(110) >= "résumé"(114) => false,
    //       "résumé"(114) >= "résumé"(114) => true, "hello"(104) >= "résumé"(114) => false,
    //       "世界"(228) >= "résumé"(114) => true
    let gte_expected = BooleanArray::from(vec![false, false, true, false, true]);
    // LtEq: all strings with first byte <= 114 should be true, "世界"(228) should be false
    let lte_expected = BooleanArray::from(vec![true, true, true, true, false]);
    assert_eq!(gte_result, gte_expected);
    assert_eq!(lte_result, lte_expected);
}

fn test_compare_equals(input: StringArray, needle: &[u8], expected: BooleanArray) {
    let compressor = LiquidByteViewArray::<FsstArray>::train_compressor(input.iter());
    let liquid_array = LiquidByteViewArray::<FsstArray>::from_string_array(&input, compressor);
    let result = liquid_array.compare_equals(needle);
    assert_eq!(result, expected);
}

#[test]
fn test_compare_equals_on_disk() {
    let input = StringArray::from(vec![
        Some("apple_orange"),
        None,
        Some("apple_orange_long_string"),
        Some("apple_b"),
        Some("apple_oo_long_string"),
        Some("apple_b"),
        Some("apple"),
    ]);
    test_compare_equals(
        input.clone(),
        b"apple",
        BooleanArray::from(vec![
            Some(false),
            None,
            Some(false),
            Some(false),
            Some(false),
            Some(false),
            Some(true),
        ]),
    );
    test_compare_equals(
        input.clone(),
        b"",
        BooleanArray::from(vec![
            Some(false),
            None,
            Some(false),
            Some(false),
            Some(false),
            Some(false),
            Some(false),
        ]),
    );
    test_compare_equals(
        input.clone(),
        b"apple_b",
        BooleanArray::from(vec![
            Some(false),
            None,
            Some(false),
            Some(true),
            Some(false),
            Some(true),
            Some(false),
        ]),
    );
    test_compare_equals(
        input.clone(),
        b"apple_oo_long_string",
        BooleanArray::from(vec![
            Some(false),
            None,
            Some(false),
            Some(false),
            Some(true),
            Some(false),
            Some(false),
        ]),
    );
}

// Benchmark tests for v2 offset compression improvements
fn generate_mixed_size_strings(count: usize, seed: u64) -> Vec<String> {
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let mut strings = Vec::with_capacity(count);

    for _ in 0..count {
        let size_type = rng.random_range(0..4);
        let string = match size_type {
            0 => {
                // Very short strings (1-3 chars) - stress test prefix optimization
                let len = rng.random_range(1..=3);
                (0..len)
                    .map(|_| rng.random_range(b'a'..=b'z') as char)
                    .collect()
            }
            1 => {
                // Medium strings (50-200 chars) - test offset compression
                let len = rng.random_range(50..=200);
                (0..len)
                    .map(|_| rng.random_range(b'a'..=b'z') as char)
                    .collect()
            }
            2 => {
                // Long strings (1000-5000 chars) - stress test linear regression
                let len = rng.random_range(1000..=5000);
                (0..len)
                    .map(|_| rng.random_range(b'a'..=b'z') as char)
                    .collect()
            }
            _ => {
                // Very long strings (10k+ chars) - edge case for offset compression
                let len = rng.random_range(10000..=50000);
                "x".repeat(len)
            }
        };
        strings.push(string);
    }

    strings
}

fn generate_zipf_strings(count: usize, base_strings: &[&str], seed: u64) -> Vec<String> {
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let mut strings = Vec::with_capacity(count);

    // Simple Zipf-like distribution: first few strings are much more common
    for _ in 0..count {
        let zipf_choice = rng.random_range(0..100);
        let base_idx = if zipf_choice < 50 {
            0 // 50% chance of first string
        } else if zipf_choice < 75 {
            1 // 25% chance of second string  
        } else if zipf_choice < 87 {
            2 // 12% chance of third string
        } else {
            rng.random_range(3..base_strings.len()) // remaining strings split rest
        };

        let base = base_strings[base_idx];

        // Add variations to create realistic patterns
        let variation = rng.random_range(0..4);
        let string = match variation {
            0 => base.to_string(),                                     // Exact duplicate
            1 => format!("{}_{}", base, rng.random_range(1000..9999)), // Common suffix
            2 => format!("{}/{}", base, rng.random_range(100..999)),   // Path-like
            _ => format!("prefix_{}", base),                           // Common prefix
        };
        strings.push(string);
    }

    strings
}

#[test]
fn test_mixed_size_offset_views() {
    let strings = generate_mixed_size_strings(16384, 42);
    let input = StringArray::from(strings.clone());

    let compressor = LiquidByteViewArray::<FsstArray>::train_compressor(input.iter());
    let liquid_array = LiquidByteViewArray::<FsstArray>::from_string_array(&input, compressor);

    // Verify correctness
    let output = liquid_array.to_arrow_array();
    assert_eq!(&input, output.as_string::<i32>());
}

#[test]
fn test_zipf_offset_views() {
    // Real-world string patterns
    let base_patterns = &[
        "error", "warning", "info", "debug", "user", "admin", "guest", "GET", "POST", "PUT",
        "DELETE", "success", "failure", "pending", "/api/v1", "/api/v2", "/health",
    ];

    let strings = generate_zipf_strings(16384, base_patterns, 123);
    let input = StringArray::from(strings.clone());

    let compressor = LiquidByteViewArray::<FsstArray>::train_compressor(input.iter());
    let liquid_array = LiquidByteViewArray::<FsstArray>::from_string_array(&input, compressor);

    // Verify correctness
    let output = liquid_array.to_arrow_array();
    assert_eq!(&input, output.as_string::<i32>());

    let offset_bytes = liquid_array.fsst_buffer.offset_bytes();

    assert!(
        offset_bytes <= 2,
        "Zipf patterns with short strings should use 1 or 2 bytes offset views, got {} bytes",
        offset_bytes
    );
}

#[test]
fn test_offset_stress() {
    let mut strings = Vec::with_capacity(16384);

    // Create strings with problematic offset patterns
    for i in 0..16384 {
        let string = match i % 8 {
            0 => "a".to_string(),                 // tiny
            1 => "x".repeat(1000 + (i % 100)),    // variable medium
            2 => "b".to_string(),                 // tiny
            3 => "y".repeat(5000 + (i % 1000)),   // variable large
            4 => "c".to_string(),                 // tiny
            5 => "medium".repeat(50 + (i % 20)),  // variable medium
            6 => "huge".repeat(2000 + (i % 500)), // variable huge
            _ => format!("string_{}", i),         // varied length based on number
        };
        strings.push(string);
    }

    let input = StringArray::from(strings);
    let compressor = LiquidByteViewArray::<FsstArray>::train_compressor(input.iter());
    let liquid_array = LiquidByteViewArray::<FsstArray>::from_string_array(&input, compressor);

    // Verify correctness
    let output = liquid_array.to_arrow_array();
    assert_eq!(&input, output.as_string::<i32>());

    // Test offset compression handles the stress case
    let offsets = liquid_array.fsst_buffer.offsets();

    // Verify offsets are monotonic
    for i in 1..offsets.len() {
        assert!(offsets[i] >= offsets[i - 1], "Offsets should be monotonic");
    }
}
