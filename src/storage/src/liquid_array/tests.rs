#[cfg(test)]
mod random_tests {
    use std::sync::Arc;

    use crate::liquid_array::byte_view_array::MemoryBuffer;
    use crate::liquid_array::{LiquidArray, LiquidByteArray, LiquidByteViewArray};
    use arrow::array::{
        Array, AsArray, BinaryViewArray, BooleanArray, DictionaryArray, StringArray,
        StringViewArray,
    };
    use arrow::buffer::BooleanBuffer;
    use arrow_schema::DataType;
    use rand::SeedableRng;
    use rand::prelude::*;

    fn make_impls_from_strings(input: &StringArray) -> Vec<(&'static str, Arc<dyn LiquidArray>)> {
        let ba = {
            let compressor = LiquidByteArray::train_compressor(input.iter());
            Arc::new(LiquidByteArray::from_string_array(input, compressor)) as Arc<dyn LiquidArray>
        };
        let bv = {
            let compressor = LiquidByteViewArray::<MemoryBuffer>::train_compressor(input.iter());
            Arc::new(LiquidByteViewArray::<MemoryBuffer>::from_string_array(
                input, compressor,
            )) as Arc<dyn LiquidArray>
        };
        vec![("byte_array", ba), ("byte_view", bv)]
    }

    // ---------- Randomized input generators (deterministic via seeds) ----------
    fn gen_random_string(rng: &mut StdRng, max_len: usize) -> String {
        const EXTRA: &[char] = &[
            '\0', '\n', '\t', 'é', 'ß', '中', '日', '本', '🙂', '🚀', '🌍',
        ];
        let len = rng.random_range(0..=max_len);
        let mut out = String::new();
        for _ in 0..len {
            let pick_extra = rng.random_bool(0.2);
            let ch = if pick_extra {
                *EXTRA.choose(rng).unwrap()
            } else {
                // Printable ASCII range
                (rng.random_range(0x20u8..=0x7Eu8)) as char
            };
            out.push(ch);
        }
        out
    }

    fn gen_vec_opt_string(
        rng: &mut StdRng,
        max_items: usize,
        max_len: usize,
    ) -> Vec<Option<String>> {
        let n = rng.random_range(0..=max_items);
        (0..n)
            .map(|_| {
                if rng.random_bool(0.2) {
                    None
                } else {
                    Some(gen_random_string(rng, max_len))
                }
            })
            .collect()
    }

    fn gen_vec_opt_bytes(
        rng: &mut StdRng,
        max_items: usize,
        max_len: usize,
    ) -> Vec<Option<Vec<u8>>> {
        let n = rng.random_range(0..=max_items);
        (0..n)
            .map(|_| {
                if rng.random_bool(0.2) {
                    None
                } else {
                    let m = rng.random_range(0..=max_len);
                    let mut v = vec![0u8; m];
                    rng.fill_bytes(&mut v);
                    Some(v)
                }
            })
            .collect()
    }

    #[test]
    fn randomized_utf8_roundtrip_byte_and_view() {
        for seed in 0..50u64 {
            let mut rng = StdRng::seed_from_u64(0xC0FFEE + seed);
            let vals = gen_vec_opt_string(&mut rng, 64, 64);
            let input = StringArray::from(vals);
            for (_name, la) in make_impls_from_strings(&input) {
                assert_eq!(la.to_arrow_array().as_string::<i32>(), &input);
            }
        }
    }

    #[test]
    fn randomized_binaryview_roundtrip_byte_and_view() {
        for seed in 0..50u64 {
            let mut rng = StdRng::seed_from_u64(0xB1A5E + seed);
            let vals = gen_vec_opt_bytes(&mut rng, 64, 64);
            let opt_slices: Vec<Option<&[u8]>> = vals.iter().map(|o| o.as_deref()).collect();
            let input = BinaryViewArray::from(opt_slices);

            let (_compressor_ba, original_ba) = LiquidByteArray::train_from_binary_view(&input);
            assert_eq!(original_ba.to_arrow_array().as_binary_view(), &input);

            let (_compressor_bv, original_bv) =
                LiquidByteViewArray::<MemoryBuffer>::train_from_binary_view(&input);
            let output_bv = original_bv.to_arrow_array().expect("InMemoryFsstBuffer");
            assert_eq!(output_bv.as_binary_view(), &input);
        }
    }

    #[test]
    fn randomized_stringview_roundtrip_byte_and_view() {
        for seed in 0..50u64 {
            let mut rng = StdRng::seed_from_u64(0x53_54_52_49 + seed);
            let vals = gen_vec_opt_string(&mut rng, 64, 64);
            let opt_slices: Vec<Option<&str>> = vals.iter().map(|o| o.as_deref()).collect();
            let input = StringViewArray::from(opt_slices);

            let (_compressor_ba, original_ba) = LiquidByteArray::train_from_string_view(&input);
            assert_eq!(original_ba.to_arrow_array().as_string_view(), &input);

            let (_compressor_bv, original_bv) =
                LiquidByteViewArray::<MemoryBuffer>::train_from_string_view(&input);
            let output_bv = original_bv.to_arrow_array().expect("InMemoryFsstBuffer");
            assert_eq!(output_bv.as_string_view(), &input);
        }
    }

    #[test]
    fn randomized_to_bytes_and_from_bytes_roundtrip() {
        for seed in 0..50u64 {
            let mut rng = StdRng::seed_from_u64(0xB17E5 + seed);
            let vals = gen_vec_opt_string(&mut rng, 64, 64);
            let input = StringArray::from(vals);

            // ByteArray
            {
                let compressor = LiquidByteArray::train_compressor(input.iter());
                let original = LiquidByteArray::from_string_array(&input, compressor.clone());
                let bytes = original.to_bytes();
                let decoded = LiquidByteArray::from_bytes(bytes.into(), compressor);
                assert_eq!(decoded.to_arrow_array().as_string::<i32>(), &input);
            }

            // ByteViewArray
            {
                let compressor =
                    LiquidByteViewArray::<MemoryBuffer>::train_compressor(input.iter());
                let original = LiquidByteViewArray::<MemoryBuffer>::from_string_array(
                    &input,
                    compressor.clone(),
                );
                let bytes = original.to_bytes();
                let decoded =
                    LiquidByteViewArray::<MemoryBuffer>::from_bytes(bytes.into(), compressor);
                let dec = decoded.to_arrow_array().expect("InMemory");
                assert_eq!(dec.as_string::<i32>(), &input);
            }
        }
    }

    #[test]
    fn randomized_filter_shared() {
        for seed in 0..50u64 {
            let mut rng = StdRng::seed_from_u64(0xF1_7E_E0 + seed);
            let vals = gen_vec_opt_string(&mut rng, 64, 64);
            let input = StringArray::from(vals);
            let mask_bits: Vec<bool> = (0..input.len()).map(|_| rng.random()).collect();
            let mask = BooleanBuffer::from_iter(mask_bits.iter().copied());

            for (_name, la) in make_impls_from_strings(&input) {
                let arr = la.filter(&mask).as_string::<i32>().clone();

                let expected_vals: Vec<Option<&str>> = (0..input.len())
                    .zip(mask_bits.iter())
                    .filter_map(|(i, &keep)| if keep { Some(i) } else { None })
                    .map(|i| {
                        if input.is_null(i) {
                            None
                        } else {
                            Some(input.value(i))
                        }
                    })
                    .collect();
                assert_eq!(arr, StringArray::from(expected_vals));
            }
        }
    }

    #[test]
    fn randomized_predicate_eq_shared() {
        use datafusion::logical_expr::Operator;
        use datafusion::physical_plan::expressions::{BinaryExpr, Literal};
        use datafusion::scalar::ScalarValue;

        for seed in 0..50u64 {
            let mut rng = StdRng::seed_from_u64(0xE0 + seed);
            let vals = gen_vec_opt_string(&mut rng, 64, 64);
            let needle = gen_random_string(&mut rng, 32);
            let input = StringArray::from(vals);
            let mask = BooleanBuffer::new_set(input.len());
            let lit: Arc<dyn datafusion::physical_plan::PhysicalExpr> =
                Arc::new(Literal::new(ScalarValue::Utf8(Some(needle.clone()))));
            let expr: Arc<dyn datafusion::physical_plan::PhysicalExpr> =
                Arc::new(BinaryExpr::new(lit.clone(), Operator::Eq, lit));

            for (_name, la) in make_impls_from_strings(&input) {
                if let Some(result) = la.try_eval_predicate(&expr, &mask) {
                    let expected: Vec<Option<bool>> =
                        input.iter().map(|o| o.map(|s| s == needle)).collect();
                    assert_eq!(result, BooleanArray::from(expected));
                }
            }
        }
    }

    #[test]
    fn randomized_byte_view_squeeze_and_soak_roundtrip() {
        for seed in 0..25u64 {
            // slightly fewer to keep runtime reasonable
            let mut rng = StdRng::seed_from_u64(0x50_55_45_33 + seed);
            let vals = gen_vec_opt_string(&mut rng, 32, 64);
            let input = StringArray::from(vals);
            let compressor = LiquidByteViewArray::<MemoryBuffer>::train_compressor(input.iter());
            let liquid = LiquidByteViewArray::<MemoryBuffer>::from_string_array(&input, compressor);

            let baseline = liquid.to_bytes();

            if let Some((hybrid, bytes)) = liquid.squeeze(None) {
                let range = hybrid.to_liquid().range;
                assert!(range.start < range.end);
                assert!((range.end as usize) <= bytes.len());

                let fsst_bytes = bytes.slice(range.start as usize..range.end as usize);
                let restored = hybrid.soak(fsst_bytes);

                let a1 = LiquidArray::to_arrow_array(&liquid);
                let a2 = restored.to_arrow_array();
                assert_eq!(a1.as_ref(), a2.as_ref());
                assert_eq!(baseline, restored.to_bytes());
            }
        }
    }

    #[test]
    fn utf8_roundtrip_byte_and_view() {
        let cases: Vec<Vec<Option<&str>>> = vec![
            vec![
                Some("hello"),
                Some("world"),
                None,
                Some("liquid"),
                Some("byte"),
                Some("array"),
                Some("hello"),
                Some(""),
            ],
            vec![
                Some("This is a very long string that should be compressed well"),
                Some("Another long string with some common patterns"),
                Some("This is a very long string that should be compressed well"),
                Some("Some unique text here to mix things up"),
                Some("Another long string with some common patterns"),
            ],
            vec![
                Some("こんにちは"),
                Some("世界"),
                None,
                Some("rust"),
                Some(""),
                Some("🌍"),
            ],
        ];

        for vals in cases {
            let input = StringArray::from(vals);
            for (_name, la) in make_impls_from_strings(&input) {
                assert_eq!(la.to_arrow_array().as_string::<i32>(), &input);
            }
        }
    }

    #[test]
    fn binaryview_roundtrip_byte_and_view() {
        let input = BinaryViewArray::from(vec![
            Some(b"hello".as_slice()),
            Some(b"world".as_slice()),
            Some(b"hello".as_slice()),
            Some(b"rust\x00".as_slice()),
            None,
            Some(b"This is a very long string that should be compressed well"),
            Some(b""),
            Some(b"This is a very long string that should be compressed well"),
        ]);

        // LiquidByteArray via BinaryView
        let (_compressor_ba, original_ba) = LiquidByteArray::train_from_binary_view(&input);
        let output_ba = original_ba.to_arrow_array();
        assert_eq!(output_ba.as_binary_view(), &input);

        // LiquidByteViewArray via BinaryView
        let (_compressor_bv, original_bv) =
            LiquidByteViewArray::<MemoryBuffer>::train_from_binary_view(&input);
        let output_bv = original_bv.to_arrow_array().expect("InMemoryFsstBuffer");
        assert_eq!(output_bv.as_binary_view(), &input);
    }

    #[test]
    fn stringview_roundtrip_byte_and_view() {
        let input = StringViewArray::from(vec![
            Some("hello"),
            Some("world"),
            Some("hello"),
            Some("rust"),
            None,
            Some("long text"),
            Some(""),
        ]);

        // ByteArray via StringView
        let (_compressor_ba, original_ba) = LiquidByteArray::train_from_string_view(&input);
        let output_ba = original_ba.to_arrow_array();
        assert_eq!(output_ba.as_string_view(), &input);

        // ByteViewArray via StringView
        let (_compressor_bv, original_bv) =
            LiquidByteViewArray::<MemoryBuffer>::train_from_string_view(&input);
        let output_bv = original_bv.to_arrow_array().expect("InMemoryFsstBuffer");
        assert_eq!(output_bv.as_string_view(), &input);
    }

    #[test]
    fn to_dict_arrow_preserves_value_type_shared() {
        // String values
        let input_str = StringArray::from(vec!["hello", "world", "test"]);
        {
            let (_c, ba) = LiquidByteArray::train_from_arrow(&input_str);
            let dict = ba.to_dict_arrow();
            assert_eq!(dict.values().data_type(), &DataType::Utf8);
        }
        {
            let (_c, bv) = LiquidByteViewArray::<MemoryBuffer>::train_from_arrow(&input_str);
            let dict = bv.to_dict_arrow().expect("InMemory");
            assert_eq!(dict.values().data_type(), &DataType::Utf8);
        }

        // Binary values
        let input_bin = arrow::compute::cast(&input_str, &DataType::Binary)
            .unwrap()
            .as_binary::<i32>()
            .clone();
        {
            let (_c, ba) = LiquidByteArray::train_from_arrow(&input_bin);
            let dict = ba.to_dict_arrow();
            assert_eq!(dict.values().data_type(), &DataType::Binary);
        }
        {
            let (_c, bv) = LiquidByteViewArray::<MemoryBuffer>::train_from_arrow(&input_bin);
            let dict = bv.to_dict_arrow().expect("InMemory");
            assert_eq!(dict.values().data_type(), &DataType::Binary);
        }

        // Dictionary<String> values
        let dict_array: DictionaryArray<arrow::datatypes::UInt16Type> =
            DictionaryArray::from_iter(input_str.iter());
        {
            let (_c, ba) = LiquidByteArray::train_from_arrow_dict(&dict_array);
            let dict = ba.to_dict_arrow();
            assert_eq!(dict.values().data_type(), &DataType::Utf8);
        }
        {
            let (_c, bv) = LiquidByteViewArray::<MemoryBuffer>::train_from_arrow_dict(&dict_array);
            let dict = bv.to_dict_arrow().expect("InMemory");
            assert_eq!(dict.values().data_type(), &DataType::Utf8);
        }
    }

    #[test]
    fn compare_equals_shared() {
        let cases = vec![
            (
                vec![Some("hello"), Some("world"), Some("hello"), Some("rust")],
                "hello",
                vec![Some(true), Some(false), Some(true), Some(false)],
            ),
            (
                vec![Some("hello"), Some("world"), Some("hello"), Some("rust")],
                "nonexistent",
                vec![Some(false), Some(false), Some(false), Some(false)],
            ),
            (
                vec![Some("hello"), None, Some("hello"), None, Some("world")],
                "hello",
                vec![Some(true), None, Some(true), None, Some(false)],
            ),
            (
                vec![Some(""), Some("hello"), Some(""), Some("world")],
                "",
                vec![Some(true), Some(false), Some(true), Some(false)],
            ),
        ];

        for (vals, needle, expected) in cases {
            let input = StringArray::from(vals);
            // Build a predicate: array == literal(needle)
            use datafusion::logical_expr::Operator;
            use datafusion::physical_plan::expressions::{BinaryExpr, Literal};
            use datafusion::scalar::ScalarValue;
            let mask = BooleanBuffer::new_set(input.len());
            let lit: Arc<dyn datafusion::physical_plan::PhysicalExpr> =
                Arc::new(Literal::new(ScalarValue::Utf8(Some(needle.to_string()))));
            let expr: Arc<dyn datafusion::physical_plan::PhysicalExpr> =
                Arc::new(BinaryExpr::new(lit.clone(), Operator::Eq, lit));

            for (_name, la) in make_impls_from_strings(&input) {
                if let Some(result) = la.try_eval_predicate(&expr, &mask) {
                    assert_eq!(result, BooleanArray::from(expected.clone()));
                }
            }
        }
    }

    #[test]
    fn to_bytes_and_from_bytes_roundtrip_shared() {
        let input = StringArray::from(vec![
            Some("a"),
            None,
            Some("b"),
            Some("a"),
            Some("longer text"),
            Some(""),
        ]);

        // ByteArray
        {
            let compressor = LiquidByteArray::train_compressor(input.iter());
            let original = LiquidByteArray::from_string_array(&input, compressor.clone());
            let bytes = original.to_bytes();
            let decoded = LiquidByteArray::from_bytes(bytes.into(), compressor);
            assert_eq!(decoded.to_arrow_array().as_string::<i32>(), &input);
        }

        // ByteViewArray
        {
            let compressor = LiquidByteViewArray::<MemoryBuffer>::train_compressor(input.iter());
            let original =
                LiquidByteViewArray::<MemoryBuffer>::from_string_array(&input, compressor.clone());
            let bytes = original.to_bytes();
            let decoded = LiquidByteViewArray::<MemoryBuffer>::from_bytes(bytes.into(), compressor);
            let dec = decoded.to_arrow_array().expect("InMemory");
            let ori = original.to_arrow_array().expect("InMemory");
            assert_eq!(dec.as_ref(), ori.as_ref());
        }
    }

    #[test]
    fn filter_shared_even_indices() {
        let input = StringArray::from(vec![
            Some("x"),
            Some("y"),
            None,
            Some("z"),
            Some("x"),
            Some("y"),
            Some("z"),
        ]);
        let mask = BooleanBuffer::from_iter((0..input.len()).map(|i| i.is_multiple_of(2)));

        for (_name, la) in make_impls_from_strings(&input) {
            let arr = la.filter(&mask).as_string::<i32>().clone();
            // Build expected via filtering input directly
            let expected_vals: Vec<Option<&str>> = (0..input.len())
                .filter(|i| i.is_multiple_of(2))
                .map(|i| {
                    if input.is_null(i) {
                        None
                    } else {
                        Some(input.value(i))
                    }
                })
                .collect();
            assert_eq!(arr, StringArray::from(expected_vals));
        }
    }

    #[test]
    fn predicate_eq_shared() {
        use datafusion::logical_expr::Operator;
        use datafusion::physical_plan::expressions::{BinaryExpr, Literal};
        use datafusion::scalar::ScalarValue;
        let input = StringArray::from(vec![
            Some("hello"),
            None,
            Some("world"),
            Some("hello"),
            Some(""),
            Some("rust"),
        ]);
        let mask = BooleanBuffer::new_set(input.len());
        let lit: Arc<dyn datafusion::physical_plan::PhysicalExpr> =
            Arc::new(Literal::new(ScalarValue::Utf8(Some("hello".to_string()))));
        let expr: Arc<dyn datafusion::physical_plan::PhysicalExpr> =
            Arc::new(BinaryExpr::new(lit.clone(), Operator::Eq, lit));

        for (_name, la) in make_impls_from_strings(&input) {
            if let Some(result) = la.try_eval_predicate(&expr, &mask) {
                // Build expected
                let expected = BooleanArray::from(vec![
                    Some(true),
                    None,
                    Some(false),
                    Some(true),
                    Some(false),
                    Some(false),
                ]);
                assert_eq!(result, expected);
            }
        }
    }

    #[test]
    fn byte_view_squeeze_and_soak_roundtrip() {
        // Build a small array
        let input = StringArray::from(vec![
            Some("hello"),
            Some("world"),
            Some("hello"),
            None,
            Some("byteview"),
        ]);
        let compressor = LiquidByteViewArray::<MemoryBuffer>::train_compressor(input.iter());
        let liquid = LiquidByteViewArray::<MemoryBuffer>::from_string_array(&input, compressor);

        // Full IPC bytes as baseline
        let baseline = liquid.to_bytes();

        // Squeeze
        let Some((hybrid, bytes)) = liquid.squeeze(None) else {
            panic!("squeeze should succeed");
        };

        let range = hybrid.to_liquid().range;
        // Sanity: range bounds are valid
        assert!(range.start < range.end);
        assert!(range.end as usize <= bytes.len());

        // Soak back to memory with raw FSST bytes
        let fsst_bytes = bytes.slice(range.start as usize..range.end as usize);
        let restored = hybrid.soak(fsst_bytes);

        // Arrow equality check
        let a1 = LiquidArray::to_arrow_array(&liquid);
        let a2 = restored.to_arrow_array();
        assert_eq!(a1.as_ref(), a2.as_ref());

        // IPC bytes should match as well
        assert_eq!(baseline, restored.to_bytes());
    }
}
