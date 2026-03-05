#[cfg(test)]
mod byte_view_tests {
    use std::sync::Arc;

    use arrow::array::{
        Array, AsArray, BinaryViewArray, BooleanArray, DictionaryArray, StringArray,
    };
    use arrow::buffer::BooleanBuffer;
    use arrow_schema::DataType;
    use datafusion::logical_expr::Operator;
    use datafusion::physical_plan::expressions::{BinaryExpr, Column, Literal};
    use datafusion::scalar::ScalarValue;
    use rand::prelude::*;
    use rand::SeedableRng;

    use crate::cache::{CacheExpression, TestSqueezeIo};
    use crate::liquid_array::raw::FsstArray;
    use crate::liquid_array::{LiquidArray, LiquidByteViewArray};

    fn make_byte_view(input: &StringArray) -> Arc<dyn LiquidArray> {
        let compressor = LiquidByteViewArray::<FsstArray>::train_compressor(input.iter());
        Arc::new(LiquidByteViewArray::<FsstArray>::from_string_array(
            input, compressor,
        ))
    }

    fn gen_random_string(rng: &mut StdRng, max_len: usize) -> String {
        let len = rng.random_range(0..=max_len);
        let mut out = String::new();
        for _ in 0..len {
            let ch = (rng.random_range(0x20u8..=0x7Eu8)) as char;
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
    fn randomized_utf8_roundtrip() {
        for seed in 0..50u64 {
            let mut rng = StdRng::seed_from_u64(0xC0FFEE + seed);
            let vals = gen_vec_opt_string(&mut rng, 64, 64);
            let input = StringArray::from(vals);
            let liquid = make_byte_view(&input);
            assert_eq!(liquid.to_arrow_array().as_string::<i32>(), &input);
        }
    }

    #[test]
    fn randomized_binaryview_roundtrip() {
        for seed in 0..50u64 {
            let mut rng = StdRng::seed_from_u64(0xB1A5E + seed);
            let vals = gen_vec_opt_bytes(&mut rng, 64, 64);
            let opt_slices: Vec<Option<&[u8]>> = vals.iter().map(|o| o.as_deref()).collect();
            let input = BinaryViewArray::from(opt_slices);
            let (_compressor, original) =
                LiquidByteViewArray::<FsstArray>::train_from_binary_view(&input);
            let output = original.to_arrow_array();
            assert_eq!(output.as_binary_view(), &input);
        }
    }

    #[test]
    fn to_dict_arrow_preserves_value_type() {
        let input_str = StringArray::from(vec!["hello", "world", "test"]);
        let (_c, bv) = LiquidByteViewArray::<FsstArray>::train_from_arrow(&input_str);
        let dict = bv.to_dict_arrow();
        assert_eq!(dict.values().data_type(), &DataType::Utf8);

        let input_bin = arrow::compute::cast(&input_str, &DataType::Binary)
            .unwrap()
            .as_binary::<i32>()
            .clone();
        let (_c, bv) = LiquidByteViewArray::<FsstArray>::train_from_arrow(&input_bin);
        let dict = bv.to_dict_arrow();
        assert_eq!(dict.values().data_type(), &DataType::Binary);

        let dict_array: DictionaryArray<arrow::datatypes::UInt16Type> =
            DictionaryArray::from_iter(input_str.iter());
        let (_c, bv) = LiquidByteViewArray::<FsstArray>::train_from_arrow_dict(&dict_array);
        let dict = bv.to_dict_arrow();
        assert_eq!(dict.values().data_type(), &DataType::Utf8);
    }

    #[test]
    fn to_bytes_and_from_bytes_roundtrip() {
        let input = StringArray::from(vec![
            Some("a"),
            None,
            Some("b"),
            Some("a"),
            Some("longer text"),
            Some(""),
        ]);

        let compressor = LiquidByteViewArray::<FsstArray>::train_compressor(input.iter());
        let original =
            LiquidByteViewArray::<FsstArray>::from_string_array(&input, compressor.clone());
        let bytes = original.to_bytes();
        let decoded = LiquidByteViewArray::<FsstArray>::from_bytes(bytes.into(), compressor);
        let output = decoded.to_arrow_array();
        assert_eq!(output.as_string::<i32>(), &input);
    }

    #[test]
    fn filter_even_indices() {
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
        let liquid = make_byte_view(&input);
        let filtered = liquid.filter(&mask).as_string::<i32>().clone();

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
        assert_eq!(filtered, StringArray::from(expected_vals));
    }

    #[test]
    fn predicate_eq() {
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
        let col: Arc<dyn datafusion::physical_plan::PhysicalExpr> = Arc::new(Column::new("c", 0));
        let expr: Arc<dyn datafusion::physical_plan::PhysicalExpr> =
            Arc::new(BinaryExpr::new(col, Operator::Eq, lit));

        let liquid = make_byte_view(&input);
        let result =
            liquid.try_eval_predicate(&crate::cache::LiquidExpr::new_unchecked(expr), &mask);
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

    #[test]
    fn squeeze_and_full_read_roundtrip() {
        let input = StringArray::from(vec![
            Some("hello"),
            Some("world"),
            Some("hello"),
            None,
            Some("byteview"),
        ]);
        let compressor = LiquidByteViewArray::<FsstArray>::train_compressor(input.iter());
        let decode_compressor = compressor.clone();
        let liquid = LiquidByteViewArray::<FsstArray>::from_string_array(&input, compressor);

        let baseline = liquid.to_bytes();
        let Some((_hybrid, bytes)) = liquid.squeeze(
            Arc::new(TestSqueezeIo::default()),
            Some(&CacheExpression::PredicateColumn),
        ) else {
            panic!("squeeze should succeed");
        };

        let restored = crate::liquid_array::ipc::read_from_bytes(
            bytes.clone(),
            &crate::liquid_array::ipc::LiquidIPCContext::new(Some(decode_compressor)),
        );

        let a1 = LiquidArray::to_arrow_array(&liquid);
        let a2 = restored.to_arrow_array();
        assert_eq!(a1.as_ref(), a2.as_ref());
        assert_eq!(baseline, restored.to_bytes());
    }
}
