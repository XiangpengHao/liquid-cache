#![no_main]

use arbitrary::{Arbitrary, Unstructured};
use arrow::array::cast::AsArray;
use arrow::array::{Array, StringArray};
use arrow::compute::kernels::cmp;
use arrow::error::ArrowError;
use datafusion::logical_expr::Operator;
use libfuzzer_sys::fuzz_target;
use liquid_cache_storage::liquid_array::LiquidByteViewArray;

#[derive(Debug, Clone, Arbitrary)]
struct FuzzInput {
    strings: Vec<Option<String>>,
    compare_operations: [CompareOperation; 5],
}

#[derive(Debug, Clone, Arbitrary)]
struct CompareOperation {
    needle: String,
    operator: FuzzOperator,
}

#[derive(Debug, Clone, Arbitrary)]
enum FuzzOperator {
    Eq,
    NotEq,
    Lt,
    LtEq,
    Gt,
    GtEq,
}

impl FuzzOperator {
    fn to_datafusion_operator(&self) -> Operator {
        match self {
            FuzzOperator::Eq => Operator::Eq,
            FuzzOperator::NotEq => Operator::NotEq,
            FuzzOperator::Lt => Operator::Lt,
            FuzzOperator::LtEq => Operator::LtEq,
            FuzzOperator::Gt => Operator::Gt,
            FuzzOperator::GtEq => Operator::GtEq,
        }
    }
}

fuzz_target!(|data: &[u8]| {
    let mut u = Unstructured::new(data);
    let input = match FuzzInput::arbitrary(&mut u) {
        Ok(input) => input,
        Err(_) => return,
    };

    if input.strings.is_empty() {
        return;
    }

    // Test roundtrip from and to arrow StringArray
    let (liquid_array, original_array) = test_roundtrip(&input.strings);

    // Test compare_with function equivalence
    test_compare_with(&liquid_array, &original_array, &input.compare_operations);
});

fn test_roundtrip(strings: &[Option<String>]) -> (LiquidByteViewArray, StringArray) {
    let original_array = StringArray::from(strings.to_vec());

    // Train compressor and create LiquidByteViewArray
    let compressor = LiquidByteViewArray::train_compressor(original_array.iter());
    let liquid_array = LiquidByteViewArray::from_string_array(&original_array, compressor);

    // Convert back to StringArray
    let roundtrip_array = liquid_array.to_arrow_array();
    let roundtrip_string_array = roundtrip_array.as_string::<i32>();

    assert_eq!(&original_array, roundtrip_string_array);

    (liquid_array, original_array)
}

fn test_compare_with(
    liquid_array: &LiquidByteViewArray,
    arrow_array: &StringArray,
    operations: &[CompareOperation],
) {
    for op in operations {
        let needle_bytes = op.needle.as_bytes();
        let operator = op.operator.to_datafusion_operator();

        // Get expected result from Arrow operations
        let arrow_result = compute_arrow_comparison(arrow_array, &op.needle, &operator);

        // Get result from LiquidByteViewArray
        let liquid_result = liquid_array.compare_with(needle_bytes, &operator);
        if let Ok(arrow_result) = arrow_result {
            assert_eq!(arrow_result, liquid_result.unwrap());
        } else {
            assert!(liquid_result.is_err());
        }
    }
}

fn compute_arrow_comparison(
    array: &StringArray,
    needle: &str,
    operator: &Operator,
) -> Result<arrow::array::BooleanArray, ArrowError> {
    let needle_array = StringArray::from(vec![needle; array.len()]);

    let result = match operator {
        Operator::Eq => cmp::eq(array, &needle_array)?,
        Operator::NotEq => cmp::neq(array, &needle_array)?,
        Operator::Lt => cmp::lt(array, &needle_array)?,
        Operator::LtEq => cmp::lt_eq(array, &needle_array)?,
        Operator::Gt => cmp::gt(array, &needle_array)?,
        Operator::GtEq => cmp::gt_eq(array, &needle_array)?,
        _ => unreachable!(),
    };

    Ok(result)
}
