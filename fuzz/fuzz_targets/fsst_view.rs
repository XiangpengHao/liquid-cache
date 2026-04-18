#![no_main]

use arbitrary::{Arbitrary, Unstructured};
use arrow::array::cast::AsArray;
use arrow::array::{Array, StringArray};
use arrow::compute::kernels::cmp;
use arrow::error::ArrowError;
use libfuzzer_sys::fuzz_target;
use liquid_cache::liquid_array::LiquidByteViewArray;
use liquid_cache::liquid_array::byte_view_array::{ByteViewOperator, Comparison, Equality};
use liquid_cache::liquid_array::raw::FsstArray;
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
    fn to_byte_view_operator(&self) -> ByteViewOperator {
        match self {
            FuzzOperator::Eq => ByteViewOperator::Equality(Equality::Eq),
            FuzzOperator::NotEq => ByteViewOperator::Equality(Equality::NotEq),
            FuzzOperator::Lt => ByteViewOperator::Comparison(Comparison::Lt),
            FuzzOperator::LtEq => ByteViewOperator::Comparison(Comparison::LtEq),
            FuzzOperator::Gt => ByteViewOperator::Comparison(Comparison::Gt),
            FuzzOperator::GtEq => ByteViewOperator::Comparison(Comparison::GtEq),
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

fn test_roundtrip(strings: &[Option<String>]) -> (LiquidByteViewArray<FsstArray>, StringArray) {
    let original_array = StringArray::from(strings.to_vec());

    // Train compressor and create LiquidByteViewArray
    let compressor = LiquidByteViewArray::<FsstArray>::train_compressor(original_array.iter());
    let liquid_array =
        LiquidByteViewArray::<FsstArray>::from_string_array(&original_array, compressor);

    // Convert back to StringArray
    let roundtrip_array = liquid_array.to_arrow_array();
    let roundtrip_string_array = roundtrip_array.as_string::<i32>();

    assert_eq!(&original_array, roundtrip_string_array);

    (liquid_array, original_array)
}

fn test_compare_with(
    liquid_array: &LiquidByteViewArray<FsstArray>,
    arrow_array: &StringArray,
    operations: &[CompareOperation],
) {
    for op in operations {
        let needle_bytes = op.needle.as_bytes();
        let operator = op.operator.to_byte_view_operator();

        // Get expected result from Arrow operations
        let arrow_result = compute_arrow_comparison(arrow_array, &op.needle, &op.operator);

        // Get result from LiquidByteViewArray
        let liquid_result = liquid_array.compare_with(needle_bytes, &operator);
        assert_eq!(arrow_result.unwrap(), liquid_result);
    }
}

fn compute_arrow_comparison(
    array: &StringArray,
    needle: &str,
    operator: &FuzzOperator,
) -> Result<arrow::array::BooleanArray, ArrowError> {
    let needle_array = StringArray::from(vec![needle; array.len()]);

    let result = match operator {
        FuzzOperator::Eq => cmp::eq(array, &needle_array)?,
        FuzzOperator::NotEq => cmp::neq(array, &needle_array)?,
        FuzzOperator::Lt => cmp::lt(array, &needle_array)?,
        FuzzOperator::LtEq => cmp::lt_eq(array, &needle_array)?,
        FuzzOperator::Gt => cmp::gt(array, &needle_array)?,
        FuzzOperator::GtEq => cmp::gt_eq(array, &needle_array)?,
    };

    Ok(result)
}
