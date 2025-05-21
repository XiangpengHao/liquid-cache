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
            builder.append_value(value as i64);
        }
    }
    let array = builder.finish();
    cast(&array, &data_type)
        .unwrap()
        .as_primitive::<T>()
        .clone()
}
