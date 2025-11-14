//! Credit: this is copied from <https://github.com/datafusion-contrib/datafusion-variant/blob/main/src/variant_get.rs>
//! Full credit to the original authors.
//! We need to copy it here because we have different datafusion versions, and can't easily include that crate in our workspace.
//! But eventually, we will use whatever the official datafusion has to offer.

use std::sync::Arc;

use arrow::{
    array::{Array, ArrayRef, AsArray, StringViewArray, StructArray},
    compute::concat,
};
use arrow_schema::{DataType, Field, FieldRef, Fields, extension::ExtensionType};
use datafusion::{
    common::{exec_datafusion_err, exec_err},
    error::{DataFusionError, Result},
    logical_expr::{
        ColumnarValue, ReturnFieldArgs, ScalarFunctionArgs, ScalarUDFImpl, Signature,
        TypeSignature, Volatility,
    },
    scalar::ScalarValue,
};
use parquet::variant::VariantPath;
use parquet::variant::{GetOptions, VariantArray, VariantType, variant_get};
use parquet_variant_json::VariantToJson;

pub fn try_field_as_variant_array(field: &Field) -> Result<()> {
    assert!(
        matches!(field.extension_type(), VariantType),
        "field does not have extension type VariantType"
    );

    let variant_type = VariantType;
    variant_type.supports_data_type(field.data_type())?;

    Ok(())
}

pub fn _try_field_as_binary(field: &Field) -> Result<()> {
    match field.data_type() {
        DataType::Binary | DataType::BinaryView | DataType::LargeBinary => {}
        unsupported => return exec_err!("expected binary field, got {unsupported} field"),
    }

    Ok(())
}

pub fn try_parse_string_columnar(array: &Arc<dyn Array>) -> Result<Vec<Option<&str>>> {
    if let Some(string_array) = array.as_string_opt::<i32>() {
        return Ok(string_array.into_iter().collect::<Vec<_>>());
    }

    if let Some(string_view_array) = array.as_string_view_opt() {
        return Ok(string_view_array.into_iter().collect::<Vec<_>>());
    }

    if let Some(large_string_array) = array.as_string_opt::<i64>() {
        return Ok(large_string_array.into_iter().collect::<Vec<_>>());
    }

    Err(exec_datafusion_err!("expected string array"))
}

pub fn try_parse_string_scalar(scalar: &ScalarValue) -> Result<Option<&String>> {
    let b = match scalar {
        ScalarValue::Utf8(s) | ScalarValue::Utf8View(s) | ScalarValue::LargeUtf8(s) => s,
        unsupported => {
            return exec_err!(
                "expected binary scalar value, got data type: {}",
                unsupported.data_type()
            );
        }
    };

    Ok(b.as_ref())
}

fn parse_type_hint(spec: &str) -> Result<DataType> {
    if let Ok(data_type) = spec.parse::<DataType>() {
        Ok(data_type)
    } else {
        exec_err!("invalid type hint: {spec}")
    }
}

fn type_hint_from_scalar(field_name: &str, scalar: &ScalarValue) -> Result<FieldRef> {
    let type_name = match scalar {
        ScalarValue::Utf8(Some(value))
        | ScalarValue::Utf8View(Some(value))
        | ScalarValue::LargeUtf8(Some(value)) => value.as_str(),
        other => {
            return exec_err!(
                "type hint must be a non-null UTF8 literal, got {}",
                other.data_type()
            );
        }
    };

    let data_type = parse_type_hint(type_name)?;
    Ok(Arc::new(Field::new(field_name, data_type, true)))
}

fn type_hint_from_value(field_name: &str, arg: &ColumnarValue) -> Result<FieldRef> {
    match arg {
        ColumnarValue::Scalar(value) => type_hint_from_scalar(field_name, value),
        ColumnarValue::Array(_) => {
            exec_err!("type hint argument must be a scalar UTF8 literal")
        }
    }
}

fn build_get_options<'a>(path: VariantPath<'a>, as_type: &Option<FieldRef>) -> GetOptions<'a> {
    match as_type {
        Some(field) => GetOptions::new_with_path(path).with_as_type(Some(field.clone())),
        None => GetOptions::new_with_path(path),
    }
}

/// UDF for getting a variant from a variant array or scalar.
#[derive(Debug, Hash, PartialEq, Eq)]
pub struct VariantGetUdf {
    signature: Signature,
}

impl Default for VariantGetUdf {
    fn default() -> Self {
        Self {
            signature: Signature::new(
                TypeSignature::OneOf(vec![TypeSignature::Any(2), TypeSignature::Any(3)]),
                Volatility::Immutable,
            ),
        }
    }
}

impl ScalarUDFImpl for VariantGetUdf {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn name(&self) -> &str {
        "variant_get"
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn return_type(&self, _arg_types: &[arrow_schema::DataType]) -> Result<arrow_schema::DataType> {
        Err(DataFusionError::Internal(
            "implemented return_field_from_args instead".into(),
        ))
    }

    fn return_field_from_args(&self, args: ReturnFieldArgs) -> Result<Arc<Field>> {
        if let Some(maybe_scalar) = args.scalar_arguments.get(2) {
            let scalar = maybe_scalar.ok_or_else(|| {
                exec_datafusion_err!("type hint argument to variant_get must be a literal")
            })?;
            return type_hint_from_scalar(self.name(), scalar);
        }

        let data_type = DataType::Struct(Fields::from(vec![
            Field::new("metadata", DataType::BinaryView, false),
            Field::new("value", DataType::BinaryView, true),
        ]));

        Ok(Arc::new(
            Field::new(self.name(), data_type, true).with_extension_type(VariantType),
        ))
    }

    fn invoke_with_args(
        &self,
        args: datafusion::logical_expr::ScalarFunctionArgs,
    ) -> Result<ColumnarValue> {
        let (variant_arg, variant_path, type_arg) = match args.args.as_slice() {
            [variant_arg, variant_path] => (variant_arg, variant_path, None),
            [variant_arg, variant_path, type_arg] => (variant_arg, variant_path, Some(type_arg)),
            _ => return exec_err!("expected 2 or 3 arguments"),
        };

        let variant_field = args
            .arg_fields
            .first()
            .ok_or_else(|| exec_datafusion_err!("expected argument field"))?;

        try_field_as_variant_array(variant_field.as_ref())?;

        let type_field = type_arg
            .map(|arg| type_hint_from_value(self.name(), arg))
            .transpose()?;

        let out = match (variant_arg, variant_path) {
            (ColumnarValue::Array(variant_array), ColumnarValue::Scalar(variant_path)) => {
                let variant_path = try_parse_string_scalar(variant_path)?
                    .map(|s| s.as_str())
                    .unwrap_or_default();

                let res = variant_get(
                    variant_array,
                    build_get_options(VariantPath::from(variant_path), &type_field),
                )?;

                ColumnarValue::Array(res)
            }
            (ColumnarValue::Scalar(scalar_variant), ColumnarValue::Scalar(variant_path)) => {
                let ScalarValue::Struct(variant_array) = scalar_variant else {
                    return exec_err!("expected struct array");
                };

                let variant_array = Arc::clone(variant_array) as ArrayRef;

                let variant_path = try_parse_string_scalar(variant_path)?
                    .map(|s| s.as_str())
                    .unwrap_or_default();

                let res = variant_get(
                    &variant_array,
                    build_get_options(VariantPath::from(variant_path), &type_field),
                )?;

                let scalar = ScalarValue::try_from_array(res.as_ref(), 0)?;
                ColumnarValue::Scalar(scalar)
            }
            (ColumnarValue::Array(variant_array), ColumnarValue::Array(variant_paths)) => {
                if variant_array.len() != variant_paths.len() {
                    return exec_err!(
                        "expected variant_array and variant paths to be of same length"
                    );
                }

                let variant_paths = try_parse_string_columnar(variant_paths)?;
                let variant_array = VariantArray::try_new(variant_array.as_ref())?;

                let mut out = Vec::with_capacity(variant_array.len());

                for (i, path) in variant_paths.iter().enumerate() {
                    let v = variant_array.value(i);
                    // todo: is there a better way to go from Variant -> VariantArray?
                    let singleton_variant_array: StructArray = VariantArray::from_iter([v]).into();

                    let arr = Arc::new(singleton_variant_array) as ArrayRef;

                    let res = variant_get(
                        &arr,
                        build_get_options(VariantPath::from(path.unwrap_or_default()), &type_field),
                    )?;

                    out.push(res);
                }

                let out_refs: Vec<&dyn Array> = out.iter().map(|a| a.as_ref()).collect();
                ColumnarValue::Array(concat(&out_refs)?)
            }
            (ColumnarValue::Scalar(scalar_variant), ColumnarValue::Array(variant_paths)) => {
                let ScalarValue::Struct(variant_array) = scalar_variant else {
                    return exec_err!("expected struct array");
                };

                let variant_array = Arc::clone(variant_array) as ArrayRef;
                let variant_paths = try_parse_string_columnar(variant_paths)?;

                let mut out = Vec::with_capacity(variant_paths.len());

                for path in variant_paths {
                    let path = path.unwrap_or_default();
                    let res = variant_get(
                        &variant_array,
                        build_get_options(VariantPath::from(path), &type_field),
                    )?;

                    out.push(res);
                }

                let out_refs: Vec<&dyn Array> = out.iter().map(|a| a.as_ref()).collect();
                ColumnarValue::Array(concat(&out_refs)?)
            }
        };

        Ok(out)
    }
}

/// Returns a pretty-printed JSON string from a VariantArray
#[derive(Debug, Hash, PartialEq, Eq)]
pub struct VariantPretty {
    signature: Signature,
}

impl Default for VariantPretty {
    fn default() -> Self {
        Self {
            signature: Signature::new(TypeSignature::Any(1), Volatility::Immutable),
        }
    }
}

impl ScalarUDFImpl for VariantPretty {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn name(&self) -> &str {
        "variant_pretty"
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> {
        Ok(DataType::Utf8View)
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        let field = args
            .arg_fields
            .first()
            .ok_or_else(|| exec_datafusion_err!("empty argument, expected 1 argument"))?;

        try_field_as_variant_array(field.as_ref())?;

        let arg = args
            .args
            .first()
            .ok_or_else(|| exec_datafusion_err!("empty argument, expected 1 argument"))?;

        let out = match arg {
            ColumnarValue::Scalar(scalar) => {
                let ScalarValue::Struct(variant_array) = scalar else {
                    return exec_err!("Unsupported data type: {}", scalar.data_type());
                };

                let variant_array = VariantArray::try_new(variant_array.as_ref())?;
                let v = variant_array.value(0);

                ColumnarValue::Scalar(ScalarValue::Utf8View(Some(format!("{:?}", v))))
            }
            ColumnarValue::Array(arr) => match arr.data_type() {
                DataType::Struct(_) => {
                    let variant_array = VariantArray::try_new(arr.as_ref())?;

                    let out = variant_array
                        .iter()
                        .map(|v| v.map(|v| format!("{:?}", v)))
                        .collect::<Vec<_>>();

                    let out: StringViewArray = out.into();

                    ColumnarValue::Array(Arc::new(out))
                }
                unsupported => return exec_err!("Invalid data type: {unsupported}"),
            },
        };

        Ok(out)
    }
}

/// Returns a JSON string from a VariantArray
///
/// ## Arguments
/// - expr: a DataType::Struct expression that represents a VariantArray
/// - options: an optional MAP (note, it seems arrow-rs' parquet-variant is pretty restrictive about the options)
#[derive(Debug, Hash, PartialEq, Eq)]
pub struct VariantToJsonUdf {
    signature: Signature,
}

impl Default for VariantToJsonUdf {
    fn default() -> Self {
        Self {
            signature: Signature::new(
                TypeSignature::OneOf(vec![TypeSignature::Any(1), TypeSignature::Any(2)]),
                Volatility::Immutable,
            ),
        }
    }
}

impl ScalarUDFImpl for VariantToJsonUdf {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn name(&self) -> &str {
        "variant_to_json"
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> {
        Ok(DataType::Utf8View)
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        let field = args
            .arg_fields
            .first()
            .ok_or_else(|| exec_datafusion_err!("empty argument, expected 1 argument"))?;

        try_field_as_variant_array(field.as_ref())?;

        let arg = args
            .args
            .first()
            .ok_or_else(|| exec_datafusion_err!("empty argument, expected 1 argument"))?;

        let out = match arg {
            ColumnarValue::Scalar(scalar) => {
                let ScalarValue::Struct(variant_array) = scalar else {
                    return exec_err!("Unsupported data type: {}", scalar.data_type());
                };

                let variant_array = VariantArray::try_new(variant_array.as_ref())?;
                let v = variant_array.value(0);

                ColumnarValue::Scalar(ScalarValue::Utf8View(Some(v.to_json_string()?)))
            }
            ColumnarValue::Array(arr) => match arr.data_type() {
                DataType::Struct(_) => {
                    let variant_array = VariantArray::try_new(arr.as_ref())?;

                    let out: StringViewArray = variant_array
                        .iter()
                        .map(|v| v.map(|v| v.to_json_string()).transpose())
                        .collect::<Result<Vec<_>, _>>()?
                        .into();

                    ColumnarValue::Array(Arc::new(out))
                }
                unsupported => return exec_err!("Invalid data type: {unsupported}"),
            },
        };

        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use arrow::array::{Array, BinaryViewArray};
    use arrow_schema::{Field, Fields};
    use datafusion::logical_expr::{ReturnFieldArgs, ScalarFunctionArgs};
    use parquet::variant::Variant;
    use parquet::variant::{VariantArrayBuilder, VariantType};
    use parquet_variant_json::JsonToVariant;

    use super::*;

    #[test]
    fn test_get_variant_scalar() {
        let expected_json = serde_json::json!({
            "name": "norm",
            "age": 50,
            "list": [false, true, ()]
        });

        let json_str = expected_json.to_string();
        let mut builder = VariantArrayBuilder::new(1);
        builder.append_json(json_str.as_str()).unwrap();

        let input = builder.build().into();

        let variant_input = ScalarValue::Struct(Arc::new(input));
        let path = "name";

        let udf = VariantGetUdf::default();

        let arg_field = Arc::new(
            Field::new("input", DataType::Struct(Fields::empty()), true)
                .with_extension_type(VariantType),
        );
        let arg_field2 = Arc::new(Field::new("path", DataType::Utf8, true));

        let return_field = udf
            .return_field_from_args(ReturnFieldArgs {
                arg_fields: &[arg_field.clone(), arg_field2.clone()],
                scalar_arguments: &[],
            })
            .unwrap();

        let args = ScalarFunctionArgs {
            args: vec![
                ColumnarValue::Scalar(variant_input),
                ColumnarValue::Scalar(ScalarValue::Utf8(Some(path.to_string()))),
            ],
            return_field,
            arg_fields: vec![arg_field],
            number_rows: Default::default(),
            config_options: Default::default(),
        };

        let result = udf.invoke_with_args(args).unwrap();

        let ColumnarValue::Scalar(ScalarValue::Struct(struct_arr)) = result else {
            panic!("expected ScalarValue struct");
        };

        assert_eq!(struct_arr.len(), 1);

        let metadata_arr = struct_arr
            .column(0)
            .as_any()
            .downcast_ref::<BinaryViewArray>()
            .unwrap();
        let value_arr = struct_arr
            .column(1)
            .as_any()
            .downcast_ref::<BinaryViewArray>()
            .unwrap();

        let metadata = metadata_arr.value(0);
        let value = value_arr.value(0);

        let v = Variant::try_new(metadata, value).unwrap();

        assert_eq!(v, Variant::from("norm"))
    }

    #[test]
    fn test_get_variant_scalar_typed() {
        let expected_json = serde_json::json!({
            "name": "norm",
            "age": 50,
            "list": [false, true, ()]
        });

        let json_str = expected_json.to_string();
        let mut builder = VariantArrayBuilder::new(1);
        builder.append_json(json_str.as_str()).unwrap();

        let input = builder.build().into();

        let variant_input = ScalarValue::Struct(Arc::new(input));
        let path = "name";

        let udf = VariantGetUdf::default();

        let arg_field = Arc::new(
            Field::new("input", DataType::Struct(Fields::empty()), true)
                .with_extension_type(VariantType),
        );
        let arg_field2 = Arc::new(Field::new("path", DataType::Utf8, true));
        let arg_field3 = Arc::new(Field::new("type_hint", DataType::Utf8, true));

        let path_scalar = ScalarValue::Utf8(Some(path.to_string()));
        let type_hint = ScalarValue::Utf8(Some("Utf8".to_string()));
        let scalar_arguments: [Option<&ScalarValue>; 3] =
            [None, Some(&path_scalar), Some(&type_hint)];

        let return_field = udf
            .return_field_from_args(ReturnFieldArgs {
                arg_fields: &[arg_field.clone(), arg_field2, arg_field3],
                scalar_arguments: &scalar_arguments,
            })
            .unwrap();
        assert_eq!(return_field.data_type(), &DataType::Utf8);

        let args = ScalarFunctionArgs {
            args: vec![
                ColumnarValue::Scalar(variant_input),
                ColumnarValue::Scalar(path_scalar.clone()),
                ColumnarValue::Scalar(type_hint.clone()),
            ],
            return_field,
            arg_fields: vec![arg_field],
            number_rows: Default::default(),
            config_options: Default::default(),
        };

        let result = udf.invoke_with_args(args).unwrap();

        let ColumnarValue::Scalar(ScalarValue::Utf8(value)) = result else {
            panic!("expected Utf8 scalar");
        };

        assert_eq!(value.as_deref(), Some("norm"));
    }

    #[test]
    fn test_parse_type_hint_invalid() {
        let err = parse_type_hint("uuid").unwrap_err();
        assert!(err.to_string().contains("unsupported type hint"));
    }
}
