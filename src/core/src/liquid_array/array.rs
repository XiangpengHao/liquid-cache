use std::mem::size_of;
use std::sync::LazyLock;

use arrow::array::{Array, ArrayRef, BooleanArray};
use arrow::buffer::{BooleanBuffer, NullBuffer};
use arrow_schema::{DataType, Field};
use bytes::Bytes;
use datafusion_common::ScalarValue;
use datafusion_expr_common::operator::Operator as DataFusionOperator;
use datafusion_physical_expr::expressions::{BinaryExpr, Column, LikeExpr, Literal};
use vortex_array::arrays::ConstantArray;
use vortex_array::builtins::ArrayBuiltins;
use vortex_array::dtype::DType;
use vortex_array::scalar_fn::fns::like::{Like, LikeOptions};
use vortex_array::scalar_fn::fns::operators::Operator;
use vortex_array::{IntoArray, VortexSessionExecute};
use vortex_arrow::ArrowSessionExt;
use vortex_btrblocks::BtrBlocksCompressor;
use vortex_buffer::BitBuffer;
use vortex_error::VortexResult;
use vortex_ipc::messages::{BufMessageReader, DecoderMessage, EncoderMessage, MessageEncoder};
use vortex_mask::Mask;
use vortex_session::VortexSession;

use crate::cache::LiquidExpr;

use super::eval_predicate_on_array;

const MAGIC: u32 = 0x4C51_4441; // "LQDA" for LiQuid Data Array
const VERSION: u16 = 1;
const HEADER_SIZE: usize = 16;

static SESSION: LazyLock<VortexSession> = LazyLock::new(session);
static COMPRESSOR: LazyLock<BtrBlocksCompressor> = LazyLock::new(BtrBlocksCompressor::default);

fn session() -> VortexSession {
    let session = vortex_array::array_session();
    vortex_arrow::initialize(&session);
    vortex_fastlanes::initialize(&session);
    vortex_fsst::initialize(&session);
    vortex_alp::initialize(&session);
    vortex_runend::initialize(&session);
    vortex_sparse::initialize(&session);
    vortex_zigzag::initialize(&session);
    vortex_datetime_parts::initialize(&session);
    vortex_decimal_byte_parts::initialize(&session);
    vortex_bytebool::initialize(&session);
    vortex_sequence::initialize(&session);
    session
}

/// A Liquid array backed by an in-memory compressed Vortex array.
#[derive(Debug)]
pub struct LiquidArray {
    array: vortex_array::ArrayRef,
    arrow_type: DataType,
}

impl LiquidArray {
    /// Transcode a supported Arrow array into Vortex.
    pub fn from_arrow_array(array: &ArrayRef) -> Result<Self, &ArrayRef> {
        if !is_supported(array.data_type()) {
            return Err(array);
        }

        let arrow_type = array.data_type().clone();
        let imported = match import_array(array) {
            Ok(imported) => imported,
            Err(error) => {
                log::warn!("failed to import {arrow_type:?} into Vortex: {error}");
                return Err(array);
            }
        };
        let mut ctx = SESSION.create_execution_ctx();
        let array = match COMPRESSOR.compress(&imported, &mut ctx) {
            Ok(compressed) => compressed,
            Err(error) => {
                log::warn!("failed to compress {arrow_type:?} with Vortex: {error}");
                return Err(array);
            }
        };
        Ok(Self { array, arrow_type })
    }

    /// Deserialize a Vortex-backed Liquid array.
    pub fn from_bytes(bytes: Bytes) -> Self {
        validate_header(&bytes);
        let header_size = HEADER_SIZE;
        let type_len = u32::from_le_bytes(
            bytes[header_size..header_size + 4]
                .try_into()
                .expect("Vortex Arrow type length"),
        ) as usize;
        let type_start = header_size + 4;
        let type_end = type_start + type_len;
        let arrow_type = serde_json::from_slice(&bytes[type_start..type_end])
            .expect("valid serialized Arrow data type");
        let ipc_start = type_end.next_multiple_of(8);
        let mut reader = BufMessageReader::new(bytes.slice(ipc_start..));
        let dtype = match reader
            .next()
            .expect("Vortex dtype message")
            .expect("valid Vortex IPC")
        {
            DecoderMessage::DType(dtype) => {
                DType::from_flatbuffer(dtype, &SESSION).expect("valid Vortex dtype")
            }
            message => panic!("expected Vortex dtype message, got {message:?}"),
        };
        let array = match reader
            .next()
            .expect("Vortex array message")
            .expect("valid Vortex IPC")
        {
            DecoderMessage::Array((array, read_ctx, row_count)) => array
                .decode(&dtype, row_count, &read_ctx, &SESSION)
                .expect("valid Vortex array"),
            message => panic!("expected Vortex array message, got {message:?}"),
        };
        Self { array, arrow_type }
    }

    fn export(&self, array: vortex_array::ArrayRef) -> ArrayRef {
        let field = Field::new("", self.arrow_type.clone(), array.dtype().is_nullable());
        let mut ctx = SESSION.create_execution_ctx();
        SESSION
            .arrow()
            .execute_arrow(array, Some(&field), &mut ctx)
            .expect("Vortex array must export to its import Arrow type")
    }

    fn filtered_vortex(&self, filter: &BooleanBuffer) -> vortex_array::ArrayRef {
        assert_eq!(filter.len(), self.len(), "filter length must match array");
        let mask = Mask::from_buffer(BitBuffer::from(filter.clone()));
        self.array.filter(mask).expect("Vortex filter must succeed")
    }

    fn try_vortex_predicate(
        &self,
        array: &vortex_array::ArrayRef,
        predicate: &LiquidExpr,
    ) -> Option<vortex_array::ArrayRef> {
        let expr = predicate.physical_expr();
        if let Some(binary) = expr.downcast_ref::<BinaryExpr>() {
            binary.left().downcast_ref::<Column>()?;
            let literal = binary.right().downcast_ref::<Literal>()?;
            return match binary.op() {
                DataFusionOperator::Eq
                | DataFusionOperator::NotEq
                | DataFusionOperator::Lt
                | DataFusionOperator::LtEq
                | DataFusionOperator::Gt
                | DataFusionOperator::GtEq => {
                    let op = vortex_operator(binary.op())?;
                    let constant = self.constant(literal.value(), array.len())?;
                    array.binary(constant, op).ok()
                }
                DataFusionOperator::LikeMatch | DataFusionOperator::NotLikeMatch => self.try_like(
                    array,
                    literal.value(),
                    LikeOptions {
                        negated: matches!(binary.op(), DataFusionOperator::NotLikeMatch),
                        case_insensitive: false,
                    },
                ),
                _ => None,
            };
        }

        if let Some(like) = expr.downcast_ref::<LikeExpr>() {
            if like.case_insensitive() || like.expr().downcast_ref::<Column>().is_none() {
                return None;
            }
            let pattern = like.pattern().downcast_ref::<Literal>()?;
            return self.try_like(
                array,
                pattern.value(),
                LikeOptions {
                    negated: like.negated(),
                    case_insensitive: like.case_insensitive(),
                },
            );
        }
        None
    }

    fn constant(&self, value: &ScalarValue, len: usize) -> Option<vortex_array::ArrayRef> {
        let value = value.cast_to(&self.arrow_type).ok()?;
        let arrow = value.to_array_of_size(1).ok()?;
        let vortex = SESSION
            .arrow()
            .from_arrow_array(arrow, value.is_null())
            .ok()?;
        let mut ctx = SESSION.create_execution_ctx();
        let scalar = vortex.execute_scalar(0, &mut ctx).ok()?;
        Some(ConstantArray::new(scalar, len).into_array())
    }

    fn try_like(
        &self,
        array: &vortex_array::ArrayRef,
        pattern: &ScalarValue,
        options: LikeOptions,
    ) -> Option<vortex_array::ArrayRef> {
        let pattern = self.constant(pattern, array.len())?;
        Some(
            Like::try_new(array.clone(), pattern, options)
                .ok()?
                .into_array(),
        )
    }

    fn try_eval_vortex_predicate(
        &self,
        array: &vortex_array::ArrayRef,
        predicate: &LiquidExpr,
    ) -> Option<BooleanArray> {
        let result = self.try_vortex_predicate(array, predicate)?;
        let field = Field::new("", DataType::Boolean, result.dtype().is_nullable());
        let mut ctx = SESSION.create_execution_ctx();
        let result = SESSION
            .arrow()
            .execute_arrow(result, Some(&field), &mut ctx)
            .ok()?;
        let result = result.as_any().downcast_ref::<BooleanArray>()?;
        let source_validity = array
            .validity()
            .ok()?
            .execute_mask(array.len(), &mut ctx)
            .ok()?;
        let source_nulls = vortex_arrow::to_null_buffer(source_validity);
        // Vortex compare on extension dtypes returns non-null false for null input rows.
        // Liquid's contract is null where the input is null (see Timestamp = literal).
        let nulls = NullBuffer::union(result.nulls(), source_nulls.as_ref());
        Some(BooleanArray::new(result.values().clone(), nulls))
    }
}

impl LiquidArray {
    /// Get the memory size of the Liquid array.
    pub fn get_array_memory_size(&self) -> usize {
        self.array.nbytes() as usize + size_of::<Self>()
    }

    /// Get the length of the Liquid array.
    pub fn len(&self) -> usize {
        self.array.len()
    }

    /// Check whether the Liquid array is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Convert the Liquid array to an Arrow array.
    pub fn to_arrow_array(&self) -> ArrayRef {
        self.export(self.array.clone())
    }

    /// Get the original Arrow data type.
    pub fn original_arrow_data_type(&self) -> DataType {
        self.arrow_type.clone()
    }

    /// Serialize the Liquid array.
    pub fn to_bytes(&self) -> Vec<u8> {
        let arrow_type = serde_json::to_vec(&self.arrow_type).expect("Arrow data type serializes");
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&header_bytes());
        bytes.extend_from_slice(&(arrow_type.len() as u32).to_le_bytes());
        bytes.extend_from_slice(&arrow_type);
        bytes.resize(bytes.len().next_multiple_of(8), 0);

        let mut encoder = MessageEncoder::new(SESSION.clone());
        for message in [
            EncoderMessage::DType(self.array.dtype()),
            EncoderMessage::Array(&self.array),
        ] {
            for buffer in encoder.encode(message).expect("Vortex IPC serialization") {
                bytes.extend_from_slice(&buffer);
            }
        }
        bytes
    }

    /// Filter the Liquid array and return an Arrow array.
    pub fn filter(&self, selection: &BooleanBuffer) -> ArrayRef {
        self.export(self.filtered_vortex(selection))
    }

    /// Evaluate a predicate on selected rows.
    pub fn try_eval_predicate(
        &self,
        predicate: &LiquidExpr,
        filter: &BooleanBuffer,
    ) -> BooleanArray {
        let filtered = self.filtered_vortex(filter);
        if let Some(value) = predicate
            .physical_expr()
            .downcast_ref::<Literal>()
            .and_then(|literal| match literal.value() {
                ScalarValue::Boolean(value) => *value,
                _ => None,
            })
        {
            let values = if value {
                BooleanBuffer::new_set(filtered.len())
            } else {
                BooleanBuffer::new_unset(filtered.len())
            };
            let mut ctx = SESSION.create_execution_ctx();
            let validity = filtered
                .validity()
                .and_then(|validity| validity.execute_mask(filtered.len(), &mut ctx))
                .expect("Vortex array validity must execute");
            return BooleanArray::new(values, vortex_arrow::to_null_buffer(validity));
        }
        let Some(result) = self.try_eval_vortex_predicate(&filtered, predicate) else {
            return eval_predicate_on_array(self.export(filtered), predicate);
        };
        result
    }
}

fn header_bytes() -> [u8; HEADER_SIZE] {
    let mut bytes = [0; HEADER_SIZE];
    bytes[0..4].copy_from_slice(&MAGIC.to_le_bytes());
    bytes[4..6].copy_from_slice(&VERSION.to_le_bytes());
    bytes
}

fn validate_header(bytes: &[u8]) {
    assert!(
        bytes.len() >= HEADER_SIZE,
        "value too small for Liquid array header, expected at least {HEADER_SIZE} bytes, got {}",
        bytes.len()
    );
    let magic = u32::from_le_bytes(bytes[0..4].try_into().unwrap());
    let version = u16::from_le_bytes(bytes[4..6].try_into().unwrap());
    assert_eq!(magic, MAGIC, "Invalid Liquid array magic number");
    assert_eq!(version, VERSION, "Unsupported Liquid array version");
}

fn import_array(array: &ArrayRef) -> VortexResult<vortex_array::ArrayRef> {
    SESSION
        .arrow()
        .from_arrow_array(array.clone(), array.nulls().is_some())
}

fn is_supported(data_type: &DataType) -> bool {
    matches!(
        data_type,
        DataType::Int8
            | DataType::Int16
            | DataType::Int32
            | DataType::Int64
            | DataType::UInt8
            | DataType::UInt16
            | DataType::UInt32
            | DataType::UInt64
            | DataType::Date32
            | DataType::Date64
            | DataType::Timestamp(_, None)
            | DataType::Float32
            | DataType::Float64
            | DataType::Decimal128(_, _)
            | DataType::Decimal256(_, _)
            | DataType::Utf8
            | DataType::Utf8View
            | DataType::Binary
            | DataType::BinaryView
    ) || matches!(
        data_type,
        DataType::Dictionary(key, value)
            if key.as_ref() == &DataType::UInt16
                && matches!(value.as_ref(), DataType::Utf8 | DataType::Binary)
    )
}

fn vortex_operator(op: &DataFusionOperator) -> Option<Operator> {
    match op {
        DataFusionOperator::Eq => Some(Operator::Eq),
        DataFusionOperator::NotEq => Some(Operator::NotEq),
        DataFusionOperator::Lt => Some(Operator::Lt),
        DataFusionOperator::LtEq => Some(Operator::Lte),
        DataFusionOperator::Gt => Some(Operator::Gt),
        DataFusionOperator::GtEq => Some(Operator::Gte),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow::array::{
        BinaryArray, BinaryViewArray, BooleanArray, Date32Array, Decimal128Array, DictionaryArray,
        Float64Array, Int32Array, Int64Array, StringArray, StringViewArray, StructArray,
        TimestampMicrosecondArray, UInt16Array,
    };
    use arrow::datatypes::UInt16Type;
    use datafusion_physical_expr::PhysicalExpr;
    use datafusion_physical_expr::expressions::CastExpr;

    use super::*;

    fn roundtrip(array: ArrayRef) {
        let vortex = LiquidArray::from_arrow_array(&array).unwrap();
        assert_eq!(vortex.to_arrow_array().as_ref(), array.as_ref());
    }

    #[test]
    fn roundtrips_supported_arrow_types() {
        let decimal = Decimal128Array::from(vec![Some(12_345), None, Some(-9_876)])
            .with_precision_and_scale(15, 3)
            .unwrap();
        let dictionary = DictionaryArray::<UInt16Type>::from_iter([
            Some("red"),
            None,
            Some("blue"),
            Some("red"),
        ]);
        let arrays: Vec<ArrayRef> = vec![
            Arc::new(Int32Array::from_iter_values(0..1_100)),
            Arc::new(Int64Array::from(vec![Some(-1), None, Some(7)])),
            Arc::new(UInt16Array::from(vec![0, 1, u16::MAX])),
            Arc::new(Date32Array::from(vec![Some(0), None, Some(20_000)])),
            Arc::new(TimestampMicrosecondArray::from(vec![
                Some(1_000),
                None,
                Some(9_000),
            ])),
            Arc::new(Float64Array::from(vec![Some(1.5), None, Some(-0.0)])),
            Arc::new(decimal),
            Arc::new(StringArray::from(vec![Some("alpha"), None, Some("")])),
            Arc::new(StringViewArray::from(vec![
                Some(""),
                None,
                Some("a longer string value"),
            ])),
            Arc::new(BinaryArray::from(vec![
                Some(b"valid utf8".as_ref()),
                None,
                Some(b"".as_ref()),
            ])),
            Arc::new(BinaryViewArray::from_iter_values([
                vec![0xff, 0x00],
                vec![],
                vec![0x80],
            ])),
            Arc::new(dictionary),
        ];
        for array in arrays {
            roundtrip(array);
        }
    }

    #[test]
    fn roundtrips_empty_array() {
        roundtrip(Arc::new(StringViewArray::from(Vec::<Option<&str>>::new())));
    }

    #[test]
    fn rejects_unsupported_types() {
        let boolean: ArrayRef = Arc::new(BooleanArray::from(vec![true, false]));
        assert!(LiquidArray::from_arrow_array(&boolean).is_err());

        let values: ArrayRef = Arc::new(Int32Array::from(vec![1, 2]));
        let structure: ArrayRef = Arc::new(StructArray::from(vec![(
            Arc::new(Field::new("x", DataType::Int32, false)),
            values,
        )]));
        assert!(LiquidArray::from_arrow_array(&structure).is_err());
    }

    #[test]
    fn filters_like_arrow() {
        let array: ArrayRef = Arc::new(Int64Array::from(vec![Some(1), None, Some(3), Some(4)]));
        let selection = BooleanBuffer::from(vec![true, false, true, false]);
        let vortex = LiquidArray::from_arrow_array(&array).unwrap();
        let expected =
            arrow::compute::filter(&array, &BooleanArray::new(selection.clone(), None)).unwrap();
        assert_eq!(vortex.filter(&selection).as_ref(), expected.as_ref());
    }

    fn comparison_expr(
        op: DataFusionOperator,
        value: ScalarValue,
        data_type: &DataType,
    ) -> LiquidExpr {
        let expr: Arc<dyn PhysicalExpr> = Arc::new(BinaryExpr::new(
            Arc::new(Column::new("c", 0)),
            op,
            Arc::new(Literal::new(value)),
        ));
        LiquidExpr::try_new(expr, data_type).unwrap()
    }

    fn like_expr(pattern: ScalarValue, negated: bool, data_type: &DataType) -> LiquidExpr {
        let expr: Arc<dyn PhysicalExpr> = Arc::new(LikeExpr::new(
            negated,
            false,
            Arc::new(Column::new("c", 0)),
            Arc::new(Literal::new(pattern)),
        ));
        LiquidExpr::try_new(expr, data_type).unwrap()
    }

    fn expected_predicate(
        array: &ArrayRef,
        expr: &LiquidExpr,
        selection: &BooleanBuffer,
    ) -> BooleanArray {
        let filtered =
            arrow::compute::filter(array, &BooleanArray::new(selection.clone(), None)).unwrap();
        eval_predicate_on_array(filtered, expr)
    }

    fn assert_predicate_matches(name: &str, array: &ArrayRef, expr: &LiquidExpr) {
        let selection = BooleanBuffer::new_set(array.len());
        let vortex = LiquidArray::from_arrow_array(array).unwrap();
        let filtered = vortex.filtered_vortex(&selection);
        if vortex.try_eval_vortex_predicate(&filtered, expr).is_none() {
            eprintln!("Vortex predicate fell back to Arrow: {name}");
        }
        assert_eq!(
            vortex.try_eval_predicate(expr, &selection),
            expected_predicate(array, expr, &selection),
            "predicate case {name}"
        );
    }

    #[test]
    fn evaluates_comparisons_with_vortex() {
        let integers: ArrayRef = Arc::new(Int64Array::from(vec![
            Some(-2),
            None,
            Some(0),
            Some(3),
            Some(9),
        ]));
        let strings: ArrayRef = Arc::new(StringViewArray::from(vec![
            Some("x"),
            None,
            Some("prefix-x"),
            Some("z"),
            Some("x-suffix"),
        ]));
        let selections = [
            BooleanBuffer::new_set(5),
            BooleanBuffer::from(vec![true, false, true, false, true]),
        ];
        let ops = [
            DataFusionOperator::Eq,
            DataFusionOperator::NotEq,
            DataFusionOperator::Lt,
            DataFusionOperator::LtEq,
            DataFusionOperator::Gt,
            DataFusionOperator::GtEq,
        ];

        for selection in selections {
            let vortex = LiquidArray::from_arrow_array(&integers).unwrap();
            for op in ops {
                for literal in [-1, 3] {
                    let expr = comparison_expr(
                        op,
                        ScalarValue::Int64(Some(literal)),
                        integers.data_type(),
                    );
                    assert_eq!(
                        vortex.try_eval_predicate(&expr, &selection),
                        expected_predicate(&integers, &expr, &selection)
                    );
                }
            }

            let vortex = LiquidArray::from_arrow_array(&strings).unwrap();
            for op in ops {
                for literal in ["x", "z"] {
                    let expr = comparison_expr(
                        op,
                        ScalarValue::Utf8View(Some(literal.into())),
                        strings.data_type(),
                    );
                    assert_eq!(
                        vortex.try_eval_predicate(&expr, &selection),
                        expected_predicate(&strings, &expr, &selection)
                    );
                }
            }
        }
    }

    #[test]
    fn evaluates_tricky_predicates() {
        let dictionary: ArrayRef = Arc::new(DictionaryArray::<UInt16Type>::from_iter([
            Some("x"),
            None,
            Some("prefix-x"),
            Some("z"),
        ]));
        for op in [
            DataFusionOperator::Eq,
            DataFusionOperator::NotEq,
            DataFusionOperator::Lt,
        ] {
            let name = format!("Dictionary(UInt16, Utf8) {op} Utf8");
            let expr = comparison_expr(
                op,
                ScalarValue::Utf8(Some("x".into())),
                dictionary.data_type(),
            );
            assert_predicate_matches(&name, &dictionary, &expr);
        }
        let expr = like_expr(
            ScalarValue::Utf8(Some("%x%".into())),
            false,
            dictionary.data_type(),
        );
        assert_predicate_matches("Dictionary(UInt16, Utf8) LIKE Utf8", &dictionary, &expr);

        let binary: ArrayRef = Arc::new(BinaryArray::from(vec![
            Some(b"ax".as_slice()),
            None,
            Some(b"by".as_slice()),
            Some(b"cz".as_slice()),
        ]));
        for op in [DataFusionOperator::Eq, DataFusionOperator::Lt] {
            let name = format!("Binary {op} Binary");
            let expr = comparison_expr(
                op,
                ScalarValue::Binary(Some(b"by".to_vec())),
                binary.data_type(),
            );
            assert_predicate_matches(&name, &binary, &expr);
        }

        let binary_view: ArrayRef = Arc::new(BinaryViewArray::from(vec![
            Some(b"\xff\x00".as_slice()),
            None,
            Some(b"\x80".as_slice()),
            Some(b"valid".as_slice()),
        ]));
        let expr = comparison_expr(
            DataFusionOperator::Eq,
            ScalarValue::BinaryView(Some(b"\x80".to_vec())),
            binary_view.data_type(),
        );
        assert_predicate_matches("invalid BinaryView = BinaryView", &binary_view, &expr);

        let dates: ArrayRef = Arc::new(Date32Array::from(vec![Some(0), None, Some(20_000)]));
        for (op, value) in [
            (DataFusionOperator::Gt, 10_000),
            (DataFusionOperator::Eq, 20_000),
        ] {
            let name = format!("Date32 {op} Date32");
            let expr = comparison_expr(op, ScalarValue::Date32(Some(value)), dates.data_type());
            assert_predicate_matches(&name, &dates, &expr);
        }

        let timestamps: ArrayRef = Arc::new(TimestampMicrosecondArray::from(vec![
            Some(1_000),
            None,
            Some(9_000),
        ]));
        for (op, value) in [
            (DataFusionOperator::Gt, 5_000),
            (DataFusionOperator::Eq, 9_000),
        ] {
            let name = format!("Timestamp(Microsecond) {op} Timestamp(Microsecond)");
            let expr = comparison_expr(
                op,
                ScalarValue::TimestampMicrosecond(Some(value), None),
                timestamps.data_type(),
            );
            assert_predicate_matches(&name, &timestamps, &expr);
        }

        let decimals: ArrayRef = Arc::new(
            Decimal128Array::from(vec![Some(1_000), None, Some(12_345)])
                .with_precision_and_scale(15, 3)
                .unwrap(),
        );
        let expr = comparison_expr(
            DataFusionOperator::GtEq,
            ScalarValue::Decimal128(Some(5_000), 15, 3),
            decimals.data_type(),
        );
        assert_predicate_matches("Decimal128(15, 3) >= Decimal128", &decimals, &expr);

        let integers: ArrayRef = Arc::new(Int64Array::from(vec![Some(1), None, Some(3)]));
        let physical: Arc<dyn PhysicalExpr> = Arc::new(BinaryExpr::new(
            Arc::new(CastExpr::new(
                Arc::new(Column::new("c", 0)),
                DataType::Int32,
                None,
            )),
            DataFusionOperator::Eq,
            Arc::new(Literal::new(ScalarValue::Int32(Some(3)))),
        ));
        let expr = LiquidExpr::try_new(physical, integers.data_type()).unwrap();
        assert_predicate_matches("Int64 = Int32", &integers, &expr);

        let strings: ArrayRef = Arc::new(StringViewArray::from(vec![
            Some("alpha"),
            None,
            Some("beta"),
        ]));
        let physical: Arc<dyn PhysicalExpr> = Arc::new(BinaryExpr::new(
            Arc::new(CastExpr::new(
                Arc::new(Column::new("c", 0)),
                DataType::Utf8,
                None,
            )),
            DataFusionOperator::Eq,
            Arc::new(Literal::new(ScalarValue::Utf8(Some("beta".into())))),
        ));
        let expr = LiquidExpr::try_new(physical, strings.data_type()).unwrap();
        assert_predicate_matches("Utf8View = Utf8", &strings, &expr);
    }

    #[test]
    fn evaluates_like_with_vortex() {
        let array: ArrayRef = Arc::new(StringViewArray::from(vec![
            Some("x"),
            None,
            Some("prefix-x"),
            Some("x-suffix"),
            Some("none"),
        ]));
        let vortex = LiquidArray::from_arrow_array(&array).unwrap();
        for (pattern, negated) in [("%x%", false), ("x%", false), ("%x", false), ("%x%", true)] {
            let expr = like_expr(
                ScalarValue::Utf8View(Some(pattern.into())),
                negated,
                &DataType::Utf8View,
            );
            let selection = BooleanBuffer::new_set(array.len());
            assert_eq!(
                vortex.try_eval_predicate(&expr, &selection),
                expected_predicate(&array, &expr, &selection)
            );
        }
    }

    #[test]
    fn boolean_literal_preserves_string_nulls() {
        let strings = (0..3_000)
            .map(|index| (index % 11 != 0).then_some("a nullable string value"))
            .collect::<Vec<_>>();
        let array: ArrayRef = Arc::new(StringViewArray::from(strings));
        let physical: Arc<dyn PhysicalExpr> =
            Arc::new(Literal::new(ScalarValue::Boolean(Some(true))));
        let predicate = LiquidExpr::try_new(physical, array.data_type()).unwrap();
        let vortex = LiquidArray::from_arrow_array(&array).unwrap();
        assert_eq!(
            vortex.try_eval_predicate(&predicate, &BooleanBuffer::new_set(array.len())),
            BooleanArray::new(BooleanBuffer::new_set(array.len()), array.nulls().cloned())
        );
    }

    #[test]
    fn roundtrips_dictionary_export_branches() {
        let identical_values = vec!["same"; 64];
        let identical: ArrayRef = Arc::new(DictionaryArray::<UInt16Type>::from_iter(
            identical_values.iter().map(|value| Some(*value)),
        ));

        let unique_values = (0..256)
            .map(|index| format!("unique-{index}"))
            .collect::<Vec<_>>();
        let unique: ArrayRef = Arc::new(DictionaryArray::<UInt16Type>::from_iter(
            unique_values.iter().map(|value| Some(value.as_str())),
        ));

        let long_values = (0..1_500)
            .map(|index| format!("value-{}", index % 7))
            .collect::<Vec<_>>();
        let long: ArrayRef = Arc::new(DictionaryArray::<UInt16Type>::from_iter(
            long_values.iter().map(|value| Some(value.as_str())),
        ));

        for array in [identical, unique, long] {
            roundtrip(array);
        }
    }

    #[test]
    fn ipc_roundtrips() {
        let arrays: Vec<ArrayRef> = vec![
            Arc::new(Int64Array::from(vec![Some(1), None, Some(3)])),
            Arc::new(StringViewArray::from(vec![
                Some("one"),
                None,
                Some("three"),
            ])),
        ];
        for array in arrays {
            let vortex = LiquidArray::from_arrow_array(&array).unwrap();
            let decoded = LiquidArray::from_bytes(Bytes::from(vortex.to_bytes()));
            assert_eq!(decoded.to_arrow_array().as_ref(), array.as_ref());
        }
    }

    #[test]
    #[should_panic(expected = "Invalid Liquid array magic number")]
    fn rejects_bad_ipc_magic() {
        LiquidArray::from_bytes(Bytes::from_static(&[0; HEADER_SIZE]));
    }

    #[test]
    #[should_panic(expected = "Unsupported Liquid array version")]
    fn rejects_bad_ipc_version() {
        let mut header = header_bytes();
        header[4..6].copy_from_slice(&(VERSION + 1).to_le_bytes());
        LiquidArray::from_bytes(Bytes::copy_from_slice(&header));
    }
}
