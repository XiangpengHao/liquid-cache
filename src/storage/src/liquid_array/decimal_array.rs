use std::any::Any;
use std::mem::size_of;
use std::num::NonZero;
use std::sync::Arc;

use arrow::array::{Array, ArrayRef, AsArray, BooleanArray, PrimitiveArray};
use arrow::buffer::{BooleanBuffer, ScalarBuffer};
use arrow::datatypes::{Decimal128Type, Decimal256Type, DecimalType, UInt64Type, i256};
use arrow_schema::DataType;
use bytes::Bytes;
use datafusion::physical_plan::PhysicalExpr;
use datafusion::physical_plan::expressions::{
    BinaryExpr, Column, DynamicFilterPhysicalExpr, Literal,
};
use num_traits::ToPrimitive;

use super::{
    LiquidArray, LiquidDataType, LiquidSqueezedArray, LiquidSqueezedArrayRef, NeedsBacking,
    Operator, SqueezeIoHandler, SqueezeResult,
};
use crate::cache::CacheExpression;
use crate::liquid_array::ipc::{LiquidIPCHeader, get_physical_type_id};
use crate::liquid_array::raw::BitPackedArray;
use crate::utils::get_bit_width;

#[derive(Debug, Clone, Copy)]
struct DecimalMeta {
    precision: u8,
    scale: i8,
    is_256: bool,
}

impl DecimalMeta {
    fn from_data_type(data_type: &DataType) -> Self {
        match data_type {
            DataType::Decimal128(precision, scale) => Self {
                precision: *precision,
                scale: *scale,
                is_256: false,
            },
            DataType::Decimal256(precision, scale) => Self {
                precision: *precision,
                scale: *scale,
                is_256: true,
            },
            _ => panic!("unsupported decimal data type: {data_type:?}"),
        }
    }

    fn data_type(&self) -> DataType {
        if self.is_256 {
            DataType::Decimal256(self.precision, self.scale)
        } else {
            DataType::Decimal128(self.precision, self.scale)
        }
    }

    fn arrow_code(&self) -> u8 {
        if self.is_256 { 1 } else { 0 }
    }
}

#[repr(C)]
struct DecimalArrayHeader {
    arrow_type: u8, // 0 for Decimal128, 1 for Decimal256
    precision: u8,
    scale: i8,
    __padding: u8,
    __reserved: u32,
}

impl DecimalArrayHeader {
    const fn size() -> usize {
        8
    }

    fn from_meta(meta: DecimalMeta) -> Self {
        Self {
            arrow_type: meta.arrow_code(),
            precision: meta.precision,
            scale: meta.scale,
            __padding: 0,
            __reserved: 0,
        }
    }

    fn to_bytes(&self) -> [u8; Self::size()] {
        let mut bytes = [0; Self::size()];
        bytes[0] = self.arrow_type;
        bytes[1] = self.precision;
        bytes[2] = self.scale as u8;
        bytes
    }

    fn from_bytes(bytes: &[u8]) -> Self {
        if bytes.len() < Self::size() {
            panic!(
                "value too small for DecimalArrayHeader, expected at least {} bytes, got {}",
                Self::size(),
                bytes.len()
            );
        }
        Self {
            arrow_type: bytes[0],
            precision: bytes[1],
            scale: bytes[2] as i8,
            __padding: 0,
            __reserved: 0,
        }
    }
}

/// Liquid decimal array stored as a compressed u64 primitive.
#[derive(Debug)]
pub struct LiquidDecimalArray {
    meta: DecimalMeta,
    bit_packed: BitPackedArray<UInt64Type>,
    reference_value: u64,
}

impl LiquidDecimalArray {
    pub(crate) fn fits_u64<T: DecimalType>(array: &PrimitiveArray<T>) -> bool
    where
        T::Native: ToPrimitive,
    {
        array.iter().flatten().all(|v| v.to_u64().is_some())
    }

    pub(crate) fn from_decimal_array<T: DecimalType>(array: &PrimitiveArray<T>) -> Self
    where
        T::Native: ToPrimitive,
    {
        debug_assert!(Self::fits_u64(array));
        let meta = DecimalMeta::from_data_type(array.data_type());
        if array.null_count() == array.len() {
            return Self {
                meta,
                bit_packed: BitPackedArray::new_null_array(array.len()),
                reference_value: 0,
            };
        }

        let nulls = array.nulls().cloned();
        let mut min = u64::MAX;
        let mut max = 0u64;
        let values: Vec<u64> = array
            .iter()
            .map(|v| match v {
                Some(v) => {
                    let value = v.to_u64().expect("decimal fits u64");
                    if value < min {
                        min = value;
                    }
                    if value > max {
                        max = value;
                    }
                    value
                }
                None => 0,
            })
            .collect();

        let bit_width = get_bit_width(max - min);
        let offsets = ScalarBuffer::from_iter(values.iter().map(|v| v.saturating_sub(min)));
        let unsigned_array = PrimitiveArray::<UInt64Type>::new(offsets, nulls);
        let bit_packed = BitPackedArray::from_primitive(unsigned_array, bit_width);

        Self {
            meta,
            bit_packed,
            reference_value: min,
        }
    }

    fn bit_pack_starting_loc() -> usize {
        let header_size = LiquidIPCHeader::size() + DecimalArrayHeader::size();
        (header_size + size_of::<u64>() + 7) & !7
    }

    fn to_u64_array(&self) -> PrimitiveArray<UInt64Type> {
        let unsigned_array = self.bit_packed.to_primitive();
        let (_data_type, values, _nulls) = unsigned_array.into_parts();
        let nulls = self.bit_packed.nulls();
        let values = if self.reference_value != 0 {
            let reference_value = self.reference_value;
            ScalarBuffer::from_iter(values.iter().map(|v| v.wrapping_add(reference_value)))
        } else {
            values
        };
        PrimitiveArray::<UInt64Type>::new(values, nulls.cloned())
    }

    pub(crate) fn to_bytes_inner(&self) -> Vec<u8> {
        let header_size = LiquidIPCHeader::size() + DecimalArrayHeader::size();
        let mut result = Vec::with_capacity(Self::bit_pack_starting_loc() + 256);
        result.resize(header_size, 0);

        let logical_type_id = LiquidDataType::Decimal as u16;
        let physical_type_id = get_physical_type_id::<UInt64Type>();
        let ipc_header = LiquidIPCHeader::new(logical_type_id, physical_type_id);
        result[0..LiquidIPCHeader::size()].copy_from_slice(&ipc_header.to_bytes());

        let decimal_header = DecimalArrayHeader::from_meta(self.meta);
        result[LiquidIPCHeader::size()..header_size].copy_from_slice(&decimal_header.to_bytes());

        result.extend_from_slice(&self.reference_value.to_le_bytes());
        while result.len() < Self::bit_pack_starting_loc() {
            result.push(0);
        }
        self.bit_packed.to_bytes(&mut result);
        result
    }

    pub(crate) fn from_bytes(bytes: Bytes) -> Self {
        let header_size = LiquidIPCHeader::size() + DecimalArrayHeader::size();
        let header = LiquidIPCHeader::from_bytes(&bytes);

        assert_eq!(header.logical_type_id, LiquidDataType::Decimal as u16);
        assert_eq!(
            header.physical_type_id,
            get_physical_type_id::<UInt64Type>()
        );

        let decimal_header =
            DecimalArrayHeader::from_bytes(&bytes[LiquidIPCHeader::size()..header_size]);
        let meta = DecimalMeta {
            precision: decimal_header.precision,
            scale: decimal_header.scale,
            is_256: match decimal_header.arrow_type {
                0 => false,
                1 => true,
                _ => panic!(
                    "unsupported decimal type code: {}",
                    decimal_header.arrow_type
                ),
            },
        };

        let ref_start = header_size;
        let ref_end = ref_start + size_of::<u64>();
        let reference_value = u64::from_le_bytes(bytes[ref_start..ref_end].try_into().unwrap());

        let bit_packed_data = bytes.slice(Self::bit_pack_starting_loc()..);
        let bit_packed = BitPackedArray::<UInt64Type>::from_bytes(bit_packed_data);

        Self {
            meta,
            bit_packed,
            reference_value,
        }
    }
}

impl LiquidArray for LiquidDecimalArray {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn get_array_memory_size(&self) -> usize {
        self.bit_packed.get_array_memory_size() + size_of::<u64>() + size_of::<DecimalMeta>()
    }

    fn len(&self) -> usize {
        self.bit_packed.len()
    }

    fn to_arrow_array(&self) -> ArrayRef {
        let u64_array = self.to_u64_array();
        let (_data_type, values, nulls) = u64_array.into_parts();
        let data_type = self.meta.data_type();
        if self.meta.is_256 {
            let values_i256 =
                ScalarBuffer::from_iter(values.iter().map(|v| i256::from_i128(*v as i128)));
            let array = PrimitiveArray::<Decimal256Type>::new(values_i256, nulls);
            Arc::new(array.with_data_type(data_type))
        } else {
            let values_i128 = ScalarBuffer::from_iter(values.iter().map(|v| *v as i128));
            let array = PrimitiveArray::<Decimal128Type>::new(values_i128, nulls);
            Arc::new(array.with_data_type(data_type))
        }
    }

    fn original_arrow_data_type(&self) -> DataType {
        self.meta.data_type()
    }

    fn to_bytes(&self) -> Vec<u8> {
        self.to_bytes_inner()
    }

    fn data_type(&self) -> LiquidDataType {
        LiquidDataType::Decimal
    }

    fn squeeze(
        &self,
        io: Arc<dyn SqueezeIoHandler>,
        expression_hint: Option<&CacheExpression>,
    ) -> Option<(LiquidSqueezedArrayRef, Bytes)> {
        let _expression_hint = expression_hint?;
        let full_bytes = Bytes::from(self.to_bytes_inner());
        let disk_range = 0u64..(full_bytes.len() as u64);

        let orig_bw = self.bit_packed.bit_width()?;
        if orig_bw.get() < 8 {
            return None;
        }

        let new_bw_u8 = NonZero::new((orig_bw.get() / 2).max(1)).unwrap();
        let unsigned_array = self.bit_packed.to_primitive();
        let (_dt, values, nulls) = unsigned_array.into_parts();

        let max_offset = values.iter().copied().max().unwrap_or(0);
        let bucket_count_u64 = 1u64 << (new_bw_u8.get() as u64);
        let range_size = max_offset.saturating_add(1);
        let bucket_width_u64 = (range_size.div_ceil(bucket_count_u64)).max(1);

        let quantized_values: ScalarBuffer<u64> =
            ScalarBuffer::from_iter(values.iter().map(|&v| {
                let mut idx_u64 = v / bucket_width_u64;
                if idx_u64 >= bucket_count_u64 {
                    idx_u64 = bucket_count_u64 - 1;
                }
                idx_u64
            }));
        let quantized_unsigned = PrimitiveArray::<UInt64Type>::new(quantized_values, nulls);
        let quantized_bitpacked = BitPackedArray::from_primitive(quantized_unsigned, new_bw_u8);

        let hybrid = LiquidDecimalQuantizedArray {
            quantized: quantized_bitpacked,
            reference_value: self.reference_value,
            bucket_width: bucket_width_u64,
            disk_range,
            io,
            meta: self.meta,
        };
        Some((Arc::new(hybrid) as LiquidSqueezedArrayRef, full_bytes))
    }
}

#[derive(Debug, Clone)]
pub(crate) struct LiquidDecimalQuantizedArray {
    quantized: BitPackedArray<UInt64Type>,
    reference_value: u64,
    bucket_width: u64,
    disk_range: std::ops::Range<u64>,
    io: Arc<dyn SqueezeIoHandler>,
    meta: DecimalMeta,
}

impl LiquidDecimalQuantizedArray {
    fn len(&self) -> usize {
        self.quantized.len()
    }

    fn new_from_filtered(&self, filtered: PrimitiveArray<UInt64Type>) -> Self {
        let bit_width = self
            .quantized
            .bit_width()
            .expect("quantized bit width must exist");
        let quantized = BitPackedArray::from_primitive(filtered, bit_width);
        Self {
            quantized,
            reference_value: self.reference_value,
            bucket_width: self.bucket_width,
            disk_range: self.disk_range.clone(),
            io: self.io.clone(),
            meta: self.meta,
        }
    }

    fn filter_inner(&self, selection: &BooleanBuffer) -> Self {
        let q_prim: PrimitiveArray<UInt64Type> = self.quantized.to_primitive();
        let selection = BooleanArray::new(selection.clone(), None);
        let filtered = arrow::compute::kernels::filter::filter(&q_prim, &selection).unwrap();
        let filtered = filtered.as_primitive::<UInt64Type>().clone();
        self.new_from_filtered(filtered)
    }

    async fn hydrate_full_arrow(&self) -> ArrayRef {
        let bytes = self
            .io
            .read(Some(self.disk_range.clone()))
            .await
            .expect("read squeezed backing");
        let liquid = crate::liquid_array::ipc::read_from_bytes(
            bytes,
            &crate::liquid_array::ipc::LiquidIPCContext::new(None),
        );
        liquid.to_arrow_array()
    }

    fn literal_to_u64(&self, literal: &Literal) -> Option<u64> {
        use datafusion::common::ScalarValue;
        match literal.value() {
            ScalarValue::Decimal128(Some(v), _precision, scale) => {
                if *scale != self.meta.scale {
                    return None;
                }
                v.to_u64()
            }
            ScalarValue::Decimal256(Some(v), _precision, scale) => {
                if *scale != self.meta.scale {
                    return None;
                }
                v.to_u64()
            }
            _ => None,
        }
    }

    fn try_eval_predicate_inner(
        &self,
        op: &Operator,
        literal: &Literal,
    ) -> SqueezeResult<Option<BooleanArray>> {
        let k = match self.literal_to_u64(literal) {
            Some(k) => k,
            None => return Ok(None),
        };

        let q_prim = self.quantized.to_primitive();
        let (_dt, values, _nulls) = q_prim.into_parts();
        let nulls_opt = self.quantized.nulls();

        let mut out_vals: Vec<bool> = Vec::with_capacity(values.len());

        let push_const_for_below = |op: &Operator| -> bool {
            match op {
                Operator::Eq => false,
                Operator::NotEq => true,
                Operator::Lt => false,
                Operator::LtEq => false,
                Operator::Gt => true,
                Operator::GtEq => true,
            }
        };

        if k < self.reference_value {
            let const_val = push_const_for_below(op);
            if let Some(n) = nulls_opt {
                for (i, _b) in values.iter().enumerate() {
                    out_vals.push(n.is_valid(i) && const_val);
                }
            } else {
                out_vals.resize(values.len(), const_val);
            }
        } else {
            let rel = k - self.reference_value;
            let bw = self.bucket_width;
            let q = rel / bw;
            let r = rel % bw;

            let less_side: bool = matches!(
                op,
                Operator::Eq | Operator::NotEq | Operator::Lt | Operator::LtEq
            );
            let greater_side: bool = matches!(op, Operator::NotEq | Operator::Gt | Operator::GtEq);
            let on_equal_bucket = |r: u64, bw: u64| -> Option<bool> {
                match op {
                    Operator::Eq | Operator::NotEq => None,
                    Operator::Lt => (r == 0).then_some(false),
                    Operator::LtEq => (r + 1 == bw).then_some(true),
                    Operator::Gt => (r + 1 == bw).then_some(false),
                    Operator::GtEq => (r == 0).then_some(true),
                }
            };

            if let Some(n) = nulls_opt {
                for (i, &b) in values.iter().enumerate() {
                    if !n.is_valid(i) {
                        out_vals.push(false);
                        continue;
                    }
                    let v = if b < q {
                        less_side
                    } else if b > q {
                        greater_side
                    } else {
                        match on_equal_bucket(r, bw) {
                            Some(val) => val,
                            None => return Err(NeedsBacking),
                        }
                    };
                    out_vals.push(v);
                }
            } else {
                for &b in values.iter() {
                    let v = if b < q {
                        less_side
                    } else if b > q {
                        greater_side
                    } else {
                        match on_equal_bucket(r, bw) {
                            Some(val) => val,
                            None => return Err(NeedsBacking),
                        }
                    };
                    out_vals.push(v);
                }
            }
        }

        let bool_buf = BooleanBuffer::from_iter(out_vals);
        let out = BooleanArray::new(bool_buf, self.quantized.nulls().cloned());
        Ok(Some(out))
    }
}

#[async_trait::async_trait]
impl LiquidSqueezedArray for LiquidDecimalQuantizedArray {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn get_array_memory_size(&self) -> usize {
        self.quantized.get_array_memory_size() + size_of::<u64>() + size_of::<DecimalMeta>()
    }

    fn len(&self) -> usize {
        LiquidDecimalQuantizedArray::len(self)
    }

    async fn to_arrow_array(&self) -> ArrayRef {
        self.hydrate_full_arrow().await
    }

    fn data_type(&self) -> LiquidDataType {
        LiquidDataType::Decimal
    }

    fn original_arrow_data_type(&self) -> DataType {
        self.meta.data_type()
    }

    async fn try_eval_predicate(
        &self,
        expr: &Arc<dyn PhysicalExpr>,
        filter: &BooleanBuffer,
    ) -> Option<BooleanArray> {
        let filtered = self.filter_inner(filter);

        let expr = unwrap_dynamic_filter(expr)?;
        let binary_expr = expr.as_any().downcast_ref::<BinaryExpr>()?;
        if binary_expr
            .left()
            .as_any()
            .downcast_ref::<Column>()
            .is_none()
        {
            return None;
        }
        let literal = binary_expr.right().as_any().downcast_ref::<Literal>()?;

        let op = Operator::from_datafusion(binary_expr.op())?;
        match filtered.try_eval_predicate_inner(&op, literal) {
            Ok(Some(mask)) => {
                self.io.trace_io_saved();
                return Some(mask);
            }
            Ok(None) => return None,
            Err(NeedsBacking) => {}
        }

        use arrow::array::cast::AsArray;
        use datafusion::logical_expr::ColumnarValue;
        use datafusion::physical_expr_common::datum::apply_cmp;

        let full = self.hydrate_full_arrow().await;
        let selection_array = BooleanArray::new(filter.clone(), None);
        let filtered_arr = arrow::compute::filter(&full, &selection_array).ok()?;
        let filtered_len = filtered_arr.len();

        let lhs = ColumnarValue::Array(filtered_arr);
        let rhs = ColumnarValue::Scalar(literal.value().clone());
        let result = match binary_expr.op() {
            datafusion::logical_expr::Operator::NotEq => {
                apply_cmp(datafusion::logical_expr::Operator::NotEq, &lhs, &rhs)
            }
            datafusion::logical_expr::Operator::Eq => {
                apply_cmp(datafusion::logical_expr::Operator::Eq, &lhs, &rhs)
            }
            datafusion::logical_expr::Operator::Lt => {
                apply_cmp(datafusion::logical_expr::Operator::Lt, &lhs, &rhs)
            }
            datafusion::logical_expr::Operator::LtEq => {
                apply_cmp(datafusion::logical_expr::Operator::LtEq, &lhs, &rhs)
            }
            datafusion::logical_expr::Operator::Gt => {
                apply_cmp(datafusion::logical_expr::Operator::Gt, &lhs, &rhs)
            }
            datafusion::logical_expr::Operator::GtEq => {
                apply_cmp(datafusion::logical_expr::Operator::GtEq, &lhs, &rhs)
            }
            _ => return None,
        };
        let result = result.ok()?;
        Some(result.into_array(filtered_len).ok()?.as_boolean().clone())
    }
}

fn unwrap_dynamic_filter(expr: &Arc<dyn PhysicalExpr>) -> Option<Arc<dyn PhysicalExpr>> {
    if let Some(dynamic_filter) = expr.as_any().downcast_ref::<DynamicFilterPhysicalExpr>() {
        dynamic_filter.current().ok()
    } else {
        Some(expr.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cache::{CacheExpression, TestSqueezeIo};
    use arrow::array::Decimal128Builder;
    use arrow::buffer::BooleanBuffer;
    use datafusion::logical_expr::Operator as DFOperator;
    use datafusion::physical_plan::expressions::{BinaryExpr, Column, Literal};
    use datafusion::scalar::ScalarValue;
    use futures::executor::block_on;
    use std::sync::Arc;

    #[test]
    fn decimal_u64_roundtrip() {
        let mut builder = Decimal128Builder::new();
        builder.append_value(100_i128);
        builder.append_null();
        builder.append_value(250_i128);
        let original = builder.finish().with_precision_and_scale(10, 2).unwrap();

        let liquid = LiquidDecimalArray::from_decimal_array(&original);
        let arrow = liquid.to_arrow_array();
        assert_eq!(arrow.as_ref(), &original);
    }

    #[test]
    fn decimal_u64_ipc_roundtrip() {
        let mut builder = Decimal128Builder::new();
        builder.append_value(12345_i128);
        builder.append_value(67890_i128);
        let original = builder.finish().with_precision_and_scale(12, 3).unwrap();

        let liquid = LiquidDecimalArray::from_decimal_array(&original);
        let bytes = liquid.to_bytes();
        let decoded = LiquidDecimalArray::from_bytes(bytes.into());
        let arrow = decoded.to_arrow_array();
        assert_eq!(arrow.as_ref(), &original);
    }

    #[test]
    fn decimal_quantized_predicate_eval() {
        let mut builder = Decimal128Builder::new();
        builder.append_value(100_i128);
        builder.append_value(200_i128);
        builder.append_null();
        builder.append_value(300_i128);
        let original = builder.finish().with_precision_and_scale(10, 2).unwrap();

        let liquid = LiquidDecimalArray::from_decimal_array(&original);
        let hint = CacheExpression::PredicateColumn;
        let io = Arc::new(TestSqueezeIo::default());
        let (hybrid, bytes) = liquid.squeeze(io.clone(), Some(&hint)).expect("squeezable");
        io.set_bytes(bytes);

        let mask = BooleanBuffer::new_set(original.len());
        let lit = Arc::new(Literal::new(ScalarValue::Decimal128(Some(100_i128), 10, 2)));
        let col = Arc::new(Column::new("col", 0));
        let expr: Arc<dyn PhysicalExpr> = Arc::new(BinaryExpr::new(col, DFOperator::GtEq, lit));

        let got = block_on(hybrid.try_eval_predicate(&expr, &mask)).expect("supported");
        let expected = BooleanArray::from(vec![Some(true), Some(true), None, Some(true)]);
        assert_eq!(got, expected);
        assert_eq!(io.reads(), 0);
    }
}
