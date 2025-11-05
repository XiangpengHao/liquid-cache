use std::any::Any;
use std::sync::Arc;

use arrow::array::ArrowNativeTypeOp;
use arrow::array::{ArrayRef, BooleanArray, PrimitiveArray, cast::AsArray};
use arrow::buffer::{BooleanBuffer, ScalarBuffer};
use arrow::datatypes::{ArrowNativeType, ArrowPrimitiveType};
use arrow_schema::DataType;
use datafusion::physical_plan::PhysicalExpr;
use datafusion::physical_plan::expressions::{BinaryExpr, Literal};
use num_traits::{AsPrimitive, FromPrimitive};

use crate::liquid_array::raw::BitPackedArray;

use super::primitive_array::{LiquidPrimitiveArray, LiquidPrimitiveType};
use super::{IoRange, LiquidArrayRef, LiquidDataType, LiquidHybridArray, Operator, PrimitiveKind};

#[derive(Debug, Clone)]
pub(crate) struct LiquidPrimitiveClampedArray<T: LiquidPrimitiveType> {
    pub(crate) squeezed: BitPackedArray<T::UnSignedType>,
    pub(crate) reference_value: T::Native,
    // Range in the on-disk payload needed to reconstruct the full array (we use full bytes)
    pub(crate) disk_range: std::ops::Range<u64>,
}

impl<T> LiquidPrimitiveClampedArray<T>
where
    T: LiquidPrimitiveType + PrimitiveKind,
{
    #[inline]
    pub(crate) fn len(&self) -> usize {
        self.squeezed.len()
    }

    pub(crate) fn new_from_filtered(
        &self,
        filtered: PrimitiveArray<<T as LiquidPrimitiveType>::UnSignedType>,
    ) -> Self {
        let bit_width = self
            .squeezed
            .bit_width()
            .expect("squeezed bit width must exist");
        let squeezed = BitPackedArray::from_primitive(filtered, bit_width);
        Self {
            squeezed,
            reference_value: self.reference_value,
            disk_range: self.disk_range.clone(),
        }
    }

    pub(crate) fn filter_inner(&self, selection: &BooleanBuffer) -> Self {
        let unsigned_array: PrimitiveArray<T::UnSignedType> = self.squeezed.to_primitive();
        let selection = BooleanArray::new(selection.clone(), None);
        let filtered_values =
            arrow::compute::kernels::filter::filter(&unsigned_array, &selection).unwrap();
        let filtered_values = filtered_values.as_primitive::<T::UnSignedType>().clone();
        self.new_from_filtered(filtered_values)
    }

    pub(crate) fn to_arrow_known_only(&self) -> Option<ArrayRef> {
        // Convert squeezed to primitive and ensure no sentinel exists.
        type U<TT> = <<TT as LiquidPrimitiveType>::UnSignedType as ArrowPrimitiveType>::Native;
        let squeezed_prim = self.squeezed.to_primitive();
        let (_dt, values, nulls) = squeezed_prim.into_parts();
        let bw = self.squeezed.bit_width().expect("bit width").get();
        let sentinel: U<T> = U::<T>::usize_as((1usize << bw) - 1);

        // If any valid value equals sentinel, cannot fully materialize without disk
        if let Some(n) = self.squeezed.nulls() {
            for (i, v) in values.iter().enumerate() {
                if n.is_valid(i) && *v == sentinel {
                    return None;
                }
            }
        } else if values.contains(&sentinel) {
            return None;
        }

        // All values are known; reconstruct to full Arrow by adding reference
        let ref_u: U<T> = self.reference_value.as_();
        let restored_vals: ScalarBuffer<T::Native> =
            ScalarBuffer::from_iter(values.iter().map(|&u| {
                let t_val: T::Native = u.add_wrapping(ref_u).as_();
                t_val
            }));
        let arr = PrimitiveArray::<T>::new(restored_vals, nulls);
        Some(Arc::new(arr))
    }

    // Evaluate a simple comparison if fully decidable without disk; otherwise return Err(IoRange)
    pub(crate) fn try_eval_predicate_inner(
        &self,
        op: &Operator,
        literal: &Literal,
    ) -> Result<Option<BooleanArray>, IoRange> {
        use datafusion::common::ScalarValue;

        // Extract scalar value as T::Native
        let k_opt: Option<T::Native> = match literal.value() {
            ScalarValue::Int8(Some(v)) => T::Native::from_i8(*v),
            ScalarValue::Int16(Some(v)) => T::Native::from_i16(*v),
            ScalarValue::Int32(Some(v)) => T::Native::from_i32(*v),
            ScalarValue::Int64(Some(v)) => T::Native::from_i64(*v),
            ScalarValue::UInt8(Some(v)) => T::Native::from_u8(*v),
            ScalarValue::UInt16(Some(v)) => T::Native::from_u16(*v),
            ScalarValue::UInt32(Some(v)) => T::Native::from_u32(*v),
            ScalarValue::UInt64(Some(v)) => T::Native::from_u64(*v),
            ScalarValue::Date32(Some(v)) => T::Native::from_i32(*v),
            ScalarValue::Date64(Some(v)) => T::Native::from_i64(*v),
            _ => None,
        };
        let Some(k) = k_opt else { return Ok(None) };

        // Prepare squeezed data and thresholds
        type U<TT> = <<TT as LiquidPrimitiveType>::UnSignedType as ArrowPrimitiveType>::Native;
        let squeezed_prim = self.squeezed.to_primitive();
        let (_dt, values, _nulls) = squeezed_prim.into_parts();
        let bw = self.squeezed.bit_width().expect("bit width").get();
        let sentinel: U<T> = U::<T>::usize_as((1usize << bw) - 1);

        // Precompute whether sentinel rows can be resolved under this operator and literal
        let is_unsigned = <T as PrimitiveKind>::IS_UNSIGNED;
        let resolves_on_sentinel: bool = if is_unsigned {
            let ref_u: U<T> = self.reference_value.as_();
            let k_u: U<T> = k.as_();
            let ref_u64: u64 = num_traits::AsPrimitive::<u64>::as_(ref_u);
            let sent_u64: u64 = num_traits::AsPrimitive::<u64>::as_(sentinel);
            let k_u64: u64 = num_traits::AsPrimitive::<u64>::as_(k_u);
            let sent_abs: u64 = ref_u64 + sent_u64;
            match op {
                Operator::Eq | Operator::NotEq | Operator::Gt | Operator::LtEq => k_u64 < sent_abs,
                Operator::Lt | Operator::GtEq => k_u64 <= sent_abs,
            }
        } else {
            // signed types (including Date32/Date64)
            let ref_i: i64 = self.reference_value.as_();
            let k_i: i64 = k.as_();
            let sent_abs: i64 = ref_i + (num_traits::AsPrimitive::<u64>::as_(sentinel) as i64);
            match op {
                Operator::Eq | Operator::NotEq | Operator::Gt | Operator::LtEq => k_i < sent_abs,
                Operator::Lt | Operator::GtEq => k_i <= sent_abs,
            }
        };

        // Build boolean values in a single pass; if an unresolved sentinel is seen, return IO range
        let ref_u: U<T> = self.reference_value.as_();
        let k_t: T::Native = k;
        let mut out_vals: Vec<bool> = Vec::with_capacity(values.len());
        if let Some(n) = self.squeezed.nulls() {
            for (i, &u) in values.iter().enumerate() {
                if !n.is_valid(i) {
                    out_vals.push(false);
                    continue;
                }
                if u == sentinel {
                    if !resolves_on_sentinel {
                        return Err(IoRange {
                            range: self.disk_range.clone(),
                        });
                    }
                    let b = match op {
                        Operator::Eq => false,
                        Operator::NotEq => true,
                        Operator::Lt => false,
                        Operator::LtEq => false,
                        Operator::Gt => true,
                        Operator::GtEq => true,
                    };
                    out_vals.push(b);
                } else {
                    let actual: T::Native = u.add_wrapping(ref_u).as_();
                    let b = match op {
                        Operator::Eq => actual == k_t,
                        Operator::NotEq => actual != k_t,
                        Operator::Lt => actual < k_t,
                        Operator::LtEq => actual <= k_t,
                        Operator::Gt => actual > k_t,
                        Operator::GtEq => actual >= k_t,
                    };
                    out_vals.push(b);
                }
            }
        } else {
            for &u in values.iter() {
                if u == sentinel {
                    if !resolves_on_sentinel {
                        return Err(IoRange {
                            range: self.disk_range.clone(),
                        });
                    }
                    let b = match op {
                        Operator::Eq => false,
                        Operator::NotEq => true,
                        Operator::Lt => false,
                        Operator::LtEq => false,
                        Operator::Gt => true,
                        Operator::GtEq => true,
                    };
                    out_vals.push(b);
                } else {
                    let actual: T::Native = u.add_wrapping(ref_u).as_();
                    let b = match op {
                        Operator::Eq => actual == k_t,
                        Operator::NotEq => actual != k_t,
                        Operator::Lt => actual < k_t,
                        Operator::LtEq => actual <= k_t,
                        Operator::Gt => actual > k_t,
                        Operator::GtEq => actual >= k_t,
                    };
                    out_vals.push(b);
                }
            }
        }

        let bool_buf = BooleanBuffer::from_iter(out_vals);
        let out = BooleanArray::new(bool_buf, self.squeezed.nulls().cloned());
        Ok(Some(out))
    }
}

impl<T> LiquidHybridArray for LiquidPrimitiveClampedArray<T>
where
    T: LiquidPrimitiveType,
{
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn get_array_memory_size(&self) -> usize {
        self.squeezed.get_array_memory_size() + std::mem::size_of::<T::Native>()
    }

    fn len(&self) -> usize {
        LiquidPrimitiveClampedArray::<T>::len(self)
    }

    fn to_arrow_array(&self) -> Result<ArrayRef, IoRange> {
        if let Some(arr) = self.to_arrow_known_only() {
            Ok(arr)
        } else {
            Err(IoRange {
                range: self.disk_range.clone(),
            })
        }
    }

    fn data_type(&self) -> LiquidDataType {
        LiquidDataType::Integer
    }

    fn original_arrow_data_type(&self) -> DataType {
        T::DATA_TYPE.clone()
    }

    fn to_bytes(&self) -> Result<Vec<u8>, IoRange> {
        Err(IoRange {
            range: self.disk_range.clone(),
        })
    }

    fn filter(&self, selection: &BooleanBuffer) -> Result<ArrayRef, IoRange> {
        let filtered = self.filter_inner(selection);
        filtered.to_arrow_array()
    }

    fn try_eval_predicate(
        &self,
        expr: &Arc<dyn PhysicalExpr>,
        filter: &BooleanBuffer,
    ) -> Result<Option<BooleanArray>, IoRange> {
        // Apply selection first to reduce input rows
        let filtered = self.filter_inner(filter);

        if let Some(binary_expr) = expr.as_any().downcast_ref::<BinaryExpr>()
            && let Some(literal) = binary_expr.right().as_any().downcast_ref::<Literal>()
        {
            let op = binary_expr.op();
            let supported_op = Operator::from_datafusion(op);
            if let Some(supported_op) = supported_op {
                return filtered.try_eval_predicate_inner(&supported_op, literal);
            }
        }
        Ok(None)
    }

    fn soak(&self, data: bytes::Bytes) -> LiquidArrayRef {
        // `data` is the full IPC payload for primitive array
        let arr = LiquidPrimitiveArray::<T>::from_bytes(data);
        Arc::new(arr)
    }

    fn to_liquid(&self) -> IoRange {
        IoRange {
            range: self.disk_range.clone(),
        }
    }
}

// Quantized hybrid array: stores bucket indices of value offsets
#[derive(Debug, Clone)]
pub(crate) struct LiquidPrimitiveQuantizedArray<T: LiquidPrimitiveType> {
    pub(crate) quantized: BitPackedArray<T::UnSignedType>,
    pub(crate) reference_value: T::Native,
    // bucket width in terms of absolute offset units
    pub(crate) bucket_width: u64,
    pub(crate) disk_range: std::ops::Range<u64>,
}

impl<T> LiquidPrimitiveQuantizedArray<T>
where
    T: LiquidPrimitiveType + PrimitiveKind,
{
    #[inline]
    pub(crate) fn len(&self) -> usize {
        self.quantized.len()
    }

    pub(crate) fn new_from_filtered(
        &self,
        filtered: PrimitiveArray<<T as LiquidPrimitiveType>::UnSignedType>,
    ) -> Self {
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
        }
    }

    pub(crate) fn filter_inner(&self, selection: &BooleanBuffer) -> Self {
        let q_prim: PrimitiveArray<T::UnSignedType> = self.quantized.to_primitive();
        let selection = BooleanArray::new(selection.clone(), None);
        let filtered = arrow::compute::kernels::filter::filter(&q_prim, &selection).unwrap();
        let filtered = filtered.as_primitive::<T::UnSignedType>().clone();
        self.new_from_filtered(filtered)
    }

    // Evaluate using bucket interval semantics; return Err if any ambiguous bucket is encountered
    pub(crate) fn try_eval_predicate_inner(
        &self,
        op: &Operator,
        literal: &Literal,
    ) -> Result<Option<BooleanArray>, IoRange> {
        use datafusion::common::ScalarValue;
        type U<TT> = <<TT as LiquidPrimitiveType>::UnSignedType as ArrowPrimitiveType>::Native;

        // Extract scalar value as T::Native
        let k_opt: Option<T::Native> = match literal.value() {
            ScalarValue::Int8(Some(v)) => T::Native::from_i8(*v),
            ScalarValue::Int16(Some(v)) => T::Native::from_i16(*v),
            ScalarValue::Int32(Some(v)) => T::Native::from_i32(*v),
            ScalarValue::Int64(Some(v)) => T::Native::from_i64(*v),
            ScalarValue::UInt8(Some(v)) => T::Native::from_u8(*v),
            ScalarValue::UInt16(Some(v)) => T::Native::from_u16(*v),
            ScalarValue::UInt32(Some(v)) => T::Native::from_u32(*v),
            ScalarValue::UInt64(Some(v)) => T::Native::from_u64(*v),
            ScalarValue::Date32(Some(v)) => T::Native::from_i32(*v),
            ScalarValue::Date64(Some(v)) => T::Native::from_i64(*v),
            _ => None,
        };
        let Some(k) = k_opt else { return Ok(None) };

        let q_prim = self.quantized.to_primitive();
        let (_dt, values, _nulls) = q_prim.into_parts();

        let mut out_vals: Vec<bool> = Vec::with_capacity(values.len());
        let nulls_opt = self.quantized.nulls();

        // Common fast-path constants when literal is below the minimum (reference)
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

        // Minimal signed/unsigned split: only to compute below_ref and relative offset
        let (below_ref, rel_opt): (bool, Option<u64>) = if T::IS_UNSIGNED {
            let ref_u_native: U<T> = self.reference_value.as_();
            let ref_u: u64 = num_traits::AsPrimitive::<u64>::as_(ref_u_native);
            let k_u_native: U<T> = k.as_();
            let k_u: u64 = num_traits::AsPrimitive::<u64>::as_(k_u_native);
            if k_u < ref_u {
                (true, None)
            } else {
                (false, Some(k_u - ref_u))
            }
        } else {
            let ref_i: i64 = self.reference_value.as_();
            let k_i: i64 = k.as_();
            if k_i < ref_i {
                (true, None)
            } else {
                (false, Some((k_i - ref_i) as u64))
            }
        };

        if below_ref {
            let const_val = push_const_for_below(op);
            if let Some(n) = nulls_opt {
                for (i, _b) in values.iter().enumerate() {
                    out_vals.push(n.is_valid(i) && const_val);
                }
            } else {
                out_vals.resize(values.len(), const_val);
            }
        } else {
            let rel = rel_opt.expect("rel must exist when not below_ref");
            let bw: u64 = self.bucket_width;
            debug_assert!(bw > 0, "bucket_width must be > 0");
            let q = rel / bw; // target bucket index for k
            let r = rel % bw; // position of k within its bucket

            // Precompute decisions outside the loop
            let less_side: bool = match op {
                Operator::Eq => false,
                Operator::NotEq => true,
                Operator::Lt => true,
                Operator::LtEq => true,
                Operator::Gt => false,
                Operator::GtEq => false,
            };
            let greater_side: bool = match op {
                Operator::Eq => false,
                Operator::NotEq => true,
                Operator::Lt => false,
                Operator::LtEq => false,
                Operator::Gt => true,
                Operator::GtEq => true,
            };
            let on_equal_bucket = |r: u64, bw: u64| -> Option<bool> {
                match op {
                    Operator::Eq | Operator::NotEq => None,
                    Operator::Lt => {
                        if r == 0 {
                            Some(false)
                        } else {
                            None
                        }
                    }
                    Operator::LtEq => {
                        if r + 1 == bw {
                            Some(true)
                        } else {
                            None
                        }
                    }
                    Operator::Gt => {
                        if r + 1 == bw {
                            Some(false)
                        } else {
                            None
                        }
                    }
                    Operator::GtEq => {
                        if r == 0 {
                            Some(true)
                        } else {
                            None
                        }
                    }
                }
            };

            if let Some(n) = nulls_opt {
                for (i, &b_native) in values.iter().enumerate() {
                    if !n.is_valid(i) {
                        out_vals.push(false);
                        continue;
                    }
                    let b: u64 = num_traits::AsPrimitive::<u64>::as_(b_native);
                    let v = if b < q {
                        less_side
                    } else if b > q {
                        greater_side
                    } else {
                        match on_equal_bucket(r, bw) {
                            Some(val) => val,
                            None => {
                                return Err(IoRange {
                                    range: self.disk_range.clone(),
                                });
                            }
                        }
                    };
                    out_vals.push(v);
                }
            } else {
                for &b_native in values.iter() {
                    let b: u64 = num_traits::AsPrimitive::<u64>::as_(b_native);
                    let v = if b < q {
                        less_side
                    } else if b > q {
                        greater_side
                    } else {
                        match on_equal_bucket(r, bw) {
                            Some(val) => val,
                            None => {
                                return Err(IoRange {
                                    range: self.disk_range.clone(),
                                });
                            }
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

impl<T> LiquidHybridArray for LiquidPrimitiveQuantizedArray<T>
where
    T: LiquidPrimitiveType + PrimitiveKind,
{
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn get_array_memory_size(&self) -> usize {
        self.quantized.get_array_memory_size() + std::mem::size_of::<T::Native>()
    }

    fn len(&self) -> usize {
        LiquidPrimitiveQuantizedArray::<T>::len(self)
    }

    fn to_arrow_array(&self) -> Result<ArrayRef, IoRange> {
        Err(IoRange {
            range: self.disk_range.clone(),
        })
    }

    fn data_type(&self) -> LiquidDataType {
        LiquidDataType::Integer
    }

    fn original_arrow_data_type(&self) -> DataType {
        T::DATA_TYPE.clone()
    }

    fn to_bytes(&self) -> Result<Vec<u8>, IoRange> {
        Err(IoRange {
            range: self.disk_range.clone(),
        })
    }

    fn try_eval_predicate(
        &self,
        expr: &Arc<dyn PhysicalExpr>,
        filter: &BooleanBuffer,
    ) -> Result<Option<BooleanArray>, IoRange> {
        // Apply selection first to reduce input rows
        let filtered = self.filter_inner(filter);

        if let Some(binary_expr) = expr.as_any().downcast_ref::<BinaryExpr>()
            && let Some(literal) = binary_expr.right().as_any().downcast_ref::<Literal>()
        {
            let op = binary_expr.op();
            let supported_op = Operator::from_datafusion(op);
            if let Some(supported_op) = supported_op {
                return filtered.try_eval_predicate_inner(&supported_op, literal);
            }
        }
        Ok(None)
    }

    fn soak(&self, data: bytes::Bytes) -> LiquidArrayRef {
        // `data` is the full IPC payload for primitive array
        let arr = LiquidPrimitiveArray::<T>::from_bytes(data);
        Arc::new(arr)
    }

    fn to_liquid(&self) -> IoRange {
        IoRange {
            range: self.disk_range.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::liquid_array::LiquidArray;
    use crate::liquid_array::primitive_array::{IntegerSqueezePolicy, LiquidPrimitiveArray};
    use crate::utils::get_bit_width;
    use arrow::array::{Array, BooleanArray, PrimitiveArray};
    use arrow::buffer::BooleanBuffer;
    use arrow::datatypes::{Int32Type, UInt32Type};
    use datafusion::logical_expr::Operator;
    use datafusion::physical_plan::expressions::{BinaryExpr, Literal};
    use datafusion::scalar::ScalarValue;
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};
    use std::sync::Arc;

    // ---------- Hybrid (squeeze) tests ----------

    fn make_i32_array_with_range(
        len: usize,
        base_min: i32,
        range: i32,
        null_prob: f32,
        rng: &mut StdRng,
    ) -> PrimitiveArray<Int32Type> {
        let mut vals: Vec<Option<i32>> = Vec::with_capacity(len);
        for _ in 0..len {
            if rng.random_bool(null_prob as f64) {
                vals.push(None);
            } else {
                let delta = rng.random_range(0..=range);
                vals.push(Some(base_min.saturating_add(delta)));
            }
        }
        PrimitiveArray::<Int32Type>::from(vals)
    }

    fn make_u32_array_with_range(
        len: usize,
        base_min: u32,
        range: u32,
        null_prob: f32,
        rng: &mut StdRng,
    ) -> PrimitiveArray<UInt32Type> {
        let mut vals: Vec<Option<u32>> = Vec::with_capacity(len);
        for _ in 0..len {
            if rng.random_bool(null_prob as f64) {
                vals.push(None);
            } else {
                let delta = rng.random_range(0..=range);
                vals.push(Some(base_min.saturating_add(delta)));
            }
        }
        PrimitiveArray::<UInt32Type>::from(vals)
    }

    fn compute_boundary_i32(arr: &PrimitiveArray<Int32Type>) -> Option<i32> {
        // boundary = min + ((1 << (bit_width(range)/2)) - 1)
        let min = arrow::compute::kernels::aggregate::min(arr)?;
        let max = arrow::compute::kernels::aggregate::max(arr)?;
        let range = (max as i64 - min as i64) as u64;
        let bw = get_bit_width(range);
        let half = (bw.get() / 2) as u32;
        let sentinel = if half == 0 { 0 } else { (1u64 << half) - 1 } as i64;
        (min as i64 + sentinel).try_into().ok()
    }

    fn compute_boundary_u32(arr: &PrimitiveArray<UInt32Type>) -> Option<u32> {
        let min = arrow::compute::kernels::aggregate::min(arr)?;
        let max = arrow::compute::kernels::aggregate::max(arr)?;
        let range = (max as u128 - min as u128) as u64;
        let bw = get_bit_width(range);
        let half = (bw.get() / 2) as u32;
        let sentinel = if half == 0 { 0 } else { (1u128 << half) - 1 } as u128;
        let b = (min as u128 + sentinel) as u64 as u32;
        Some(b)
    }

    #[test]
    fn clamp_unsqueezable_small_range() {
        // range < 512 -> bit width < 10 => None
        let mut rng = StdRng::seed_from_u64(0x51_71);
        let arr = make_i32_array_with_range(64, 10_000, 100, 0.1, &mut rng);
        let liquid = LiquidPrimitiveArray::<Int32Type>::from_arrow_array(arr)
            .with_squeeze_policy(IntegerSqueezePolicy::Clamp);
        assert!(liquid.squeeze(None).is_none());
    }

    #[test]
    fn clamp_squeeze_and_soak_roundtrip_i32() {
        let mut rng = StdRng::seed_from_u64(0x51_72);
        let arr = make_i32_array_with_range(128, -50_000, 1 << 16, 0.1, &mut rng);
        let liq = LiquidPrimitiveArray::<Int32Type>::from_arrow_array(arr.clone())
            .with_squeeze_policy(IntegerSqueezePolicy::Clamp);
        let bytes_baseline = liq.to_bytes();
        let (hybrid, bytes) = liq.squeeze(None).expect("squeezable");
        // ensure we can recover the original using soak
        let recovered = hybrid.soak(bytes.clone());
        assert_eq!(recovered.to_arrow_array().as_primitive::<Int32Type>(), &arr);
        assert_eq!(bytes_baseline, recovered.to_bytes());

        // If we filter to only known values, hybrid can materialize without IO
        let boundary = compute_boundary_i32(&arr).unwrap();
        let mask_bits: Vec<bool> = (0..arr.len())
            .map(|i| {
                if arr.is_null(i) {
                    true
                } else {
                    arr.value(i) < boundary
                }
            })
            .collect();
        let mask = BooleanBuffer::from_iter(mask_bits.iter().copied());
        let filtered_arrow = hybrid
            .filter(&mask)
            .expect("known-only selection should be materializable");

        let expected = {
            let vals: Vec<Option<i32>> = (0..arr.len())
                .zip(mask_bits.iter())
                .filter(|&(_, &keep)| keep)
                .map(|(i, &_keep)| {
                    if arr.is_null(i) {
                        None
                    } else {
                        Some(arr.value(i))
                    }
                })
                .collect();
            PrimitiveArray::<Int32Type>::from(vals)
        };
        assert_eq!(filtered_arrow.as_primitive::<Int32Type>(), &expected);
    }

    #[test]
    fn clamp_predicate_eval_i32_resolvable_and_unresolvable() {
        let mut rng = StdRng::seed_from_u64(0x51_73);
        let arr = make_i32_array_with_range(200, -1_000_000, 1 << 16, 0.2, &mut rng);
        let liq = LiquidPrimitiveArray::<Int32Type>::from_arrow_array(arr.clone())
            .with_squeeze_policy(IntegerSqueezePolicy::Clamp);
        let (hybrid, _bytes) = liq.squeeze(None).expect("squeezable");

        let boundary = compute_boundary_i32(&arr).unwrap();
        // selection mask: random subset
        let mask_bits: Vec<bool> = (0..arr.len()).map(|_| rng.random()).collect();
        let mask = BooleanBuffer::from_iter(mask_bits.iter().copied());

        let build_expr =
            |op: Operator, k: i32| -> Arc<dyn datafusion::physical_plan::PhysicalExpr> {
                let lit = Arc::new(Literal::new(ScalarValue::Int32(Some(k))));
                Arc::new(BinaryExpr::new(lit.clone(), op, lit))
            };

        // Helper to compute expected boolean array on selected rows
        let expected_for = |op: Operator, k: i32| -> BooleanArray {
            let vals: Vec<Option<bool>> = (0..arr.len())
                .zip(mask_bits.iter())
                .filter(|&(_, &keep)| keep)
                .map(|(i, &_keep)| {
                    if arr.is_null(i) {
                        None
                    } else {
                        let v = arr.value(i);
                        Some(match op {
                            Operator::Eq => v == k,
                            Operator::NotEq => v != k,
                            Operator::Lt => v < k,
                            Operator::LtEq => v <= k,
                            Operator::Gt => v > k,
                            Operator::GtEq => v >= k,
                            _ => unreachable!(),
                        })
                    }
                })
                .collect();
            BooleanArray::from(vals)
        };

        // Resolvable cases: K strictly less than boundary for Eq,Neq,LtEq,Gt; K <= boundary for Lt,GtEq
        let resolvable_cases: Vec<(Operator, i32)> = vec![
            (Operator::Eq, boundary - 1),
            (Operator::NotEq, boundary - 1),
            (Operator::Lt, boundary),
            (Operator::LtEq, boundary - 1),
            (Operator::Gt, boundary - 1),
            (Operator::GtEq, boundary),
        ];

        for (op, k) in resolvable_cases {
            let expr = build_expr(op, k);
            let got = hybrid.try_eval_predicate(&expr, &mask).expect("no IO");
            let expected = expected_for(op, k);
            assert_eq!(got.unwrap(), expected);
        }

        // Unresolvable: choose constants >= boundary for ops that require disk
        let unresolvable_cases: Vec<(Operator, i32)> = vec![
            (Operator::Eq, boundary),
            (Operator::NotEq, boundary),
            (Operator::Lt, boundary + 1),
            (Operator::LtEq, boundary),
            (Operator::Gt, boundary + 1),
            (Operator::GtEq, boundary + 1),
        ];
        for (op, k) in unresolvable_cases {
            let expr = build_expr(op, k);
            let err = hybrid
                .try_eval_predicate(&expr, &mask)
                .expect_err("should request IO");
            let io = hybrid.to_liquid();
            assert_eq!(err.range(), io.range());
        }
    }

    #[test]
    fn clamp_predicate_eval_u32_resolvable_and_unresolvable() {
        let mut rng = StdRng::seed_from_u64(0x51_74);
        let arr = make_u32_array_with_range(180, 1_000_000, 1 << 16, 0.15, &mut rng);
        let liq = LiquidPrimitiveArray::<UInt32Type>::from_arrow_array(arr.clone())
            .with_squeeze_policy(IntegerSqueezePolicy::Clamp);
        let (hybrid, _bytes) = liq.squeeze(None).expect("squeezable");

        let boundary = compute_boundary_u32(&arr).unwrap();
        let mask_bits: Vec<bool> = (0..arr.len()).map(|_| rng.random()).collect();
        let mask = BooleanBuffer::from_iter(mask_bits.iter().copied());

        let build_expr =
            |op: Operator, k: u32| -> Arc<dyn datafusion::physical_plan::PhysicalExpr> {
                let lit = Arc::new(Literal::new(ScalarValue::UInt32(Some(k))));
                Arc::new(BinaryExpr::new(lit.clone(), op, lit))
            };

        let expected_for = |op: Operator, k: u32| -> BooleanArray {
            let vals: Vec<Option<bool>> = (0..arr.len())
                .zip(mask_bits.iter())
                .filter(|&(_, &keep)| keep)
                .map(|(i, &_keep)| {
                    if arr.is_null(i) {
                        None
                    } else {
                        let v = arr.value(i);
                        Some(match op {
                            Operator::Eq => v == k,
                            Operator::NotEq => v != k,
                            Operator::Lt => v < k,
                            Operator::LtEq => v <= k,
                            Operator::Gt => v > k,
                            Operator::GtEq => v >= k,
                            _ => unreachable!(),
                        })
                    }
                })
                .collect();
            BooleanArray::from(vals)
        };

        let resolvable_cases: Vec<(Operator, u32)> = vec![
            (Operator::Eq, boundary - 1),
            (Operator::NotEq, boundary - 1),
            (Operator::Lt, boundary),
            (Operator::LtEq, boundary - 1),
            (Operator::Gt, boundary - 1),
            (Operator::GtEq, boundary),
        ];
        for (op, k) in resolvable_cases {
            let expr = build_expr(op, k);
            let got = hybrid.try_eval_predicate(&expr, &mask).expect("no IO");
            let expected = expected_for(op, k);
            assert_eq!(got.unwrap(), expected);
        }

        let unresolvable_cases: Vec<(Operator, u32)> = vec![
            (Operator::Eq, boundary),
            (Operator::NotEq, boundary),
            (Operator::Lt, boundary + 1),
            (Operator::LtEq, boundary),
            (Operator::Gt, boundary + 1),
            (Operator::GtEq, boundary + 1),
        ];
        for (op, k) in unresolvable_cases {
            let expr = build_expr(op, k);
            let err = hybrid
                .try_eval_predicate(&expr, &mask)
                .expect_err("should request IO");
            assert_eq!(err.range(), hybrid.to_liquid().range());
        }
    }

    #[test]
    fn quantize_predicate_eval_u32_resolvable_and_unresolvable() {
        let mut rng = StdRng::seed_from_u64(0x51_84);
        let arr = make_u32_array_with_range(200, 1_000_000, 1 << 16, 0.2, &mut rng);
        let liq = LiquidPrimitiveArray::<UInt32Type>::from_arrow_array(arr.clone())
            .with_squeeze_policy(IntegerSqueezePolicy::Quantize);
        let (hybrid, _bytes) = liq.squeeze(None).expect("squeezable");

        let min = arrow::compute::kernels::aggregate::min(&arr).unwrap();

        let mask = BooleanBuffer::from(vec![true; arr.len()]);
        let build_expr =
            |op: Operator, k: u32| -> Arc<dyn datafusion::physical_plan::PhysicalExpr> {
                let lit = Arc::new(Literal::new(ScalarValue::UInt32(Some(k))));
                Arc::new(BinaryExpr::new(lit.clone(), op, lit))
            };

        // Expect resolvable results without IO
        let resolvable_cases: Vec<(Operator, u32, bool)> = vec![
            (Operator::Eq, min.saturating_sub(1), false), // eq false everywhere
            (Operator::NotEq, min.saturating_sub(1), true), // neq true everywhere
            (Operator::Lt, min, false),                   // lt false everywhere
            (Operator::LtEq, min.saturating_sub(1), false), // lte false everywhere
            (Operator::Gt, min.saturating_sub(1), true),  // gt true everywhere
            (Operator::GtEq, min, true),                  // gte true everywhere
        ];
        for (op, k, expected_const) in resolvable_cases {
            let expr = build_expr(op, k);
            let got = hybrid.try_eval_predicate(&expr, &mask).expect("no IO");
            let expected = {
                let vals: Vec<Option<bool>> = (0..arr.len())
                    .map(|i| {
                        if arr.is_null(i) {
                            None
                        } else {
                            Some(expected_const)
                        }
                    })
                    .collect();
                BooleanArray::from(vals)
            };
            assert_eq!(got.unwrap(), expected);
        }

        // Unresolvable for Eq: pick a present value (ensures ambiguous bucket)
        let k_present = (0..arr.len())
            .find_map(|i| {
                if arr.is_null(i) {
                    None
                } else {
                    Some(arr.value(i))
                }
            })
            .unwrap();
        let expr_eq_present = build_expr(Operator::Eq, k_present);
        let err = hybrid
            .try_eval_predicate(&expr_eq_present, &mask)
            .expect_err("quantized should request IO on ambiguous eq bucket");
        assert_eq!(err.range(), hybrid.to_liquid().range());
    }

    #[test]
    fn quantize_predicate_eval_i32_resolvable_and_unresolvable() {
        let mut rng = StdRng::seed_from_u64(0x51_85);
        let arr = make_i32_array_with_range(220, -1_000_000, 1 << 16, 0.2, &mut rng);
        let liq = LiquidPrimitiveArray::<Int32Type>::from_arrow_array(arr.clone())
            .with_squeeze_policy(IntegerSqueezePolicy::Quantize);
        let (hybrid, _bytes) = liq.squeeze(None).expect("squeezable");

        let min = arrow::compute::kernels::aggregate::min(&arr).unwrap();
        let mask = BooleanBuffer::from(vec![true; arr.len()]);
        let build_expr =
            |op: Operator, k: i32| -> Arc<dyn datafusion::physical_plan::PhysicalExpr> {
                let lit = Arc::new(Literal::new(ScalarValue::Int32(Some(k))));
                Arc::new(BinaryExpr::new(lit.clone(), op, lit))
            };

        let resolvable_cases: Vec<(Operator, i32, bool)> = vec![
            (Operator::Eq, min - 1, false), // eq false everywhere
            (Operator::NotEq, min - 1, true),
            (Operator::Lt, min, false),
            (Operator::LtEq, min - 1, false),
            (Operator::Gt, min - 1, true),
            (Operator::GtEq, min, true),
        ];
        for (op, k, expected_const) in resolvable_cases {
            let expr = build_expr(op, k);
            let got = hybrid.try_eval_predicate(&expr, &mask).expect("no IO");
            let expected = {
                let vals: Vec<Option<bool>> = (0..arr.len())
                    .map(|i| {
                        if arr.is_null(i) {
                            None
                        } else {
                            Some(expected_const)
                        }
                    })
                    .collect();
                BooleanArray::from(vals)
            };
            assert_eq!(got.unwrap(), expected);
        }

        // Unresolvable for Eq: pick a present value
        let k_present = (0..arr.len())
            .find_map(|i| {
                if arr.is_null(i) {
                    None
                } else {
                    Some(arr.value(i))
                }
            })
            .unwrap();
        let expr_eq_present = build_expr(Operator::Eq, k_present);
        let err = hybrid
            .try_eval_predicate(&expr_eq_present, &mask)
            .expect_err("quantized should request IO on ambiguous eq bucket");
        assert_eq!(err.range(), hybrid.to_liquid().range());
    }

    #[test]
    fn quantize_to_arrow_is_err() {
        let mut rng = StdRng::seed_from_u64(0x51_86);
        let arr = make_u32_array_with_range(64, 1000, 1 << 12, 0.0, &mut rng);
        let liq = LiquidPrimitiveArray::<UInt32Type>::from_arrow_array(arr)
            .with_squeeze_policy(IntegerSqueezePolicy::Quantize);
        let (hybrid, _bytes) = liq.squeeze(None).expect("squeezable");
        // Quantized hybrid cannot materialize to Arrow without IO
        assert!(hybrid.to_arrow_array().is_err());
    }
}
