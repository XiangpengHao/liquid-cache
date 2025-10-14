use arrow::array::{ArrayRef, BooleanArray, PrimitiveArray, cast::AsArray};
use arrow::buffer::{BooleanBuffer, ScalarBuffer};
use arrow::datatypes::{ArrowPrimitiveType, Date32Type, Int32Type, UInt32Type};
use arrow_schema::DataType;
use std::sync::Arc;

use super::LiquidArray;
use super::primitive_array::LiquidPrimitiveArray;
use super::{IoRange, LiquidArrayRef, LiquidDataType, LiquidHybridArray, LiquidHybridArrayRef};
use crate::liquid_array::raw::BitPackedArray;
use crate::utils::get_bit_width;

/// Which component to extract from a `Date32` (days since UNIX epoch).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Date32Field {
    /// Year component
    Year,
    /// Month component
    Month,
    /// Day component
    Day,
}

/// A bit-packed array that stores a single extracted component (YEAR/MONTH/DAY)
/// from a `Date32` array.
///
/// Values are stored as unsigned offsets from `reference_value`, using the same
/// bit-packing machinery as primitive arrays.
#[derive(Debug, Clone)]
pub struct SqueezedDate32Array {
    field: Date32Field,
    bit_packed: BitPackedArray<UInt32Type>,
    /// The minimum extracted value used as reference for offsetting.
    reference_value: i32,
}

impl SqueezedDate32Array {
    /// Build a squeezed representation (YEAR/MONTH/DAY) from a `LiquidPrimitiveArray<Date32Type>`.
    pub fn from_liquid_date32(
        array: &LiquidPrimitiveArray<Date32Type>,
        field: Date32Field,
    ) -> Self {
        // Decode the logical Date32 array (i32: days since epoch) from the liquid array.
        let arrow_array: PrimitiveArray<Date32Type> =
            array.to_arrow_array().as_primitive::<Date32Type>().clone();

        let (_dt, values, nulls) = arrow_array.into_parts();

        // Compute min and max for the extracted component, skipping nulls.
        let mut has_value = false;
        let mut min_component: i32 = i32::MAX;
        let mut max_component: i32 = i32::MIN;

        // Fast path: if all nulls, return a null bit-packed array of the same length.
        if let Some(nulls_buf) = &nulls {
            if nulls_buf.null_count() == values.len() {
                return Self {
                    field,
                    bit_packed: BitPackedArray::new_null_array(values.len()),
                    reference_value: 0,
                };
            }
        }

        for (idx, &days) in values.iter().enumerate() {
            if nulls.as_ref().is_some_and(|n| n.is_null(idx)) {
                continue;
            }
            let (year, month, day) = ymd_from_epoch_days(days);
            let comp = match field {
                Date32Field::Year => year,
                Date32Field::Month => month as i32,
                Date32Field::Day => day as i32,
            };
            has_value = true;
            if comp < min_component {
                min_component = comp;
            }
            if comp > max_component {
                max_component = comp;
            }
        }

        // If no non-null values found, return an all-null structure (defensive)
        if !has_value {
            return Self {
                field,
                bit_packed: BitPackedArray::new_null_array(values.len()),
                reference_value: 0,
            };
        }

        // Compute bit width from the value range.
        let max_offset = (max_component as i64 - min_component as i64) as u64;
        let bit_width = get_bit_width(max_offset);

        // Build unsigned offsets for packing; placeholders are fine for nulls.
        let offsets: ScalarBuffer<<UInt32Type as ArrowPrimitiveType>::Native> =
            ScalarBuffer::from_iter((0..values.len()).map(|idx| {
                if nulls.as_ref().is_some_and(|n| n.is_null(idx)) {
                    0u32
                } else {
                    let (year, month, day) = ymd_from_epoch_days(values[idx]);
                    let comp = match field {
                        Date32Field::Year => year,
                        Date32Field::Month => month as i32,
                        Date32Field::Day => day as i32,
                    };
                    (comp - min_component) as u32
                }
            }));

        let unsigned_array = PrimitiveArray::<UInt32Type>::new(offsets, nulls);
        let bit_packed = BitPackedArray::from_primitive(unsigned_array, bit_width);

        Self {
            field,
            bit_packed,
            reference_value: min_component,
        }
    }

    /// Length of the array.
    pub fn len(&self) -> usize {
        self.bit_packed.len()
    }

    /// Whether the array has no elements.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Memory size of the bit-packed representation plus reference value.
    pub fn get_array_memory_size(&self) -> usize {
        self.bit_packed.get_array_memory_size() + std::mem::size_of::<i32>()
    }

    /// The extracted component type.
    pub fn field(&self) -> Date32Field {
        self.field
    }

    /// Convert back to an Arrow `Int32` array representing the extracted component values.
    /// Useful for verification or future pushdown logic.
    pub fn to_component_int32(&self) -> PrimitiveArray<Int32Type> {
        let unsigned: PrimitiveArray<UInt32Type> = self.bit_packed.to_primitive();
        let (_dt, values, nulls) = unsigned.into_parts();
        let ref_v = self.reference_value;
        let signed_values: ScalarBuffer<<Int32Type as ArrowPrimitiveType>::Native> =
            ScalarBuffer::from_iter(values.iter().map(|&v| (v as i32).saturating_add(ref_v)));
        PrimitiveArray::<Int32Type>::new(signed_values, nulls)
    }

    /// Lossy reconstruction to Arrow `Date32` (days since epoch).
    ///
    /// Mapping used:
    /// - Year: (year, 1, 1)
    /// - Month: (1970, month, 1)
    /// - Day: (1970, 1, day)
    pub fn to_arrow_date32_lossy(&self) -> PrimitiveArray<Date32Type> {
        let unsigned: PrimitiveArray<UInt32Type> = self.bit_packed.to_primitive();
        let (_dt, values, nulls) = unsigned.into_parts();

        let ref_v = self.reference_value;
        let days_values: ScalarBuffer<<Date32Type as ArrowPrimitiveType>::Native> =
            ScalarBuffer::from_iter(values.iter().enumerate().map(|(i, &off)| {
                if nulls.as_ref().is_some_and(|n| n.is_null(i)) {
                    0i32
                } else {
                    match self.field {
                        Date32Field::Year => {
                            let y = ref_v + off as i32;
                            ymd_to_epoch_days(y, 1, 1)
                        }
                        Date32Field::Month => {
                            let m = (ref_v + off as i32) as u32;
                            ymd_to_epoch_days(1970, m, 1)
                        }
                        Date32Field::Day => {
                            let d = (ref_v + off as i32) as u32;
                            ymd_to_epoch_days(1970, 1, d)
                        }
                    }
                }
            }));

        PrimitiveArray::<Date32Type>::new(days_values, nulls)
    }
}

/// Convert days since UNIX epoch (1970-01-01) to (year, month, day) in the
/// proleptic Gregorian calendar using a branchless integer algorithm.
fn ymd_from_epoch_days(days_since_epoch: i32) -> (i32, u32, u32) {
    // Based on Howard Hinnant's civil_from_days algorithm.
    let z = days_since_epoch as i64 + 719_468; // shift to civil epoch
    let era = if z >= 0 {
        z / 146_097
    } else {
        (z - 146_096) / 146_097
    };
    let doe = (z - era * 146_097) as i64; // [0, 146096]
    let yoe = (doe - doe / 1_460 + doe / 36_524 - doe / 146_096) / 365; // [0, 399]
    let mut y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100); // [0, 365]
    let mp = (5 * doy + 2) / 153; // [0, 11]
    let d = (doy - (153 * mp + 2) / 5) + 1; // [1, 31]
    let m = mp + if mp < 10 { 3 } else { -9 }; // [1, 12]
    if m <= 2 {
        y += 1;
    }
    (y as i32, m as u32, d as u32)
}

/// Convert a date (year, month, day) in proleptic Gregorian calendar to
/// days since UNIX epoch (1970-01-01).
fn ymd_to_epoch_days(year: i32, month: u32, day: u32) -> i32 {
    // Based on Howard Hinnant's civil_to_days algorithm.
    let y = year as i64 - if month <= 2 { 1 } else { 0 };
    let era = if y >= 0 { y / 400 } else { (y - 399) / 400 };
    let yoe = y - era * 400; // [0, 399]
    let m = month as i64;
    let d = day as i64;
    let mp = m + if m > 2 { -3 } else { 9 }; // Mar=0..Jan=10,Feb=11
    let doy = (153 * mp + 2) / 5 + d - 1; // [0, 365]
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy; // [0, 146096]
    (era * 146_097 + doe - 719_468) as i32
}

impl LiquidHybridArray for SqueezedDate32Array {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn get_array_memory_size(&self) -> usize {
        self.get_array_memory_size()
    }

    fn len(&self) -> usize {
        self.len()
    }

    fn to_arrow_array(&self) -> Result<ArrayRef, IoRange> {
        let arr = self.to_arrow_date32_lossy();
        Ok(Arc::new(arr))
    }

    fn data_type(&self) -> LiquidDataType {
        LiquidDataType::Integer
    }

    fn original_arrow_data_type(&self) -> DataType {
        DataType::Date32
    }

    fn to_bytes(&self) -> Result<Vec<u8>, IoRange> {
        todo!("Not implemented");
    }

    fn filter(&self, selection: &BooleanBuffer) -> Result<LiquidHybridArrayRef, IoRange> {
        let unsigned_array: PrimitiveArray<UInt32Type> = self.bit_packed.to_primitive();
        let selection = BooleanArray::new(selection.clone(), None);
        let filtered_values =
            arrow::compute::kernels::filter::filter(&unsigned_array, &selection).unwrap();
        let filtered_values = filtered_values.as_primitive::<UInt32Type>().clone();
        let filtered = if let Some(bit_width) = self.bit_packed.bit_width() {
            let squeezed = BitPackedArray::from_primitive(filtered_values, bit_width);
            SqueezedDate32Array {
                field: self.field,
                bit_packed: squeezed,
                reference_value: self.reference_value,
            }
        } else {
            SqueezedDate32Array {
                field: self.field,
                bit_packed: BitPackedArray::new_null_array(filtered_values.len()),
                reference_value: self.reference_value,
            }
        };
        Ok(Arc::new(filtered) as LiquidHybridArrayRef)
    }

    fn filter_to_arrow(&self, selection: &BooleanBuffer) -> Result<ArrayRef, IoRange> {
        let unsigned_array: PrimitiveArray<UInt32Type> = self.bit_packed.to_primitive();
        let selection = BooleanArray::new(selection.clone(), None);
        let filtered_values =
            arrow::compute::kernels::filter::filter(&unsigned_array, &selection).unwrap();
        let filtered_values = filtered_values.as_primitive::<UInt32Type>().clone();
        // Reconstruct lossy Date32 directly from filtered offsets
        let (_dt, values, nulls) = filtered_values.into_parts();
        let ref_v = self.reference_value;
        let days_values: ScalarBuffer<<Date32Type as ArrowPrimitiveType>::Native> =
            ScalarBuffer::from_iter(values.iter().enumerate().map(|(i, &off)| {
                if nulls.as_ref().is_some_and(|n| n.is_null(i)) {
                    0i32
                } else {
                    match self.field {
                        Date32Field::Year => {
                            let y = ref_v + off as i32;
                            ymd_to_epoch_days(y, 1, 1)
                        }
                        Date32Field::Month => {
                            let m = (ref_v + off as i32) as u32;
                            ymd_to_epoch_days(1970, m, 1)
                        }
                        Date32Field::Day => {
                            let d = (ref_v + off as i32) as u32;
                            ymd_to_epoch_days(1970, 1, d)
                        }
                    }
                }
            }));
        let arr = PrimitiveArray::<Date32Type>::new(days_values, nulls);
        Ok(Arc::new(arr))
    }

    fn try_eval_predicate(
        &self,
        _predicate: &Arc<dyn datafusion::physical_plan::PhysicalExpr>,
        _filter: &BooleanBuffer,
    ) -> Result<Option<BooleanArray>, IoRange> {
        Ok(None)
    }

    fn soak(&self, _data: bytes::Bytes) -> LiquidArrayRef {
        todo!("Not implemented");
    }

    fn to_liquid(&self) -> IoRange {
        todo!("Not implemented");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::PrimitiveArray;
    use std::sync::Arc;

    fn dates(vals: &[Option<i32>]) -> PrimitiveArray<Date32Type> {
        PrimitiveArray::<Date32Type>::from(vals.to_vec())
    }

    fn assert_prim_eq<T: ArrowPrimitiveType>(a: PrimitiveArray<T>, b: PrimitiveArray<T>) {
        let a_ref: arrow::array::ArrayRef = Arc::new(a);
        let b_ref: arrow::array::ArrayRef = Arc::new(b);
        assert_eq!(a_ref.as_ref(), b_ref.as_ref());
    }

    fn extract(field: Date32Field, input: Vec<Option<i32>>) -> PrimitiveArray<Int32Type> {
        let arr = dates(&input);
        let liquid = LiquidPrimitiveArray::<Date32Type>::from_arrow_array(arr);
        let squeezed = SqueezedDate32Array::from_liquid_date32(&liquid, field);
        squeezed.to_component_int32()
    }

    fn lossy(field: Date32Field, input: Vec<Option<i32>>) -> PrimitiveArray<Date32Type> {
        let arr = dates(&input);
        let liquid = LiquidPrimitiveArray::<Date32Type>::from_arrow_array(arr);
        let squeezed = SqueezedDate32Array::from_liquid_date32(&liquid, field);
        squeezed.to_arrow_date32_lossy()
    }

    #[test]
    fn test_extraction_correctness() {
        // YEAR
        let input = vec![
            Some(-1),
            Some(0),
            Some(ymd_to_epoch_days(1971, 7, 15)),
            None,
        ];
        let expected =
            PrimitiveArray::<Int32Type>::from(vec![Some(1969), Some(1970), Some(1971), None]);
        assert_prim_eq(extract(Date32Field::Year, input), expected);

        // MONTH
        let input = vec![
            Some(ymd_to_epoch_days(1970, 1, 31)),
            Some(ymd_to_epoch_days(1970, 2, 1)),
            Some(ymd_to_epoch_days(1970, 12, 31)),
            None,
        ];
        let expected = PrimitiveArray::<Int32Type>::from(vec![Some(1), Some(2), Some(12), None]);
        assert_prim_eq(extract(Date32Field::Month, input), expected);

        // DAY
        let input = vec![
            Some(ymd_to_epoch_days(1970, 1, 1)),
            Some(ymd_to_epoch_days(1970, 1, 31)),
            Some(ymd_to_epoch_days(1970, 2, 1)),
            None,
        ];
        let expected = PrimitiveArray::<Int32Type>::from(vec![Some(1), Some(31), Some(1), None]);
        assert_prim_eq(extract(Date32Field::Day, input), expected);
    }

    #[test]
    fn test_lossy_reconstruction_mapping() {
        // YEAR → (y,1,1)
        let input = vec![
            Some(ymd_to_epoch_days(1999, 12, 31)),
            Some(ymd_to_epoch_days(2000, 6, 1)),
            None,
        ];
        let expected = PrimitiveArray::<Date32Type>::from(vec![
            Some(ymd_to_epoch_days(1999, 1, 1)),
            Some(ymd_to_epoch_days(2000, 1, 1)),
            None,
        ]);
        assert_prim_eq(lossy(Date32Field::Year, input), expected);

        // MONTH → (1970,m,1)
        let input = vec![
            Some(ymd_to_epoch_days(1980, 3, 14)),
            Some(ymd_to_epoch_days(1977, 12, 5)),
            None,
        ];
        let expected = PrimitiveArray::<Date32Type>::from(vec![
            Some(ymd_to_epoch_days(1970, 3, 1)),
            Some(ymd_to_epoch_days(1970, 12, 1)),
            None,
        ]);
        assert_prim_eq(lossy(Date32Field::Month, input), expected);

        // DAY → (1970,1,d)
        let input = vec![
            Some(ymd_to_epoch_days(1980, 3, 14)),
            Some(ymd_to_epoch_days(1977, 12, 5)),
            None,
        ];
        let expected = PrimitiveArray::<Date32Type>::from(vec![
            Some(ymd_to_epoch_days(1970, 1, 14)),
            Some(ymd_to_epoch_days(1970, 1, 5)),
            None,
        ]);
        assert_prim_eq(lossy(Date32Field::Day, input), expected);
    }

    #[test]
    fn test_roundtrip_idempotence() {
        let input = vec![
            Some(ymd_to_epoch_days(1969, 12, 31)),
            Some(ymd_to_epoch_days(1970, 1, 1)),
            Some(ymd_to_epoch_days(1970, 1, 31)),
            Some(ymd_to_epoch_days(1970, 2, 1)),
            Some(ymd_to_epoch_days(1971, 7, 15)),
            None,
        ];

        for &field in &[Date32Field::Year, Date32Field::Month, Date32Field::Day] {
            let comp1 = extract(field, input.clone());
            let lossy_dt = lossy(field, input.clone());
            let liquid2 = LiquidPrimitiveArray::<Date32Type>::from_arrow_array(lossy_dt);
            let comp2 =
                SqueezedDate32Array::from_liquid_date32(&liquid2, field).to_component_int32();
            assert_prim_eq(comp1, comp2);
        }
    }

    #[test]
    fn test_all_nulls_behavior() {
        let input = vec![None, None, None];

        for &field in &[Date32Field::Year, Date32Field::Month, Date32Field::Day] {
            let comp = extract(field, input.clone());
            let expected_comp = PrimitiveArray::<Int32Type>::from(vec![None, None, None]);
            assert_prim_eq(comp, expected_comp);

            let lossy_dt = lossy(field, input.clone());
            let expected_dt = PrimitiveArray::<Date32Type>::from(vec![None, None, None]);
            assert_prim_eq(lossy_dt, expected_dt);
        }
    }
}
