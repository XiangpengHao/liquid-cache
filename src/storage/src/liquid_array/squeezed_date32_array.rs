use arrow::array::{Array, ArrayRef, BooleanArray, PrimitiveArray, cast::AsArray};
use arrow::buffer::{BooleanBuffer, ScalarBuffer};
use arrow::datatypes::{ArrowPrimitiveType, Date32Type, Int32Type, UInt32Type};
use arrow_schema::DataType;
use std::sync::Arc;

use super::LiquidArray;
use super::primitive_array::LiquidPrimitiveArray;
use super::{IoRange, LiquidArrayRef, LiquidDataType, LiquidHybridArray};
use crate::cache::DatePartSet;
use crate::liquid_array::LiquidPrimitiveType;
use crate::liquid_array::raw::BitPackedArray;
use crate::utils::get_bit_width;
use arrow::compute::DatePart;

#[derive(Debug, Clone)]
struct ComponentChunk {
    part: DatePart,
    bit_packed: BitPackedArray<UInt32Type>,
    reference_value: i32,
}

impl ComponentChunk {
    fn len(&self) -> usize {
        self.bit_packed.len()
    }

    fn get_array_memory_size(&self) -> usize {
        self.bit_packed.get_array_memory_size() + std::mem::size_of::<i32>()
    }

    fn new_null(part: DatePart, len: usize) -> Self {
        Self {
            part,
            bit_packed: BitPackedArray::new_null_array(len),
            reference_value: 0,
        }
    }

    fn from_components(part: DatePart, comps_array: PrimitiveArray<Int32Type>) -> Self {
        let len = comps_array.len();
        let min_component =
            arrow::compute::kernels::aggregate::min(&comps_array).unwrap_or(i32::MAX);
        let max_component =
            arrow::compute::kernels::aggregate::max(&comps_array).unwrap_or(i32::MIN);

        let has_value = min_component != i32::MAX && max_component != i32::MIN;
        if !has_value {
            return Self::new_null(part, len);
        }

        let max_offset = (max_component as i64 - min_component as i64) as u64;
        let bit_width = get_bit_width(max_offset);

        let (_dt, comps_values, comps_nulls) = comps_array.into_parts();
        let offsets: ScalarBuffer<u32> = ScalarBuffer::from_iter(
            comps_values
                .iter()
                .map(|&v| (v.saturating_sub(min_component)) as u32),
        );

        let unsigned_array = PrimitiveArray::<UInt32Type>::new(offsets, comps_nulls);
        let bit_packed = BitPackedArray::from_primitive(unsigned_array, bit_width);

        Self {
            part,
            bit_packed,
            reference_value: min_component,
        }
    }

    fn to_component_array(&self) -> PrimitiveArray<Date32Type> {
        let unsigned: PrimitiveArray<UInt32Type> = self.bit_packed.to_primitive();
        let (_dt, values, nulls) = unsigned.into_parts();
        let signed_values: ScalarBuffer<<Int32Type as ArrowPrimitiveType>::Native> =
            ScalarBuffer::from_iter(
                values
                    .iter()
                    .map(|&v| (v as i32).saturating_add(self.reference_value)),
            );
        PrimitiveArray::<Date32Type>::new(signed_values, nulls)
    }

    fn to_lossy_date32(&self) -> PrimitiveArray<Date32Type> {
        let unsigned: PrimitiveArray<UInt32Type> = self.bit_packed.to_primitive();
        let (_dt, values, nulls) = unsigned.into_parts();
        let ref_v = self.reference_value;
        let days_values: ScalarBuffer<<Date32Type as ArrowPrimitiveType>::Native> =
            ScalarBuffer::from_iter(values.iter().enumerate().map(|(i, &off)| {
                if nulls.as_ref().is_some_and(|n| n.is_null(i)) {
                    0i32
                } else {
                    match self.part {
                        DatePart::Year => {
                            let y = ref_v + off as i32;
                            ymd_to_epoch_days(y, 1, 1)
                        }
                        DatePart::Month => {
                            let m = (ref_v + off as i32) as u32;
                            ymd_to_epoch_days(1970, m, 1)
                        }
                        DatePart::Day => {
                            let d = (ref_v + off as i32) as u32;
                            ymd_to_epoch_days(1970, 1, d)
                        }
                        _ => unreachable!("Stored field should be Year/Month/Day"),
                    }
                }
            }));
        PrimitiveArray::<Date32Type>::new(days_values, nulls)
    }

    fn derive_with_arrow(&self, requested: DatePart) -> PrimitiveArray<Date32Type> {
        let lossy = self.to_lossy_date32();
        let derived_array = arrow::compute::date_part(&lossy, requested).unwrap();
        let derived_int32 = derived_array.as_primitive::<Int32Type>();
        let (_dt, derived_values, derived_nulls) = derived_int32.clone().into_parts();
        let derived_date32_values: ScalarBuffer<<Date32Type as ArrowPrimitiveType>::Native> =
            ScalarBuffer::from_iter(derived_values.iter().copied());
        PrimitiveArray::<Date32Type>::new(derived_date32_values, derived_nulls)
    }
}

/// A bit-packed array that stores extracted components (YEAR/MONTH/DAY)
/// from a `Date32` array.
///
/// Each component is stored as unsigned offsets from its component-specific `reference_value`.
#[derive(Debug, Clone)]
pub struct SqueezedDate32Array {
    parts: DatePartSet,
    chunks: Vec<ComponentChunk>,
}

impl SqueezedDate32Array {
    /// Build a squeezed representation (YEAR/MONTH/DAY) from a `LiquidPrimitiveArray<Date32Type>`.
    pub fn from_liquid_date32<T: LiquidPrimitiveType>(
        array: &LiquidPrimitiveArray<T>,
        field: impl Into<DatePartSet>,
    ) -> Self {
        let parts = field.into();
        let arrow_array: PrimitiveArray<Date32Type> =
            array.to_arrow_array().as_primitive::<Date32Type>().clone();
        let len = arrow_array.len();

        let all_null = arrow_array.null_count() == len && len > 0;
        let chunks = parts
            .iter()
            .map(|part| {
                if all_null {
                    ComponentChunk::new_null(part, len)
                } else {
                    Self::build_chunk(part, &arrow_array)
                }
            })
            .collect();

        Self { parts, chunks }
    }

    fn build_chunk(part: DatePart, arrow_array: &PrimitiveArray<Date32Type>) -> ComponentChunk {
        let comps = arrow::compute::date_part(arrow_array, part).unwrap();
        let comps_array = comps.as_primitive::<Int32Type>().clone();
        ComponentChunk::from_components(part, comps_array)
    }

    fn chunk(&self, part: DatePart) -> Option<&ComponentChunk> {
        self.chunks.iter().find(|chunk| chunk.part == part)
    }

    fn primary_chunk(&self) -> Option<&ComponentChunk> {
        self.parts.iter().next().and_then(|part| self.chunk(part))
    }

    /// Length of the array.
    pub fn len(&self) -> usize {
        self.chunks.first().map(|chunk| chunk.len()).unwrap_or(0)
    }

    /// Whether the array has no elements.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Memory size of the bit-packed representation plus reference values.
    pub fn get_array_memory_size(&self) -> usize {
        self.chunks
            .iter()
            .map(|chunk| chunk.get_array_memory_size())
            .sum()
    }

    /// The extracted component set.
    pub fn field(&self) -> DatePartSet {
        self.parts
    }

    /// Whether the array can compute the requested derived field.
    /// Quarter can be derived from Month, DayOfYear from Month+Day.
    pub fn can_derive_field(&self, requested: DatePart) -> bool {
        if self.chunk(requested).is_some() {
            return true;
        }
        match requested {
            DatePart::Quarter => self.chunk(DatePart::Month).is_some(),
            DatePart::DayOfYear => {
                self.chunk(DatePart::Month).is_some() && self.chunk(DatePart::Day).is_some()
            }
            _ => false,
        }
    }

    /// Compute a DatePart from the stored components.
    /// Returns Date32Type array (same pattern as to_component_date32) for consistency.
    pub fn compute_derived(&self, requested: DatePart) -> Option<PrimitiveArray<Date32Type>> {
        if let Some(chunk) = self.chunk(requested) {
            return Some(chunk.to_component_array());
        }

        match requested {
            DatePart::Quarter => self
                .chunk(DatePart::Month)
                .map(|chunk| chunk.derive_with_arrow(DatePart::Quarter)),
            _ => None,
        }
    }

    /// Convert back to an Arrow `Int32` array representing the extracted component values.
    pub fn to_component_date32(&self, part: DatePart) -> Option<PrimitiveArray<Date32Type>> {
        self.chunk(part).map(|chunk| chunk.to_component_array())
    }

    /// Lossy reconstruction to Arrow `Date32` (days since epoch) using the primary component.
    pub fn to_arrow_date32_lossy(&self) -> PrimitiveArray<Date32Type> {
        self.primary_chunk()
            .map(|chunk| chunk.to_lossy_date32())
            .unwrap_or_else(|| PrimitiveArray::<Date32Type>::new_null(0))
    }
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

    fn filter(&self, selection: &BooleanBuffer) -> Result<ArrayRef, IoRange> {
        let Some(chunk) = self.primary_chunk() else {
            return Ok(Arc::new(PrimitiveArray::<Date32Type>::new_null(0)));
        };

        let unsigned_array: PrimitiveArray<UInt32Type> = chunk.bit_packed.to_primitive();
        let selection = BooleanArray::new(selection.clone(), None);
        let filtered_values =
            arrow::compute::kernels::filter::filter(&unsigned_array, &selection).unwrap();
        let filtered_values = filtered_values.as_primitive::<UInt32Type>().clone();
        // Reconstruct lossy Date32 directly from filtered offsets
        let (_dt, values, nulls) = filtered_values.into_parts();
        let ref_v = chunk.reference_value;
        let days_values: ScalarBuffer<<Date32Type as ArrowPrimitiveType>::Native> =
            ScalarBuffer::from_iter(values.iter().enumerate().map(|(i, &off)| {
                if nulls.as_ref().is_some_and(|n| n.is_null(i)) {
                    0i32
                } else {
                    match chunk.part {
                        DatePart::Year => {
                            let y = ref_v + off as i32;
                            ymd_to_epoch_days(y, 1, 1)
                        }
                        DatePart::Month => {
                            let m = (ref_v + off as i32) as u32;
                            ymd_to_epoch_days(1970, m, 1)
                        }
                        DatePart::Day => {
                            let d = (ref_v + off as i32) as u32;
                            ymd_to_epoch_days(1970, 1, d)
                        }

                        _ => unreachable!("Stored field should be Year/Month/Day"),
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
    use crate::cache::DatePartSet;
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

    fn extract(field: DatePart, input: Vec<Option<i32>>) -> PrimitiveArray<Date32Type> {
        let arr = dates(&input);
        let liquid = LiquidPrimitiveArray::<Date32Type>::from_arrow_array(arr);
        let squeezed = SqueezedDate32Array::from_liquid_date32(&liquid, field);
        squeezed
            .to_component_date32(field)
            .expect("component should exist")
    }

    fn lossy(field: DatePart, input: Vec<Option<i32>>) -> PrimitiveArray<Date32Type> {
        let arr = dates(&input);
        let liquid = LiquidPrimitiveArray::<Date32Type>::from_arrow_array(arr);
        let squeezed = SqueezedDate32Array::from_liquid_date32(&liquid, field);
        squeezed.to_arrow_date32_lossy()
    }

    #[test]
    fn test_year_month_components() {
        let input = vec![
            Some(ymd_to_epoch_days(1970, 1, 1)),
            Some(ymd_to_epoch_days(1981, 12, 31)),
            None,
            Some(ymd_to_epoch_days(1999, 7, 4)),
        ];

        let arr = dates(&input);
        let liquid = LiquidPrimitiveArray::<Date32Type>::from_arrow_array(arr);
        let squeezed = SqueezedDate32Array::from_liquid_date32(&liquid, DatePartSet::YearMonth);

        assert_eq!(squeezed.field(), DatePartSet::YearMonth);
        let years = squeezed
            .to_component_date32(DatePart::Year)
            .expect("year chunk");
        let months = squeezed
            .to_component_date32(DatePart::Month)
            .expect("month chunk");
        assert!(squeezed.to_component_date32(DatePart::Day).is_none());

        let expected_years =
            PrimitiveArray::<Date32Type>::from(vec![Some(1970), Some(1981), None, Some(1999)]);
        let expected_months =
            PrimitiveArray::<Date32Type>::from(vec![Some(1), Some(12), None, Some(7)]);

        assert_prim_eq(years, expected_years);
        assert_prim_eq(months, expected_months);

        let quarters = squeezed
            .compute_derived(DatePart::Quarter)
            .expect("quarter derivation");
        let expected_quarters =
            PrimitiveArray::<Date32Type>::from(vec![Some(1), Some(4), None, Some(3)]);
        assert_prim_eq(quarters, expected_quarters);
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
            PrimitiveArray::<Date32Type>::from(vec![Some(1969), Some(1970), Some(1971), None]);
        assert_prim_eq(extract(DatePart::Year, input), expected);

        // MONTH
        let input = vec![
            Some(ymd_to_epoch_days(1970, 1, 31)),
            Some(ymd_to_epoch_days(1970, 2, 1)),
            Some(ymd_to_epoch_days(1970, 12, 31)),
            None,
        ];
        let expected = PrimitiveArray::<Date32Type>::from(vec![Some(1), Some(2), Some(12), None]);
        assert_prim_eq(extract(DatePart::Month, input), expected);

        // DAY
        let input = vec![
            Some(ymd_to_epoch_days(1970, 1, 1)),
            Some(ymd_to_epoch_days(1970, 1, 31)),
            Some(ymd_to_epoch_days(1970, 2, 1)),
            None,
        ];
        let expected = PrimitiveArray::<Date32Type>::from(vec![Some(1), Some(31), Some(1), None]);
        assert_prim_eq(extract(DatePart::Day, input), expected);
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
        assert_prim_eq(lossy(DatePart::Year, input), expected);

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
        assert_prim_eq(lossy(DatePart::Month, input), expected);

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
        assert_prim_eq(lossy(DatePart::Day, input), expected);
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

        for &field in &[DatePart::Year, DatePart::Month, DatePart::Day] {
            let comp1 = extract(field, input.clone());
            let lossy_dt = lossy(field, input.clone());
            let liquid2 = LiquidPrimitiveArray::<Date32Type>::from_arrow_array(lossy_dt);
            let comp2 = SqueezedDate32Array::from_liquid_date32(&liquid2, field)
                .to_component_date32(field)
                .expect("component should exist");
            assert_prim_eq(comp1, comp2);
        }
    }

    #[test]
    fn test_all_nulls_behavior() {
        let input = vec![None, None, None];

        for &field in &[DatePart::Year, DatePart::Month, DatePart::Day] {
            let comp = extract(field, input.clone());
            let expected_comp = PrimitiveArray::<Date32Type>::from(vec![None, None, None]);
            assert_prim_eq(comp, expected_comp);

            let lossy_dt = lossy(field, input.clone());
            let expected_dt = PrimitiveArray::<Date32Type>::from(vec![None, None, None]);
            assert_prim_eq(lossy_dt, expected_dt);
        }
    }

    #[test]
    fn test_month_day_components() {
        let input = vec![
            Some(ymd_to_epoch_days(1970, 1, 1)),
            Some(ymd_to_epoch_days(1981, 12, 31)),
            None,
            Some(ymd_to_epoch_days(1999, 7, 4)),
        ];

        let arr = dates(&input);
        let liquid = LiquidPrimitiveArray::<Date32Type>::from_arrow_array(arr);
        let squeezed = SqueezedDate32Array::from_liquid_date32(&liquid, DatePartSet::MonthDay);

        assert_eq!(squeezed.field(), DatePartSet::MonthDay);
        let months = squeezed
            .to_component_date32(DatePart::Month)
            .expect("month chunk");
        let days = squeezed
            .to_component_date32(DatePart::Day)
            .expect("day chunk");
        assert!(squeezed.to_component_date32(DatePart::Year).is_none());

        let expected_months =
            PrimitiveArray::<Date32Type>::from(vec![Some(1), Some(12), None, Some(7)]);
        let expected_days =
            PrimitiveArray::<Date32Type>::from(vec![Some(1), Some(31), None, Some(4)]);

        assert_prim_eq(months, expected_months);
        assert_prim_eq(days, expected_days);
    }

    #[test]
    fn test_year_day_components() {
        let input = vec![
            Some(ymd_to_epoch_days(1970, 1, 1)),
            Some(ymd_to_epoch_days(1981, 12, 31)),
            None,
            Some(ymd_to_epoch_days(1999, 7, 4)),
        ];

        let arr = dates(&input);
        let liquid = LiquidPrimitiveArray::<Date32Type>::from_arrow_array(arr);
        let squeezed = SqueezedDate32Array::from_liquid_date32(&liquid, DatePartSet::YearDay);

        assert_eq!(squeezed.field(), DatePartSet::YearDay);
        let years = squeezed
            .to_component_date32(DatePart::Year)
            .expect("year chunk");
        let days = squeezed
            .to_component_date32(DatePart::Day)
            .expect("day chunk");
        assert!(squeezed.to_component_date32(DatePart::Month).is_none());

        let expected_years =
            PrimitiveArray::<Date32Type>::from(vec![Some(1970), Some(1981), None, Some(1999)]);
        let expected_days =
            PrimitiveArray::<Date32Type>::from(vec![Some(1), Some(31), None, Some(4)]);

        assert_prim_eq(years, expected_years);
        assert_prim_eq(days, expected_days);
    }

    #[test]
    fn test_can_derive_field() {
        let input = vec![
            Some(ymd_to_epoch_days(1970, 3, 15)),
            Some(ymd_to_epoch_days(1981, 7, 4)),
            None,
        ];

        let arr = dates(&input);
        let liquid = LiquidPrimitiveArray::<Date32Type>::from_arrow_array(arr);

        // YearMonth can derive Quarter (from Month)
        let squeezed = SqueezedDate32Array::from_liquid_date32(&liquid, DatePartSet::YearMonth);
        assert!(squeezed.can_derive_field(DatePart::Year));
        assert!(squeezed.can_derive_field(DatePart::Month));
        assert!(squeezed.can_derive_field(DatePart::Quarter));
        assert!(!squeezed.can_derive_field(DatePart::Day));
        assert!(!squeezed.can_derive_field(DatePart::DayOfYear));

        // Month only can derive Quarter
        let squeezed = SqueezedDate32Array::from_liquid_date32(&liquid, DatePartSet::Month);
        assert!(squeezed.can_derive_field(DatePart::Month));
        assert!(squeezed.can_derive_field(DatePart::Quarter));
        assert!(!squeezed.can_derive_field(DatePart::Year));
        assert!(!squeezed.can_derive_field(DatePart::Day));

        // MonthDay can derive Quarter
        let squeezed = SqueezedDate32Array::from_liquid_date32(&liquid, DatePartSet::MonthDay);
        assert!(squeezed.can_derive_field(DatePart::Month));
        assert!(squeezed.can_derive_field(DatePart::Day));
        assert!(squeezed.can_derive_field(DatePart::Quarter));
        assert!(!squeezed.can_derive_field(DatePart::Year));
    }

    #[test]
    fn test_cannot_compute_derived_fields() {
        let input = vec![
            Some(ymd_to_epoch_days(1970, 3, 15)),
            Some(ymd_to_epoch_days(1981, 7, 4)),
            None,
        ];

        let arr = dates(&input);
        let liquid = LiquidPrimitiveArray::<Date32Type>::from_arrow_array(arr);

        // YearMonth cannot derive Day
        let squeezed = SqueezedDate32Array::from_liquid_date32(&liquid, DatePartSet::YearMonth);
        assert!(squeezed.compute_derived(DatePart::Day).is_none());
        assert!(squeezed.compute_derived(DatePart::Year).is_some());
        assert!(squeezed.compute_derived(DatePart::Month).is_some());

        // Month only cannot derive Year or Day
        let squeezed = SqueezedDate32Array::from_liquid_date32(&liquid, DatePartSet::Month);
        assert!(squeezed.compute_derived(DatePart::Year).is_none());
        assert!(squeezed.compute_derived(DatePart::Day).is_none());
        assert!(squeezed.compute_derived(DatePart::Quarter).is_some());
    }

    #[test]
    fn test_filter_multi_part() {
        let input = vec![
            Some(ymd_to_epoch_days(1970, 1, 1)),
            Some(ymd_to_epoch_days(1981, 12, 31)),
            Some(ymd_to_epoch_days(1999, 7, 4)),
            None,
        ];

        let arr = dates(&input);
        let liquid = LiquidPrimitiveArray::<Date32Type>::from_arrow_array(arr);
        let squeezed = SqueezedDate32Array::from_liquid_date32(&liquid, DatePartSet::YearMonth);

        // Filter: keep first and third elements
        let selection = BooleanBuffer::from(vec![true, false, true, false]);
        let filtered = squeezed.filter(&selection).expect("filter should succeed");
        let filtered_arr = filtered
            .as_any()
            .downcast_ref::<PrimitiveArray<Date32Type>>()
            .expect("should be Date32 array");

        assert_eq!(filtered_arr.len(), 2);
        // Lossy reconstruction uses primary component (Year), so should be (1970,1,1) and (1999,1,1)
        assert_eq!(filtered_arr.value(0), ymd_to_epoch_days(1970, 1, 1));
        assert_eq!(filtered_arr.value(1), ymd_to_epoch_days(1999, 1, 1));
    }

    #[test]
    fn test_memory_size_multi_part() {
        let input = vec![
            Some(ymd_to_epoch_days(1970, 1, 1)),
            Some(ymd_to_epoch_days(1981, 12, 31)),
            Some(ymd_to_epoch_days(1999, 7, 4)),
        ];

        let arr = dates(&input);
        let liquid = LiquidPrimitiveArray::<Date32Type>::from_arrow_array(arr);

        // Single part
        let single = SqueezedDate32Array::from_liquid_date32(&liquid, DatePartSet::Year);
        let single_size = single.get_array_memory_size();

        // Multi-part should be larger
        let multi = SqueezedDate32Array::from_liquid_date32(&liquid, DatePartSet::YearMonth);
        let multi_size = multi.get_array_memory_size();

        assert!(
            multi_size > single_size,
            "multi-part should use more memory"
        );
    }

    #[test]
    fn test_empty_array() {
        let input: Vec<Option<i32>> = vec![];

        let arr = dates(&input);
        let liquid = LiquidPrimitiveArray::<Date32Type>::from_arrow_array(arr);
        let squeezed = SqueezedDate32Array::from_liquid_date32(&liquid, DatePartSet::YearMonth);

        assert_eq!(squeezed.len(), 0);
        assert!(squeezed.is_empty());
    }

    #[test]
    fn test_all_nulls_multi_part() {
        let input = vec![None, None, None];

        let arr = dates(&input);
        let liquid = LiquidPrimitiveArray::<Date32Type>::from_arrow_array(arr);
        let squeezed = SqueezedDate32Array::from_liquid_date32(&liquid, DatePartSet::YearMonth);

        assert_eq!(squeezed.len(), 3);
        assert_eq!(squeezed.field(), DatePartSet::YearMonth);

        // Should still be able to get components (all null)
        let years = squeezed
            .to_component_date32(DatePart::Year)
            .expect("year chunk should exist");
        assert_eq!(years.null_count(), 3);
        assert_eq!(years.len(), 3);

        let months = squeezed
            .to_component_date32(DatePart::Month)
            .expect("month chunk should exist");
        assert_eq!(months.null_count(), 3);
        assert_eq!(months.len(), 3);
    }
}
