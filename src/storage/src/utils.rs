use std::num::NonZero;

use arrow::{
    array::{
        ArrayAccessor, ArrayIter, BinaryViewArray, DictionaryArray, GenericByteArray,
        GenericByteDictionaryBuilder, PrimitiveArray, PrimitiveDictionaryBuilder, StringViewArray,
    },
    datatypes::{BinaryType, ByteArrayType, DecimalType, UInt16Type, Utf8Type},
};
use arrow_schema::DataType;

use crate::liquid_array::get_bit_width;

/// A wrapper around `DictionaryArray<UInt16Type>` that ensures the values are unique.
/// This is because we leverage the fact that the values are unique in the dictionary to short cut the
/// comparison process, i.e., return the index on first match.
/// If the values are not unique, we are screwed.
pub struct CheckedDictionaryArray {
    val: DictionaryArray<UInt16Type>,
}

impl CheckedDictionaryArray {
    pub fn new_checked(array: &DictionaryArray<UInt16Type>) -> Self {
        gc_dictionary_array(array)
    }

    pub fn from_byte_array<T: ByteArrayType>(array: &GenericByteArray<T>) -> Self {
        let iter = array.iter();
        byte_array_to_dict_array::<T, _>(iter)
    }

    pub fn from_string_view_array(array: &StringViewArray) -> Self {
        let iter = array.iter();
        byte_array_to_dict_array::<Utf8Type, _>(iter)
    }

    pub fn from_binary_view_array(array: &BinaryViewArray) -> Self {
        let iter = array.iter();
        byte_array_to_dict_array::<BinaryType, _>(iter)
    }

    pub fn from_decimal_array<T: DecimalType>(array: &PrimitiveArray<T>) -> Self {
        decimal_array_to_dict_array(array)
    }

    /// # Safety
    /// The caller must ensure that the values in the dictionary are unique.
    pub unsafe fn new_unchecked_i_know_what_i_am_doing(
        array: &DictionaryArray<UInt16Type>,
    ) -> Self {
        #[cfg(debug_assertions)]
        {
            let gc_ed = gc_dictionary_array(array).val;
            assert_eq!(
                gc_ed.values().len(),
                array.values().len(),
                "the input dictionary values are not unique"
            );
        }
        Self { val: array.clone() }
    }

    pub fn into_inner(self) -> DictionaryArray<UInt16Type> {
        self.val
    }

    pub fn as_ref(&self) -> &DictionaryArray<UInt16Type> {
        &self.val
    }

    pub fn bit_width_for_key(&self) -> NonZero<u8> {
        let distinct_count = self.as_ref().values().len();
        get_bit_width(distinct_count as u64)
    }
}

fn gc_dictionary_array(array: &DictionaryArray<UInt16Type>) -> CheckedDictionaryArray {
    let value_type = array.values().data_type();
    if let DataType::Binary = value_type {
        let typed = array
            .downcast_dict::<GenericByteArray<BinaryType>>()
            .unwrap();
        let iter = typed.into_iter();
        byte_array_to_dict_array::<BinaryType, _>(iter)
    } else if let DataType::Utf8 = value_type {
        let typed = array.downcast_dict::<GenericByteArray<Utf8Type>>().unwrap();
        let iter = typed.into_iter();
        byte_array_to_dict_array::<Utf8Type, _>(iter)
    } else {
        unreachable!("Unsupported dictionary type: {:?}", value_type);
    }
}

fn decimal_array_to_dict_array<T: DecimalType>(
    array: &PrimitiveArray<T>,
) -> CheckedDictionaryArray {
    let iter = array.iter();
    let mut builder =
        PrimitiveDictionaryBuilder::<UInt16Type, T>::with_capacity(array.len(), array.len());
    for s in iter {
        builder.append_option(s);
    }
    let dict = builder.finish();
    CheckedDictionaryArray { val: dict }
}

fn byte_array_to_dict_array<'a, T: ByteArrayType, I: ArrayAccessor<Item = &'a T::Native>>(
    input: ArrayIter<I>,
) -> CheckedDictionaryArray {
    let mut builder = GenericByteDictionaryBuilder::<UInt16Type, T>::with_capacity(
        input.size_hint().0,
        input.size_hint().0,
        input.size_hint().0,
    );
    for s in input {
        builder.append_option(s);
    }
    let dict = builder.finish();
    CheckedDictionaryArray { val: dict }
}

pub(crate) fn yield_now_if_shuttle() {
    #[cfg(all(feature = "shuttle", test))]
    shuttle::thread::yield_now();
}

#[cfg(all(feature = "shuttle", test))]
pub(crate) fn shuttle_test(test: impl Fn() + Send + Sync + 'static) {
    _ = tracing_subscriber::fmt()
        .with_ansi(true)
        .with_thread_names(false)
        .with_target(false)
        .try_init();

    let mut runner = shuttle::PortfolioRunner::new(true, Default::default());

    let available_cores = std::thread::available_parallelism().unwrap().get().min(4);

    for _i in 0..available_cores {
        runner.add(shuttle::scheduler::PctScheduler::new(10, 1_000));
    }
    runner.run(test);
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::{BinaryArray, DictionaryArray};
    use std::sync::Arc;

    fn create_test_dictionary(values: Vec<&[u8]>) -> DictionaryArray<UInt16Type> {
        let binary_array = BinaryArray::from_iter_values(values);
        DictionaryArray::new(vec![0u16, 1, 2, 3].into(), Arc::new(binary_array))
    }

    #[test]
    fn test_gc_behavior() {
        // Test duplicate removal
        let dup_dict = create_test_dictionary(vec![b"a", b"a", b"b", b"b"]);
        let checked = CheckedDictionaryArray::new_checked(&dup_dict);
        let dict_values = checked.as_ref().values();
        assert_eq!(dict_values.len(), 2);
        assert_eq!(
            dict_values
                .as_any()
                .downcast_ref::<BinaryArray>()
                .unwrap()
                .value(0),
            b"a"
        );
        assert_eq!(
            dict_values
                .as_any()
                .downcast_ref::<BinaryArray>()
                .unwrap()
                .value(1),
            b"b"
        );

        // Test already unique values
        let unique_dict = create_test_dictionary(vec![b"a", b"b", b"c", b"d"]);
        let checked_unique = CheckedDictionaryArray::new_checked(&unique_dict);
        assert_eq!(checked_unique.as_ref().values().len(), 4);
    }

    #[test]
    #[cfg(debug_assertions)]
    #[should_panic(expected = "the input dictionary values are not unique")]
    fn test_unchecked_duplicates_panic() {
        let dup_dict = create_test_dictionary(vec![b"a", b"a", b"b", b"b"]);
        unsafe {
            CheckedDictionaryArray::new_unchecked_i_know_what_i_am_doing(&dup_dict);
        }
    }
}
