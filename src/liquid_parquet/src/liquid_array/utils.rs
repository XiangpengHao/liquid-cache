use arrow::{
    array::{
        ArrayAccessor, ArrayIter, DictionaryArray, GenericByteArray, GenericByteDictionaryBuilder,
        StringViewArray,
    },
    datatypes::{BinaryType, ByteArrayType, UInt16Type, Utf8Type},
};
use arrow_schema::DataType;

/// A wrapper around `DictionaryArray<UInt16Type>` that ensures the values are unique.
/// This is because we leverage the fact that the values are unique in the dictionary to short cut the
/// comparison process, i.e., return the index on first match.
/// If the values are not unique, we are screwed.
pub(crate) struct CheckedDictionaryArray {
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

fn byte_array_to_dict_array<'a, T: ByteArrayType, I: ArrayAccessor<Item = &'a T::Native>>(
    input: ArrayIter<I>,
) -> CheckedDictionaryArray {
    let mut builder = GenericByteDictionaryBuilder::<UInt16Type, T>::new();
    for s in input {
        builder.append_option(s);
    }
    let dict = builder.finish();
    CheckedDictionaryArray { val: dict }
}
