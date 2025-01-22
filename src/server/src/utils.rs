use std::{
    pin::Pin,
    sync::Arc,
    task::{Context, Poll},
};

use arrow::array::{
    Array, RecordBatch, StringArray, builder::StringDictionaryBuilder, cast::AsArray,
    types::UInt16Type,
};
use datafusion::error::Result;
use futures::{Stream, ready};
use futures::{StreamExt, stream::BoxStream};

/// A stream that garbage collects the memory of the record batches.
/// Applies to DictionaryArray and StringViewArray where the data may not be compact.
/// Useful before sending the data over the network.
pub struct GcStream {
    inner: BoxStream<'static, Result<RecordBatch>>,
}

impl GcStream {
    pub fn new<S: Stream<Item = Result<RecordBatch>> + Send + 'static>(inner: S) -> Self {
        Self {
            inner: inner.boxed(),
        }
    }
}

impl Stream for GcStream {
    type Item = Result<RecordBatch>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let batch = ready!(self.inner.poll_next_unpin(cx));
        match batch {
            Some(Ok(batch)) => {
                let batch = gc_batch(batch);
                Poll::Ready(Some(Ok(batch)))
            }
            Some(Err(e)) => Poll::Ready(Some(Err(e))),
            None => Poll::Ready(None),
        }
    }
}

fn gc_batch(batch: RecordBatch) -> RecordBatch {
    let new_columns = batch.columns().iter().map(|column| {
        if let Some(dict_array) = column.as_dictionary_opt::<UInt16Type>() {
            if let Some(typed_dict_array) = dict_array.downcast_dict::<StringArray>() {
                let array_len = dict_array.len();
                let values_len = typed_dict_array.values().len();
                if values_len > array_len {
                    let mut gc_array = StringDictionaryBuilder::<UInt16Type>::with_capacity(
                        array_len, values_len, 1024,
                    );
                    for v in typed_dict_array.into_iter() {
                        gc_array.append_option(v);
                    }
                    let gc_array = gc_array.finish();
                    return Arc::new(gc_array) as _;
                }
            }
        }
        column.clone()
    });

    RecordBatch::try_new(batch.schema(), new_columns.collect()).unwrap()
}
