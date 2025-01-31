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

use crate::StatsCollector;

/// A stream that finalizes the record batches.
/// It currently do two things:
/// 1. Gc the record batches, especially for arrays after filtering.
/// 2. Collect stats for the execution plan.
pub struct FinalStream {
    inner: BoxStream<'static, Result<RecordBatch>>,
    stats_collector: Vec<Arc<dyn StatsCollector>>,
}

impl FinalStream {
    pub fn new<S: Stream<Item = Result<RecordBatch>> + Send + 'static>(
        inner: S,
        mut stats_collector: Vec<Arc<dyn StatsCollector>>,
    ) -> Self {
        for collector in stats_collector.iter_mut() {
            collector.start();
        }
        Self {
            inner: inner.boxed(),
            stats_collector,
        }
    }
}

impl Drop for FinalStream {
    fn drop(&mut self) {
        for collector in self.stats_collector.iter_mut() {
            collector.stop();
        }
    }
}

impl Stream for FinalStream {
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
