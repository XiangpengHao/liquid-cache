use arrow::{array::RecordBatch, compute::concat_batches};
use datafusion::error::Result;
use futures::{Stream, ready};
use futures::{StreamExt, stream::BoxStream};
use std::{
    pin::Pin,
    task::{Context, Poll},
};

/// A stream that finalizes the record batches.
/// It currently do two things:
/// 1. Gc the record batches, especially for arrays after filtering.
/// 2. Merge small batches into a large one.
pub struct FinalStream {
    inner: BoxStream<'static, Result<RecordBatch>>,
    target_batch_size: usize,
    buffered_batches: Vec<RecordBatch>,
    current_buffered_rows: usize,
    span: fastrace::Span,
}

impl FinalStream {
    pub fn new<S: Stream<Item = Result<RecordBatch>> + Send + 'static>(
        inner: S,
        target_batch_size: usize,
        span: fastrace::Span,
    ) -> Self {
        Self {
            inner: inner.boxed(),
            target_batch_size,
            buffered_batches: Vec::new(),
            current_buffered_rows: 0,
            span,
        }
    }
}

impl Stream for FinalStream {
    type Item = Result<RecordBatch>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let this = &mut *self;
        let _guard = this.span.set_local_parent();
        loop {
            let threshold = (this.target_batch_size * 3) / 4;
            if this.current_buffered_rows > threshold {
                this.current_buffered_rows = 0;
                let batches = std::mem::take(&mut this.buffered_batches);
                let schema = batches[0].schema();
                let result = concat_batches(&schema, batches.iter());
                return Poll::Ready(Some(Ok(result?)));
            }

            match ready!(this.inner.poll_next_unpin(cx)).transpose()? {
                Some(batch) => {
                    let num_rows = batch.num_rows();
                    this.current_buffered_rows += num_rows;
                    this.buffered_batches.push(batch);
                }
                None => {
                    if this.buffered_batches.is_empty() {
                        return Poll::Ready(None);
                    }
                    this.current_buffered_rows = 0;
                    let batches = std::mem::take(&mut this.buffered_batches);
                    let schema = batches[0].schema();
                    let result = concat_batches(&schema, batches.iter());
                    return Poll::Ready(Some(Ok(result?)));
                }
            }
        }
    }
}
