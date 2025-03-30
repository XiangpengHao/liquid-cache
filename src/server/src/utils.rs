use crate::StatsCollector;
use arrow::{array::RecordBatch, compute::concat_batches};
use datafusion::{error::Result, physical_plan::ExecutionPlan};
use fastrace::Span;
use futures::{Stream, ready};
use futures::{StreamExt, stream::BoxStream};
use std::{
    pin::Pin,
    sync::Arc,
    task::{Context, Poll},
};

/// A stream that finalizes the record batches.
/// It currently do two things:
/// 1. Gc the record batches, especially for arrays after filtering.
/// 2. Collect stats for the execution plan.
/// 3. Merge small batches into a large one.
pub struct FinalStream {
    inner: BoxStream<'static, Result<RecordBatch>>,
    stats_collector: Vec<Arc<dyn StatsCollector>>,
    target_batch_size: usize,
    buffered_batches: Vec<RecordBatch>,
    current_buffered_rows: usize,
    partition: usize,
    execution_plan: Arc<dyn ExecutionPlan>,
    span: fastrace::Span,
}

impl FinalStream {
    #[fastrace::trace]
    pub fn new<S: Stream<Item = Result<RecordBatch>> + Send + 'static>(
        inner: S,
        mut stats_collector: Vec<Arc<dyn StatsCollector>>,
        target_batch_size: usize,
        partition: usize,
        execution_plan: Arc<dyn ExecutionPlan>,
    ) -> Self {
        for collector in stats_collector.iter_mut() {
            collector.start(partition, &execution_plan);
        }
        let span = fastrace::Span::enter_with_local_parent("final_stream");

        Self {
            inner: inner.boxed(),
            stats_collector,
            target_batch_size,
            buffered_batches: Vec::new(),
            current_buffered_rows: 0,
            partition,
            execution_plan,
            span,
        }
    }
}

impl Drop for FinalStream {
    fn drop(&mut self) {
        for collector in self.stats_collector.iter_mut() {
            collector.stop(self.partition, &self.execution_plan);
        }
    }
}

impl Stream for FinalStream {
    type Item = Result<RecordBatch>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let this = &mut *self;
        let span = Span::enter_with_parent("final_stream_poll_next", &this.span);
        let _guard = span.set_local_parent();
        loop {
            let threshold = (this.target_batch_size * 3) / 4;
            if this.current_buffered_rows > threshold {
                let batches = std::mem::take(&mut this.buffered_batches);
                this.current_buffered_rows = 0;
                let schema = batches[0].schema();
                let result = concat_batches(&schema, batches.iter()).unwrap();
                return Poll::Ready(Some(Ok(result)));
            }

            match ready!(this.inner.poll_next_unpin(cx)) {
                Some(Ok(batch)) => {
                    let num_rows = batch.num_rows();
                    this.current_buffered_rows += num_rows;
                    this.buffered_batches.push(batch);
                }
                Some(Err(e)) => {
                    panic!("Poll next batch error: {:?}", e);
                }
                None => {
                    if this.buffered_batches.is_empty() {
                        return Poll::Ready(None);
                    }
                    let batches = std::mem::take(&mut this.buffered_batches);
                    this.current_buffered_rows = 0;
                    let schema = batches[0].schema();
                    let result = concat_batches(&schema, batches.iter()).unwrap();
                    return Poll::Ready(Some(Ok(result)));
                }
            }
        }
    }
}
