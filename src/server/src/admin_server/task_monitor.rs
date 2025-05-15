use std::ops::Deref;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use tokio_metrics::{RuntimeMonitor, TaskMonitor};

pub(crate) struct LiquidTaskMonitor {
    monitor: TaskMonitor,
    rt_monitor: RuntimeMonitor,
    cancellation_token: Arc<AtomicBool>,
    liquid_task_metrics: Arc<LiquidTaskMetrics>,
}

pub(crate) struct LiquidTaskMetrics {
    total_tasks_n: AtomicU64,
    total_slow_poll_n: AtomicU64,
    total_poll_n: AtomicU64,
    io_usage_n: AtomicU64,
}

#[derive(Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct LiquidTaskMetricsResponse {
    pub total_tasks_n: u64,
    pub total_slow_poll_n: u64,
    pub total_poll_n: u64,
    pub io_usage_n: u64,
}

impl LiquidTaskMonitor {
    pub fn new() -> Self {
        let handle = tokio::runtime::Handle::current();
        let rt_monitor = tokio_metrics::RuntimeMonitor::new(&handle);
        let liquid_task_metrics = LiquidTaskMetrics {
            total_tasks_n: AtomicU64::new(0),
            total_slow_poll_n: AtomicU64::new(0),
            total_poll_n: AtomicU64::new(0),
            io_usage_n: AtomicU64::new(0),
        };

        Self {
            rt_monitor,
            monitor: TaskMonitor::new(),
            // TODO: Use channel or CancellationToken from tokio-utils (https://docs.rs/tokio-util/latest/tokio_util/sync/struct.CancellationToken.html)
            cancellation_token: Arc::new(AtomicBool::new(false)),
            liquid_task_metrics: Arc::new(liquid_task_metrics),
        }
    }

    pub async fn run_collector(&self) {
        let monitor_interval = self.monitor.intervals();
        let rt_monitor_interval = self.rt_monitor.intervals();

        let cancellation_token = self.cancellation_token.clone();
        let liquid_task_metrics = self.liquid_task_metrics.clone();

        tokio::spawn(async move {
            fn update_high_mark(counter: &AtomicU64, new_value: u64) {
                let mut curr = counter.load(Ordering::Relaxed);
                loop {
                    if new_value <= curr {
                        break;
                    }

                    match counter.compare_exchange(curr, new_value, Ordering::Relaxed, Ordering::Relaxed) {
                        Ok(_) => break,
                        Err(actual) => curr = actual,
                    }
                }
            }

            let mut rt_stream = Box::pin(rt_monitor_interval);
            let mut metrics_stream = Box::pin(monitor_interval);

            loop {
                if cancellation_token.load(Ordering::Acquire) {
                    return;
                }

                tokio::select! {
                Some(rt_metrics) = rt_stream.next() => {
                    update_high_mark(&liquid_task_metrics.total_tasks_n, rt_metrics.live_tasks_count as u64);
                }

                Some(metrics) = metrics_stream.next() => {
                    update_high_mark(&liquid_task_metrics.total_slow_poll_n, metrics.total_slow_poll_count);
                    update_high_mark(&liquid_task_metrics.total_poll_n, metrics.total_poll_count);
                }

                _ = tokio::time::sleep(tokio::time::Duration::from_millis(500)) => {
                }
            }
            }
        });
    }

    pub async fn stop_collector(&self) {
        self.cancellation_token.store(true, Ordering::Release);
    }

    pub async fn get_metrics(&self) -> LiquidTaskMetricsResponse {
        LiquidTaskMetricsResponse {
            total_tasks_n: self.liquid_task_metrics.total_tasks_n.load(Ordering::Acquire),
            total_slow_poll_n: self.liquid_task_metrics.total_slow_poll_n.load(Ordering::Acquire),
            total_poll_n: self.liquid_task_metrics.total_poll_n.load(Ordering::Acquire),
            io_usage_n: self.liquid_task_metrics.io_usage_n.load(Ordering::Acquire),
        }
    }

    pub fn reset_counters(&self) {
        self.liquid_task_metrics.total_slow_poll_n.store(0, Ordering::Relaxed);
        self.liquid_task_metrics.total_poll_n.store(0, Ordering::Relaxed);
        self.liquid_task_metrics.io_usage_n.store(0, Ordering::Relaxed);
        self.liquid_task_metrics.total_tasks_n.store(0, Ordering::Relaxed);
        self.cancellation_token.store(false, Ordering::Relaxed);
    }
}
