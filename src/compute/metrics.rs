// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

use datafusion::{
    common::instant::Instant,
    physical_plan::metrics::{Count, ExecutionPlanMetricsSet, MetricBuilder, Time},
};

/// A timer that can be started and stopped.
pub struct StartableTime {
    pub(crate) metrics: Time,
    // use for record each part cost time, will eventually add into 'metrics'.
    pub(crate) start: Option<Instant>,
}

impl StartableTime {
    pub(crate) fn start(&mut self) {
        assert!(self.start.is_none());
        self.start = Some(Instant::now());
    }

    pub(crate) fn stop(&mut self) {
        if let Some(start) = self.start.take() {
            self.metrics.add_elapsed(start);
        }
    }
}

pub(crate) struct FlightTableMetrics {
    pub time_creating_client: StartableTime,
    pub time_getting_stream: StartableTime,
}

impl FlightTableMetrics {
    pub(crate) fn new(metrics: &ExecutionPlanMetricsSet, partition: usize) -> Self {
        Self {
            time_creating_client: StartableTime {
                metrics: MetricBuilder::new(metrics).subset_time("time_creating_client", partition),
                start: None,
            },
            time_getting_stream: StartableTime {
                metrics: MetricBuilder::new(metrics).subset_time("time_getting_stream", partition),
                start: None,
            },
        }
    }
}

pub(crate) struct FlightStreamMetrics {
    pub time_processing: StartableTime,
    pub time_reading_total: StartableTime,
    pub poll_count: Count,
    pub output_rows: Count,
    pub bytes_transferred: Count,
}

impl FlightStreamMetrics {
    pub(crate) fn new(metrics: &ExecutionPlanMetricsSet, partition: usize) -> Self {
        Self {
            time_processing: StartableTime {
                metrics: MetricBuilder::new(metrics).subset_time("time_processing", partition),
                start: None,
            },
            time_reading_total: StartableTime {
                metrics: MetricBuilder::new(metrics).subset_time("time_reading_total", partition),
                start: None,
            },
            output_rows: MetricBuilder::new(metrics).output_rows(partition),
            poll_count: MetricBuilder::new(metrics).counter("poll_count", partition),
            bytes_transferred: MetricBuilder::new(metrics).counter("bytes_transferred", partition),
        }
    }
}
