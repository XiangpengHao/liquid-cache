//! Execution metrics reported for a query plan.

#[derive(Clone, PartialEq, serde::Serialize, serde::Deserialize, Debug)]
pub struct ExecutionMetricsResponse {
    pub pushdown_eval_time: u64,
    pub cache_memory_usage: u64,
    pub liquid_cache_usage: u64,
}

impl ExecutionMetricsResponse {
    pub fn zero() -> Self {
        Self {
            pushdown_eval_time: 0,
            cache_memory_usage: 0,
            liquid_cache_usage: 0,
        }
    }
}
