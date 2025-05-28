use serde::{Deserialize, Serialize};

/// Parameters for the set_execution_stats endpoint
#[derive(Deserialize, Serialize, Clone)]
pub struct ExecutionStats {
    /// Plan ID for the execution plan
    pub plan_ids: Vec<String>,
    /// Display name for the execution plan
    pub display_name: String,
    /// Flamegraph SVG for the execution plan
    pub flamegraph_svg: Option<String>,
    /// Network traffic bytes for the execution plan
    pub network_traffic_bytes: u64,
    /// Execution time in milliseconds
    pub execution_time_ms: u64,
    /// User input SQL
    pub user_sql: String,
}

/// Execution stats with plan
#[derive(Serialize)]
pub struct ExecutionStatsWithPlan {
    /// Execution stats
    pub execution_stats: ExecutionStats,
    /// Plan info
    pub plans: Vec<PlanInfo>,
}

/// Response for the admin server
#[derive(Serialize, Deserialize)]
pub struct ApiResponse {
    /// Message for the response
    pub message: String,
    /// Status for the response
    pub status: String,
}

/// Schema field
#[derive(Serialize)]
pub struct SchemaField {
    /// Field name
    pub name: String,
    /// Field data type
    pub data_type: String,
}

/// Column statistics
#[derive(Serialize)]
pub struct ColumnStatistics {
    /// Column name
    pub name: String,
    /// Null count
    pub null: Option<String>,
    /// Max value
    pub max: Option<String>,
    /// Min value
    pub min: Option<String>,
    /// Sum value
    pub sum: Option<String>,
    /// Distinct count
    pub distinct_count: Option<String>,
}

/// Statistics
#[derive(Serialize)]
pub struct Statistics {
    /// Number of rows
    pub num_rows: String,
    /// Total byte size
    pub total_byte_size: String,
    /// Column statistics
    pub column_statistics: Vec<ColumnStatistics>,
}

/// Metric
#[derive(Serialize)]
pub struct MetricValues {
    /// Metric name
    pub name: String,
    /// Metric value
    pub value: String,
}

/// Execution plan with stats
#[derive(Serialize)]
pub struct ExecutionPlanWithStats {
    /// Execution plan name
    pub name: String,
    /// Schema fields
    pub schema: Vec<SchemaField>,
    /// Statistics
    pub statistics: Statistics,
    /// Metrics
    pub metrics: Vec<MetricValues>,
    /// Children
    pub children: Vec<ExecutionPlanWithStats>,
}

/// Plan info
#[derive(Serialize)]
pub struct PlanInfo {
    /// Created at
    pub created_at: u64,
    /// Execution plan
    pub plan: ExecutionPlanWithStats,
    /// ID
    pub id: String,
    /// Predicate
    pub predicate: Option<String>,
}
