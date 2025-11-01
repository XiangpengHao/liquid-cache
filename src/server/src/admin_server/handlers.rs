use std::{
    fs,
    path::{Path, PathBuf},
    sync::Arc,
};

use axum::{
    Json,
    extract::{Query, State},
};
use datafusion::{
    catalog::memory::DataSourceExec,
    common::{
        stats::Precision,
        tree_node::{TreeNode, TreeNodeRecursion},
    },
    datasource::physical_plan::FileScanConfig,
    physical_plan::ExecutionPlan,
};
use liquid_cache_common::rpc::ExecutionMetricsResponse;
use liquid_cache_parquet::LiquidParquetSource;
use log::info;
use serde::Serialize;
use uuid::Uuid;

use crate::{
    ColumnStatistics, ExecutionPlanWithStats, ExecutionStatsWithPlan, MetricValues, PlanInfo,
    SchemaField, Statistics,
};

use super::{
    AppState,
    models::{ApiResponse, ExecutionStats},
};

pub(crate) async fn shutdown_handler() -> Json<ApiResponse> {
    info!("Shutdown request received, shutting down server...");

    tokio::spawn(async {
        tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
        std::process::exit(0);
    });

    Json(ApiResponse {
        message: "Server shutting down...".to_string(),
        status: "success".to_string(),
    })
}

pub(crate) async fn reset_cache_handler(State(state): State<Arc<AppState>>) -> Json<ApiResponse> {
    info!("Resetting cache...");
    let cache = state.liquid_cache.cache();
    unsafe {
        cache.reset();
    }

    Json(ApiResponse {
        message: "Cache reset successfully".to_string(),
        status: "success".to_string(),
    })
}

#[derive(Serialize)]
pub(crate) struct ParquetCacheUsage {
    directory: String,
    file_count: usize,
    total_size_bytes: u64,
    status: String,
}

fn get_parquet_cache_usage_inner(cache_dir: &Path) -> ParquetCacheUsage {
    let mut file_count = 0;
    let mut total_size: u64 = 0;

    fn walk_dir(dir: &Path) -> Result<(usize, u64), std::io::Error> {
        let mut count = 0;
        let mut size = 0;

        if dir.exists() {
            for entry in fs::read_dir(dir)? {
                let entry = entry?;
                let path = entry.path();

                if path.is_file() {
                    count += 1;
                    let metadata = fs::metadata(&path)?;
                    size += metadata.len();
                } else if path.is_dir() {
                    let (sub_count, sub_size) = walk_dir(&path)?;
                    count += sub_count;
                    size += sub_size;
                }
            }
        }

        Ok((count, size))
    }

    if let Ok((count, size)) = walk_dir(cache_dir) {
        file_count = count;
        total_size = size;
    }

    ParquetCacheUsage {
        directory: cache_dir.to_string_lossy().to_string(),
        file_count,
        total_size_bytes: total_size,
        status: "success".to_string(),
    }
}

pub(crate) async fn get_parquet_cache_usage_handler(
    State(state): State<Arc<AppState>>,
) -> Json<ParquetCacheUsage> {
    info!("Getting parquet cache usage...");
    let cache_dir = state.liquid_cache.get_parquet_cache_dir();
    let usage = get_parquet_cache_usage_inner(cache_dir);
    Json(usage)
}

#[derive(Serialize)]
pub(crate) struct CacheInfo {
    batch_size: usize,
    max_cache_bytes: u64,
    memory_usage_bytes: u64,
    disk_usage_bytes: u64,
}

pub(crate) async fn get_cache_info_handler(State(state): State<Arc<AppState>>) -> Json<CacheInfo> {
    info!("Getting cache info...");
    let cache = state.liquid_cache.cache();
    let batch_size = cache.batch_size();
    let max_cache_bytes = cache.max_cache_bytes() as u64;
    let memory_usage_bytes = cache.memory_usage_bytes() as u64;
    let disk_usage_bytes = cache.disk_usage_bytes() as u64;
    Json(CacheInfo {
        batch_size,
        max_cache_bytes,
        memory_usage_bytes,
        disk_usage_bytes,
    })
}

#[derive(Serialize)]
pub(crate) struct SystemInfo {
    total_memory_bytes: u64,
    used_memory_bytes: u64,
    available_memory_bytes: u64,
    name: String,
    kernel: String,
    os: String,
    host_name: String,
    cpu_cores: usize,
    server_resident_memory_bytes: u64,
    server_virtual_memory_bytes: u64,
}

pub(crate) async fn get_system_info_handler(
    State(_state): State<Arc<AppState>>,
) -> Json<SystemInfo> {
    info!("Getting system info...");
    let mut sys = sysinfo::System::new_all();
    sys.refresh_all();
    let current_pid = sysinfo::get_current_pid().unwrap();
    let process = sys.process(current_pid).unwrap();
    let resident_memory = process.memory();
    let virtual_memory = process.virtual_memory();
    Json(SystemInfo {
        total_memory_bytes: sys.total_memory(),
        used_memory_bytes: sys.used_memory(),
        available_memory_bytes: sys.available_memory(),
        name: sysinfo::System::name().unwrap_or_default(),
        kernel: sysinfo::System::kernel_version().unwrap_or_default(),
        os: sysinfo::System::os_version().unwrap_or_default(),
        host_name: sysinfo::System::host_name().unwrap_or_default(),
        cpu_cores: sysinfo::System::physical_core_count().unwrap_or(0),
        server_resident_memory_bytes: resident_memory,
        server_virtual_memory_bytes: virtual_memory,
    })
}

#[derive(serde::Deserialize)]
pub(crate) struct TraceParams {
    path: String,
}

#[derive(serde::Deserialize)]
pub(crate) struct ExecutionMetricsParams {
    plan_id: String,
}

#[derive(serde::Deserialize)]
pub(crate) struct CacheStatsParams {
    path: String,
}

pub(crate) async fn start_trace_handler(State(state): State<Arc<AppState>>) -> Json<ApiResponse> {
    info!("Starting cache trace collection...");
    let cache = state.liquid_cache.cache();
    cache.enable_trace();

    Json(ApiResponse {
        message: "Cache trace collection started".to_string(),
        status: "success".to_string(),
    })
}

pub(crate) async fn stop_trace_handler(
    Query(params): Query<TraceParams>,
    State(state): State<Arc<AppState>>,
) -> Json<ApiResponse> {
    info!("Stopping cache trace collection...");
    let save_path = Path::new(&params.path);

    match save_trace_to_file(save_path, &state) {
        Ok(_) => Json(ApiResponse {
            message: format!(
                "Cache trace collection stopped, saved to {}",
                save_path.display()
            ),
            status: "success".to_string(),
        }),
        Err(e) => Json(ApiResponse {
            message: format!("Failed to save trace: {e}"),
            status: "error".to_string(),
        }),
    }
}

pub(crate) fn save_trace_to_file(
    save_dir: &Path,
    state: &AppState,
) -> Result<(), Box<dyn std::error::Error>> {
    let now = std::time::SystemTime::now();
    let datetime = now.duration_since(std::time::UNIX_EPOCH).unwrap();
    let minute = (datetime.as_secs() / 60) % 60;
    let second = datetime.as_secs() % 60;
    let trace_id = state
        .trace_id
        .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let filename = format!("cache-trace-id{trace_id:02}-{minute:02}-{second:03}.parquet",);

    // Ensure directory exists
    if !save_dir.exists() {
        fs::create_dir_all(save_dir)?;
    }

    let file_path = save_dir.join(filename);
    let cache = state.liquid_cache.cache();
    cache.disable_trace();
    cache.flush_trace(&file_path);
    Ok(())
}

pub(crate) async fn get_execution_metrics_handler(
    State(state): State<Arc<AppState>>,
    Query(params): Query<ExecutionMetricsParams>,
) -> Json<Option<ExecutionMetricsResponse>> {
    let Ok(uuid) = Uuid::parse_str(&params.plan_id) else {
        return Json(None);
    };
    let metrics = state.liquid_cache.inner().get_metrics(&uuid);
    Json(metrics)
}

pub(crate) fn get_cache_stats_inner(
    cache: &liquid_cache_parquet::LiquidCacheRef,
    save_dir: impl AsRef<Path>,
    state: &AppState,
) -> Result<PathBuf, Box<dyn std::error::Error>> {
    let now = std::time::SystemTime::now();
    let datetime = now.duration_since(std::time::UNIX_EPOCH).unwrap();
    let minute = (datetime.as_secs() / 60) % 60;
    let second = datetime.as_secs() % 60;
    let trace_id = state
        .stats_id
        .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let filename = format!("cache-stats-id{trace_id:02}-{minute:02}-{second:03}.parquet",);
    let file_path = save_dir.as_ref().join(filename);
    cache.write_stats(&file_path)?;
    Ok(file_path)
}

pub(crate) async fn get_cache_stats_handler(
    State(state): State<Arc<AppState>>,
    Query(params): Query<CacheStatsParams>,
) -> Json<ApiResponse> {
    let cache = state.liquid_cache.cache();
    match get_cache_stats_inner(cache, &params.path, &state) {
        Ok(file_path) => {
            info!("Cache stats saved to {}", file_path.display());
            Json(ApiResponse {
                message: format!("Cache stats saved to {}", file_path.display()),
                status: "success".to_string(),
            })
        }
        Err(e) => Json(ApiResponse {
            message: format!("Failed to get cache stats: {e}"),
            status: "error".to_string(),
        }),
    }
}

pub(crate) async fn start_flamegraph_handler(
    State(state): State<Arc<AppState>>,
) -> Json<ApiResponse> {
    info!("Starting flamegraph collection...");
    state.flamegraph.start();
    Json(ApiResponse {
        message: "Flamegraph collection started".to_string(),
        status: "success".to_string(),
    })
}

impl From<&Arc<dyn ExecutionPlan>> for ExecutionPlanWithStats {
    fn from(plan: &Arc<dyn ExecutionPlan>) -> Self {
        let metrics = plan.metrics().unwrap().aggregate_by_name();
        let mut metric_values = Vec::new();
        for metric in metrics.iter() {
            metric_values.push(MetricValues {
                name: metric.value().name().to_string(),
                value: metric.value().to_string(),
            });
        }

        let mut column_statistics = Vec::new();
        for (i, cs) in plan
            .partition_statistics(None)
            .unwrap()
            .column_statistics
            .iter()
            .enumerate()
        {
            let min = if cs.min_value != Precision::Absent {
                Some(cs.min_value.to_string())
            } else {
                None
            };
            let max = if cs.max_value != Precision::Absent {
                Some(cs.max_value.to_string())
            } else {
                None
            };
            let sum = if cs.sum_value != Precision::Absent {
                Some(cs.sum_value.to_string())
            } else {
                None
            };
            let distinct = if cs.distinct_count != Precision::Absent {
                Some(cs.distinct_count.to_string())
            } else {
                None
            };
            let null = if cs.null_count != Precision::Absent {
                Some(cs.null_count.to_string())
            } else {
                None
            };
            column_statistics.push(ColumnStatistics {
                name: format!("col_{i}"),
                null,
                min,
                max,
                sum,
                distinct_count: distinct,
            });
        }

        ExecutionPlanWithStats {
            name: plan.name().to_string(),
            schema: plan
                .schema()
                .fields()
                .iter()
                .map(|field| SchemaField {
                    name: field.name().to_string(),
                    data_type: field.data_type().to_string(),
                })
                .collect(),
            statistics: Statistics {
                num_rows: plan
                    .partition_statistics(None)
                    .unwrap()
                    .num_rows
                    .to_string(),
                total_byte_size: plan
                    .partition_statistics(None)
                    .unwrap()
                    .total_byte_size
                    .to_string(),
                column_statistics,
            },
            metrics: metric_values,
            children: plan
                .children()
                .iter()
                .map(|child| (*child).into())
                .collect(),
        }
    }
}

fn get_liquid_exec_info(plan: &Arc<dyn ExecutionPlan>) -> Option<String> {
    let mut rv = None;
    plan.apply(|node| {
        let Some(data_source) = node.as_any().downcast_ref::<DataSourceExec>() else {
            return Ok(TreeNodeRecursion::Continue);
        };
        let file_scan_config = data_source
            .data_source()
            .as_any()
            .downcast_ref::<FileScanConfig>()
            .expect("FileScanConfig not found");
        let Some(liquid_source) = file_scan_config
            .file_source()
            .as_any()
            .downcast_ref::<LiquidParquetSource>()
        else {
            return Ok(TreeNodeRecursion::Continue);
        };
        let predicate = liquid_source.predicate();

        rv = predicate.map(|v| v.to_string());
        Ok(TreeNodeRecursion::Stop)
    })
    .unwrap();
    rv
}

pub(crate) async fn get_execution_stats(
    State(state): State<Arc<AppState>>,
) -> Json<Vec<ExecutionStatsWithPlan>> {
    let execution_stats = state.liquid_cache.inner().get_execution_stats();
    let mut rv = Vec::new();
    for execution_stat in execution_stats {
        let mut plans = Vec::new();
        for plan_id in execution_stat.plan_ids.iter() {
            let uuid = Uuid::parse_str(plan_id).expect("Invalid plan ID");
            let plan = state
                .liquid_cache
                .inner()
                .get_plan(&uuid)
                .expect("Plan not found");
            let model_plan = ExecutionPlanWithStats::from(&plan.plan);
            let plan_info = PlanInfo {
                id: plan_id.to_string(),
                created_at: plan
                    .created_at
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_secs(),
                plan: model_plan,
                predicate: get_liquid_exec_info(&plan.plan),
            };
            plans.push(plan_info);
        }
        let execution_stats_with_plan = ExecutionStatsWithPlan {
            execution_stats: execution_stat,
            plans,
        };
        rv.push(execution_stats_with_plan);
    }
    Json(rv)
}

pub(crate) async fn stop_flamegraph_handler(
    State(state): State<Arc<AppState>>,
) -> Json<ApiResponse> {
    let svg_content = if let Ok(svg_content) = state.flamegraph.stop_to_string() {
        svg_content
    } else {
        return Json(ApiResponse {
            message: "Flamegraph not generated".to_string(),
            status: "error".to_string(),
        });
    };
    Json(ApiResponse {
        message: svg_content,
        status: "success".to_string(),
    })
}

pub(crate) async fn add_execution_stats_handler(
    State(state): State<Arc<AppState>>,
    Json(params): Json<ExecutionStats>,
) -> Json<ApiResponse> {
    let message = format!(
        "Execution stats added for execution {}",
        params.display_name
    );
    state.liquid_cache.inner().add_execution_stats(params);
    Json(ApiResponse {
        message,
        status: "success".to_string(),
    })
}

pub(crate) async fn start_disk_usage_monitor_handler(
    State(state): State<Arc<AppState>>,
) -> Json<ApiResponse> {
    state.disk_monitor.clone().start_recording();
    let message = "Successfully started disk usage monitoring";
    Json(ApiResponse {
        message: message.to_string(),
        status: "success".to_string(),
    })
}

pub(crate) async fn stop_disk_usage_monitor_handler(
    State(state): State<Arc<AppState>>,
) -> Json<ApiResponse> {
    state.disk_monitor.clone().stop_recording();
    let message = "Stopped disk usage monitoring";
    Json(ApiResponse {
        message: message.to_string(),
        status: "success".to_string(),
    })
}

#[cfg(test)]
mod tests {
    use std::{io::Write, path::PathBuf};

    use tempfile::tempdir;

    use super::*;

    #[test]
    fn test_get_parquet_cache_usage_inner() {
        let temp_dir = tempdir().unwrap();
        let temp_path = temp_dir.path();

        let file1_path = temp_path.join("file1.parquet");
        let file2_path = temp_path.join("file2.parquet");

        let subdir_path = temp_path.join("subdir");
        std::fs::create_dir(&subdir_path).unwrap();
        let file3_path = subdir_path.join("file3.parquet");

        let data1 = [1u8; 1000];
        let data2 = [2u8; 2000];
        let data3 = [3u8; 3000];

        let mut file1 = std::fs::File::create(&file1_path).unwrap();
        file1.write_all(&data1).unwrap();

        let mut file2 = std::fs::File::create(&file2_path).unwrap();
        file2.write_all(&data2).unwrap();

        let mut file3 = std::fs::File::create(&file3_path).unwrap();
        file3.write_all(&data3).unwrap();

        // Expected total size: 6000 bytes (1000 + 2000 + 3000)

        let result = get_parquet_cache_usage_inner(temp_path);
        assert_eq!(result.directory, temp_path.to_string_lossy().to_string());
        assert_eq!(result.file_count, 3);
        assert_eq!(result.total_size_bytes, 6000);
        assert_eq!(result.status, "success");
    }

    #[test]
    fn test_get_parquet_cache_usage_inner_empty_dir() {
        let temp_dir = tempdir().unwrap();
        let temp_path = temp_dir.path();
        let result = get_parquet_cache_usage_inner(temp_path);
        assert_eq!(result.file_count, 0);
        assert_eq!(result.total_size_bytes, 0);
    }

    #[test]
    fn test_get_parquet_cache_usage_inner_nonexistent_dir() {
        let nonexistent_path = PathBuf::from("/path/does/not/exist");
        let result = get_parquet_cache_usage_inner(&nonexistent_path);
        assert_eq!(result.file_count, 0);
        assert_eq!(result.total_size_bytes, 0);
    }
}
